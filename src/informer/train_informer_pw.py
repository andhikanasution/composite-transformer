#!/usr/bin/env python3
"""
train_informer_pointwise.py

Pointwise Informer training: predict the full stress sequence from exogenous inputs
(6 strains + θ + 4 LPs). Uses padded sequences with masks from CompositeStressDataset.

Key points
----------
- Encoder/decoder both consume exogenous inputs (no lagged stress channels here).
- Decoder receives a zero start token (length LABEL_LEN) plus known exogenous inputs.
- Loss is masked MSE over real (un-padded) timesteps.
- Reports full-sequence and last-step metrics (in physical units via saved scaler).
- Logs wall-clock + RSS memory and persists scalers alongside best weights.
"""

from __future__ import annotations

import os
import time
import json
import psutil
from typing import Tuple, List

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score

from informer.models.model import Informer
from src.dataloader import make_train_val_loaders


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[float], List[float]]:
    """RMSE and R^2 per channel for arrays with shape [N, C]."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2 = [float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    return rmse, r2


def main() -> None:
    # ----------------- device & hyperparameters -----------------
    DEVICE = select_device()
    print(f"Using device: {DEVICE}")

    BATCH_SIZE = 32
    MAX_EPOCHS = 10
    LR = 1e-4
    SEQ_LEN = 200
    LABEL_LEN = 1           # decoder start token length
    PRED_LEN = SEQ_LEN      # predict full sequence (pointwise mapping)

    INPUT_CSV = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/"
        "2. Data Science MSc/Modules/Data Science Project/"
        "composite_stress_prediction/data/IM78552_DATABASEInput.csv"
    )
    DATA_DIR = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/"
        "2. Data Science MSc/Modules/Data Science Project/"
        "composite_stress_prediction/data/_CSV"
    )
    MODEL_DIR = "models/informer_pointwise_v3"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ----------------- DataLoaders (no lagged stress) -----------------
    train_loader, val_loader, input_scaler, target_scaler = make_train_val_loaders(
        INPUT_CSV,
        DATA_DIR,
        max_seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        split_ratio=0.8,
        seed=42,
        use_lagged_stress=False,
    )

    # ----------------- Model wrapper -----------------
    class InformerForPointwise(nn.Module):
        """
        Informer configured to map exogenous inputs → stress over the full sequence.

        Encoder input: x_enc = exogenous [B, T, enc_in]
        Decoder input: x_dec = [zeros LABEL_LEN; exogenous future] [B, LABEL_LEN+T, dec_in]
        Output:        [B, T, c_out]
        """
        def __init__(self, enc_in: int, dec_in: int, c_out: int) -> None:
            super().__init__()
            self.dec_in = int(dec_in)
            self.seq_len = SEQ_LEN
            self.label_len = LABEL_LEN
            self.pred_len = PRED_LEN
            self.net = Informer(
                enc_in=enc_in,
                dec_in=dec_in,
                c_out=c_out,
                seq_len=self.seq_len,
                label_len=self.label_len,
                out_len=self.pred_len,
                factor=5,
                d_model=128,
                n_heads=4,
                e_layers=3,
                d_layers=2,
                d_ff=512,
                dropout=0.1,
                attn="prob",
                embed="fixed",
                freq="s",
                activation="gelu",
                output_attention=False,
                distil=True,
                mix=True,
            )

        def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x_enc: [B, T, enc_in] exogenous inputs

            Returns:
                pred:  [B, T, c_out] stress predictions
            """
            B = int(x_enc.shape[0])

            # time marks are zeros (embed='fixed')
            x_mark_enc = x_enc.new_zeros((B, self.seq_len, 5))
            x_mark_dec = x_enc.new_zeros((B, self.label_len + self.pred_len, 5))

            # decoder input: [zero start token ; known exogenous for future]
            x_dec = x_enc.new_zeros((B, self.label_len + self.pred_len, self.dec_in))
            x_dec[:, self.label_len:, :] = x_enc  # copy encoder exogenous into decoder future

            return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len(=T), c_out]

    # Instantiate
    enc_in = train_loader.dataset.inputs[0].shape[1]  # 11 = 6 strains + θ + 4 LPs
    dec_in = enc_in
    model = InformerForPointwise(enc_in, dec_in, c_out=6).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # ----------------- bookkeeping -----------------
    best_val_loss = float("inf")
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss
    t_start = time.time()

    # ----------------- training -----------------
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x, pad_mask, times, y in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]"):
            x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            pred = model(x)  # [B, T, C]
            Bsz, T, C = pred.shape

            # masked MSE over real timesteps
            pred_flat = pred.view(-1, C)
            y_flat = y.view(-1, C)
            mask_flat = pad_mask.view(-1, 1).expand(-1, C)
            loss = ((pred_flat - y_flat) ** 2)[mask_flat].mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += float(loss.item()) * Bsz

        train_loss /= len(train_loader.dataset)

        # ----------------- validation -----------------
        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []
        last_preds, last_trues = [], []

        with torch.no_grad():
            for x, pad_mask, times, y in tqdm(val_loader, desc=f"Epoch {epoch} [VAL]"):
                x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
                pred = model(x)  # [B, T, C]

                Bsz, T, C = pred.shape
                pred_flat = pred.reshape(-1, C)
                y_flat = y[:, :T, :].reshape(-1, C)
                mask_flat = pad_mask[:, :T].reshape(-1, 1).expand(-1, C)
                loss = ((pred_flat - y_flat) ** 2)[mask_flat].mean()
                val_loss += float(loss.item()) * Bsz

                # collect for metrics in physical units
                pred_np = pred.detach().cpu().numpy()
                y_np = y[:, :T, :].detach().cpu().numpy()
                mask_np = pad_mask[:, :T].detach().cpu().numpy()

                # full sequence (keep only real timesteps)
                real_idx = mask_np.reshape(-1).astype(bool)
                all_preds.append(pred_np.reshape(-1, C)[real_idx])
                all_trues.append(y_np.reshape(-1, C)[real_idx])

                # last step metrics (only samples with real last timestep)
                valid_final = mask_np[:, -1].astype(bool)
                if valid_final.any():
                    last_preds.append(pred_np[valid_final, -1, :])
                    last_trues.append(y_np[valid_final, -1, :])

        val_loss /= len(val_loader.dataset)

        # metrics
        all_preds = np.vstack(all_preds)
        all_trues = np.vstack(all_trues)
        last_preds = np.vstack(last_preds) if last_preds else np.zeros((0, 6))
        last_trues = np.vstack(last_trues) if last_trues else np.zeros((0, 6))

        all_preds_phys = target_scaler.inverse_transform(all_preds)
        all_trues_phys = target_scaler.inverse_transform(all_trues)
        rmse_full, r2_full = compute_metrics(all_trues_phys, all_preds_phys)

        if last_preds.size:
            last_preds_phys = target_scaler.inverse_transform(last_preds)
            last_trues_phys = target_scaler.inverse_transform(last_trues)
            rmse_last, r2_last = compute_metrics(last_trues_phys, last_preds_phys)
        else:
            rmse_last, r2_last = ([0.0] * 6, [0.0] * 6)

        print(f"\nEpoch {epoch}: Train {train_loss:.4f}  Val {val_loss:.4f}")
        print(f"  Full-seq RMSE: {[f'{r:.3f}' for r in rmse_full]}")
        print(f"  Full-seq R^2 : {[f'{r:.3f}' for r in r2_full]}")
        print(f"  Last-step RMSE: {[f'{r:.3f}' for r in rmse_last]}")
        print(f"  Last-step R^2 : {[f'{r:.3f}' for r in r2_last]}")

        # save best by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_pointwise.pt"))
            print("Saved new best model.")

    # ----------------- resources & scalers -----------------
    t_end = time.time()
    mem_after = proc.memory_info().rss
    resources = {
        "train_time_s": round(t_end - t_start, 2),
        "train_mem_increase_mb": round((mem_after - mem_before) / 1024**2, 2),
    }
    with open(os.path.join(MODEL_DIR, "resource_log.json"), "w") as f:
        json.dump(resources, f, indent=2)

    with open(os.path.join(MODEL_DIR, "scalers.json"), "w") as f:
        json.dump({"input": input_scaler.to_dict(), "target": target_scaler.to_dict()}, f, indent=2)

    print("Resources:", resources)


if __name__ == "__main__":
    main()
