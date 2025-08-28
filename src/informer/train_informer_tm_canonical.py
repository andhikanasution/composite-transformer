#!/usr/bin/env python3
"""
train_informer_tm_canonical.py

Canonical Informer training with windowed (encoder/decoder) inputs:
- Encoder: [past y(6); past exo(11)] → 17 channels
- Decoder: [last LABEL_LEN y(6); exo(11)] with future y zeros → 17 channels
- Windowed dataset (no padding/masks); masked losses unnecessary.
"""

from __future__ import annotations

import os
import time
import json
import psutil
from typing import Tuple, List

import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score
import contextlib

from informer.models.model import Informer
from informer.models.attn import TriangularCausalMask
from src.dataloader import make_informer_window_loaders


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[float], List[float]]:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2 = [float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    return rmse, r2


def main() -> None:
    DEVICE = select_device()
    print(f"Using device: {DEVICE}")

    def amp_context():
        if DEVICE.type == "cuda":
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    # ---------------- hyperparameters ----------------
    BATCH, EPOCHS, LR = 32, 30, 1e-4
    SEQ_LEN = 96
    LABEL_LEN = 48
    PRED_LEN = 96
    STRIDE = 96

    # data paths
    INPUT_CSV = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/2. Data Science MSc/Modules/"
        "Data Science Project/composite_stress_prediction/data/IM78552_DATABASEInput.csv"
    )
    DATA_DIR = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/2. Data Science MSc/Modules/"
        "Data Science Project/composite_stress_prediction/data/_CSV"
    )
    MODEL_DIR = "models/informer_tm_canonical_v2"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -------- loaders: windowed, no padding/masks --------
    train_loader, val_loader, exo_scaler, target_scaler = make_informer_window_loaders(
        INPUT_CSV,
        DATA_DIR,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        stride=STRIDE,
        batch_size=BATCH,
        split_ratio=0.8,
        seed=42,
        scale=True,
    )

    # ---------------- model wrapper ----------------
    class InformerCanonicalTM(nn.Module):
        """
        Standard Informer wrapper.

        Forward signature mirrors the Informer API:
          forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, pred_len, c_out]
        """
        def __init__(self, enc_in: int = 17, dec_in: int = 17, c_out: int = 6,
                     seq_len: int = SEQ_LEN, label_len: int = LABEL_LEN, pred_len: int = PRED_LEN) -> None:
            super().__init__()
            self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
            self.dec_in = dec_in
            self.net = Informer(
                enc_in=enc_in,
                dec_in=dec_in,
                c_out=c_out,
                seq_len=seq_len,
                label_len=label_len,
                out_len=pred_len,
                factor=5,
                d_model=192,
                n_heads=3,
                e_layers=2,
                d_layers=1,
                d_ff=768,
                dropout=0.1,
                attn="full",
                distil=True,
                embed="fixed",
                freq="s",
                activation="gelu",
                output_attention=False,
                mix=True,
            )

        def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                    x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:
            B = x_enc.size(0)
            tdec = x_dec.size(1)
            dec_self_mask = TriangularCausalMask(B, tdec, device=x_enc.device)
            return self.net(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=dec_self_mask, dec_enc_mask=None
            )

    # -------- instantiate --------
    enc_in = 17
    dec_in = 17
    c_out = 6
    model = InformerCanonicalTM(enc_in, dec_in, c_out).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val = float("inf")
    proc = psutil.Process(os.getpid())
    mem_before, t0 = proc.memory_info().rss, time.time()

    # ---------------- training ----------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x_enc, x_dec, xme, xmd, y_true in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]"):
            x_enc, x_dec, xme, xmd, y_true = [t.to(DEVICE) for t in (x_enc, x_dec, xme, xmd, y_true)]
            optimizer.zero_grad(set_to_none=True)

            with amp_context():
                pred = model(x_enc, xme, x_dec, xmd)  # [B, PRED_LEN, 6]
                # data term
                mse = ((pred - y_true) ** 2).mean()
                # smoothness (finite differences vs ground truth)
                d_pred = pred[:, 1:] - pred[:, :-1]
                d_true = y_true[:, 1:] - y_true[:, :-1]
                smooth = ((d_pred - d_true) ** 2).mean()
                loss = mse + 0.05 * smooth

            if DEVICE.type == "cuda":
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += float(loss.item()) * x_enc.size(0)

        train_loss /= len(train_loader.dataset)

        # ------------- validation -------------
        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x_enc, x_dec, xme, xmd, y_true in tqdm(val_loader, desc=f"Epoch {epoch} [VAL]"):
                x_enc, x_dec, xme, xmd, y_true = [t.to(DEVICE) for t in (x_enc, x_dec, xme, xmd, y_true)]
                with amp_context():
                    pred = model(x_enc, xme, x_dec, xmd)  # [B, PRED_LEN, 6]
                mse = ((pred - y_true) ** 2).mean()
                d_pred = pred[:, 1:] - pred[:, :-1]
                d_true = y_true[:, 1:] - y_true[:, :-1]
                smooth = ((d_pred - d_true) ** 2).mean()
                loss = mse + 0.05 * smooth
                val_loss += float(loss.item()) * x_enc.size(0)

                all_preds.append(pred.reshape(-1, pred.size(-1)).detach().cpu().numpy())
                all_trues.append(y_true.reshape(-1, y_true.size(-1)).detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_trues = np.vstack(all_trues)
        phys_preds = target_scaler.inverse_transform(all_preds)
        phys_trues = target_scaler.inverse_transform(all_trues)
        rmse, r2 = compute_metrics(phys_trues, phys_preds)

        print(f"\nEpoch {epoch}  Train {train_loss:.4f}  Val {val_loss:.4f}")
        print(f"  Val RMSE: {[f'{v:.3f}' for v in rmse]}")
        print(f"  Val R^2 : {[f'{v:.3f}' for v in r2]}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_tm.pt"))
            print("Saved new best model.")

    # -------- persist scalers & resources --------
    with open(os.path.join(MODEL_DIR, "scalers.json"), "w") as f:
        json.dump({"input": exo_scaler.to_dict(), "target": target_scaler.to_dict()}, f, indent=2)

    t1, mem_after = time.time(), proc.memory_info().rss
    with open(os.path.join(MODEL_DIR, "resources.json"), "w") as f:
        json.dump({
            "train_time_s": round(t1 - t0, 2),
            "train_mem_increase_mb": round((mem_after - mem_before) / 1024**2, 2),
        }, f, indent=2)


if __name__ == "__main__":
    main()
