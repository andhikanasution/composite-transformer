#!/usr/bin/env python3
"""
train_informer_tm.py

Time-marching Informer training with scheduled sampling.

Design
------
- Train loader includes lagged stress channels (teacher forcing).
- Validation simulates inference: no lagged channels; multi-refinement loop builds lag from predictions.
- Loss = SmoothL1 (data) + λ * smoothness on finite differences.
- Tracks grad norm, roughness, RMSE/R^2, and logs resources; persists best weights and scalers.
"""

from __future__ import annotations

import os
import time
import json
import psutil
from typing import Dict, List, Tuple

import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from informer.models.model import Informer
from src.dataloader import make_train_val_loaders, get_dataloader
from informer.models.attn import TriangularCausalMask


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_history() -> Dict[str, list]:
    return {
        "epoch": [],
        "train_loss": [],
        "val_infer_loss": [],
        "grad_norm": [],
        "roughness": [],
        "rmse": [],
        "r2": [],
    }


def save_history_json(history: Dict[str, list], path: str) -> None:
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def plot_training_curves(history: Dict[str, list], out_dir: str) -> None:
    if not history["epoch"]:
        return
    ep = history["epoch"]

    plt.figure(figsize=(6, 4))
    plt.plot(ep, history["train_loss"], label="Train")
    plt.plot(ep, history["val_infer_loss"], label="Val-Infer")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_loss.png"), dpi=200); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, history["grad_norm"])
    plt.xlabel("Epoch"); plt.ylabel("Grad L2 norm"); plt.title("Gradient norm")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_gradnorm.png"), dpi=200); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, history["roughness"])
    plt.xlabel("Epoch"); plt.ylabel("Mean squared diff"); plt.title("Prediction roughness")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_roughness.png"), dpi=200); plt.close()

    r2 = np.array(history["r2"])
    if r2.ndim == 2 and r2.shape[1] == 6:
        plt.figure(figsize=(7, 5))
        for i in range(6):
            plt.plot(ep, r2[:, i], label=f"S{i+1}")
        plt.ylim(-0.5, 1.0)
        plt.xlabel("Epoch"); plt.ylabel("R^2"); plt.title("R^2 by component")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_r2_by_comp.png"), dpi=200); plt.close()

    rmse = np.array(history["rmse"])
    if rmse.ndim == 2 and rmse.shape[1] == 6:
        plt.figure(figsize=(7, 5))
        for i in range(6):
            plt.plot(ep, rmse[:, i], label=f"S{i+1}")
        plt.xlabel("Epoch"); plt.ylabel("RMSE (phys units)"); plt.title("RMSE by component")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_rmse_by_comp.png"), dpi=200); plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[float], List[float]]:
    """Per-channel RMSE and R^2 for [N, C] arrays."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2 = [float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    return rmse, r2


class SimpleMask:
    """Wrap a [B,1,Lq,Lk] boolean tensor with a .mask attribute (Informer expects this)."""
    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask


def main() -> None:
    DEVICE = select_device()
    print(f"Using device: {DEVICE}")

    BATCH = 32
    EPOCHS = 30
    LR = 1e-4
    SEQ_LEN = 200
    LABEL_LEN = 1                 # start token is y[:, 0]
    PRED_LEN = SEQ_LEN - LABEL_LEN
    N_REFINE = 8                  # inference refinements during validation

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
    MODEL_DIR = "models/informer_timemarching_v3"
    os.makedirs(MODEL_DIR, exist_ok=True)

    history = init_history()

    # --- Train loader: lagged stress ON (teacher forcing) ---
    train_loader, _val_ignored, input_scaler, target_scaler = make_train_val_loaders(
        INPUT_CSV,
        DATA_DIR,
        max_seq_len=SEQ_LEN,
        batch_size=BATCH,
        split_ratio=0.8,
        seed=42,
        use_lagged_stress=True,
    )

    # --- Validation loader (inference-style): lagged stress OFF, reuse scalers ---
    val_infer_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=SEQ_LEN,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        scale=True,
        split="val",
        split_ratio=0.8,
        seed=42,
        use_lagged_stress=False,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )

    class InformerTimeMarching(nn.Module):
        """
        Informer that predicts y(t=1..T-1) autoregressively with a single start token y(t=0).

        Encoder: exogenous+lag channels [B, T, enc_in]
        Decoder: start token y0 then zeros [B, LABEL_LEN+PRED_LEN, dec_in]
        """
        def __init__(self, enc_in: int, dec_in: int, c_out: int,
                     seq_len: int = 200, label_len: int = 1, pred_len: int = 199) -> None:
            super().__init__()
            self.dec_in = dec_in
            self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
            self.net = Informer(
                enc_in=enc_in,
                dec_in=dec_in,
                c_out=c_out,
                seq_len=seq_len,
                label_len=label_len,
                out_len=pred_len,
                factor=5,
                d_model=192,
                n_heads=4,
                e_layers=3,
                d_layers=2,
                d_ff=1024,
                dropout=0.1,
                attn="full",
                embed="fixed",
                freq="s",
                activation="gelu",
                output_attention=False,
                distil=False,
                mix=True,
            )

        def forward(self, x_enc: torch.Tensor, y_start_token: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
            """
            Args:
                x_enc:         [B, seq_len, enc_in]
                y_start_token: [B, 1, dec_in]
                pad_mask:      [B, seq_len] boolean (True = real, False = pad)

            Returns:
                pred:          [B, pred_len, c_out] (for t=1..T-1)
            """
            B = int(x_enc.shape[0])
            device = x_enc.device

            # time marks (zeros; embed='fixed')
            x_mark_enc = x_enc.new_zeros((B, self.seq_len, 5))
            Tdec = self.label_len + self.pred_len
            x_mark_dec = x_enc.new_zeros((B, Tdec, 5))

            # decoder input = [start token ; zeros]
            x_dec = x_enc.new_zeros((B, Tdec, self.dec_in))
            x_dec[:, :self.label_len, :] = y_start_token

            # encoder self-mask: causal + pad keys
            enc_tri = TriangularCausalMask(B, self.seq_len, device=device).mask
            if pad_mask is not None:
                enc_key_pad = (~pad_mask[:, :self.seq_len]).unsqueeze(1).unsqueeze(1).expand(B, 1, self.seq_len, self.seq_len)
                enc_self_mask = SimpleMask(enc_tri | enc_key_pad)
            else:
                enc_self_mask = SimpleMask(enc_tri)

            # decoder self-mask: causal
            dec_self_mask = TriangularCausalMask(B, Tdec, device=device)

            # decoder-encoder mask: pad keys on encoder side
            if pad_mask is not None:
                dec_enc_pad = (~pad_mask[:, :self.seq_len]).unsqueeze(1).unsqueeze(1).expand(B, 1, Tdec, self.seq_len)
                dec_enc_mask = SimpleMask(dec_enc_pad)
            else:
                dec_enc_mask = None

            return self.net(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=enc_self_mask,
                dec_self_mask=dec_self_mask,
                dec_enc_mask=dec_enc_mask,
            )

    # Model
    enc_in = train_loader.dataset.inputs[0].shape[1]   # 17 when lagged stress ON
    dec_in = train_loader.dataset.targets[0].shape[1]  # 6
    model = InformerTimeMarching(enc_in, dec_in, dec_in,
                                 seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_infer = float("inf")
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss
    t0 = time.time()

    # scheduled sampling schedule (teacher-forcing ratio)
    def tf_ratio(epoch: int, total: int) -> float:
        start, end = 1.0, 0.0
        if total <= 1:
            return end
        alpha = (epoch - 1) / (total - 1)
        return start + (end - start) * alpha

    lag_slice = slice(enc_in - dec_in, enc_in)  # last 6 channels hold lagged stress

    # ----------------- training loop -----------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        grad_norm_accum = 0.0
        num_steps = 0
        tfp = tf_ratio(epoch, EPOCHS)  # keep-true probability for lag channels

        for x, pad_mask, times, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
            B = x.size(0)

            # pass 1 (no grad): predictions with teacher-forced lag for constructing candidate lag
            y0 = y[:, :1, :]  # start token (true σ at t=0)
            with torch.no_grad():
                pred_tf = model(x, y0, pad_mask)   # [B, PRED_LEN, 6] for t=1..T-1
                lag_pred = x.new_zeros((B, SEQ_LEN, dec_in))
                lag_pred[:, 1:, :] = pred_tf

            optimizer.zero_grad(set_to_none=True)

            # scheduled sampling mix: true lag vs predicted lag
            if tfp < 1.0:
                keep_true = x.new_zeros((B, SEQ_LEN, 1)) + 1.0
                keep_true[:, 1:, :] = (torch.rand(B, SEQ_LEN - 1, 1, device=x.device) < tfp).float()
                mixed_lag = keep_true * x[:, :, lag_slice] + (1.0 - keep_true) * lag_pred
                x_mixed = x.clone()
                x_mixed[:, :, lag_slice] = mixed_lag
            else:
                x_mixed = x

            # zero-lag augmentation (simulate cold start)
            if torch.rand(()) < 0.30:
                x_for_train = x.clone()
                x_for_train[:, :, lag_slice] = 0.0
            else:
                x_for_train = x_mixed

            # pass 2: trainable forward
            pred = model(x_for_train, y0, pad_mask)  # [B, PRED_LEN, 6]
            y_target = y[:, 1:, :]                   # align to t=1..T-1
            mask_pred = pad_mask[:, 1:]

            # data term (SmoothL1) on real timesteps
            huber = torch.nn.SmoothL1Loss(reduction="none")
            loss_data = huber(pred, y_target)
            loss_data = loss_data[mask_pred.unsqueeze(-1).expand_as(loss_data)].mean()

            # smoothness term on finite differences
            if pred.size(1) > 1:
                diff = pred[:, 1:, :] - pred[:, :-1, :]
                mask_diff = mask_pred[:, 1:].unsqueeze(-1).expand_as(diff)
                loss_smooth = (diff ** 2)[mask_diff].mean()
            else:
                loss_smooth = torch.tensor(0.0, device=pred.device)

            loss = loss_data + 0.01 * loss_smooth

            loss.backward()
            grad_norm_batch = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            grad_norm_accum += float(grad_norm_batch)
            num_steps += 1
            train_loss += float(loss.item()) * B

        train_loss /= len(train_loader.dataset)
        avg_grad_norm = grad_norm_accum / max(1, num_steps)

        # ----------------- inference-style validation -----------------
        model.eval()
        val_infer_loss = 0.0
        all_preds, all_trues = [], []
        rough_epoch_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_base, pad_mask, times, y in tqdm(val_infer_loader, desc=f"Val-Infer Epoch {epoch}"):
                x_base, pad_mask, y = x_base.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
                B = int(x_base.shape[0])
                y0 = y[:, :1, :]
                lag = x_base.new_zeros((B, SEQ_LEN, dec_in))  # start with zero lag

                # multi-refinement loop
                for _ in range(N_REFINE):
                    x_full = torch.cat([x_base, lag], dim=-1)  # append lag channels
                    pred = model(x_full, y0, pad_mask)         # [B, PRED_LEN, 6]
                    lag = lag.clone()
                    lag[:, 1:, :] = pred

                # compute loss against y[:, 1:, :]
                y_target = y[:, 1:, :]
                mask_pred = pad_mask[:, 1:]
                loss = ((pred - y_target) ** 2)[mask_pred.unsqueeze(-1).expand_as(pred)].mean()
                val_infer_loss += float(loss.item()) * B

                # roughness (mean squared finite difference on valid steps)
                if pred.size(1) > 1:
                    diff = pred[:, 1:, :] - pred[:, :-1, :]
                    mask_d = mask_pred[:, 1:].unsqueeze(-1).expand_as(diff)
                    rough_batch = (diff ** 2)[mask_d].mean().item()
                else:
                    rough_batch = 0.0

                rough_epoch_accum += rough_batch
                val_batches += 1

                # collect for metrics
                pred_np = pred.detach().cpu().numpy().reshape(-1, dec_in)
                y_np = y_target.detach().cpu().numpy().reshape(-1, dec_in)
                m_np = mask_pred.detach().cpu().numpy().reshape(-1).astype(bool)
                all_preds.append(pred_np[m_np])
                all_trues.append(y_np[m_np])

        val_infer_loss /= len(val_infer_loader.dataset)
        all_preds = np.vstack(all_preds) if len(all_preds) else np.zeros((0, dec_in))
        all_trues = np.vstack(all_trues) if len(all_trues) else np.zeros((0, dec_in))
        phys_preds = target_scaler.inverse_transform(all_preds) if len(all_preds) else all_preds
        phys_trues = target_scaler.inverse_transform(all_trues) if len(all_trues) else all_trues
        rmse, r2 = compute_metrics(phys_trues, phys_preds) if len(all_preds) else ([0.0] * dec_in, [0.0] * dec_in)

        print(f"\nEpoch {epoch}  Train {train_loss:.4f}  Val-Infer {val_infer_loss:.4f}")
        print(f"  Val-Infer RMSE: {[f'{v:.3f}' for v in rmse]}")
        print(f"  Val-Infer R^2 : {[f'{v:.3f}' for v in r2]}")

        # checkpoint on inference-style validation loss
        if val_infer_loss < best_val_infer:
            best_val_infer = val_infer_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_tm.pt"))
            print("Saved new best (inference-style) model.")

        # epoch bookkeeping
        avg_roughness = rough_epoch_accum / max(1, val_batches)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_infer_loss"].append(float(val_infer_loss))
        history["grad_norm"].append(float(avg_grad_norm))
        history["roughness"].append(float(avg_roughness))
        history["rmse"].append([float(v) for v in rmse])
        history["r2"].append([float(v) for v in r2])

        save_history_json(history, os.path.join(MODEL_DIR, "history.json"))
        plot_training_curves(history, MODEL_DIR)

    # ----------------- resources & scalers -----------------
    t1 = time.time()
    mem_after = proc.memory_info().rss
    resources = {
        "train_time_s": round(t1 - t0, 2),
        "train_mem_increase_mb": round((mem_after - mem_before) / 1024**2, 2),
    }
    with open(os.path.join(MODEL_DIR, "resources.json"), "w") as f:
        json.dump(resources, f, indent=2)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_tm.pt"))

    if input_scaler is not None and target_scaler is not None:
        with open(os.path.join(MODEL_DIR, "scalers.json"), "w") as f:
            json.dump({"input": input_scaler.to_dict(), "target": target_scaler.to_dict()}, f, indent=2)

    print("Done. Resources:", resources)


if __name__ == "__main__":
    main()
