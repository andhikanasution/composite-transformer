#!/usr/bin/env python3
"""
evaluate_timemarching.py

Evaluation for the *time-marching* Informer (teacher-forcing during training,
autoregressive-style evaluation with iterative lag refinement here).

Pipeline
--------
1) Load saved scalers (required) and build val DataLoader WITHOUT lagged channels.
2) Append zero lag channels and iteratively refine them using model predictions (N_REFINE steps).
3) Invert scaling and compute:
   - Full-sequence RMSE/R^2 across real timesteps
   - Final-step RMSE/R^2 (last real timestep per sequence)
4) Save metrics JSON and plots (final-step scatter, per-component time series, histogram).
"""

from __future__ import annotations

import os
import json
import time
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from sklearn.metrics import r2_score

from src.dataloader import get_dataloader
from src.normalisation import StandardScaler
from informer.models.model import Informer
from informer.models.attn import TriangularCausalMask


# ----------------------------- utils ------------------------------------------

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[float], List[float]]:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2   = [float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    return rmse, r2


class SimpleMask:
    """Wrap a [B,1,Lq,Lk] boolean tensor with a .mask attribute (Informer expects this)."""
    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask


# ----------------------------- model wrapper ----------------------------------

class InformerTimeMarching(nn.Module):
    """
    Informer that predicts y(t=1..T-1) with a single start token y(t=0).
    Matches `train_informer_tm.py` hyperparameters.
    """
    def __init__(self, enc_in: int, dec_in: int, c_out: int,
                 seq_len: int, label_len: int, pred_len: int) -> None:
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
        x_enc: [B, seq_len, enc_in] (exogenous + lag)
        y_start_token: [B, 1, dec_in] (true σ at t=0)
        pad_mask: [B, seq_len] boolean (True=real, False=pad)
        """
        B = int(x_enc.shape[0])
        device = x_enc.device

        x_mark_enc = x_enc.new_zeros((B, self.seq_len, 5))
        Tdec = self.label_len + self.pred_len
        x_mark_dec = x_enc.new_zeros((B, Tdec, 5))

        # decoder input: start token then zeros
        x_dec = x_enc.new_zeros((B, Tdec, self.dec_in))
        x_dec[:, :self.label_len, :] = y_start_token

        # encoder self-attn: causal + pad keys
        enc_tri = TriangularCausalMask(B, self.seq_len, device=device).mask  # [B,1,L,L]
        if pad_mask is not None:
            enc_key_pad = (~pad_mask[:, :self.seq_len]).unsqueeze(1).unsqueeze(1).expand(B, 1, self.seq_len, self.seq_len)
            enc_self_mask = SimpleMask(enc_tri | enc_key_pad)
        else:
            enc_self_mask = SimpleMask(enc_tri)

        # decoder self-attn: causal
        dec_self_mask = TriangularCausalMask(B, Tdec, device=device)

        # cross-attn: block padded encoder keys
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
        )  # [B, pred_len, c_out]


# ----------------------------- evaluation -------------------------------------

def main() -> None:
    device = select_device()
    print(f"Using device: {device}")

    # Paths and constants
    INPUT_CSV = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/2. Data Science MSc/Modules/"
        "Data Science Project/composite_stress_prediction/data/IM78552_DATABASEInput.csv"
    )
    DATA_DIR = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/2. Data Science MSc/Modules/"
        "Data Science Project/composite_stress_prediction/data/_CSV"
    )
    MODEL_DIR  = "models/informer_timemarching_v3"
    MODEL_PATH = os.path.join(MODEL_DIR, "best_tm.pt")

    SEQ_LEN = 200
    LABEL_LEN = 1
    PRED_LEN = SEQ_LEN - LABEL_LEN
    BATCH = 32
    N_REFINE = 8  # refinement depth (match validation setting)

    # Load scalers saved at train time
    scalers_path = os.path.join(MODEL_DIR, "scalers.json")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"{scalers_path} not found. Re-run training to save scalers.")
    with open(scalers_path, "r") as f:
        sc = json.load(f)
    input_scaler  = StandardScaler.from_dict(sc["input"])
    target_scaler = StandardScaler.from_dict(sc["target"])

    # Build val loader WITHOUT lagged stress (append and refine inside loop)
    val_loader = get_dataloader(
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

    # Instantiate & load model (enc_in = 11 + 6 = 17 for lag-augmented input)
    base_in = val_loader.dataset.inputs[0].shape[1]  # 11 (no lag in val loader)
    dec_in  = val_loader.dataset.targets[0].shape[1] # 6
    enc_in  = base_in + dec_in                       # add lag channels
    model = InformerTimeMarching(enc_in, dec_in, dec_in, SEQ_LEN, LABEL_LEN, PRED_LEN).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Collections for metrics/plots
    seq_trues, seq_preds = [], []
    flat_trues, flat_preds = [], []
    flat_final_trues, flat_final_preds = [], []

    t0 = time.time()
    with torch.no_grad():
        for x_base, pad_mask, times, y in tqdm(val_loader, desc="Evaluating"):
            x_base, pad_mask, y = x_base.to(device), pad_mask.to(device), y.to(device)
            B = int(x_base.shape[0])
            y0 = y[:, :1, :]
            lag = x_base.new_zeros((B, SEQ_LEN, dec_in))  # start with zero lag

            # iterative refinement of lag channels
            for _ in range(N_REFINE):
                x_full = torch.cat([x_base, lag], dim=-1)
                pred = model(x_full, y0, pad_mask)   # [B, PRED_LEN, 6]
                lag = lag.clone()
                lag[:, 1:, :] = pred

            # per-sequence collections (phys units after inversion later)
            pred_np = pred.cpu().numpy()            # [B, PRED_LEN, 6]
            true_np = y[:, 1:, :].cpu().numpy()     # align to t=1..T-1
            mask_np = pad_mask[:, 1:].cpu().numpy() # valid timesteps for t>=1

            for i in range(B):
                valid_T = int(mask_np[i].sum())
                if valid_T > 0:
                    seq_preds.append(pred_np[i, :valid_T, :])
                    seq_trues.append(true_np[i, :valid_T, :])

            # flatten real timesteps for global metrics
            idx = mask_np.reshape(-1).astype(bool)
            flat_preds.append(pred_np.reshape(-1, dec_in)[idx])
            flat_trues.append(true_np.reshape(-1, dec_in)[idx])

            # final-step per sequence: last real timestep
            for i in range(B):
                valid_T = int(mask_np[i].sum())
                if valid_T > 0:
                    flat_final_preds.append(pred_np[i, valid_T - 1, :][None, :])
                    flat_final_trues.append(true_np[i, valid_T - 1, :][None, :])

    eval_time = time.time() - t0

    # Concatenate and invert scaling
    flat_preds       = np.vstack(flat_preds) if flat_preds else np.zeros((0, dec_in))
    flat_trues       = np.vstack(flat_trues) if flat_trues else np.zeros((0, dec_in))
    flat_final_preds = np.vstack(flat_final_preds) if flat_final_preds else np.zeros((0, dec_in))
    flat_final_trues = np.vstack(flat_final_trues) if flat_final_trues else np.zeros((0, dec_in))

    preds_phys       = target_scaler.inverse_transform(flat_preds) if len(flat_preds) else flat_preds
    trues_phys       = target_scaler.inverse_transform(flat_trues) if len(flat_trues) else flat_trues
    final_preds_phys = target_scaler.inverse_transform(flat_final_preds) if len(flat_final_preds) else flat_final_preds
    final_trues_phys = target_scaler.inverse_transform(flat_final_trues) if len(flat_final_trues) else flat_final_trues

    # Metrics
    rmse_full, r2_full   = compute_metrics(trues_phys, preds_phys) if len(flat_preds) else ([0]*dec_in, [0]*dec_in)
    rmse_final, r2_final = compute_metrics(final_trues_phys, final_preds_phys) if len(flat_final_preds) else ([0]*dec_in, [0]*dec_in)
    max_error = np.max(np.abs(final_trues_phys - final_preds_phys), axis=1) if len(final_trues_phys) else np.array([])
    mean_max_error = float(np.mean(max_error)) if max_error.size else 0.0
    avg_time_per_sample = float(eval_time / max(1, len(seq_trues)))

    metrics = {
        "rmse_full": rmse_full,
        "r2_full": r2_full,
        "rmse_final": rmse_final,
        "r2_final": r2_final,
        "mean_max_error": mean_max_error,
        "total_eval_time_s": round(eval_time, 4),
        "avg_time_per_sample_s": round(avg_time_per_sample, 6),
    }
    with open(os.path.join(MODEL_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics:", metrics)

    # ----------------- plots -----------------
    if len(final_preds_phys):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten()
        for i, ax in enumerate(axes):
            ax.scatter(final_trues_phys[:, i], final_preds_phys[:, i], s=5, alpha=0.4)
            mn = min(final_trues_phys[:, i].min(), final_preds_phys[:, i].min())
            mx = max(final_trues_phys[:, i].max(), final_preds_phys[:, i].max())
            ax.plot([mn, mx], [mn, mx], "--", linewidth=1)
            ax.set_title(f"S{i+1} (R^2={r2_final[i]:.3f})")
            ax.set_xlabel("True"); ax.set_ylabel("Pred")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "scatter_final_components_tm.png"), dpi=300)
        plt.close(fig)

    n_plot = min(5, len(seq_trues))
    if n_plot >= 1:
        idxs = np.random.choice(len(seq_trues), size=n_plot, replace=False)
        cmap = plt.get_cmap("tab10"); colors = cmap(np.linspace(0, 1, n_plot))
        for comp in range(dec_in):
            fig, ax = plt.subplots(figsize=(10, 6))
            for j, idx in enumerate(idxs):
                t = np.arange(seq_trues[idx].shape[0])
                ax.plot(t, seq_trues[idx][:, comp], "-", alpha=0.7, color=colors[j], label="True" if j == 0 else None)
                ax.plot(t, seq_preds[idx][:, comp], "--", alpha=0.7, color=colors[j], label="Pred" if j == 0 else None)
            ax.set_title(f"S{comp+1} time-marching")
            ax.set_xlabel("Timestep"); ax.set_ylabel("Stress")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR, f"time_series_tm_S{comp+1}.png"), dpi=300)
            plt.close(fig)

    if max_error.size:
        plt.figure()
        plt.hist(max_error, bins=50, alpha=0.7)
        plt.xlabel("Absolute final-step error"); plt.ylabel("Count")
        plt.title("Histogram of final-timestep errors")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "hist_max_error_tm.png"), dpi=300)
        plt.close()

    print("Saved plots to", MODEL_DIR)


if __name__ == "__main__":
    main()
