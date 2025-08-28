#!/usr/bin/env python3
"""
evaluate_time_marching_canonical.py

Evaluation for the *canonical windowed* Informer setup:
- Encoder: [past y(6); past exo(11)] → 17 channels
- Decoder: [last LABEL_LEN y(6); exo(11)] with future y zeros → 17 channels
- Windowed dataset (no padding/masks).
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score

from informer.models.model import Informer
from informer.models.attn import TriangularCausalMask
from src.dataset_windows import CompositeInformerWindows
from src.normalisation import StandardScaler


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


# ----------------------------- model wrapper ----------------------------------

class InformerCanonicalTM(nn.Module):
    """
    Standard Informer wrapper for windowed evaluation.

    Forward signature mirrors the Informer API:
      forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, pred_len, c_out]
    """
    def __init__(self, enc_in: int, dec_in: int, c_out: int,
                 seq_len: int, label_len: int, pred_len: int) -> None:
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


# ----------------------------- evaluation -------------------------------------

def main() -> None:
    device = select_device()
    print(f"Using device: {device}")

    SEQ_LEN = 96
    LABEL_LEN = 48
    PRED_LEN = 96
    STRIDE = 96
    BATCH = 32

    INPUT_CSV = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/2. Data Science MSc/Modules/"
        "Data Science Project/composite_stress_prediction/data/IM78552_DATABASEInput.csv"
    )
    DATA_DIR = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/2. Data Science MSc/Modules/"
        "Data Science Project/composite_stress_prediction/data/_CSV"
    )
    MODEL_DIR  = "models/informer_tm_canonical_v2"
    MODEL_PATH = os.path.join(MODEL_DIR, "best_tm.pt")

    # Load scalers saved at train time
    scalers_path = os.path.join(MODEL_DIR, "scalers.json")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"{scalers_path} not found. Re-run training to save scalers.")
    with open(scalers_path, "r") as f:
        sc = json.load(f)
    input_scaler  = StandardScaler.from_dict(sc["input"])
    target_scaler = StandardScaler.from_dict(sc["target"])

    # Windowed validation dataset (no padding/masks)
    val_ds = CompositeInformerWindows(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        stride=STRIDE,
        scale=True,
        split="val",
        split_ratio=0.8,
        seed=42,
        exo_scaler=input_scaler,
        y_scaler=target_scaler,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, persistent_workers=True)

    # Model
    enc_in, dec_in, c_out = 17, 17, 6
    model = InformerCanonicalTM(enc_in, dec_in, c_out, SEQ_LEN, LABEL_LEN, PRED_LEN).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Collections
    seq_trues, seq_preds = [], []
    flat_trues, flat_preds = [], []
    flat_final_trues, flat_final_preds = [], []

    t0 = time.time()
    with torch.no_grad():
        for x_enc, x_dec, xme, xmd, y_true in tqdm(val_loader, desc="Evaluating"):
            x_enc, x_dec, xme, xmd, y_true = [t.to(device) for t in (x_enc, x_dec, xme, xmd, y_true)]
            pred = model(x_enc, xme, x_dec, xmd)  # [B, PRED_LEN, 6]

            B = pred.size(0)
            C = pred.size(-1)
            pred_np = pred.cpu().numpy().reshape(B, PRED_LEN, C)
            true_np = y_true.cpu().numpy().reshape(B, PRED_LEN, C)

            # per-sequence (windowed)
            for i in range(B):
                seq_preds.append(pred_np[i])
                seq_trues.append(true_np[i])

            # flattened for global metrics
            flat_preds.append(pred_np.reshape(-1, C))
            flat_trues.append(true_np.reshape(-1, C))

            # final-step per window
            flat_final_preds.append(pred_np[:, -1, :])  # [B,6]
            flat_final_trues.append(true_np[:, -1, :])  # [B,6]

    eval_time = time.time() - t0

    # Concatenate and invert scaling
    flat_preds       = np.vstack(flat_preds) if flat_preds else np.zeros((0, c_out))
    flat_trues       = np.vstack(flat_trues) if flat_trues else np.zeros((0, c_out))
    flat_final_preds = np.vstack(flat_final_preds) if flat_final_preds else np.zeros((0, c_out))
    flat_final_trues = np.vstack(flat_final_trues) if flat_final_trues else np.zeros((0, c_out))

    preds_phys       = target_scaler.inverse_transform(flat_preds) if len(flat_preds) else flat_preds
    trues_phys       = target_scaler.inverse_transform(flat_trues) if len(flat_trues) else flat_trues
    final_preds_phys = target_scaler.inverse_transform(flat_final_preds) if len(flat_final_preds) else flat_final_preds
    final_trues_phys = target_scaler.inverse_transform(flat_final_trues) if len(flat_final_trues) else flat_final_trues

    # Metrics
    rmse_full, r2_full   = compute_metrics(trues_phys, preds_phys) if len(flat_preds) else ([0]*c_out, [0]*c_out)
    rmse_final, r2_final = compute_metrics(final_trues_phys, final_preds_phys) if len(flat_final_preds) else ([0]*c_out, [0]*c_out)
    max_error = np.max(np.abs(final_trues_phys - final_preds_phys), axis=1) if len(flat_final_preds) else np.array([])
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
        for comp in range(c_out):
            fig, ax = plt.subplots(figsize=(10, 6))
            for j, idx in enumerate(idxs):
                t = np.arange(seq_trues[idx].shape[0])
                ax.plot(t, seq_trues[idx][:, comp], "-", alpha=0.7, color=colors[j], label="True" if j == 0 else None)
                ax.plot(t, seq_preds[idx][:, comp], "--", alpha=0.7, color=colors[j], label="Pred" if j == 0 else None)
            ax.set_title(f"S{comp+1} time-marching (canonical)")
            ax.set_xlabel("Timestep"); ax.set_ylabel("Stress")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR, f"time_series_tm_S{comp+1}.png"), dpi=300)
            plt.close(fig)

    if max_error.size:
        plt.figure()
        plt.hist(max_error, bins=50, alpha=0.7)
        plt.xlabel("Absolute final-step error"); plt.ylabel("Count")
        plt.title("Histogram of final-timestep errors (canonical)")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "hist_max_error_tm.png"), dpi=300)
        plt.close()

    print("Saved plots to", MODEL_DIR)


if __name__ == "__main__":
    main()
