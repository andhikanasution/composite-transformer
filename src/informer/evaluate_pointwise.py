#!/usr/bin/env python3
"""
evaluate_pointwise.py

Evaluation for the *pointwise* Informer model (full-sequence mapping).
Matches `train_informer_pointwise.py`.

Pipeline
--------
1) Load saved input/target scalers and pass them to the validation DataLoader.
2) Build model with the same hyperparameters as training.
3) Forward once per batch: [B, T, C_out].
4) Invert scaling to physical units and compute metrics:
   - Full-sequence RMSE/R^2 (over all real timesteps)
   - Final-step RMSE/R^2 (per sequence, last *real* timestep)
5) Save metrics JSON and plots (final-step scatter, per-component time series, histogram).
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


# ----------------------------- config -----------------------------------------

SEQ_LEN   = 200
LABEL_LEN = 1                  # must match training
PRED_LEN  = SEQ_LEN


# ----------------------------- utils ------------------------------------------

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[float], List[float]]:
    """Per-channel RMSE and R^2 for arrays shaped [N, C]."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2   = [float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    return rmse, r2


# ----------------------------- model wrapper ----------------------------------

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
        self.net = Informer(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=SEQ_LEN,
            label_len=LABEL_LEN,
            out_len=PRED_LEN,
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
        B = x_enc.size(0)
        x_mark_enc = x_enc.new_zeros((B, SEQ_LEN, 5))
        x_mark_dec = x_enc.new_zeros((B, LABEL_LEN + PRED_LEN, 5))
        x_dec = x_enc.new_zeros((B, LABEL_LEN + PRED_LEN, self.dec_in))
        x_dec[:, LABEL_LEN:, :] = x_enc
        return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, T, C_out]


# ----------------------------- evaluation -------------------------------------

def main() -> None:
    device = select_device()
    print(f"Using device: {device}")

    # Paths
    INPUT_CSV = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/2. Data Science MSc/Modules/"
        "Data Science Project/composite_stress_prediction/data/IM78552_DATABASEInput.csv"
    )
    DATA_DIR = os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-UniversityofBristol/2. Data Science MSc/Modules/"
        "Data Science Project/composite_stress_prediction/data/_CSV"
    )
    MODEL_DIR  = "models/informer_pointwise_v3"  # adjust as needed
    MODEL_PATH = os.path.join(MODEL_DIR, "best_pointwise.pt")

    # Load scalers saved at train time (required for val splits with scale=True)
    scalers_path = os.path.join(MODEL_DIR, "scalers.json")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"{scalers_path} not found. Re-run training to save scalers.")
    with open(scalers_path, "r") as f:
        sc = json.load(f)
    input_scaler  = StandardScaler.from_dict(sc["input"])
    target_scaler = StandardScaler.from_dict(sc["target"])

    # Validation DataLoader (no lagged stress for pointwise)
    val_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=SEQ_LEN,
        batch_size=32,
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

    # Instantiate & load model
    enc_in = val_loader.dataset.inputs[0].shape[1]   # 11
    dec_in = enc_in                                   # decoder sees same exogenous
    c_out  = 6
    model  = InformerForPointwise(enc_in, dec_in, c_out).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Collections
    all_trues, all_preds = [], []
    final_trues, final_preds = [], []
    seq_trues, seq_preds, seq_masks = [], [], []

    t0 = time.time()
    with torch.no_grad():
        for x, pad_mask, times, y in tqdm(val_loader, desc="Evaluating"):
            x, pad_mask, y = x.to(device), pad_mask.to(device), y.to(device)
            pred = model(x)  # [B, T, C]

            # per-sequence storage for plotting
            B, T, C = pred.shape
            pred_np = pred.cpu().numpy()
            true_np = y.cpu().numpy()
            mask_np = pad_mask.cpu().numpy()

            for i in range(B):
                valid_T = int(mask_np[i].sum())
                if valid_T > 0:
                    seq_preds.append(pred_np[i, :valid_T, :])
                    seq_trues.append(true_np[i, :valid_T, :])
                    seq_masks.append(mask_np[i, :valid_T])

            # full-sequence metrics over real timesteps
            pred_flat = pred_np.reshape(-1, C)
            true_flat = true_np.reshape(-1, C)
            mask_flat = mask_np.reshape(-1).astype(bool)
            all_preds.append(pred_flat[mask_flat])
            all_trues.append(true_flat[mask_flat])

            # final-step metrics (last real timestep per sequence)
            for i in range(B):
                valid_T = int(mask_np[i].sum())
                if valid_T > 0:
                    final_preds.append(pred_np[i, valid_T - 1, :][None, :])
                    final_trues.append(true_np[i, valid_T - 1, :][None, :])

    eval_time = time.time() - t0

    # Concatenate
    all_preds   = np.vstack(all_preds)
    all_trues   = np.vstack(all_trues)
    final_preds = np.vstack(final_preds) if final_preds else np.zeros((0, 6))
    final_trues = np.vstack(final_trues) if final_trues else np.zeros((0, 6))

    # Invert scaling
    all_preds_phys   = target_scaler.inverse_transform(all_preds)
    all_trues_phys   = target_scaler.inverse_transform(all_trues)
    final_preds_phys = target_scaler.inverse_transform(final_preds) if len(final_preds) else final_preds
    final_trues_phys = target_scaler.inverse_transform(final_trues) if len(final_trues) else final_trues

    # Metrics
    rmse_full, r2_full = compute_metrics(all_trues_phys,   all_preds_phys)
    rmse_final, r2_final = compute_metrics(final_trues_phys, final_preds_phys) if len(final_preds) else ([0]*6, [0]*6)

    max_error = np.max(np.abs(final_trues_phys - final_preds_phys), axis=1) if len(final_preds) else np.array([])
    mean_max_error = float(np.mean(max_error)) if max_error.size else 0.0
    n_seq = max(1, len(seq_trues))
    avg_time_per_sample = float(eval_time / n_seq)

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
    # 1) Final-step scatter
    if len(final_preds):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            ax.scatter(final_trues_phys[:, i], final_preds_phys[:, i], s=5, alpha=0.4)
            mn = min(final_trues_phys[:, i].min(), final_preds_phys[:, i].min())
            mx = max(final_trues_phys[:, i].max(), final_preds_phys[:, i].max())
            ax.plot([mn, mx], [mn, mx], "--", linewidth=1)
            ax.set_title(f"S{i+1} (R^2={r2_final[i]:.3f})")
            ax.set_xlabel("True final stress"); ax.set_ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "scatter_final_timestep_components.png"), dpi=300)
        plt.close(fig)

    # 2) Time-series (per component) for a few sequences
    n_plot = min(5, len(seq_trues))
    if n_plot >= 1:
        idxs = np.random.choice(len(seq_trues), size=n_plot, replace=False)
        cmap = plt.get_cmap("tab10"); colors = cmap(np.linspace(0, 1, n_plot))
        for comp in range(6):
            fig, ax = plt.subplots(figsize=(10, 6))
            for j, idx in enumerate(idxs):
                t = np.arange(seq_trues[idx].shape[0])
                ax.plot(t, seq_trues[idx][:, comp], "-", alpha=0.7, color=colors[j], label="True" if j == 0 else None)
                ax.plot(t, seq_preds[idx][:, comp], "--", alpha=0.7, color=colors[j], label="Pred" if j == 0 else None)
            ax.set_xlabel("Timestep"); ax.set_ylabel(f"Stress S{comp+1}")
            ax.set_title(f"S{comp+1}: true vs pred")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR, f"time_series_S{comp+1}.png"), dpi=300)
            plt.close(fig)

    # 3) Histogram of final-step absolute errors
    if max_error.size:
        plt.figure()
        plt.hist(max_error, bins=50, alpha=0.7)
        plt.xlabel("Absolute final-step error"); plt.ylabel("Count")
        plt.title("Histogram of final-timestep absolute errors")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "hist_max_error.png"), dpi=300)
        plt.close()

    print("Saved plots to", MODEL_DIR)


if __name__ == "__main__":
    main()
