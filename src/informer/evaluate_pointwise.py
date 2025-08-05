#!/usr/bin/env python3
# evaluate_pointwise.py

import os
import time
import json

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from src.dataloader import get_dataloader
from informer.models.model import Informer

# ──────────────────────────────────────────────────────────────────────────────
# 1) Model wrapper (must match training)
# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN   = 200
LABEL_LEN = 200
PRED_LEN  = 200

class InformerForPointwise(nn.Module):
    def __init__(self, enc_in, dec_in, c_out):
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
            attn='prob',
            embed='fixed',
            freq='t',
            activation='gelu',
            output_attention=False,
            distil=True,
            mix=True
        )

    def forward(self, x_enc):
        B = x_enc.size(0)
        x_dec      = torch.zeros(B, LABEL_LEN, self.dec_in, device=x_enc.device)
        x_mark_enc = torch.zeros(B, SEQ_LEN, 5,        device=x_enc.device)
        x_mark_dec = torch.zeros(B, LABEL_LEN, 5,      device=x_enc.device)
        return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Metric functions
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """y_true, y_pred: [N, C]"""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    r2   = np.array([r2_score(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])])
    return rmse, r2

# ──────────────────────────────────────────────────────────────────────────────
# 3) Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
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
    MODEL_DIR  = "models/informer_pointwise_v2"
    MODEL_PATH = os.path.join(MODEL_DIR, "best_pointwise.pt")

    # 3a) Build val DataLoader
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
        use_lagged_stress=False
    )

    # scalers
    target_scaler = val_loader.dataset.target_scaler

    # 3b) Instantiate & load model
    enc_in = val_loader.dataset.inputs[0].shape[1]
    dec_in = val_loader.dataset.targets[0].shape[1]
    model  = InformerForPointwise(enc_in, dec_in, dec_in).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_trues, all_preds = [], []
    final_trues, final_preds = [], []

    start_time = time.time()
    with torch.no_grad():
        for x, pad_mask, y in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            x, pad_mask, y = x.to(device), pad_mask.to(device), y.to(device)

            # Forward pass
            pred = model(x)  # [B, SEQ_LEN, C]

            # Flatten batch+time
            B, T, C = pred.shape
            pred_flat = pred.view(-1, C).cpu().numpy()  # [B*T, C]
            y_flat    = y   .view(-1, C).cpu().numpy()  # [B*T, C]
            mask_flat = pad_mask.view(-1).cpu().numpy() # [B*T]

            # Keep only real timesteps
            real_mask = mask_flat.astype(bool)
            all_preds.append(pred_flat[real_mask])
            all_trues.append(y_flat[real_mask])

            # Final‐step (only if that timestep is real)
            pad_final = pad_mask[:, -1].cpu().numpy()  # [B]
            pred_last = pred[:, -1, :].cpu().numpy()   # [B, C]
            y_last    = y   [:, -1, :].cpu().numpy()   # [B, C]

            valid_final = pad_final.astype(bool)
            final_preds.append(pred_last[valid_final])
            final_trues.append(y_last[valid_final])

    eval_time = time.time() - start_time

    # Concatenate
    all_preds   = np.vstack(all_preds)
    all_trues   = np.vstack(all_trues)
    final_preds = np.vstack(final_preds)
    final_trues = np.vstack(final_trues)

    # Invert scaling
    all_preds_phys = target_scaler.inverse_transform(all_preds)
    all_trues_phys = target_scaler.inverse_transform(all_trues)
    final_preds_phys = target_scaler.inverse_transform(final_preds)
    final_trues_phys = target_scaler.inverse_transform(final_trues)

    # Metrics
    rmse_full, r2_full = compute_metrics(all_trues_phys, all_preds_phys)
    rmse_final, r2_final = compute_metrics(final_trues_phys, final_preds_phys)

    max_error = np.max(np.abs(final_trues_phys - final_preds_phys), axis=1)
    mean_max_error = float(np.mean(max_error))
    n_samples = len(final_trues)
    avg_time_per_sample = eval_time / n_samples

    metrics = {
        "rmse_full":       rmse_full.tolist(),
        "r2_full":         r2_full.tolist(),
        "rmse_final":      rmse_final.tolist(),
        "r2_final":        r2_final.tolist(),
        "mean_max_error":  mean_max_error,
        "total_eval_time_s":     round(eval_time, 4),
        "avg_time_per_sample_s": round(avg_time_per_sample, 6)
    }

    with open(os.path.join(MODEL_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics:", metrics)

    # 4) Scatter plot (6 panels)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(final_trues[:, i], final_preds[:, i], s=5, alpha=0.4)
        mn = min(final_trues[:, i].min(), final_preds[:, i].min())
        mx = max(final_trues[:, i].max(), final_preds[:, i].max())
        ax.plot([mn, mx], [mn, mx], '--', color='gray', linewidth=1)
        ax.set_title(f"S{i+1} (R²={r2_final[i]:.3f})", fontsize=14)
        ax.set_xlabel("True final stress");  ax.set_ylabel("Predicted")
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "scatter_final_timestep_components.png"), dpi=300)
    plt.close(fig)

    # 5) Time‐series plot: a few representative samples, with legend
    # assume all_trues_phys, all_preds_phys are shape [N_total, 6]
    # and PRED_LEN == 200
    n_plot = 5
    n_components = 6
    # how many sequences are in val set:
    n_sequences = len(val_loader.dataset)
    # pick sample indices
    sample_idxs = np.random.choice(n_sequences, size=n_plot, replace=False)

    # build a colormap
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, n_plot))
    times = np.arange(PRED_LEN)

    for comp in range(n_components):
        fig, ax = plt.subplots(figsize=(10, 6))
        for j, idx in enumerate(sample_idxs):
            start = idx * PRED_LEN
            end = start + PRED_LEN
            seq_true = all_trues_phys[start:end, comp]
            seq_pred = all_preds_phys[start:end, comp]
            ax.plot(times, seq_true,
                    color=colors[j],
                    linestyle='-',
                    alpha=0.7)
            ax.plot(times, seq_pred,
                    color=colors[j],
                    linestyle='--',
                    alpha=0.7)

        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel(f"Stress S{comp + 1}", fontsize=12)
        ax.set_title(f"Sample true vs pred (component S{comp + 1})", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # add a legend only for the line‐styles
        ax.plot([], [], color='k', linestyle='-', label='True')
        ax.plot([], [], color='k', linestyle='--', label='Pred')
        ax.legend(fontsize=12, loc='upper left')

        plt.tight_layout()
        fig.savefig(os.path.join(MODEL_DIR, f"time_series_S{comp + 1}.png"), dpi=300)
        plt.close(fig)

    # 6) Histogram of max‐error
    plt.figure()
    plt.hist(max_error, bins=50, alpha=0.7)
    plt.xlabel("Absolute final‐stress error")
    plt.ylabel("Count")
    plt.title("Histogram of final‐timestep absolute errors")
    plt.savefig(os.path.join(MODEL_DIR, "hist_max_error.png"), dpi=300)
    plt.close()

    print("Saved all plots to", MODEL_DIR)

if __name__ == "__main__":
    main()
