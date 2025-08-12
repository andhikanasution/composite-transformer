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
from src.normalisation import StandardScaler

def compute_metrics(y_true, y_pred):
    """y_true, y_pred: [N, C]"""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    r2   = np.array([r2_score(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])])
    return rmse, r2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    MODEL_DIR  = "models/informer_pointwise_v3"
    MODEL_PATH = os.path.join(MODEL_DIR, "best_pointwise.pt")

    scalers_path = os.path.join(MODEL_DIR, "scalers.json")
    with open(scalers_path, "r") as f:
        sc = json.load(f)
    input_scaler = StandardScaler.from_dict(sc["input"])
    target_scaler = StandardScaler.from_dict(sc["target"])

    # must match training
    SEQ_LEN = 200
    LABEL_LEN = 1
    PRED_LEN = SEQ_LEN

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
        use_lagged_stress=False,
        input_scaler=input_scaler,
        target_scaler = target_scaler,
    )

    class InformerForPointwise(nn.Module):
        def __init__(self, enc_in, dec_in, c_out):
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
                attn='prob',
                embed='fixed',
                freq='s',
                activation='gelu',
                output_attention=False,
                distil=True,
                mix=True
            )

        def forward(self, x_enc):
            B = int(x_enc.shape[0])
            x_mark_enc = x_enc.new_zeros((B, self.seq_len, 5))
            x_mark_dec = x_enc.new_zeros((B, self.label_len + self.pred_len, 5))
            x_dec = x_enc.new_zeros((B, self.label_len + self.pred_len, self.dec_in))
            x_dec[:, self.label_len:, :] = x_enc
            return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # scalers
    target_scaler = val_loader.dataset.target_scaler

    # 3b) Instantiate & load model
    enc_in = val_loader.dataset.inputs[0].shape[1]  # 11
    dec_in = enc_in  # decoder sees same exogenous
    model = InformerForPointwise(enc_in, dec_in, c_out=6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_trues, all_preds = [], []
    final_trues, final_preds = [], []

    start_time = time.time()
    with torch.no_grad():
        for x, pad_mask, times, y in tqdm(val_loader, desc="Evaluating"):
            x, pad_mask, y = x.to(device), pad_mask.to(device), y.to(device)

            # Forward pass
            pred = model(x)  # [B, PRED_LEN, 6]

            # —— Build per-sequence arrays for plotting ——
            if 'seq_trues' not in locals():
                seq_trues, seq_preds = [], []
            B, T, C = pred.shape
            mask_np_b = pad_mask[:, :T].cpu().numpy()  # [B, T]
            y_np_b = y[:, :T, :].cpu().numpy()  # [B, T, 6]
            pred_np_b = pred.cpu().numpy()  # [B, T, 6]
            for bi in range(B):
                valid_len = int(mask_np_b[bi].sum())
                seq_trues.append(y_np_b[bi, :valid_len])  # [valid_len, 6]
                seq_preds.append(pred_np_b[bi, :valid_len])  # [valid_len, 6]

            # Flatten batch+time
            B, T, C = pred.shape
            pred_flat = pred.reshape(-1, C).cpu().numpy()
            y_flat = y[:, :T, :].reshape(-1, C).cpu().numpy()
            mask_flat = pad_mask[:, :T].reshape(-1).cpu().numpy()

            # Keep only real timesteps
            real_mask = mask_flat.astype(bool)
            all_preds.append(pred_flat[real_mask])
            all_trues.append(y_flat[real_mask])

            # Final‐step (only if that timestep is real)
            pad_final = pad_mask[:, :T][:, -1].cpu().numpy()  # last real in window
            pred_last = pred[:, -1, :].cpu().numpy()
            y_last = y[:, :T, :][:, -1, :].cpu().numpy()

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
        ax.scatter(final_trues_phys[:, i], final_preds_phys[:, i], s=5, alpha=0.4)
        mn = min(final_trues_phys[:, i].min(), final_preds_phys[:, i].min())
        mx = max(final_trues_phys[:, i].max(), final_preds_phys[:, i].max())
        ax.plot([mn, mx], [mn, mx], '--', color='gray', linewidth=1)
        ax.set_title(f"S{i+1} (R²={r2_final[i]:.3f})", fontsize=14)
        ax.set_xlabel("True final stress");  ax.set_ylabel("Predicted")
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "scatter_final_timestep_components.png"), dpi=300)
    plt.close(fig)

    # 5) Time-series plots: sample a few full sequences (uses per-sequence lists)
    if len(seq_trues) > 0:
        n_plot = min(5, len(seq_trues))
        sample_idxs = np.random.choice(len(seq_trues), size=n_plot, replace=False)
        for comp in range(6):
            fig, ax = plt.subplots(figsize=(10, 6))
            for j, idx in enumerate(sample_idxs):
                t = np.arange(len(seq_trues[idx]))
                ax.plot(t, seq_trues[idx][:, comp], '-', alpha=0.7, label='True' if j == 0 else None)
                ax.plot(t, seq_preds[idx][:, comp], '--', alpha=0.7, label='Pred' if j == 0 else None)
            ax.set_xlabel("Timestep", fontsize=12)
            ax.set_ylabel(f"Stress S{comp + 1}", fontsize=12)
            ax.set_title(f"S{comp + 1}: sampled sequences", fontsize=14)
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
