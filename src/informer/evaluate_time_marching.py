#!/usr/bin/env python3
# evaluate_timemarching.py

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
# 1) Hyper‐params & constants (must match training)
# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN   = 200
LABEL_LEN = 200
OUT_LEN   = 200
BATCH     = 32

# ──────────────────────────────────────────────────────────────────────────────
# 2) Model wrapper (identical to train_informer_tm.py)
# ──────────────────────────────────────────────────────────────────────────────
class InformerTimeMarching(nn.Module):
    def __init__(self, enc_in, dec_in, c_out):
        super().__init__()
        self.dec_in = dec_in
        self.net = Informer(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=SEQ_LEN,
            label_len=LABEL_LEN,
            out_len=OUT_LEN,
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

    def forward(self, x_enc, x_dec):
        B = x_enc.size(0)
        x_mark_enc = torch.zeros(B, SEQ_LEN, 5,   device=x_enc.device)
        x_mark_dec = torch.zeros(B, LABEL_LEN, 5, device=x_enc.device)
        return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """y_true, y_pred: [N, C]"""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    r2   = np.array([r2_score(y_true[:,i], y_pred[:,i])
                     for i in range(y_true.shape[1])])
    return rmse, r2


# ──────────────────────────────────────────────────────────────────────────────
# 4) Full Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

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
    MODEL_DIR  = "models/informer_timemarching_v2"
    MODEL_PATH = os.path.join(MODEL_DIR, "best_tm.pt")

    # — Build val DataLoader (no lagged stress during eval) —
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
        use_lagged_stress=False
    )
    target_scaler = val_loader.dataset.target_scaler

    # — Instantiate & load model —
    enc_in = val_loader.dataset.inputs[0].shape[1]
    dec_in = val_loader.dataset.targets[0].shape[1]
    model = InformerTimeMarching(enc_in, dec_in, dec_in).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # We'll collect two parallel sets of lists:
    #  1) per‐sequence true/pred for plotting
    #  2) flattened real‐timestep true/pred for metrics
    seq_trues = []
    seq_preds = []
    flat_trues = []
    flat_preds = []
    flat_final_trues = []
    flat_final_preds = []

    t0 = time.time()
    with torch.no_grad():
        for x, pad_mask, y in tqdm(val_loader, desc="Evaluating"):
            x, pad_mask, y = x.to(device), pad_mask.to(device), y.to(device)
            B = x.size(0)

            # build decoder input: σ^(0) = true at t=0, zeros thereafter
            x_dec = torch.zeros(B, LABEL_LEN, dec_in, device=device)
            x_dec[:,0,:] = y[:,0,:]

            # forward
            pred = model(x, x_dec)  # [B, OUT_LEN, C]

            # —— 1) Collect per‐sequence (in phys units) ——
            # un‐normalize
            pred_np = target_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, dec_in))
            true_np = target_scaler.inverse_transform(y   .cpu().numpy().reshape(-1, dec_in))
            mask_np = pad_mask.cpu().numpy().reshape(-1)

            pred_np = pred_np.reshape(B, OUT_LEN, dec_in)
            true_np = true_np.reshape(B, OUT_LEN, dec_in)
            mask_np = mask_np.reshape(B, OUT_LEN)

            for i in range(B):
                valid_len = int(mask_np[i].sum())
                seq_preds.append(pred_np[i, :valid_len])
                seq_trues.append(true_np[i, :valid_len])

            # —— 2) Flatten real‐timesteps for global metrics ——
            idx = mask_np.flatten().astype(bool)
            flat_preds.append(pred_np.reshape(-1, dec_in)[idx])
            flat_trues.append(true_np.reshape(-1, dec_in)[idx])

            # also final‐step (if not padded)
            pad_final = pad_mask[:, -1].cpu().numpy().astype(bool)
            flat_final_preds.append(pred[:, -1, :].cpu().numpy()[pad_final])
            flat_final_trues.append(   y[:, -1, :].cpu().numpy()[pad_final])

    eval_time = time.time() - t0

    # Concatenate
    flat_preds       = np.vstack(flat_preds)
    flat_trues       = np.vstack(flat_trues)
    flat_final_preds = np.vstack(flat_final_preds)
    flat_final_trues = np.vstack(flat_final_trues)

    # Invert scaling for final‐step (we only inverted seq above)
    final_preds_phys = target_scaler.inverse_transform(flat_final_preds)
    final_trues_phys = target_scaler.inverse_transform(flat_final_trues)

    # —— Compute metrics ——
    rmse_full, r2_full     = compute_metrics(flat_trues, flat_preds)
    rmse_final, r2_final   = compute_metrics(final_trues_phys, final_preds_phys)
    max_error = np.max(np.abs(final_trues_phys - final_preds_phys), axis=1)
    mean_max_error = float(np.mean(max_error))
    avg_time_per_sample = eval_time / len(seq_trues)

    metrics = {
        "rmse_full":       rmse_full.tolist(),
        "r2_full":         r2_full.tolist(),
        "rmse_final":      rmse_final.tolist(),
        "r2_final":        r2_final.tolist(),
        "mean_max_error":  mean_max_error,
        "total_eval_time_s":     round(eval_time, 4),
        "avg_time_per_sample_s": round(avg_time_per_sample, 6)
    }

    # save metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics:", metrics)

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Scatter plot: final‐step
    fig, axes = plt.subplots(2, 3, figsize=(18,10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(final_trues_phys[:,i], final_preds_phys[:,i],
                   s=5, alpha=0.4)
        mn = min(final_trues_phys[:,i].min(), final_preds_phys[:,i].min())
        mx = max(final_trues_phys[:,i].max(), final_preds_phys[:,i].max())
        ax.plot([mn,mx],[mn,mx],'--',color='gray',lw=1)
        ax.set_title(f"S{i+1} (R²={r2_final[i]:.3f})", fontsize=14)
        ax.set_xlabel("True"); ax.set_ylabel("Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR,
                             "scatter_final_components_tm.png"), dpi=300)
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────
    # 6) Time‐series plots: pick 5 random sequences
    n_plot = 5
    n_comp = dec_in
    sample_idxs = np.random.choice(len(seq_trues), size=n_plot,
                                   replace=False)
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0,1,n_plot))

    for comp in range(n_comp):
        fig, ax = plt.subplots(figsize=(10, 6))
        for j, idx in enumerate(sample_idxs):
            c = colors[j]
            true_seq = seq_trues[idx][:, comp]
            pred_seq = seq_preds[idx][:, comp]
            t = np.arange(len(true_seq))
            # force same color for true & pred of this sample
            ax.plot(t, true_seq, '-', color=c, alpha=0.7,
                             label = "True" if j == 0 else None)
            ax.plot(t, pred_seq, '--', color=c, alpha=0.7,
                             label = "Pred" if j == 0 else None)
        ax.set_title(f"S{comp+1} time-marching")
        ax.set_xlabel("Timestep"); ax.set_ylabel("Stress")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR,
                                 f"time_series_tm_S{comp+1}.png"),
                    dpi=300)
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────
    # 7) Histogram of max‐error
    plt.figure()
    plt.hist(max_error, bins=50, alpha=0.7)
    plt.xlabel("Absolute final‐step error")
    plt.ylabel("Count")
    plt.title("Histogram of final‐timestep errors")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR,
                             "hist_max_error_tm.png"), dpi=300)
    plt.close()

    print("All plots saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
