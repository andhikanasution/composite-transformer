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
from informer.models.attn import TriangularCausalMask
from src.normalisation import StandardScaler


class SimpleMask:
    """Wrap a [B,1,Lq,Lk] boolean tensor with a .mask attribute (Informer expects this)."""
    def __init__(self, mask: torch.Tensor):
        self.mask = mask

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
    MODEL_DIR  = "models/informer_timemarching_v3"
    MODEL_PATH = os.path.join(MODEL_DIR, "best_tm.pt")

    SEQ_LEN = 200
    LABEL_LEN = 1
    PRED_LEN = SEQ_LEN - LABEL_LEN
    BATCH = 32
    # Use the same refinement depth you validated with during training
    N_REFINE = 8

    # Load scalers saved at train time (preferred)
    scalers_path = os.path.join(MODEL_DIR, "scalers.json")
    if os.path.exists(scalers_path):
        with open(scalers_path, "r") as f:
            sc = json.load(f)
        input_scaler = StandardScaler.from_dict(sc["input"])
        target_scaler = StandardScaler.from_dict(sc["target"])
    else:
        raise RuntimeError("scalers.json not found in MODEL_DIR; re-run training to save scalers.")

    # Build val loader WITHOUT lagged stress baked in; we'll append zeros then refine with predictions
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
        target_scaler=target_scaler
    )

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Model wrapper (identical to train_informer_tm.py)
    # ──────────────────────────────────────────────────────────────────────────────
    class InformerTimeMarching(nn.Module):
        def __init__(self, enc_in, dec_in, c_out, seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN):
            super().__init__()
            self.dec_in = dec_in
            self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
            self.net = Informer(
                enc_in = enc_in, dec_in = dec_in, c_out = c_out,
                seq_len = seq_len, label_len = label_len, out_len = pred_len,
                factor = 5,
                d_model = 192,  # MATCH TRAIN
                n_heads = 4,
                e_layers = 3,
                d_layers = 2,
                d_ff = 1024,  # MATCH TRAIN
                dropout = 0.1,
                attn = 'full',  # MATCH TRAIN
                embed = 'fixed',
                freq = 's',
                activation = 'gelu',
                output_attention = False,
                distil = False,  # MATCH TRAIN
                mix = True
            )

        def forward(self, x_enc, y_start_token, pad_mask):
            """
            x_enc: [B, seq_len, enc_in] (exogenous + lag)
            y_start_token: [B, 1, dec_in] (true σ at t=0)
            pad_mask: [B, seq_len]  True=real, False=pad
            """

            B = int(x_enc.shape[0])
            device = x_enc.device
            # time marks (zeros for embed='fixed')
            x_mark_enc = x_enc.new_zeros((B, self.seq_len, 5))
            Tdec = self.label_len + self.pred_len
            x_mark_dec = x_enc.new_zeros((B, Tdec, 5))

            # decoder input: start token then zeros
            x_dec = x_enc.new_zeros((B, Tdec, self.dec_in))
            x_dec[:, :self.label_len, :] = y_start_token

            # encoder self-attn: causal + block padded keys
            enc_tri = TriangularCausalMask(B, self.seq_len, device=device).mask  # [B,1,L,L]

            if pad_mask is not None:
                enc_key_pad = (~pad_mask[:, :self.seq_len]).unsqueeze(1).unsqueeze(1) \
                    .expand(B, 1, self.seq_len, self.seq_len)
                enc_self_mask = SimpleMask(enc_tri | enc_key_pad)
            else:
                enc_self_mask = SimpleMask(enc_tri)

            # decoder self-attn: causal
            dec_self_mask = TriangularCausalMask(B, Tdec, device=device)

            # cross-attn: block padded encoder keys
            if pad_mask is not None:
                dec_enc_pad = (~pad_mask[:, :self.seq_len]).unsqueeze(1).unsqueeze(1) \
                    .expand(B, 1, Tdec, self.seq_len)
                dec_enc_mask = SimpleMask(dec_enc_pad)
            else:
                dec_enc_mask = None
            return self.net(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=enc_self_mask,
                dec_self_mask = dec_self_mask,
                dec_enc_mask = dec_enc_mask
            )  # [B, pred_len, c_out]

    target_scaler = val_loader.dataset.target_scaler

    # — Instantiate & load model —
    # we trained with lag channels -> enc_in = 11 + 6 = 17
    base_in = val_loader.dataset.inputs[0].shape[1]  # 11 (no lag in val loader)
    dec_in = val_loader.dataset.targets[0].shape[1]  # 6
    enc_in = base_in + dec_in
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
        for x_base, pad_mask, times, y in tqdm(val_loader, desc="Evaluating"):
            # append zero lag channels, then refine once using predicted σ(t-1)
            x_base, pad_mask, y = x_base.to(device), pad_mask.to(device), y.to(device)
            B = int(x_base.shape[0])
            y0 = y[:, :1, :]
            lag = x_base.new_zeros((B, SEQ_LEN, dec_in))
            # multi-refinement (same as training val)
            for _ in range(N_REFINE):
                x_full = torch.cat([x_base, lag], dim=-1)
                pred = model(x_full, y0, pad_mask)  # [B, PRED_LEN, 6]
                lag = lag.clone()
                lag[:, 1:, :] = pred  # update σ(t-1)

            # —— 1) Collect per‐sequence (in phys units) ——
            pred_np = pred.cpu().numpy().reshape(-1, dec_in)
            true_np = y[:, 1:, :].cpu().numpy().reshape(-1, dec_in)
            mask_np = pad_mask[:, 1:].cpu().numpy().reshape(-1)

            pred_np = pred_np.reshape(B, PRED_LEN, dec_in)
            true_np = true_np.reshape(B, PRED_LEN, dec_in)
            mask_np = mask_np.reshape(B, PRED_LEN)

            for i in range(B):
                valid_len = int(mask_np[i].sum())
                seq_preds.append(pred_np[i, :valid_len])
                seq_trues.append(true_np[i, :valid_len])

            # —— 2) Flatten real‐timesteps for global metrics ——
            idx = mask_np.flatten().astype(bool)
            flat_preds.append(pred_np.reshape(-1, dec_in)[idx])
            flat_trues.append(true_np.reshape(-1, dec_in)[idx])

            # final‐step among real timesteps (map final real idx per sample)
            # approximate: take last valid index per sample (mask sum - 1)
            for i in range(B):
                valid_len = int(mask_np[i].sum())
                if valid_len > 0:
                    flat_final_preds.append(pred_np[i, valid_len - 1, :][None, :])
                    flat_final_trues.append(true_np[i, valid_len - 1, :][None, :])

    eval_time = time.time() - t0

    # Concatenate
    flat_preds       = np.vstack(flat_preds)
    flat_trues       = np.vstack(flat_trues)
    flat_final_preds = np.vstack(flat_final_preds) if len(flat_final_preds) else np.zeros((0, dec_in))
    flat_final_trues = np.vstack(flat_final_trues) if len(flat_final_trues) else np.zeros((0, dec_in))

    # Invert scaling
    preds_phys = target_scaler.inverse_transform(flat_preds)
    trues_phys = target_scaler.inverse_transform(flat_trues)
    final_preds_phys = target_scaler.inverse_transform(flat_final_preds) if len(
        flat_final_preds) else flat_final_preds
    final_trues_phys = target_scaler.inverse_transform(flat_final_trues) if len(
        flat_final_trues) else flat_final_trues

    # —— Compute metrics ——
    rmse_full, r2_full     = compute_metrics(trues_phys, preds_phys)
    rmse_final, r2_final   = compute_metrics(final_trues_phys, final_preds_phys)
    max_error = np.max(np.abs(final_trues_phys - final_preds_phys), axis=1) if len(final_trues_phys) else np.array([])
    mean_max_error = float(np.mean(max_error)) if max_error.size else 0.0
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
    if max_error.size:
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
