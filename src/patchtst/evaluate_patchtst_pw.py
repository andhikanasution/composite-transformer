#!/usr/bin/env python3
"""
evaluate_patchtst_pw.py

Point-wise evaluation (no autoregression) for PatchTST on composite stress prediction.

Steps:
  1) Read mode_config.json → determines 11ch vs 17ch
  2) Instantiate matching PatchTSTConfig & load weights
  3) Build val DataLoader with use_lagged_stress flag
  4) For each batch:
       - pad zeros if loader channels < model channels
       - forward pass → [B,1,6] → squeeze → [B,6]
       - inverse‐scale back to MPa
  5) Compute per‐component MAE, RMSE, R² + macro averages
  6) Save metrics JSON & scatter‐plot grid
"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from transformers import PatchTSTConfig, PatchTSTForRegression
from src.dataloader import get_dataloader      # your updated DataLoader
from src.normalisation import StandardScaler   # your scaler class


def main():
    # ──────────────────────────────────────────────
    # 1) Paths & hyperparams
    # ──────────────────────────────────────────────
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

    # Read which mode we trained in (11ch vs 17ch)
    MODEL_DIR   = "models/patchtst_5pc_v5_11ch"
    with open(os.path.join(MODEL_DIR, "mode_config.json")) as f:
        mode = json.load(f)
    USE_LAGGED  = mode.get("use_lagged_stress", False)           # bool

    WEIGHTS    = os.path.join(MODEL_DIR, "patchtst_best.pt")
    SCALER_JSON= os.path.join(MODEL_DIR, "target_scaler.json")
    METRICS_OUT= os.path.join(MODEL_DIR, f"eval_metrics_pw_{'17ch' if USE_LAGGED else '11ch'}.json")
    PLOT_OUT   = os.path.join(MODEL_DIR, f"scatter_pw_{'17ch' if USE_LAGGED else '11ch'}.png")

    BATCH_SIZE = 32
    NUM_WORKERS= 4
    MAX_SEQ_LEN= 1800
    SPLIT_RATIO= 0.8
    SEED       = 42

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ──────────────────────────────────────────────
    # 2) Instantiate & load model
    # ──────────────────────────────────────────────
    in_ch = 17 if USE_LAGGED else 11
    config = PatchTSTConfig(
        num_input_channels=in_ch,   # match training
        num_targets=6,
        context_length=MAX_SEQ_LEN,
        prediction_length=1,
        patch_length=16,
        patch_stride=16,
        d_model=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        ffn_dim=512,
        norm_type="layernorm",
        loss="mse",
        scaling=None,
        share_embedding=True,
        positional_encoding_type="sincos",
        attention_dropout=0.1,
        ff_dropout=0.1,
    )
    model = PatchTSTForRegression(config)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device).eval()

    # load the target-stress scaler to invert predictions
    with open(SCALER_JSON) as f:
        target_scaler = StandardScaler.from_dict(json.load(f))

    # ──────────────────────────────────────────────
    # 3) Prepare validation DataLoader
    # ──────────────────────────────────────────────
    val_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=False,
        scale=True,                # feed raw inputs → HF model will standardize
        num_workers=NUM_WORKERS,
        split="val",
        split_ratio=SPLIT_RATIO,
        seed=SEED,
        use_lagged_stress=USE_LAGGED
    )

    # ──────────────────────────────────────────────
    # 4) Inference
    # ──────────────────────────────────────────────
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for bx, by in tqdm(val_loader, desc="Evaluating"):
            # bx: [B, T, C_loader],    by: [B, T, 6]
            bx = bx.to(device)
            C_loader = bx.size(2)

            # if model expects more channels than loader provides, pad zeros
            if in_ch > C_loader:
                pad = torch.zeros(bx.size(0), bx.size(1), in_ch - C_loader, device=device)
                bx = torch.cat([bx, pad], dim=2)

            out = model(past_values=bx)
            # → out.regression_outputs: [B, 1, 6]
            preds_std = out.regression_outputs.squeeze(1).cpu().numpy()
            preds = target_scaler.inverse_transform(preds_std)

            # true stress is at final step (point-wise)
            trues_std = by[:, -1, :].cpu().numpy()
            trues     = target_scaler.inverse_transform(trues_std)

            all_preds.append(preds)
            all_trues.append(trues)

    all_preds = np.vstack(all_preds)   # [N,6]
    all_trues = np.vstack(all_trues)   # [N,6]

    # ──────────────────────────────────────────────
    # 5) Compute metrics
    # ──────────────────────────────────────────────
    per_comp = {}
    maes, rmses, r2s = [], [], []
    for i in range(6):
        y_t = all_trues[:, i]
        y_p = all_preds[:, i]
        mae  = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        r2   = r2_score(y_t, y_p)
        per_comp[f"S{i+1}"] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        maes.append(mae); rmses.append(rmse); r2s.append(r2)

    metrics = {
        "per_component": per_comp,
        "macro_avg": {
            "MAE":  float(np.mean(maes)),
            "RMSE": float(np.mean(rmses)),
            "R2":   float(np.mean(r2s)),
        }
    }

    with open(METRICS_OUT, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"✅ Saved metrics to {METRICS_OUT}")

    # ──────────────────────────────────────────────
    # 6) Scatter‐plot matrix
    # ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18,10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(all_trues[:,i], all_preds[:,i], alpha=0.3, s=5)
        mn = min(all_trues[:,i].min(), all_preds[:,i].min())
        mx = max(all_trues[:,i].max(), all_preds[:,i].max())
        ax.plot([mn,mx],[mn,mx], "--", color="red")
        ax.set_title(f"S{i+1} (R²={per_comp[f'S{i+1}']['R2']:.3f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    plt.tight_layout()
    fig.savefig(PLOT_OUT)
    print(f"✅ Saved scatter plot to {PLOT_OUT}")


if __name__ == "__main__":
    main()
