#!/usr/bin/env python3
"""
evaluate_patchtst.py

Evaluation script for the trained PatchTST model on composite stress prediction.

This script:
  - Loads saved PatchTSTForRegression model weights
  - Builds the validation DataLoader (unscaled inputs, deterministic split)
  - Runs a point-wise inference pass on the 'val' split
  - Computes MAE, RMSE, and R² per stress component + macro-average
  - Saves metrics to JSON
  - Generates & saves scatter plots of predictions vs. ground truth
"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from transformers import PatchTSTConfig, PatchTSTForRegression
from src.dataloader import get_dataloader  # your updated DataLoader with split & scale args
from src.normalisation import StandardScaler


def main():
    # ──────────────────────────────────────────────
    # 1) Paths & Hyperparameters
    # ──────────────────────────────────────────────
    # Data & model locations
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
    MODEL_DIR = "models/patchtst_5pc_v2"
    os.makedirs(MODEL_DIR, exist_ok=True)

    WEIGHTS_PATH       = os.path.join(MODEL_DIR, "patchtst_5pc.pt")
    METRICS_JSON_PATH  = os.path.join(MODEL_DIR, "patchtst_5pc_eval_metrics.json")
    PLOT_PATH          = os.path.join(MODEL_DIR, "patchtst_5pc_scatter.png")

    # DataLoader & split settings
    BATCH_SIZE  = 32
    NUM_WORKERS = 4
    MAX_SEQ_LEN = 1800
    SPLIT_RATIO = 0.8        # same as training
    SEED        = 42         # for deterministic split

    # Device (Apple MPS if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ──────────────────────────────────────────────
    # 2) Instantiate & Load Model
    # ──────────────────────────────────────────────
    # Must match the config used during training
    config = PatchTSTConfig(
        num_input_channels=11,           # 6 strain + θ + 4 lam params
        num_targets=6,                   # 6 stress outputs
        context_length=MAX_SEQ_LEN,      # full history
        prediction_length=1,             # point-wise
        patch_length=16,
        patch_stride=16,
        d_model=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        ffn_dim=512,
        norm_type="batchnorm",
        loss="mse",
        scaling=None,                   # let model apply its internal scaler
        share_embedding=True,
        positional_encoding_type="sincos"
    )
    model = PatchTSTForRegression(config)
    # load the trained weights
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    with open(os.path.join(MODEL_DIR, "target_scaler.json")) as f:
        target_scaler = StandardScaler.from_dict(json.load(f))

    # ──────────────────────────────────────────────
    # 3) Prepare Validation DataLoader
    # ──────────────────────────────────────────────
    # scale=False → we feed raw inputs, the model’s internal scaler will normalize them
    val_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=False,
        scale=True,
        num_workers=NUM_WORKERS,
        split="val",
        split_ratio=SPLIT_RATIO,
        seed=SEED,
    )

    # ──────────────────────────────────────────────
    # 4) Inference on Validation Split
    # ──────────────────────────────────────────────
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)      # [B, T, 11]
            outputs = model(past_values=batch_x)
            # regression_outputs: [B, 1, 6] → squeeze to [B, 6]
            preds_std = outputs.regression_outputs.squeeze(1).cpu().numpy()
            # Invert back to real stress units
            preds = target_scaler.inverse_transform(preds_std)

            # ground truth is the stress at the final time-step: [B, T, 6] → select [:, -1, :]
            trues_std = batch_y[:, -1, :].cpu().numpy()
            trues = target_scaler.inverse_transform(trues_std)
            all_preds.append(preds)
            all_trues.append(trues)

    # stack into [N_samples, 6]
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    # ──────────────────────────────────────────────
    # 5) Compute Metrics
    # ──────────────────────────────────────────────
    per_comp = {}
    maes, rmses, r2s = [], [], []
    n_components = all_preds.shape[1]
    for i in range(n_components):
        y_true = all_trues[:, i]
        y_pred = all_preds[:, i]
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, y_pred)
        per_comp[f"S{i+1}"] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        maes.append(mae); rmses.append(rmse); r2s.append(r2)

    metrics = {
        "per_component": per_comp,
        "macro_avg": {
            "MAE":  float(np.mean(maes)),
            "RMSE": float(np.mean(rmses)),
            "R2":   float(np.mean(r2s))
        }
    }

    # save to JSON
    with open(METRICS_JSON_PATH, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"✅ Saved evaluation metrics to: {METRICS_JSON_PATH}")

    # ──────────────────────────────────────────────
    # 6) Visualization: Scatter Plots Pred vs True
    # ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(all_trues[:, i], all_preds[:, i], alpha=0.3, s=5)
        # 45° reference line
        mn = min(all_trues[:, i].min(), all_preds[:, i].min())
        mx = max(all_trues[:, i].max(), all_preds[:, i].max())
        ax.plot([mn, mx], [mn, mx], ls="--", c="red")
        ax.set_title(f"S{i+1} (R²={per_comp[f'S{i+1}']['R2']:.3f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    print(f"✅ Saved scatter plot to: {PLOT_PATH}")

if __name__ == "__main__":
    main()
