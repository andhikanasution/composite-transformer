#!/usr/bin/env python3
"""
evaluate_patchtst.py

Pointwise (non-autoregressive) evaluation for PatchTST on composite stress prediction.

Pipeline
--------
1) Read mode_config.json → determines 11ch vs 17ch
2) Instantiate PatchTST and load best weights
3) Load both input & target scalers; pass them to the val DataLoader (scale=True)
4) Forward pass: [B,T,C] → [B,1,6] → squeeze → [B,6]
5) Invert scaling to physical units; compute per-component MAE/RMSE/R^2 and macro means
6) Save metrics JSON and a 2×3 scatter plot grid
"""

from __future__ import annotations

import os
import json
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import PatchTSTConfig, PatchTSTForRegression

from src.dataloader import get_dataloader
from src.normalisation import StandardScaler


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    # ──────────────────────────────────────────────
    # 1) Paths & constants
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

    MODEL_DIR = "models/patchtst_5pc_v5_11ch"  # adjust if evaluating a 17ch run
    with open(os.path.join(MODEL_DIR, "mode_config.json")) as f:
        mode = json.load(f)
    USE_LAGGED = bool(mode.get("use_lagged_stress", False))

    WEIGHTS = os.path.join(MODEL_DIR, "patchtst_best.pt")
    INPUT_SCALER_JSON = os.path.join(MODEL_DIR, "input_scaler.json")
    TARGET_SCALER_JSON = os.path.join(MODEL_DIR, "target_scaler.json")

    METRICS_OUT = os.path.join(MODEL_DIR, f"eval_metrics_pw_{'17ch' if USE_LAGGED else '11ch'}.json")
    PLOT_OUT = os.path.join(MODEL_DIR, f"scatter_pw_{'17ch' if USE_LAGGED else '11ch'}.png")

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_SEQ_LEN = 1800
    SPLIT_RATIO = 0.8
    SEED = 42

    device = select_device()
    print(f"Using device: {device.type}")

    # ──────────────────────────────────────────────
    # 2) Instantiate & load model
    # ──────────────────────────────────────────────
    in_ch = 17 if USE_LAGGED else 11
    config = PatchTSTConfig(
        num_input_channels=in_ch,
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
    state = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # Load scalers (required for val split with scale=True)
    with open(INPUT_SCALER_JSON) as f:
        input_scaler = StandardScaler.from_dict(json.load(f))
    with open(TARGET_SCALER_JSON) as f:
        target_scaler = StandardScaler.from_dict(json.load(f))

    # ──────────────────────────────────────────────
    # 3) Validation DataLoader (scaled using train scalers)
    # ──────────────────────────────────────────────
    val_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        scale=True,
        split="val",
        split_ratio=SPLIT_RATIO,
        seed=SEED,
        use_lagged_stress=USE_LAGGED,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )

    # ──────────────────────────────────────────────
    # 4) Inference
    # ──────────────────────────────────────────────
    preds_all = []
    trues_all = []

    with torch.no_grad():
        for bx, _bm, _bt, by in tqdm(val_loader, desc="Evaluating"):
            bx = bx.to(device, non_blocking=True)  # [B, T, C]
            out = model(past_values=bx)            # [B, 1, 6]
            y_std = out.regression_outputs.squeeze(1).cpu().numpy()   # standardized
            y_hat = target_scaler.inverse_transform(y_std)            # to physical units

            t_std = by[:, -1, :].cpu().numpy()
            t_hat = target_scaler.inverse_transform(t_std)

            preds_all.append(y_hat)
            trues_all.append(t_hat)

    preds_all = np.vstack(preds_all)  # [N,6]
    trues_all = np.vstack(trues_all)  # [N,6]

    # ──────────────────────────────────────────────
    # 5) Metrics
    # ──────────────────────────────────────────────
    per_comp: Dict[str, Dict[str, float]] = {}
    maes, rmses, r2s = [], [], []

    for i in range(6):
        yt = trues_all[:, i]
        yp = preds_all[:, i]
        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = r2_score(yt, yp)
        per_comp[f"S{i+1}"] = {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    metrics = {
        "per_component": per_comp,
        "macro_avg": {
            "MAE": float(np.mean(maes)),
            "RMSE": float(np.mean(rmses)),
            "R2": float(np.mean(r2s)),
        },
    }

    with open(METRICS_OUT, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved metrics: {METRICS_OUT}")

    # ──────────────────────────────────────────────
    # 6) Scatter plot (2×3)
    # ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(trues_all[:, i], preds_all[:, i], alpha=0.3, s=6)
        mn = float(min(trues_all[:, i].min(), preds_all[:, i].min()))
        mx = float(max(trues_all[:, i].max(), preds_all[:, i].max()))
        ax.plot([mn, mx], [mn, mx], "--")
        ax.set_title(f"S{i+1} (R^2={per_comp[f'S{i+1}']['R2']:.3f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=200)
    print(f"Saved scatter plot: {PLOT_OUT}")


if __name__ == "__main__":
    main()
