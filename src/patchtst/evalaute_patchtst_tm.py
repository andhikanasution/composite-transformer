#!/usr/bin/env python3
"""
evaluate_patchtst_tm.py

Time-marching (autoregressive) evaluation script for PatchTST on composite stress prediction.

For a model trained with lagged-stress channels (17ch), this script will:
  1) Load the best pretrained weights + scalers + mode config
  2) Iterate over the validation set (no teacher-forcing):
       • At each time step, feed in the past context (strains+metadata) plus
         the model’s previous predictions as the lagged-stress channels
       • Collect the one-step forecast
  3) Invert scaling to physical units
  4) Compute MAE, RMSE, R² per stress component and macro-averages
  5) Save metrics to JSON and a global scatter plot of true vs. predicted
"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from transformers import PatchTSTConfig, PatchTSTForRegression
from src.dataloader import get_dataloader
from src.normalisation import StandardScaler


def main():
    # ──────────────────────────────────────────────
    # 1) Paths & Mode
    # ──────────────────────────────────────────────
    MODEL_DIR = "models/patchtst_5pc_v5_17ch"  # set this to your trained‐17ch model folder
    # read how the model was trained
    with open(os.path.join(MODEL_DIR, "mode_config.json")) as f:
        mode = json.load(f)
    USE_LAGGED = mode.get("use_lagged_stress", False)
    if not USE_LAGGED:
        raise RuntimeError(
            "Time-marching evaluation requires a model trained with lagged-stress channels (17ch)."
        )

    # data & model artifacts
    INPUT_CSV       = os.path.expanduser("~/Library/CloudStorage/OneDrive-UniversityofBristol/"
                                         "2. Data Science MSc/Modules/Data Science Project/"
                                         "composite_stress_prediction/data/IM78552_DATABASEInput.csv")
    DATA_DIR        = os.path.expanduser("~/Library/CloudStorage/OneDrive-UniversityofBristol/"
                                         "2. Data Science MSc/Modules/Data Science Project/"
                                         "composite_stress_prediction/data/_CSV")
    WEIGHTS_PATH    = os.path.join(MODEL_DIR, "patchtst_best.pt")
    IN_SCALER_PATH  = os.path.join(MODEL_DIR, "input_scaler.json")
    OUT_SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.json")
    METRICS_OUT     = os.path.join(MODEL_DIR, "eval_metrics_tm.json")
    SCATTER_OUT     = os.path.join(MODEL_DIR, "scatter_tm.png")

    # hyperparams
    BATCH_SIZE   = 1        # we step sample-by-sample
    NUM_WORKERS  = 2
    MAX_SEQ_LEN  = 1800
    SPLIT_RATIO  = 0.8      # must match training
    SEED         = 42

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ──────────────────────────────────────────────
    # 2) Load model + scalers
    # ──────────────────────────────────────────────
    config = PatchTSTConfig(
        num_input_channels=17,
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
        ff_dropout=0.1
    )
    model = PatchTSTForRegression(config)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device).eval()

    # load scalers to invert predictions
    with open(IN_SCALER_PATH) as f:
        input_scaler = StandardScaler.from_dict(json.load(f))
    with open(OUT_SCALER_PATH) as f:
        target_scaler = StandardScaler.from_dict(json.load(f))

    # ──────────────────────────────────────────────
    # 3) Prepare validation dataset (no teacher forcing!)
    # ──────────────────────────────────────────────
    val_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=False,
        scale=True,               # we will feed scaled inputs
        num_workers=NUM_WORKERS,
        split="val",
        split_ratio=SPLIT_RATIO,
        seed=SEED,
        use_lagged_stress=False   # disable builtin lagged channels
    )

    # ──────────────────────────────────────────────
    # 4) Time-marching inference
    # ──────────────────────────────────────────────
    all_preds = []
    all_trues = []
    for batch_x, batch_y in tqdm(val_loader, desc="TM Eval"):
        # batch_x: [1, T, 11], batch_y: [1, T, 6], both already scaled
        x11 = batch_x[0].cpu().numpy()  # (T,11)
        y6  = batch_y[0].cpu().numpy()  # (T,6)

        # we will keep a rolling window of past predictions as the lagged channels
        lagged = np.zeros((MAX_SEQ_LEN, 6), dtype=np.float32)

        # store scaled preds
        pred_list = []

        # step through every time index
        for t in range(MAX_SEQ_LEN):
            # build full 17ch window
            #   - inputs: padded & scaled [MAX_SEQ_LEN,11]
            #   - lagged: our autoregressive preds so far [MAX_SEQ_LEN,6]
            inp17 = np.concatenate([x11, lagged], axis=1)

            # run model
            with torch.no_grad():
                t_in = torch.tensor(inp17, dtype=torch.float32, device=device)[None]  # [1, L,17]
                out = model(past_values=t_in)
                p_scaled = out.regression_outputs.squeeze(1).cpu().numpy()  # shape (6,)

            pred_list.append(p_scaled)

            # roll the lagged buffer up by one timestep
            lagged[:-1] = lagged[1:]
            lagged[-1]  = p_scaled

        # stack & invert scale
        preds_scaled = np.vstack(pred_list)              # (L,6)
        preds = target_scaler.inverse_transform(preds_scaled)

        # true stresses: invert scale on the padded ground truth
        trues = target_scaler.inverse_transform(y6)

        all_preds.append(preds)
        all_trues.append(trues)

    # flatten across all samples & time
    all_preds = np.vstack(all_preds)  # (N_total, 6)
    all_trues = np.vstack(all_trues)

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
            "R2":   float(np.mean(r2s))
        }
    }
    with open(METRICS_OUT, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"✅ Saved TM metrics to {METRICS_OUT}")

    # ──────────────────────────────────────────────
    # 6) Scatter plot true vs predicted
    # ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(all_trues[:, i], all_preds[:, i], alpha=0.3, s=5)
        mn, mx = np.min([all_trues[:, i].min(), all_preds[:, i].min()]), \
                 np.max([all_trues[:, i].max(), all_preds[:, i].max()])
        ax.plot([mn, mx], [mn, mx], "--", color="red")
        ax.set_title(f"S{i+1} (R²={per_comp[f'S{i+1}']['R2']:.3f})")
        ax.set_xlabel("True"); ax.set_ylabel("Pred")
    plt.tight_layout()
    fig.savefig(SCATTER_OUT)
    print(f"✅ Saved TM scatter to {SCATTER_OUT}")


if __name__ == "__main__":
    main()
