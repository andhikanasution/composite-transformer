#!/usr/bin/env python3
"""
train_patchtst.py

Training script for point-wise PatchTST regression on composite stress prediction.
Includes:
  - Hugging Face PatchTSTConfig & PatchTSTForRegression setup
  - Apple MPS / CPU device handling
  - DataLoader integration (filtering, padding, standardisation)
  - Training loop with tqdm progress bars
  - Single-batch inference profiling
  - Wall-clock time & memory (RSS) logging via psutil
  - Model summary capture and saving
  - Saving model weights, input scaler, and resource log to JSON
"""

import os
import time
import json
import psutil
import torch
from torchinfo import summary
from tqdm import tqdm
from transformers import PatchTSTConfig, PatchTSTForRegression, get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
import numpy as np
from sklearn.metrics import r2_score

# Project-specific data loader (applies utils_parsing & normalization)
from src.dataloader import get_dataloader


def bytes_to_mb(bytes_val: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_val / (1024 * 1024)


def capture_model_summary(model: torch.nn.Module, input_size: tuple) -> str:
    """
    Capture the textual model summary from torchinfo.summary into a string.

    Args:
        model: The PyTorch model to summarize.
        input_size: Input tensor shape for summary (e.g. (1, seq_len, channels)).
    Returns:
        A string containing the printed model summary.
    """
    from io import StringIO
    import sys

    buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer
    summary(model, input_size=input_size)
    sys.stdout = original_stdout
    return buffer.getvalue()


def main():
    # ──────────────────────────────────────────────
    # 1) Paths & Hyperparameters
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
    USE_LAGGED_STRESS = True  # Set to False for 11-channel input (no teacher-forcing)
    MODEL_DIR = f"models/patchtst_5pc_v5_{'17ch' if USE_LAGGED_STRESS else '11ch'}"
    os.makedirs(MODEL_DIR, exist_ok=True)

    BATCH_SIZE = 32
    MAX_EPOCHS = 100  # maximum number of epochs to run
    WARMUP_PCT = 0.05  # 5% of total steps
    PATIENCE = 10  # stop if no val-loss improvement after this many epochs
    best_val_r2   = -float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(MODEL_DIR, "patchtst_best.pt")
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    MAX_SEQ_LEN = 1800
    SEED = 42  # Ensures reproducible train/val split

    # ──────────────────────────────────────────────
    # 2) Configure & Instantiate PatchTST
    # ──────────────────────────────────────────────
    config = PatchTSTConfig(
        num_input_channels = 17 if USE_LAGGED_STRESS else 11,
        num_targets=6,  # 6 stress outputs
        context_length=MAX_SEQ_LEN,  # look-back window
        prediction_length=1,  # point-wise
        patch_length=16,  # patch size
        patch_stride=16,  # no overlap
        d_model=128,  # embedding dim
        num_hidden_layers=3,  # transformer depth
        num_attention_heads=4,  # heads per layer
        ffn_dim=512,  # feed-forward dim
        norm_type="layernorm",  # batch normalization
        loss="mse",  # objective
        scaling=None,  # enable internal standardisation
        share_embedding=True,  # share embeddings across channels
        positional_encoding_type="sincos",  # sinusoidal positional encodings
        attention_dropout=0.1,
        ff_dropout=0.1
    )
    model = PatchTSTForRegression(config)

    # Capture & print model summary
    model_summary = capture_model_summary(model, input_size=(1, MAX_SEQ_LEN, config.num_input_channels))
    print("=== Model Summary ===\n" + model_summary)

    # ──────────────────────────────────────────────
    # 3) Prepare DataLoader
    # ──────────────────────────────────────────────
    train_loader = get_dataloader(input_csv_path=INPUT_CSV, data_dir=DATA_DIR, max_seq_len=MAX_SEQ_LEN,
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, split="train",
                                  split_ratio=0.8, seed=SEED, use_lagged_stress=USE_LAGGED_STRESS)

    # 3bis) Validation DataLoader
    val_loader = get_dataloader(input_csv_path=INPUT_CSV, data_dir=DATA_DIR, max_seq_len=MAX_SEQ_LEN,
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, split="val",
                                split_ratio=0.8, seed=SEED, use_lagged_stress=USE_LAGGED_STRESS)

    # Extract fitted scaler (for later saving)
    input_scaler = train_loader.dataset.input_scaler

    # Retrieve the target (stress) scaler so we can invert predictions later
    target_scaler = train_loader.dataset.target_scaler

    # ──────────────────────────────────────────────
    # 4) Device & Optimizer Setup
    # ──────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.SmoothL1Loss()  # Huber loss δ=1.0
    total_steps = MAX_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_PCT * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # For memory profiling
    process = psutil.Process(os.getpid())
    mem_before_train = process.memory_info().rss
    t_train_start = time.time()

    # ──────────────────────────────────────────────
    # 5) Training Loop with Early Stopping
    # ──────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for bx, by in tqdm(train_loader, desc=f"Train {epoch}/{MAX_EPOCHS}", leave=False):
            optimizer.zero_grad()
            out = model(past_values=bx.to(device))
            pred = out.regression_outputs.squeeze(1)
            true = by[:, -1, :].to(device)
            loss = criterion(pred, true)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)

        # ——— validation pass —————————————————————————————
        model.eval()
        # — single validation pass: collect preds/trues, compute val_loss & macro‐R² →
        all_v_preds, all_v_trues = [], []
        val_loss = 0.0
        for vx, vy in val_loader:
            out = model(past_values=vx.to(device))
            pred_t = out.regression_outputs.squeeze(1)
            true_t = vy[:, -1, :].to(device)
            # accumulate loss on tensor form
            val_loss += criterion(pred_t, true_t).item() * vx.size(0)

            # store for R²
            all_v_preds.append(pred_t.detach().cpu().numpy())
            all_v_trues.append(true_t.detach().cpu().numpy())

        # normalize loss
        val_loss /= len(val_loader.dataset)

        # compute macro‐R²
        all_v_preds = np.vstack(all_v_preds)
        all_v_trues = np.vstack(all_v_trues)
        val_r2 = np.mean([
            r2_score(all_v_trues[:, i], all_v_preds[:, i])
            for i in range(all_v_preds.shape[1])
        ])

        dt = time.time() - t0
        print(f"Epoch {epoch}/{MAX_EPOCHS} — train_loss: {train_loss:.4f} — val_loss: {val_loss:.4f} — time: {dt:.1f}s")

        # ——— early-stop logic ————————————————————————————
        if val_r2 > best_val_r2:  # flip to R² maximisation
            best_val_r2 = val_r2
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾  New best model (val {val_loss:.4f}) saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️  No improvement for {epochs_no_improve}/{PATIENCE} epochs")
            if epochs_no_improve >= PATIENCE:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break
        model.train()

    t_train_end = time.time()
    mem_after_train = process.memory_info().rss

    # ──────────────────────────────────────────────
    # 6) Single-Batch Inference Profiling
    # ──────────────────────────────────────────────
    model.eval()
    sample_x, _ = next(iter(train_loader))

    mem_before_inf = process.memory_info().rss
    t_inf_start = time.time()
    with torch.no_grad():
        _ = model(past_values=sample_x.to(device))
    t_inf_end = time.time()
    mem_after_inf = process.memory_info().rss

    # ──────────────────────────────────────────────
    # 7) Save Outputs & Resource Log
    # ──────────────────────────────────────────────
    # b) Input scaler
    scaler_path = os.path.join(MODEL_DIR, "input_scaler.json")
    with open(scaler_path, "w") as f:
        json.dump(input_scaler.to_dict(), f, indent=2)
    print(f"✅ Saved input scaler: {scaler_path}")

    # c) Save target scaler (so we can invert preds later)
    target_scaler_path = os.path.join(MODEL_DIR, "target_scaler.json")
    with open(target_scaler_path, "w") as f:
        json.dump(target_scaler.to_dict(), f, indent=2)
    print(f"✅ Saved target scaler:    {target_scaler_path}")

    # ───────────────────────────────────────────────────
    # Save the ASCII model summary to its own .txt
    # ───────────────────────────────────────────────────
    summary_txt_path = os.path.join(MODEL_DIR, "patchtst_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(model_summary)
    print(f"✅ Saved model summary: {summary_txt_path}")

    # d) Resource usage & model summary
    resource_log = {
        "train_time_s": round(t_train_end - t_train_start, 2),
        "train_mem_diff_mb": round(bytes_to_mb(mem_after_train - mem_before_train), 2),
        "inf_time_s": round(t_inf_end - t_inf_start, 4),
        "inf_mem_diff_mb": round(bytes_to_mb(mem_after_inf - mem_before_inf), 2)
    }
    log_path = os.path.join(MODEL_DIR, "resource_log.json")
    with open(log_path, "w") as f:
        json.dump(resource_log, f, indent=2)
    print(f"✅ Saved resource log: {log_path}\n")

    # Save training mode to config
    mode_path = os.path.join(MODEL_DIR, "mode_config.json")
    with open(mode_path, "w") as f:
        json.dump({"use_lagged_stress": USE_LAGGED_STRESS}, f, indent=2)
    print(f"✅ Saved training mode config: {mode_path}")

    # Display summary
    print("=== Resource Summary ===")
    for key, val in resource_log.items():
        if key == "model_summary":
            continue
        unit = "MB" if "mem" in key else "s"
        print(f"{key:22}: {val} {unit}")


if __name__ == "__main__":
    main()
