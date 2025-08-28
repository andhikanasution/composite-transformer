#!/usr/bin/env python3
"""
train_patchtst.py

Pointwise PatchTST training for composite stress prediction.

Key features
------------
- Hugging Face PatchTSTConfig / PatchTSTForRegression
- Robust device selection (CUDA → MPS → CPU)
- Mask-aware DataLoader integration (padded sequences, Z-score scaling)
- Cosine LR schedule with warmup, gradient clipping
- Early stopping on validation R^2
- Wall-clock and RSS memory profiling (psutil)
- Model summary capture (torchinfo)
- Saves: best weights, scalers (input/target), resource log, summary, mode config
"""

from __future__ import annotations

import os
import sys
import time
import json
import psutil
import random
from io import StringIO
from typing import Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torchinfo import summary
from tqdm import tqdm
from transformers import (
    PatchTSTConfig,
    PatchTSTForRegression,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import r2_score

from src.dataloader import get_dataloader


# ----------------------------- utilities ---------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bytes_to_mb(n_bytes: int) -> float:
    """Convert bytes → MB."""
    return float(n_bytes) / (1024.0 * 1024.0)


def capture_model_summary(model: torch.nn.Module, input_size: Tuple[int, int, int]) -> str:
    """
    Capture torchinfo.summary output as a string.

    Args:
        model: Model to summarize.
        input_size: (B, T, C) tuple for a single forward spec.
    """
    buf = StringIO()
    _stdout = sys.stdout
    try:
        sys.stdout = buf
        summary(model, input_size=input_size)
    finally:
        sys.stdout = _stdout
    return buf.getvalue()


# ----------------------------- main --------------------------------------------


def main() -> None:
    # ──────────────────────────────────────────────
    # 1) Paths & hyperparameters
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

    USE_LAGGED_STRESS = False  # False → 11 channels, True → 17 channels (teacher forcing)
    MODEL_DIR = f"models/patchtst_5pc_v5_{'17ch' if USE_LAGGED_STRESS else '11ch'}"
    os.makedirs(MODEL_DIR, exist_ok=True)

    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WARMUP_PCT = 0.05
    PATIENCE = 10
    NUM_WORKERS = 4
    MAX_SEQ_LEN = 1800
    SEED = 42

    set_seed(SEED)

    # ──────────────────────────────────────────────
    # 2) Configure & instantiate PatchTST
    # ──────────────────────────────────────────────
    config = PatchTSTConfig(
        num_input_channels=17 if USE_LAGGED_STRESS else 11,
        num_targets=6,                 # 6 stress outputs
        context_length=MAX_SEQ_LEN,    # look-back window
        prediction_length=1,           # pointwise prediction
        patch_length=16,
        patch_stride=16,               # no overlap
        d_model=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        ffn_dim=512,
        norm_type="layernorm",         # transformer uses LayerNorm (not batch norm)
        loss="mse",                    # model's internal loss (unused here; we use SmoothL1)
        scaling=None,                  # no internal scaling (we scale in the dataset)
        share_embedding=True,
        positional_encoding_type="sincos",
        attention_dropout=0.1,
        ff_dropout=0.1,
    )
    model = PatchTSTForRegression(config)

    # Capture model summary once
    model_summary_txt = capture_model_summary(model, input_size=(1, MAX_SEQ_LEN, config.num_input_channels))
    print("Model summary:\n" + "-" * 80 + f"\n{model_summary_txt}\n" + "-" * 80)

    # ──────────────────────────────────────────────
    # 3) DataLoaders (train/val)
    # ──────────────────────────────────────────────
    train_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        split="train",
        split_ratio=0.8,
        seed=SEED,
        use_lagged_stress=USE_LAGGED_STRESS,
    )
    # Reuse the fitted scalers for validation
    in_scaler = getattr(train_loader.dataset, "input_scaler", None)
    out_scaler = getattr(train_loader.dataset, "target_scaler", None)

    val_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        split="val",
        split_ratio=0.8,
        seed=SEED,
        use_lagged_stress=USE_LAGGED_STRESS,
        input_scaler=in_scaler,
        target_scaler=out_scaler,
    )

    # ──────────────────────────────────────────────
    # 4) Device & optimisation
    # ──────────────────────────────────────────────
    device = select_device()
    model.to(device)
    print(f"Using device: {device.type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.SmoothL1Loss()  # Huber loss (δ=1)

    total_steps = MAX_EPOCHS * max(1, len(train_loader))
    warmup_steps = int(WARMUP_PCT * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    process = psutil.Process(os.getpid())
    mem_before_train = process.memory_info().rss
    t_train_start = time.time()

    best_val_r2 = -float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(MODEL_DIR, "patchtst_best.pt")

    # ──────────────────────────────────────────────
    # 5) Training loop with early stopping
    # ──────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        # CompositeStressDataset yields: (inputs, mask, times, targets)
        for bx, _bm, _bt, by in tqdm(train_loader, desc=f"Train {epoch}/{MAX_EPOCHS}", leave=False):
            bx = bx.to(device, non_blocking=True)        # [B, T, C]
            by = by.to(device, non_blocking=True)        # [B, T, 6]

            optimizer.zero_grad(set_to_none=True)
            out = model(past_values=bx)                  # [B, 1, 6]
            pred = out.regression_outputs.squeeze(1)     # [B, 6]
            true = by[:, -1, :]                          # final time step

            loss = criterion(pred, true)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * bx.size(0)

        train_loss /= max(1, len(train_loader.dataset))

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        all_v_preds, all_v_trues = [], []

        with torch.no_grad():
            for vx, _vm, _vt, vy in val_loader:
                vx = vx.to(device, non_blocking=True)
                vy = vy.to(device, non_blocking=True)

                out = model(past_values=vx)
                vpred = out.regression_outputs.squeeze(1)
                vtrue = vy[:, -1, :]

                val_loss += criterion(vpred, vtrue).item() * vx.size(0)
                all_v_preds.append(vpred.detach().cpu().numpy())
                all_v_trues.append(vtrue.detach().cpu().numpy())

        val_loss /= max(1, len(val_loader.dataset))
        all_v_preds = np.vstack(all_v_preds)
        all_v_trues = np.vstack(all_v_trues)

        # macro R^2 across six components
        val_r2 = float(np.mean([r2_score(all_v_trues[:, i], all_v_preds[:, i]) for i in range(all_v_preds.shape[1])]))

        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_R2={val_r2:.4f} | {dt:.1f}s"
        )

        # early stopping on R^2 (maximize)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} (val_R2={best_val_r2:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no R^2 improvement for {PATIENCE} epochs).")
                break

    t_train_end = time.time()
    mem_after_train = process.memory_info().rss

    # ──────────────────────────────────────────────
    # 6) Single-batch inference profiling
    # ──────────────────────────────────────────────
    model.eval()
    sample_x, _sample_mask, _sample_time, _sample_y = next(iter(train_loader))

    mem_before_inf = process.memory_info().rss
    t_inf_start = time.time()
    with torch.no_grad():
        _ = model(past_values=sample_x.to(device))
    t_inf_end = time.time()
    mem_after_inf = process.memory_info().rss

    # ──────────────────────────────────────────────
    # 7) Persist artefacts
    # ──────────────────────────────────────────────
    # Scalers
    input_scaler_path = os.path.join(MODEL_DIR, "input_scaler.json")
    target_scaler_path = os.path.join(MODEL_DIR, "target_scaler.json")
    with open(input_scaler_path, "w") as f:
        json.dump(in_scaler.to_dict(), f, indent=2)
    with open(target_scaler_path, "w") as f:
        json.dump(out_scaler.to_dict(), f, indent=2)
    print(f"Saved scalers to:\n  {input_scaler_path}\n  {target_scaler_path}")

    # Model summary
    summary_txt_path = os.path.join(MODEL_DIR, "patchtst_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(model_summary_txt)
    print(f"Saved model summary: {summary_txt_path}")

    # Mode config (11ch vs 17ch)
    mode_path = os.path.join(MODEL_DIR, "mode_config.json")
    with open(mode_path, "w") as f:
        json.dump({"use_lagged_stress": USE_LAGGED_STRESS}, f, indent=2)
    print(f"Saved mode config: {mode_path}")

    # Resource usage
    resource_log = {
        "train_time_s": round(t_train_end - t_train_start, 2),
        "train_mem_diff_mb": round(bytes_to_mb(mem_after_train - mem_before_train), 2),
        "inf_time_s": round(t_inf_end - t_inf_start, 4),
        "inf_mem_diff_mb": round(bytes_to_mb(mem_after_inf - mem_before_inf), 2),
        "best_val_R2": round(best_val_r2, 4),
    }
    log_path = os.path.join(MODEL_DIR, "resource_log.json")
    with open(log_path, "w") as f:
        json.dump(resource_log, f, indent=2)
    print(f"Saved resource log: {log_path}")

    # Final summary print
    print("Resource summary:")
    for k, v in resource_log.items():
        unit = "MB" if "mem" in k else ("s" if "time" in k else "")
        print(f"  {k:18} {v} {unit}")


if __name__ == "__main__":
    main()
