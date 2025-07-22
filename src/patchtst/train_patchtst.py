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
from transformers import PatchTSTConfig, PatchTSTForRegression

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
    MODEL_DIR = "models/patchtst_5pc_v2"
    os.makedirs(MODEL_DIR, exist_ok=True)

    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    MAX_SEQ_LEN = 1800
    SEED = 42  # Ensures reproducible train/val split

    # ──────────────────────────────────────────────
    # 2) Configure & Instantiate PatchTST
    # ──────────────────────────────────────────────
    config = PatchTSTConfig(
        num_input_channels=11,  # 6 strain + 5 static metadata
        num_targets=6,  # 6 stress outputs
        context_length=MAX_SEQ_LEN,  # look-back window
        prediction_length=1,  # point-wise
        patch_length=16,  # patch size
        patch_stride=16,  # no overlap
        d_model=128,  # embedding dim
        num_hidden_layers=3,  # transformer depth
        num_attention_heads=4,  # heads per layer
        ffn_dim=512,  # feed-forward dim
        norm_type="batchnorm",  # batch normalization
        loss="mse",  # objective
        scaling="None",  # enable internal standardisation
        share_embedding=True,  # share embeddings across channels
        positional_encoding_type="sincos"  # sinusoidal positional encodings
    )
    model = PatchTSTForRegression(config)

    # Capture & print model summary
    model_summary = capture_model_summary(model, input_size=(1, MAX_SEQ_LEN, config.num_input_channels))
    print("=== Model Summary ===\n" + model_summary)

    # ──────────────────────────────────────────────
    # 3) Prepare DataLoader
    # ──────────────────────────────────────────────
    train_loader = get_dataloader(
        input_csv_path=INPUT_CSV,
        data_dir=DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        split="train",          # Only load the training portion
        split_ratio=0.8,        # 80% train, 20% val
        seed=SEED               # Fixed split for reproducibility
    )
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
    criterion = torch.nn.MSELoss()

    # For memory profiling
    process = psutil.Process(os.getpid())
    mem_before_train = process.memory_info().rss
    t_train_start = time.time()

    # ──────────────────────────────────────────────
    # 5) Training Loop with Progress Bar
    # ──────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for batch_x, batch_y in loop:
            optimizer.zero_grad()

            outputs = model(past_values=batch_x.to(device))
            preds = outputs.regression_outputs.squeeze(1)  # shape: [B, 6]
            truths = batch_y[:, -1, :].to(device)  # true stress at final step

            loss = criterion(preds, truths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:>2}/{NUM_EPOCHS} — avg_loss: {avg_loss:.6f} — time: {epoch_time:.1f}s")

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
    # a) Model weights
    model_path = os.path.join(MODEL_DIR, "patchtst_5pc.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Saved model weights: {model_path}")

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
    # Save the pretty ASCII model summary to its own .txt
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

    # Display summary
    print("=== Resource Summary ===")
    for key, val in resource_log.items():
        if key == "model_summary":
            continue
        unit = "MB" if "mem" in key else "s"
        print(f"{key:22}: {val} {unit}")


if __name__ == "__main__":
    main()
