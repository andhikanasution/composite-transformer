#!/usr/bin/env python3
# train_informer_tm.py

import os
import time
import json
import psutil

import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score

from informer.models.model import Informer
from src.dataloader import get_dataloader

def compute_metrics(y_true, y_pred):
    """Compute per-channel RMSE and R² over [N, C] arrays."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2   = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return rmse, r2

class InformerTimeMarching(nn.Module):
    def __init__(self, enc_in, dec_in, c_out,
                 seq_len=200, label_len=200, out_len=200):
        super().__init__()
        self.dec_in = dec_in
        self.seq_len, self.label_len, self.out_len = seq_len, label_len, out_len
        self.net = Informer(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            out_len=out_len,
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
        """
        x_enc: [B, seq_len, enc_in]
        x_dec: [B, label_len, dec_in]
        """
        B = x_enc.size(0)
        # time‐embeddings placeholders
        x_mark_enc = torch.zeros(B, self.seq_len, 5,      device=x_enc.device)
        x_mark_dec = torch.zeros(B, self.label_len, 5,    device=x_enc.device)
        # full seq2seq pass
        return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)

def main():
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH, EPOCHS, LR = 32, 20, 1e-4
    SEQ_LEN, LABEL_LEN, OUT_LEN = 200, 200, 200

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
    MODEL_DIR = "models/informer_timemarching_v2"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) DataLoaders
    train_loader = get_dataloader(INPUT_CSV, DATA_DIR, SEQ_LEN,
                                  batch_size=BATCH, shuffle=True,
                                  split="train", split_ratio=0.8,
                                  use_lagged_stress=False)
    val_loader   = get_dataloader(INPUT_CSV, DATA_DIR, SEQ_LEN,
                                  batch_size=BATCH, shuffle=False,
                                  split="val",   split_ratio=0.8,
                                  use_lagged_stress=False)

    # scalers for inversion
    target_scaler = train_loader.dataset.target_scaler

    # 2) Model
    enc_in = train_loader.dataset.inputs[0].shape[1]   # 11
    dec_in = train_loader.dataset.targets[0].shape[1]  # 6
    model  = InformerTimeMarching(enc_in, dec_in, dec_in,
                                  seq_len=SEQ_LEN,
                                  label_len=LABEL_LEN,
                                  out_len=OUT_LEN).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    proc     = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss
    t0         = time.time()

    # ───── Training Loop ───────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x, pad_mask, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
            B = x.size(0)

            # Build decoder input: first true stress at t=0, zeros thereafter
            x_dec = torch.zeros(B, LABEL_LEN, dec_in, device=DEVICE)
            x_dec[:, 0, :] = y[:, 0, :]  # σ^(0) = true stress at t=0

            optimizer.zero_grad()
            pred = model(x, x_dec)  # [B, OUT_LEN, C]

            # Masked MSE over all timesteps & channels
            pred_flat = pred.view(-1, dec_in)
            y_flat    = y   .view(-1, dec_in)
            mask_flat = pad_mask.view(-1, 1).expand(-1, dec_in)

            loss = ((pred_flat - y_flat)**2)[mask_flat].mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B

        train_loss /= len(train_loader.dataset)

        # ───── Validation Loop ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []

        with torch.no_grad():
            for x, pad_mask, y in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
                B = x.size(0)

                x_dec = torch.zeros(B, LABEL_LEN, dec_in, device=DEVICE)
                x_dec[:, 0, :] = y[:, 0, :]

                pred = model(x, x_dec)

                pred_flat = pred.view(-1, dec_in)
                y_flat    = y   .view(-1, dec_in)
                mask_flat = pad_mask.view(-1, 1).expand(-1, dec_in)

                loss = ((pred_flat - y_flat)**2)[mask_flat].mean()
                val_loss += loss.item() * B

                # collect real‐timesteps for metrics
                pred_np = pred.cpu().numpy().reshape(-1, dec_in)
                y_np    = y   .cpu().numpy().reshape(-1, dec_in)
                m_np    = pad_mask.cpu().numpy().reshape(-1)

                idx = m_np.astype(bool)
                all_preds.append(pred_np[idx])
                all_trues.append(y_np   [idx])

        val_loss /= len(val_loader.dataset)

        # compute & log metrics
        all_preds   = np.vstack(all_preds)
        all_trues   = np.vstack(all_trues)
        phys_preds  = target_scaler.inverse_transform(all_preds)
        phys_trues  = target_scaler.inverse_transform(all_trues)
        rmse, r2    = compute_metrics(phys_trues, phys_preds)

        print(f"\nEpoch {epoch}  Train Loss {train_loss:.4f}  Val Loss {val_loss:.4f}")
        print(f" → Val RMSE: {['{:.3f}'.format(v) for v in rmse]}")
        print(f" → Val R²  : {['{:.3f}'.format(v) for v in r2]}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_tm.pt"))

    # ───── Resource Logging & Scalers ──────────────────────────────────────────
    t1        = time.time()
    mem_after = proc.memory_info().rss
    resources = {
        "train_time_s":        round(t1 - t0, 2),
        "train_mem_increase_mb": round((mem_after - mem_before) / 1024**2, 2)
    }
    with open(os.path.join(MODEL_DIR, "resources.json"), "w") as f:
        json.dump(resources, f, indent=2)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_tm.pt"))
    print("Done. Resources:", resources)

if __name__ == "__main__":
    main()
