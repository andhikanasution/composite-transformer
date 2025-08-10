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
from src.dataloader import make_train_val_loaders

def compute_metrics(y_true, y_pred):
    """Compute per-channel RMSE and R² over [N, C] arrays."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2   = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return rmse, r2

def main():
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH, EPOCHS, LR = 32, 30, 1e-4
    SEQ_LEN = 200
    LABEL_LEN = 1  # time-marching start token = y[:,0]
    PRED_LEN = SEQ_LEN - LABEL_LEN  # predict timesteps 1..199

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

    # 1) DataLoaders (train scalers -> reused for val), use lagged stress channels
    train_loader, val_loader, input_scaler, target_scaler = make_train_val_loaders(
        INPUT_CSV, DATA_DIR, max_seq_len=SEQ_LEN, batch_size=BATCH,
        split_ratio = 0.8, seed = 42, use_lagged_stress = True
    )

    class InformerTimeMarching(nn.Module):
        def __init__(self, enc_in, dec_in, c_out,
                     seq_len=200, label_len=1, pred_len=199):
            super().__init__()
            self.dec_in = dec_in
            self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
            self.net = Informer(
                enc_in=enc_in,
                dec_in=dec_in,
                c_out=c_out,
                seq_len=seq_len,
                label_len=label_len,
                out_len=pred_len,
                factor=5,
                d_model=128,
                n_heads=4,
                e_layers=3,
                d_layers=2,
                d_ff=512,
                dropout=0.1,
                attn='prob',
                embed='fixed',
                freq='s',
                activation='gelu',
                output_attention=False,
                distil=True,
                mix=True
            )


        def forward(self, x_enc, y_start_token):
            """
            x_enc:        [B, seq_len, enc_in]  (includes lagged σ(t-1) channels)
            y_start_token:[B, 1, dec_in]       (true σ at t=0)
            Returns:      [B, pred_len, dec_in] (t=1..T-1)
            """
            B = int(x_enc.shape[0])
            # zero time-marks (embed=fixed -> unused but kept for API)
            x_mark_enc = x_enc.new_zeros((B, self.seq_len, 5))
            x_mark_dec = x_enc.new_zeros((B, self.label_len + self.pred_len, 5))
            # decoder input: start token + zeros
            x_dec = x_enc.new_zeros((B, self.label_len + self.pred_len, self.dec_in))
            x_dec[:, :self.label_len, :] = y_start_token
            # no custom masks; Informer handles causal masks internally
            out = self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, dec_in]
            return out

    # scalers for inversion
    target_scaler = target_scaler

    # 2) Model
    enc_in = train_loader.dataset.inputs[0].shape[1]   # 11
    dec_in = train_loader.dataset.targets[0].shape[1]  # 6
    model = InformerTimeMarching(enc_in, dec_in, dec_in,
                                 seq_len = SEQ_LEN,
                                 label_len = LABEL_LEN,
                                 pred_len = PRED_LEN).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    proc     = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss
    t0         = time.time()

    # scheduled sampling schedule (teacher-forcing ratio)
    def tf_ratio(epoch, total):
        start, end = 1.0, 0.3
        if total <= 1: return end
        alpha = (epoch - 1) / (total - 1)
        return start + (end - start) * alpha

    lag_slice = slice(enc_in - dec_in, enc_in)  # last 6 channels are lagged σ(t-1)

    # ───── Training Loop ───────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        tfp = tf_ratio(epoch, EPOCHS)  # probability to keep TRUE lag; (1-tfp) use predicted lag

        for x, pad_mask, times, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            # x includes lagged σ(t-1) (teacher forcing version from dataset)
            x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
            B = x.size(0)

            # --- pass 1: predictions with teacher-forced lag (no grad to pred) ---
            y0 = y[:, :1, :]  # start token (true σ at t=0)
            with torch.no_grad():
                pred_tf = model(x, y0)  # [B, PRED_LEN, 6] -> predicts t=1..T-1
                # shift to align as lag for times t>=1
                lag_pred = torch.zeros(B, SEQ_LEN, dec_in, device=DEVICE)
                lag_pred[:, 1:, :] = pred_tf  # lag for t is pred at t-1

            optimizer.zero_grad()

            # --- scheduled sampling: mix true lag (already in x) with predicted lag ---
            if tfp < 1.0:
                # Bernoulli mask per (B, T, 1) for t>=1
                keep_true = torch.ones(B, SEQ_LEN, 1, device=DEVICE)
                keep_true[:, 1:, :] = (torch.rand(B, SEQ_LEN - 1, 1, device=DEVICE) < tfp).float()
                mixed_lag = keep_true * x[:, :, lag_slice] + (1.0 - keep_true) * lag_pred
                x_mixed = x.clone()
                x_mixed[:, :, lag_slice] = mixed_lag
            else:
                x_mixed = x

            # pass 2: trainable forward with mixed lag
            pred = model(x_mixed, y0)  # [B, PRED_LEN, 6]; corresponds to t=1..T-1

            # Masked MSE over all timesteps & channels
            # Align targets/pads to t=1..T-1
            y_target = y[:, 1:, :]  # [B, PRED_LEN, 6]
            mask_pred = pad_mask[:, 1:]  # [B, PRED_LEN]
            pred_flat = pred.reshape(-1, dec_in)
            y_flat = y_target.reshape(-1, dec_in)
            mask_flat = mask_pred.reshape(-1, 1).expand(-1, dec_in)
            loss = ((pred_flat - y_flat) ** 2)[mask_flat].mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B

        train_loss /= len(train_loader.dataset)

        # ───── Validation Loop ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []

        with torch.no_grad():
            for x, pad_mask, times, y in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
                B = x.size(0)

                # *** one refinement step at val: replace lag with predicted prev ***
                y0 = y[:, :1, :]
                pred0 = model(x, y0)  # [B, PRED_LEN, 6]
                lag_pred = torch.zeros(B, SEQ_LEN, dec_in, device=DEVICE)
                lag_pred[:, 1:, :] = pred0
                x_ref = x.clone()
                x_ref[:, :, lag_slice] = lag_pred
                pred = model(x_ref, y0)  # refined

                # loss on t>=1 only
                y_target = y[:, 1:, :]
                mask_pred = pad_mask[:, 1:]
                pred_flat = pred.reshape(-1, dec_in)
                y_flat = y_target.reshape(-1, dec_in)
                mask_flat = mask_pred.reshape(-1, 1).expand(-1, dec_in)
                loss = ((pred_flat - y_flat) ** 2)[mask_flat].mean()
                val_loss += loss.item() * B

                # collect real‐timesteps for metrics (physical units)
                pred_np = pred.cpu().numpy().reshape(-1, dec_in)
                y_np = y_target.cpu().numpy().reshape(-1, dec_in)
                m_np = pad_mask[:, 1:].cpu().numpy().reshape(-1)

                idx = m_np.astype(bool)
                all_preds.append(pred_np[idx])
                all_trues.append(y_np   [idx])

        val_loss /= len(val_loader.dataset)

        # compute & log metrics
        all_preds   = np.vstack(all_preds)
        all_trues   = np.vstack(all_trues)
        phys_preds = target_scaler.inverse_transform(all_preds)
        phys_trues = target_scaler.inverse_transform(all_trues)
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
    # also persist scalers for later use
    if input_scaler is not None and target_scaler is not None:
        with open(os.path.join(MODEL_DIR, "scalers.json"), "w") as f:
            json.dump({"input": input_scaler.to_dict(), "target": target_scaler.to_dict()}, f, indent=2)
    print("Done. Resources:", resources)

if __name__ == "__main__":
    main()
