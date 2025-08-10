#!/usr/bin/env python3
# train_informer_pointwise.py

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
    """Both are [N_total, c_out]"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2 = [r2_score(y_true[:, i], y_pred[:, i])
          for i in range(y_true.shape[1])]
    return rmse, r2

def main():
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Hyperparams
    BATCH_SIZE, MAX_EPOCHS, LR = 32, 30, 1e-4
    SEQ_LEN = 200
    LABEL_LEN = 1  # start token length
    PRED_LEN = SEQ_LEN  # predict full sequence

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
    MODEL_DIR  = "models/informer_pointwise_v3"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) DataLoaders (train scalers -> reused for val), NO lagged stress for pointwise
    train_loader, val_loader, input_scaler, target_scaler = make_train_val_loaders(
        INPUT_CSV, DATA_DIR, max_seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        split_ratio = 0.8, seed = 42, use_lagged_stress = False
    )

    # 2) Model wrapper
    class InformerForPointwise(nn.Module):
        def __init__(self, enc_in, dec_in, c_out):
            super().__init__()
            self.dec_in = int(dec_in)
            self.seq_len = SEQ_LEN
            self.label_len = LABEL_LEN
            self.pred_len = PRED_LEN
            self.net = Informer(
                enc_in=enc_in,
                dec_in=dec_in,
                c_out=c_out,
                seq_len=self.seq_len,
                label_len = self.label_len,
                out_len = self.pred_len,
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

        def forward(self, x_enc):
            """
            Pointwise mapping: given encoder inputs for all timesteps,
            feed the decoder with a dummy start token + the same known exogenous inputs.
            """

            B = int(x_enc.shape[0])
            # time marks are zeros (embed='fixed')
            x_mark_enc = x_enc.new_zeros((B, self.seq_len, 5))
            x_mark_dec = x_enc.new_zeros((B, self.label_len + self.pred_len, 5))
            # decoder input: [zero start token; known exogenous per timestep]
            x_dec = x_enc.new_zeros((B, self.label_len + self.pred_len, self.dec_in))
            x_dec[:, self.label_len:, :] = x_enc  # copy encoder exogenous into decoder future
            # Informer outputs [B, pred_len, c_out]

            return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # instantiate
    enc_in = train_loader.dataset.inputs[0].shape[1]  # 11 (E1..E6 + theta + LP1..LP4)
    dec_in = enc_in  # decoder sees same exogenous
    model  = InformerForPointwise(enc_in, dec_in, dec_in).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss
    t_start    = time.time()

    # ─── TRAINING ───────────────────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x, pad_mask, times, y in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]"):
            x, pad_mask, times, y = x.to(DEVICE), pad_mask.to(DEVICE), times.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)  # [B, SEQ_LEN, C]

            # masked MSE
            B, T, C = pred.shape  # T == PRED_LEN == SEQ_LEN
            pred_flat = pred.view(-1, C)
            y_flat    = y.view(-1, C)
            mask_flat = pad_mask.view(-1, 1).expand(-1, C)

            loss = ((pred_flat - y_flat)**2)[mask_flat].mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B

        train_loss /= len(train_loader.dataset)

        # ─── VALIDATION ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []
        last_preds, last_trues = [], []

        with torch.no_grad():
            for x, pad_mask, times, y in tqdm(val_loader, desc=f"Epoch {epoch} [VAL]"):
                x, pad_mask, y = x.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
                pred = model(x)

                # masked MSE
                B, T, C = pred.shape
                pred_flat = pred.reshape(-1, C)
                y_flat = y[:, :T, :].reshape(-1, C)
                mask_flat = pad_mask[:, :T].reshape(-1, 1).expand(-1, C)
                loss = ((pred_flat - y_flat)**2)[mask_flat].mean()
                val_loss += loss.item() * B

                # bring everything to NumPy
                pred_np = pred.cpu().numpy()  # [B, T, C]
                y_np = y[:, :T, :].cpu().numpy()  # [B, T, C]
                mask_np = pad_mask[:, :T].cpu().numpy()  # [B, T]

                # flatten batch+time
                pred_flat = pred_np.reshape(-1, C)  # [B*T, C]
                y_flat = y_np.reshape(-1, C)
                mask_flat = mask_np.reshape(-1)  # [B*T]

                # only keep real timesteps for full‐seq metrics
                real_idx = mask_flat.astype(bool)
                all_preds.append(pred_flat[real_idx])
                all_trues.append(y_flat[real_idx])

                # final-step: only those samples where the last timestep is real
                valid_final = mask_np[:, -1].astype(bool)  # [B]
                pred_last_np = pred_np[valid_final, -1, :]  # [n_valid, C]
                y_last_np = y_np[valid_final, -1, :]
                last_preds.append(pred_last_np)
                last_trues.append(y_last_np)

        val_loss /= len(val_loader.dataset)

        # metrics
        all_preds   = np.vstack(all_preds)
        all_trues   = np.vstack(all_trues)
        last_preds  = np.vstack(last_preds)
        last_trues  = np.vstack(last_trues)

        all_preds_phys   = target_scaler.inverse_transform(all_preds)
        all_trues_phys   = target_scaler.inverse_transform(all_trues)
        last_preds_phys  = target_scaler.inverse_transform(last_preds)
        last_trues_phys  = target_scaler.inverse_transform(last_trues)

        rmse_full, r2_full   = compute_metrics(all_trues_phys,   all_preds_phys)
        rmse_last, r2_last   = compute_metrics(last_trues_phys,  last_preds_phys)

        print(f"\nEpoch {epoch}: Train Loss {train_loss:.4f} — Val Loss {val_loss:.4f}")
        print(f"  → Full-seq RMSE: {['{:.3f}'.format(r) for r in rmse_full]}")
        print(f"     Full-seq R² : {['{:.3f}'.format(r) for r in r2_full]}")
        print(f"  → Last-step RMSE: {['{:.3f}'.format(r) for r in rmse_last]}")
        print(f"     Last-step R² : {['{:.3f}'.format(r) for r in r2_last]}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_pointwise.pt"))
            print("  💾 Saved new best model!")

    # ─── RESOURCE LOGGING & SCALERS ─────────────────────────────────────────────
    t_end     = time.time()
    mem_after = proc.memory_info().rss
    resources = {
        "train_time_s"          : round(t_end - t_start, 2),
        "train_mem_increase_mb" : round((mem_after - mem_before) / 1024**2, 2),
    }
    with open(os.path.join(MODEL_DIR, "resource_log.json"), "w") as f:
        json.dump(resources, f, indent=2)
    print("Resources:", resources)

    # persist scalers together
    with open(os.path.join(MODEL_DIR, "scalers.json"), "w") as f:
        json.dump({"input": input_scaler.to_dict(),
                   "target": target_scaler.to_dict()}, f, indent=2)

if __name__ == "__main__":
    main()
