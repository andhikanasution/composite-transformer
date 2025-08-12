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
from src.dataloader import make_train_val_loaders, get_dataloader
from informer.models.attn import TriangularCausalMask


class SimpleMask:
    """Wrap a [B,1,Lq,Lk] boolean tensor with a .mask attribute, like Informer masks."""
    def __init__(self, mask: torch.Tensor):
        self.mask = mask


def compute_metrics(y_true, y_pred):
    """Compute per-channel RMSE and R² over [N, C] arrays."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    r2   = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return rmse, r2

def main():
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH, EPOCHS, LR = 32, 10, 1e-4
    SEQ_LEN = 200
    LABEL_LEN = 1  # time-marching start token = y[:,0]
    PRED_LEN = SEQ_LEN - LABEL_LEN  # predict timesteps 1..199
    N_REFINE = 8  # number of inference-style refinements for validation

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
    MODEL_DIR = "models/informer_timemarching_v3"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Train loader: lagged stress channels ON (for scheduled sampling)
    train_loader, _val_ignored, input_scaler, target_scaler = make_train_val_loaders(
        INPUT_CSV, DATA_DIR, max_seq_len=SEQ_LEN, batch_size=BATCH,
        split_ratio = 0.8, seed = 42, use_lagged_stress = True
    )

    # 1b) Inference-style validation loader: lagged stress OFF (use same scalers!)
    val_infer_loader = get_dataloader(
        input_csv_path = INPUT_CSV, data_dir = DATA_DIR, max_seq_len = SEQ_LEN,
        batch_size = BATCH, shuffle = False, num_workers = 2, scale = True,
        split = "val", split_ratio = 0.8, seed = 42, use_lagged_stress = False,
        input_scaler = input_scaler, target_scaler = target_scaler
    )

    class InformerTimeMarching(nn.Module):
        def __init__(self, enc_in, dec_in, c_out, seq_len=200, label_len=1, pred_len=199):
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
                d_model=192,
                n_heads=4,
                e_layers=3,
                d_layers=2,
                d_ff=1024,
                dropout=0.1,
                attn='full',
                embed='fixed',
                freq='s',
                activation='gelu',
                output_attention=False,
                distil=False,
                mix=True
            )

        def forward(self, x_enc, y_start_token, pad_mask):
            """
            x_enc: [B, seq_len, enc_in]
            y_start_token: [B, 1, dec_in]
            pad_mask: [B, seq_len]  (True=real, False=pad)
            """
            B = int(x_enc.shape[0])
            device = x_enc.device

            # time marks (unused with embed='fixed', but API requires them)
            x_mark_enc = x_enc.new_zeros((B, self.seq_len, 5))
            Tdec = self.label_len + self.pred_len
            x_mark_dec = x_enc.new_zeros((B, Tdec, 5))

            # decoder input = [start token ; zeros]
            x_dec = x_enc.new_zeros((B, Tdec, self.dec_in))
            x_dec[:, :self.label_len, :] = y_start_token

            # ── Encoder self-mask: causal + block padded encoder keys ────────────────
            # TriangularCausalMask returns shape [B,1,L,L]
            enc_tri = TriangularCausalMask(B, self.seq_len, device=device).mask
            if pad_mask is not None:
                # block padded *keys* across all queries
                enc_key_pad = (~pad_mask[:, :self.seq_len]).unsqueeze(1).unsqueeze(1) \
                    .expand(B, 1, self.seq_len, self.seq_len)  # [B,1,L,L]
                enc_self_mask = SimpleMask(enc_tri | enc_key_pad)
            else:
                enc_self_mask = SimpleMask(enc_tri)

            # ── Decoder self-mask: standard causal ───────────────────────────────────
            dec_self_mask = TriangularCausalMask(B, Tdec, device=device)  # already an object

            # ── Decoder–encoder cross-mask: block padded encoder keys ────────────────
            if pad_mask is not None:
                dec_enc_pad = (~pad_mask[:, :self.seq_len]).unsqueeze(1).unsqueeze(1) \
                    .expand(B, 1, Tdec, self.seq_len)  # [B,1,Tdec,Lenc]
                dec_enc_mask = SimpleMask(dec_enc_pad)
            else:
                dec_enc_mask = None

            # forward
            out = self.net(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=enc_self_mask,
                dec_self_mask=dec_self_mask,
                dec_enc_mask=dec_enc_mask
            )  # [B, pred_len, c_out]
            return out

    # 2) Model
    enc_in = train_loader.dataset.inputs[0].shape[1]   # 11
    dec_in = train_loader.dataset.targets[0].shape[1]  # 6
    model = InformerTimeMarching(enc_in, dec_in, dec_in,
                                 seq_len = SEQ_LEN,
                                 label_len = LABEL_LEN,
                                 pred_len = PRED_LEN).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_infer = float("inf")
    proc     = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss
    t0         = time.time()

    # scheduled sampling schedule (teacher-forcing ratio)
    def tf_ratio(epoch, total):
        start, end = 1.0, 0.0
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
                pred_tf = model(x, y0, pad_mask)  # [B, PRED_LEN, 6] -> predicts t=1..T-1
                # shift to align as lag for times t>=1
                lag_pred = x.new_zeros((B, SEQ_LEN, dec_in))
                lag_pred[:, 1:, :] = pred_tf  # lag for t is pred at t-1

            optimizer.zero_grad()

            # --- scheduled sampling: mix true lag (already in x) with predicted lag ---
            if tfp < 1.0:
                # Bernoulli mask per (B, T, 1) for t>=1
                keep_true = x.new_zeros((B, SEQ_LEN, 1)) + 1.0
                keep_true[:, 1:, :] = (torch.rand(B, SEQ_LEN - 1, 1, device=x.device) < tfp).float()
                mixed_lag = keep_true * x[:, :, lag_slice] + (1.0 - keep_true) * lag_pred
                x_mixed = x.clone()
                x_mixed[:, :, lag_slice] = mixed_lag
            else:
                x_mixed = x

            # ── Zero-lag augmentation (30% of updates): simulate a cold start
            if torch.rand(()) < 0.30:
                x_for_train = x.clone()
                x_for_train[:, :, lag_slice] = 0.0
            else:
                x_for_train = x_mixed

            # pass 2: trainable forward with mixed lag
            pred = model(x_for_train, y0, pad_mask)  # [B, PRED_LEN, 6]; corresponds to t=1..T-1

            # Masked MSE over all timesteps & channels
            # Align targets/pads to t=1..T-1
            y_target = y[:, 1:, :]  # [B, PRED_LEN, C]
            mask_pred = pad_mask[:, 1:]  # [B, PRED_LEN]
            pred_flat = pred.reshape(-1, dec_in)
            y_flat = y_target.reshape(-1, dec_in)
            mask_flat = mask_pred.reshape(-1, 1).expand(-1, dec_in)

            # Robust data term (Huber / SmoothL1) on real timesteps
            huber = torch.nn.SmoothL1Loss(reduction="none")
            loss_data = huber(pred_flat, y_flat)[mask_flat].mean()

            # Smoothness term on predicted sequence (finite differences)
            if pred.size(1) > 1:
                diff = pred[:, 1:, :] - pred[:, :-1, :]
                mask_diff = mask_pred[:, 1:].unsqueeze(-1).expand_as(diff)
                loss_smooth = (diff ** 2)[mask_diff].mean()
            else:
                loss_smooth = torch.tensor(0.0, device=pred.device)

            # Combined loss (tune λ: start 0.01, try 0.005–0.05)
            loss = loss_data + 0.01 * loss_smooth

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B

        train_loss /= len(train_loader.dataset)

        # ───── Inference-style Validation (no teacher lag, multi-refine) ─────
        model.eval()
        val_infer_loss = 0.0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x_base, pad_mask, times, y in tqdm(val_infer_loader, desc=f"Val-Infer Epoch {epoch}"):
                x_base, pad_mask, y = x_base.to(DEVICE), pad_mask.to(DEVICE), y.to(DEVICE)
                B = int(x_base.shape[0])
                y0 = y[:, :1, :]
                # start with zero lag
                lag = x_base.new_zeros((B, SEQ_LEN, dec_in))
                # multi-refinement loop
                for _ in range(N_REFINE):
                    x_full = torch.cat([x_base, lag], dim=-1)
                    pred = model(x_full, y0, pad_mask)  # [B, PRED_LEN, 6]
                    lag = lag.clone()
                    lag[:, 1:, :] = pred  # update σ(t-1)
                # compute loss on t>=1 only
                y_target = y[:, 1:, :]
                mask_pred = pad_mask[:, 1:]
                pred_flat = pred.reshape(-1, dec_in)
                y_flat = y_target.reshape(-1, dec_in)
                mask_flat = mask_pred.reshape(-1, 1).expand(-1, dec_in)
                loss = ((pred_flat - y_flat) ** 2)[mask_flat].mean()
                val_infer_loss += loss.item() * B
                # collect for metrics
                pred_np = pred.cpu().numpy().reshape(-1, dec_in)
                y_np = y_target.cpu().numpy().reshape(-1, dec_in)
                m_np = mask_pred.cpu().numpy().reshape(-1)
                idx = m_np.astype(bool)
                all_preds.append(pred_np[idx])
                all_trues.append(y_np[idx])
        val_infer_loss /= len(val_infer_loader.dataset)
        all_preds = np.vstack(all_preds) if len(all_preds) else np.zeros((0, dec_in))
        all_trues = np.vstack(all_trues) if len(all_trues) else np.zeros((0, dec_in))
        phys_preds = target_scaler.inverse_transform(all_preds) if len(all_preds) else all_preds
        phys_trues = target_scaler.inverse_transform(all_trues) if len(all_trues) else all_trues
        rmse, r2 = compute_metrics(phys_trues, phys_preds) if len(all_preds) else ([0] * dec_in, [0] * dec_in)

        print(f"\nEpoch {epoch}  Train Loss {train_loss:.4f}  Val-Infer Loss {val_infer_loss:.4f}")
        print(f" → Val-Infer RMSE: {['{:.3f}'.format(v) for v in rmse]}")
        print(f" → Val-Infer R²  : {['{:.3f}'.format(v) for v in r2]}")

        # save best on INFERENCE-STYLE validation
        if val_infer_loss < best_val_infer:
            best_val_infer = val_infer_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_tm.pt"))
            print("  💾 Saved new best (inference-style) model.")

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
