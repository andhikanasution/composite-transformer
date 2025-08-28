"""
Sliding-window Dataset for Informer-style encoder–decoder training.

For each window starting at s:
  - Encoder input  x_enc: [seq_len, 17]  = concat(past_y[6], past_exo[11])
  - Decoder input  x_dec: [label_len+pred_len, 17]
        * first label_len rows include history: y + exo
        * future rows include only exo; future y slots are zeros
  - Targets       y_true: [pred_len, 6]
Time-mark embeddings (xme/xmd) are provided as zeros (use embed='fixed' in the model).
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils_parsing import load_metadata, build_sample_mapping, load_time_series
from src.normalisation import StandardScaler


class CompositeInformerWindows(Dataset):
    """
    Windowed dataset for Informer. No padding/masks; short sequences are skipped.
    """

    def __init__(
        self,
        input_csv_path: str,
        data_dir: str,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        stride: int = 24,
        scale: bool = True,
        split: str = "train",
        split_ratio: float = 0.8,
        seed: int = 42,
        exo_scaler: Optional[StandardScaler] = None,
        y_scaler: Optional[StandardScaler] = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride
        self.scale = scale

        # --- Metadata (θ, LP1..LP4) and file mapping -----------------------------
        metadata = load_metadata(input_csv_path)  # shape: [N_cases, 5]
        mapping = build_sample_mapping(data_dir)  # dict: idx -> csv path

        # --- Split by case index (RVE), not by window --------------------------------
        n = len(mapping)
        idxs = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)
        sp = int(split_ratio * n)
        chosen = idxs[:sp] if split == "train" else idxs[sp:] if split == "val" else idxs
        self.case_indices = chosen

        # --- Fit/reuse scalers on real timesteps (no padding) ------------------------
        if scale:
            if split == "train" and (exo_scaler is None or y_scaler is None):
                exo_list, y_list = [], []
                for i in chosen:
                    E, Y, _t = load_time_series(mapping[i])  # E: [T,6], Y: [T,6]
                    exo = np.hstack([E, np.tile(metadata[i], (E.shape[0], 1))])  # [T,11]
                    exo_list.append(exo)
                    y_list.append(Y)
                exo_stack = np.vstack(exo_list).astype(np.float32)
                y_stack = np.vstack(y_list).astype(np.float32)
                self.exo_scaler = StandardScaler(exo_stack.mean(0), exo_stack.std(0))
                self.y_scaler = StandardScaler(y_stack.mean(0), y_stack.std(0))
            else:
                if exo_scaler is None or y_scaler is None:
                    raise ValueError("Val/test must be given scalers computed on the train split.")
                self.exo_scaler = exo_scaler
                self.y_scaler = y_scaler
        else:
            self.exo_scaler = None
            self.y_scaler = None

        # --- Build windows -----------------------------------------------------------
        self.x_enc: List[torch.Tensor] = []
        self.x_dec: List[torch.Tensor] = []
        self.xme: List[torch.Tensor] = []  # encoder time marks
        self.xmd: List[torch.Tensor] = []  # decoder time marks
        self.y_true: List[torch.Tensor] = []

        for i in chosen:
            E, Y, _t = load_time_series(mapping[i])  # E: [T,6], Y: [T,6]
            exo = np.hstack([E, np.tile(metadata[i], (E.shape[0], 1))]).astype(np.float32)  # [T,11]

            if scale:
                exo = self.exo_scaler.transform(exo)
                Y = self.y_scaler.transform(Y.astype(np.float32))

            T = Y.shape[0]
            needed = self.seq_len + self.pred_len
            if T < needed:
                # Skip cases that cannot provide a full window
                continue

            for s in range(0, T - needed + 1, self.stride):
                # Encoder: past_y + past_exo
                past_y = Y[s : s + self.seq_len]            # [seq_len, 6]
                past_exo = exo[s : s + self.seq_len]        # [seq_len, 11]
                xenc = np.concatenate([past_y, past_exo], axis=1)  # [seq_len, 17]

                # Decoder: history (label_len) with y+exo, then future exo only
                xdec = np.zeros((self.label_len + self.pred_len, 17), dtype=np.float32)
                hist_y = Y[s + self.seq_len - self.label_len : s + self.seq_len]
                hist_exo = exo[s + self.seq_len - self.label_len : s + self.seq_len]
                fut_exo = exo[s + self.seq_len : s + self.seq_len + self.pred_len]

                xdec[: self.label_len, :6] = hist_y
                xdec[: self.label_len, 6:] = hist_exo
                xdec[self.label_len :, 6:] = fut_exo

                # Target for the forecast horizon
                ytrue = Y[s + self.seq_len : s + self.seq_len + self.pred_len]  # [pred_len, 6]

                # Time-marks (zeros; use Informer 'fixed' embedding)
                xme = np.zeros((self.seq_len, 5), dtype=np.float32)
                xmd = np.zeros((self.label_len + self.pred_len, 5), dtype=np.float32)

                # Store tensors
                self.x_enc.append(torch.tensor(xenc, dtype=torch.float32))
                self.x_dec.append(torch.tensor(xdec, dtype=torch.float32))
                self.xme.append(torch.tensor(xme, dtype=torch.float32))
                self.xmd.append(torch.tensor(xmd, dtype=torch.float32))
                self.y_true.append(torch.tensor(ytrue, dtype=torch.float32))

    # ---- PyTorch Dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.y_true)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_enc: [seq_len, 17]
            x_dec: [label_len + pred_len, 17]
            xme:   [seq_len, 5]               (time-marks; zeros)
            xmd:   [label_len + pred_len, 5]  (time-marks; zeros)
            y_true:[pred_len, 6]
        """
        return self.x_enc[i], self.x_dec[i], self.xme[i], self.xmd[i], self.y_true[i]
