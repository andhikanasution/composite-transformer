"""
Padded, mask-aware Dataset for composite strain→stress prediction.

Each item returns:
    inputs: [T, Din]  (Din=11 or 17 if use_lagged_stress=True)
    mask:   [T]       (True for real timesteps, False for padding)
    times:  [T]
    targets:[T, 6]
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils_parsing import load_all_data
from src.normalisation import compute_stats, StandardScaler, apply_scaling


class CompositeStressDataset(Dataset):
    """
    PyTorch Dataset for padded sequences used by Transformer/LSTM baselines.

    Channels:
        - Inputs:  6 strain components + 1 θ (theta) + 4 lamination parameters = 11
                   (+6 lagged stress channels if use_lagged_stress=True → Din=17)
        - Targets: 6 stress components
    """

    def __init__(
        self,
        input_csv_path: str,
        data_dir: str,
        max_seq_len: int = 200,
        scale: bool = True,
        split: str = "all",
        split_ratio: float = 0.8,
        seed: int = 42,
        use_lagged_stress: bool = False,
        input_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
    ) -> None:
        """
        Args:
            input_csv_path: Path to IM78552_DATABASEInput.csv (metadata).
            data_dir: Directory containing per-RVE time-series CSVs.
            max_seq_len: Pad/truncate all sequences to this T.
            scale: If True, apply Z-score standardisation.
            split: {"train","val","all"}.
            split_ratio: Train/val split proportion (when split != "all").
            seed: RNG seed used for the index shuffle before splitting.
            use_lagged_stress: Append lagged stress channels for teacher forcing.
            input_scaler/target_scaler: If provided, reuse; otherwise fitted on train split.
        """
        super().__init__()

        # 1) Load raw (unpadded) sequences and masks
        inputs_raw, targets_raw, times_raw, masks_raw = load_all_data(
            input_csv_path, data_dir, max_seq_len
        )

        # 2) Train/val split by shuffled indices (reproducible with seed)
        n_samples = len(inputs_raw)
        indices = np.arange(n_samples)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        sp = int(split_ratio * n_samples)
        if split == "train":
            selected = indices[:sp]
        elif split == "val":
            selected = indices[sp:]
        else:
            selected = indices  # full set

        self.indices = selected  # helpful for audit/debug

        # 3) Subset lists
        self.inputs_raw = [inputs_raw[i] for i in selected]
        self.targets_raw = [targets_raw[i] for i in selected]
        self.masks_raw = [masks_raw[i] for i in selected]
        self.times_raw = [times_raw[i] for i in selected]

        # 4) Optional Z-score standardisation (fit on train; reuse elsewhere)
        if scale:
            if (input_scaler is None) or (target_scaler is None):
                if split != "train":
                    raise ValueError("Val/test must receive precomputed scalers from the train split.")
                in_mean, in_std = compute_stats(self.inputs_raw)
                out_mean, out_std = compute_stats(self.targets_raw)
                self.input_scaler = StandardScaler(in_mean, in_std)
                self.target_scaler = StandardScaler(out_mean, out_std)
            else:
                self.input_scaler = input_scaler
                self.target_scaler = target_scaler

            self.inputs = apply_scaling(self.inputs_raw, self.input_scaler)   # list[[T,11]]
            self.targets = apply_scaling(self.targets_raw, self.target_scaler)  # list[[T,6]]
        else:
            self.input_scaler = None
            self.target_scaler = None
            self.inputs = self.inputs_raw
            self.targets = self.targets_raw

        # 5) Optional teacher forcing: append lagged stress channels
        if use_lagged_stress:
            lagged = []
            for y in self.targets:
                pad0 = np.zeros((1, y.shape[1]), dtype=y.dtype)
                lag = np.vstack([pad0, y[:-1]])
                lagged.append(lag)
            self.inputs = [np.concatenate([x, l], axis=1) for x, l in zip(self.inputs, lagged)]
            # Din becomes 17 (11 exogenous + 6 lagged stress)

        # 6) Convert to tensors
        self.inputs = [torch.tensor(x, dtype=torch.float32) for x in self.inputs]
        self.targets = [torch.tensor(y, dtype=torch.float32) for y in self.targets]
        self.times = [torch.tensor(t, dtype=torch.float32) for t in self.times_raw]
        self.masks = [torch.tensor(m, dtype=torch.bool) for m in self.masks_raw]

    # ---- PyTorch Dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        """Number of cases (RVE sequences) in this split."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs: [T, Din]
            mask:   [T] (bool)
            times:  [T]
            targets:[T, 6]
        """
        return self.inputs[idx], self.masks[idx], self.times[idx], self.targets[idx]
