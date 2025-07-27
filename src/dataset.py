import torch
from torch.utils.data import Dataset
from utils_parsing import load_all_data
from normalisation import compute_stats, StandardScaler, apply_scaling
import numpy as np

class CompositeStressDataset(Dataset):
    """
    PyTorch Dataset class for composite material strain-to-stress prediction.

    Each sample consists of:
        - Input sequence: [max_seq_len, 11] → 6 strain components + 1 theta + 4 lamination parameters
        - Target sequence: [max_seq_len, 6] → 6 stress components

    Optionally applies Z-score standardisation (recommended for model training).
    """

    def __init__(self, input_csv_path, data_dir, max_seq_len=1800, scale=True, split="all", split_ratio=0.8, seed=42):
        """
        Args:
            input_csv_path (str): Path to the metadata file (IM78552_DATABASEInput.csv)
            data_dir (str): Path to the folder with time-series CSV files (_CSV)
            max_seq_len (int): Number of timesteps to pad/truncate each sequence to
            scale (bool): Whether to apply Z-score standardisation
            split (str): 'train', 'val', or 'all' — determines which subset to load
            split_ratio (float): Fraction of data to use for training (only used if split != 'all')
            seed (int): Random seed for reproducible split
        """
        super().__init__()

        # Load all raw sequences
        inputs_raw, targets_raw = load_all_data(input_csv_path, data_dir, max_seq_len)

        # Determine indices for splitting
        total_samples = len(inputs_raw)
        indices = list(range(total_samples))
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        split_point = int(split_ratio * total_samples)
        if split == "train":
            selected = indices[:split_point]
        elif split == "val":
            selected = indices[split_point:]
        else:
            selected = indices  # full set

        # Store selected indices for downstream access (e.g., for sanity checks)
        self.indices = selected

        # Subset the data
        self.inputs_raw = [inputs_raw[i] for i in selected]
        self.targets_raw = [targets_raw[i] for i in selected]

        # Apply optional standardisation
        if scale:
            input_mean, input_std = compute_stats(self.inputs_raw)
            target_mean, target_std = compute_stats(self.targets_raw)

            self.input_scaler = StandardScaler(input_mean, input_std)
            self.target_scaler = StandardScaler(target_mean, target_std)

            # scale both inputs and targets
            self.inputs = apply_scaling(self.inputs_raw, self.input_scaler)  # list of [T,11]
            self.targets = apply_scaling(self.targets_raw, self.target_scaler)  # list of [T, 6]
        else:
            self.inputs = self.inputs_raw
            self.targets = self.targets_raw

        # ──────────────────────────────────────────────────────
        # Teacher‐forcing: build lagged‐stress channels
        # for each sequence, prepend a zero‐row and drop its last true
        lagged = []
        for y in self.targets:  # y: [T,6]
            # first timestep has no prior stress → zero
            pad0 = np.zeros((1, y.shape[1]), dtype=y.dtype)
            lag = np.vstack([pad0, y[:-1]])  # [T,6]
            lagged.append(lag)

        # append those 6 lagged stress channels to your original 11-dim inputs
        self.inputs = [
            np.concatenate([x, lag], axis=1)  # now [T, 11+6=17]
            for x, lag in zip(self.inputs, lagged)
        ]

        self.inputs  = [torch.tensor(x, dtype=torch.float32) for x in self.inputs]   # [T,17]
        self.targets = [torch.tensor(y, dtype=torch.float32) for y in self.targets]  # [T, 6]

    def __len__(self):
        """Returns total number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """Returns a single (input_sequence, target_sequence) pair."""
        return self.inputs[idx], self.targets[idx]
