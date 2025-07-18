import torch
from torch.utils.data import Dataset
from utils_parsing import load_all_data
from normalisation import compute_stats, StandardScaler, apply_scaling

class CompositeStressDataset(Dataset):
    """
    PyTorch Dataset class for composite material strain-to-stress prediction.

    Each sample consists of:
        - Input sequence: [max_seq_len, 11] → 6 strain components + 1 theta + 4 lamination parameters
        - Target sequence: [max_seq_len, 6] → 6 stress components

    Optionally applies Z-score standardisation (recommended for model training).
    """

    def __init__(self, input_csv_path, data_dir, max_seq_len=1800, scale=True):
        """
        Args:
            input_csv_path (str): Path to the metadata file (IM78552_DATABASEInput.csv)
            data_dir (str): Path to the folder with time-series CSV files (_CSV)
            max_seq_len (int): Number of timesteps to pad/truncate each sequence to
            scale (bool): Whether to apply Z-score standardisation
        """
        super().__init__()

        # Load all raw input and target sequences from disk
        self.inputs_raw, self.targets_raw = load_all_data(input_csv_path, data_dir, max_seq_len)

        # If standardisation is enabled, compute and apply it
        if scale:
            # Compute mean and std per feature across all sequences
            input_mean, input_std = compute_stats(self.inputs_raw)
            target_mean, target_std = compute_stats(self.targets_raw)

            # Store the scalers (for inverse transformation, saving, etc.)
            self.input_scaler = StandardScaler(input_mean, input_std)
            self.target_scaler = StandardScaler(target_mean, target_std)

            # Apply standardisation to each sequence in the dataset
            self.inputs = apply_scaling(self.inputs_raw, self.input_scaler)
            self.targets = apply_scaling(self.targets_raw, self.target_scaler)
        else:
            # If not scaling, use raw values
            self.inputs = self.inputs_raw
            self.targets = self.targets_raw

        # Convert all sequences to PyTorch tensors
        self.inputs = [torch.tensor(x, dtype=torch.float32) for x in self.inputs]
        self.targets = [torch.tensor(y, dtype=torch.float32) for y in self.targets]

    def __len__(self):
        """Returns total number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """Returns a single (input_sequence, target_sequence) pair."""
        return self.inputs[idx], self.targets[idx]
