from torch.utils.data import DataLoader
from src.dataset import CompositeStressDataset


def get_dataloader(input_csv_path,
                   data_dir,
                   max_seq_len=200,
                   batch_size=32,
                   shuffle=True,
                   num_workers=2,
                   scale=True,
                   split="all",
                   split_ratio=0.8,
                   seed=42,
                   use_lagged_stress=False):
    """
    Returns a PyTorch DataLoader for the composite stress prediction dataset.

    Args:
        input_csv_path (str): Path to IM78552_DATABASEInput.csv
        data_dir (str): Path to the _CSV folder containing all time-series files
        max_seq_len (int): Length to pad/truncate sequences
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the dataset (used in training)
        num_workers (int): Number of subprocesses for data loading
        scale (bool): Whether to apply Z-score standardisation
        split (str): 'train', 'val', or 'all'
        split_ratio (float): Ratio of training data if split is not 'all'
        seed (int): Random seed for reproducible split

    Returns:
        DataLoader
    """
    dataset = CompositeStressDataset(
        input_csv_path=input_csv_path,
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        scale=scale,
        split=split,
        split_ratio=split_ratio,
        seed=seed,
        use_lagged_stress = use_lagged_stress
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

