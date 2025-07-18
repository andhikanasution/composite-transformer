from torch.utils.data import DataLoader
from dataset import CompositeStressDataset


def get_dataloader(input_csv_path, data_dir, max_seq_len=1800, batch_size=32, shuffle=True, num_workers=2):
    """
    Returns a PyTorch DataLoader for the composite stress prediction dataset.

    Args:
        input_csv_path (str): Path to IM78552_DATABASEInput.csv
        data_dir (str): Path to the _CSV folder containing all time-series files
        max_seq_len (int): Length to pad/truncate sequences
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of subprocesses for data loading

    Returns:
        DataLoader
    """
    dataset = CompositeStressDataset(input_csv_path, data_dir, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
