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
                   use_lagged_stress=False,
                   input_scaler=None,
                   target_scaler=None):

    dataset = CompositeStressDataset(
        input_csv_path=input_csv_path,
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        scale=scale,
        split=split,
        split_ratio=split_ratio,
        seed=seed,
        use_lagged_stress=use_lagged_stress,
        input_scaler=input_scaler,
        target_scaler=target_scaler
    )

    do_shuffle = (split == "train") and shuffle

    return DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers)

def make_train_val_loaders(input_csv_path,
                           data_dir,
                           max_seq_len=200,
                           batch_size=32,
                           split_ratio=0.8,
                           seed=42,
                           use_lagged_stress=True):
    """
    Convenience: builds train loader (computes scalers) and val loader (reuses them).
    """
    train_loader = get_dataloader(input_csv_path, data_dir, max_seq_len,
                                  batch_size=batch_size, shuffle=True,
                                  split="train", split_ratio=split_ratio, seed=seed,
                                  use_lagged_stress=use_lagged_stress,
                                  input_scaler=None, target_scaler=None)
    in_scaler  = train_loader.dataset.input_scaler if hasattr(train_loader.dataset, "input_scaler") else None
    out_scaler = train_loader.dataset.target_scaler if hasattr(train_loader.dataset, "target_scaler") else None
    val_loader   = get_dataloader(input_csv_path, data_dir, max_seq_len,
                                  batch_size=batch_size, shuffle=False,
                                  split="val", split_ratio=split_ratio, seed=seed,
                                  use_lagged_stress=use_lagged_stress,
                                  input_scaler=in_scaler, target_scaler=out_scaler)
    return train_loader, val_loader, in_scaler, out_scaler