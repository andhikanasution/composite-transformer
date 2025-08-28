"""
DataLoader factories for the composites stress–prediction project.

This module provides:
- get_dataloader: generic padded (mask-based) sequence loader
- make_train_val_loaders: convenience function that also returns the fitted scalers
- make_informer_window_loaders: sliding-window loaders for Informer-style training
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
from src.dataset import CompositeStressDataset
from src.dataset_windows import CompositeInformerWindows


def get_dataloader(
    input_csv_path: str,
    data_dir: str,
    max_seq_len: int = 200,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    scale: bool = True,
    split: str = "all",
    split_ratio: float = 0.8,
    seed: int = 42,
    use_lagged_stress: bool = False,
    input_scaler=None,
    target_scaler=None,
) -> DataLoader:
    """
    Build a DataLoader for padded sequences with masks.

    Args:
        input_csv_path: Path to the metadata CSV (IM78552_DATABASEInput.csv).
        data_dir: Directory containing the per-RVE time-series CSVs.
        max_seq_len: Pad/truncate each sequence to this length (T).
        batch_size: Mini-batch size.
        shuffle: Shuffle order (applied only for the train split).
        num_workers: PyTorch DataLoader workers.
        scale: If True, apply Z-score standardisation.
        split: {"train","val","all"}.
        split_ratio: Train/val split proportion (used when split != "all").
        seed: RNG seed for reproducible splits.
        use_lagged_stress: If True, append lagged stress channels (teacher forcing).
        input_scaler/target_scaler: Reuse pre-fitted scalers (required for val/test if scale=True).

    Returns:
        torch.utils.data.DataLoader
    """
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
        target_scaler=target_scaler,
    )

    do_shuffle = (split == "train") and shuffle

    # pin_memory only helps on CUDA; harmless otherwise
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def make_train_val_loaders(
    input_csv_path: str,
    data_dir: str,
    max_seq_len: int = 200,
    batch_size: int = 32,
    split_ratio: float = 0.8,
    seed: int = 42,
    use_lagged_stress: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[object], Optional[object]]:
    """
    Convenience wrapper that:
      1) builds a train loader (fitting scalers if scale=True), and
      2) builds a val loader that reuses those scalers.

    Returns:
        (train_loader, val_loader, input_scaler, target_scaler)
    """
    train_loader = get_dataloader(
        input_csv_path,
        data_dir,
        max_seq_len,
        batch_size=batch_size,
        shuffle=True,
        split="train",
        split_ratio=split_ratio,
        seed=seed,
        use_lagged_stress=use_lagged_stress,
        input_scaler=None,
        target_scaler=None,
    )

    in_scaler = getattr(train_loader.dataset, "input_scaler", None)
    out_scaler = getattr(train_loader.dataset, "target_scaler", None)

    val_loader = get_dataloader(
        input_csv_path,
        data_dir,
        max_seq_len,
        batch_size=batch_size,
        shuffle=False,
        split="val",
        split_ratio=split_ratio,
        seed=seed,
        use_lagged_stress=use_lagged_stress,
        input_scaler=in_scaler,
        target_scaler=out_scaler,
    )
    return train_loader, val_loader, in_scaler, out_scaler


def make_informer_window_loaders(
    input_csv_path: str,
    data_dir: str,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 96,
    stride: int = 24,
    batch_size: int = 32,
    split_ratio: float = 0.8,
    seed: int = 42,
    scale: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[object], Optional[object]]:
    """
    Build sliding-window DataLoaders for Informer-style training (no padding/masks).

    Returns:
        (train_loader, val_loader, exo_scaler, y_scaler)
    """
    train_ds = CompositeInformerWindows(
        input_csv_path,
        data_dir,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        stride=stride,
        scale=scale,
        split="train",
        split_ratio=split_ratio,
        seed=seed,
        exo_scaler=None,
        y_scaler=None,
    )
    val_ds = CompositeInformerWindows(
        input_csv_path,
        data_dir,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        stride=stride,
        scale=scale,
        split="val",
        split_ratio=split_ratio,
        seed=seed,
        exo_scaler=train_ds.exo_scaler,
        y_scaler=train_ds.y_scaler,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    return train_loader, val_loader, train_ds.exo_scaler, train_ds.y_scaler
