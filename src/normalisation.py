"""
Lightweight normalisation utilities.

StandardScaler mirrors scikit-learn behaviour for mean/std scaling, with:
- epsilon guard against zero std
- (de)serialisation helpers to_dict / from_dict
"""

from typing import List, Tuple
import numpy as np
import torch  # noqa: F401  # (imported for type parity with project; not used directly)


class StandardScaler:
    """Simple per-channel Z-score scaler."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        # Avoid divide-by-zero; preserves zeros where channels are constant
        self.std[self.std == 0] = 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "StandardScaler":
        return StandardScaler(np.array(d["mean"], dtype=np.float32),
                              np.array(d["std"], dtype=np.float32))


def compute_stats(data_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel (mean, std) across a list of [T, D] arrays by stacking on time.

    Args:
        data_list: list of arrays with identical second dimension D.

    Returns:
        (mean[D], std[D])
    """
    stacked = np.vstack(data_list).astype(np.float32)
    return stacked.mean(axis=0), stacked.std(axis=0)


def apply_scaling(data_list: List[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
    """Apply a given StandardScaler to each [T, D] array in a list."""
    return [scaler.transform(x) for x in data_list]
