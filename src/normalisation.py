import numpy as np
import torch

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        # Avoid division by zero
        self.std[self.std == 0] = 1e-8

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d):
        return StandardScaler(np.array(d["mean"]), np.array(d["std"]))


def compute_stats(data_list):
    """
    Compute mean and std from a list of [T, D] arrays.
    Stacks them along the time axis first.
    """
    stacked = np.vstack(data_list)
    return stacked.mean(axis=0), stacked.std(axis=0)


def apply_scaling(data_list, scaler):
    """
    Applies Z-score normalisation to a list of [T, D] arrays using the given scaler.
    """
    return [scaler.transform(x) for x in data_list]
