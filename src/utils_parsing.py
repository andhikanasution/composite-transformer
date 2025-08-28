"""
Parsing and preprocessing helpers for the composites dataset.

Expected inputs:
- Metadata file: IM78552_DATABASEInput.csv  (theta in col 7, LP1..LP4 in cols 8..11)
- Time-series files: IM78552_DATABASE_001.csv ... IM78552_DATABASE_1848.csv
"""

from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import pandas as pd


def load_metadata(input_path: str) -> np.ndarray:
    """
    Load and parse metadata (IM78552_DATABASEInput.csv).

    Extracts:
        - Column 7:  load angle θ
        - Columns 8–11: lamination parameters (LP1–LP4)

    Returns:
        (N_cases, 5) array: [theta, LP1, LP2, LP3, LP4]
    """
    metadata = pd.read_csv(input_path, header=None)
    theta = metadata.iloc[:, 6].values.reshape(-1, 1)
    lam_params = metadata.iloc[:, 7:11].values
    return np.hstack([theta, lam_params]).astype(np.float32)


def load_time_series(
    file_path: str,
    strain_threshold: float = float("inf"),
    component_thresholds: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single RVE time series and apply optional strain-based filtering.

    Extracted columns (first six of each):
        - Strain: E1..E6
        - Stress: S1..S6
        - Time:   'time' (seconds)

    Filtering behaviour:
        - If component_thresholds is provided, apply ±threshold per component (keys 0..5).
        - Else, apply a global ±strain_threshold across all six components.

    Args:
        file_path: CSV path for one RVE time series.
        strain_threshold: Global absolute threshold (ignored if component_thresholds given).
        component_thresholds: Dict of per-component thresholds, indices 0..5.

    Returns:
        (filtered_strain[T',6], filtered_stress[T',6], filtered_time[T'])
    """
    df = pd.read_csv(file_path)

    # Identify E* and S* columns, excluding any with "Solid" in the name
    strain_cols = [c for c in df.columns if c.startswith("E") and "Solid" not in c][:6]
    stress_cols = [c for c in df.columns if c.startswith("S") and "Solid" not in c][:6]

    strain_seq = df[strain_cols].to_numpy(dtype=np.float32)  # [T,6]
    stress_seq = df[stress_cols].to_numpy(dtype=np.float32)  # [T,6]

    # --- Build a boolean mask for valid timesteps ----------------------------------
    if component_thresholds:
        mask = np.ones(strain_seq.shape[0], dtype=bool)
        for i in range(6):
            if i in component_thresholds:
                thr = component_thresholds[i]
                mask &= (strain_seq[:, i] >= -thr) & (strain_seq[:, i] <= thr)
    else:
        valid = (strain_seq >= -strain_threshold) & (strain_seq <= strain_threshold)
        mask = valid.all(axis=1)

    # Apply mask to strain/stress/time
    filtered_strain = strain_seq[mask]
    filtered_stress = stress_seq[mask]

    time_seq = df["time"].to_numpy(dtype=np.float32)  # [T]
    filtered_time = time_seq[mask]

    return filtered_strain, filtered_stress, filtered_time


def build_sample_mapping(data_dir: str) -> Dict[int, str]:
    """
    Map integer indices to file paths.

    Assumes files are named IM78552_DATABASE_001.csv ... IM78552_DATABASE_1848.csv.

    Returns:
        dict: {idx: absolute_file_path}
    """
    return {i - 1: os.path.join(data_dir, f"IM78552_DATABASE_{i:03d}.csv") for i in range(1, 1849)}


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Uniformly resample or pad a sequence to target_len timesteps.

    - If T >= target_len: pick evenly spaced indices in [0, T-1] (rounded).
    - If T <  target_len: zero-pad at the end so that the last real row stays last.

    Args:
        seq: [T, D] array (or [T] vector).
        target_len: desired length.

    Returns:
        [target_len, D] (or [target_len]) array.
    """
    seq = np.asarray(seq)
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)

    T, D = seq.shape
    if T >= target_len:
        idx = np.linspace(0, T - 1, num=target_len).round().astype(int)
        out = seq[idx]
    else:
        pad = np.zeros((target_len - T, D), dtype=seq.dtype)
        out = np.vstack([seq, pad])

    return out if D > 1 else out.flatten()


def load_all_data(
    input_csv_path: str,
    data_dir: str,
    max_seq_len: int = 200,
    strain_threshold: float = float("inf"),
    min_timesteps: int = 20,
    component_thresholds: Optional[Dict[int, float]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Load and preprocess all RVE simulations, returning padded inputs/targets and masks.

    Pipeline:
        1) Load static metadata per case: [theta, LP1..4].
        2) Map case index -> CSV file.
        3) For each case:
            a) load strain/stress/time and apply filtering
            b) discard runs with too few timesteps
            c) optionally truncate at first exceedance of the global threshold
            d) concatenate static metadata to each timestep → inputs [T, 11]
            e) resample/pad both inputs [T,11] and targets [T,6] to max_seq_len
            f) construct a boolean mask of length max_seq_len (True=real, False=pad)

    Returns:
        all_inputs:  List of [max_seq_len, 11] arrays
        all_targets: List of [max_seq_len,  6] arrays
        all_times:   List of [max_seq_len] arrays (seconds)
        all_masks:   List of [max_seq_len] boolean arrays
    """
    metadata = load_metadata(input_csv_path)          # (N_cases, 5)
    mapping = build_sample_mapping(data_dir)          # {idx: csv_path}

    all_inputs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_times: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []

    for idx, file_path in mapping.items():
        # 3a) Load + filter
        strain_seq, stress_seq, time_seq = load_time_series(
            file_path,
            strain_threshold=strain_threshold,
            component_thresholds=component_thresholds,
        )

        # 3b) Discard too-short runs (helps stability and avoids degenerate masks)
        if strain_seq.shape[0] < min_timesteps:
            continue

        # 3c) (Global-threshold mode only) truncate at first exceedance if any
        exceed = np.any(np.abs(strain_seq) > strain_threshold, axis=1)
        if exceed.any():
            t_end = int(np.argmax(exceed))
            strain_seq = strain_seq[: t_end + 1]
            stress_seq = stress_seq[: t_end + 1]
            time_seq = time_seq[: t_end + 1]

        # 3d) Attach static metadata per timestep
        static_info = np.tile(metadata[idx], (strain_seq.shape[0], 1)).astype(np.float32)  # [T,5]
        input_seq = np.hstack([strain_seq, static_info])  # [T, 11]

        # 3e) Resample/pad to uniform length
        input_resampled = resample_sequence(input_seq, max_seq_len)   # [max_seq_len, 11]
        target_resampled = resample_sequence(stress_seq, max_seq_len) # [max_seq_len,  6]
        time_resampled = resample_sequence(time_seq.reshape(-1, 1), max_seq_len).flatten()

        # 3f) Mask: True for real timesteps (up to original length), else False
        real_len = min(strain_seq.shape[0], max_seq_len)
        mask = np.concatenate(
            [np.ones(real_len, dtype=bool), np.zeros(max_seq_len - real_len, dtype=bool)]
        )

        all_inputs.append(input_resampled.astype(np.float32))
        all_targets.append(target_resampled.astype(np.float32))
        all_times.append(time_resampled.astype(np.float32))
        all_masks.append(mask)

    return all_inputs, all_targets, all_times, all_masks
