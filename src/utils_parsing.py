import os
import pandas as pd
import numpy as np


def load_metadata(input_path):
    """
    Load and parse metadata file (IM78552_DATABASEInput.csv).
    Extract:
        - Column 7: Load angle (theta)
        - Columns 8–11: Lamination parameters (LP1–LP4)

    Returns:
        np.ndarray of shape (N, 5) containing [theta, LP1, LP2, LP3, LP4]
    """
    metadata = pd.read_csv(input_path, header=None)
    theta = metadata.iloc[:, 6].values.reshape(-1, 1)
    lam_params = metadata.iloc[:, 7:11].values
    return np.hstack([theta, lam_params])


def load_time_series(file_path, strain_threshold=0.05, component_thresholds=None):
    """
    Load and filter a single time-series file for composite strain–stress data.

    This function extracts:
        - Strain components: E1 to E6
        - Stress components: S1 to S6
    and applies a filtering mask to exclude time steps where strain exceeds either:
        - A global ±strain_threshold (applied to all components), or
        - Selective per-component thresholds if provided.

    Args:
        file_path (str): Path to the CSV file containing one RVE time series.
        strain_threshold (float): A global threshold to clip all strain values (±value).
                                  Ignored if component_thresholds is provided.
        component_thresholds (dict or None): Optional dictionary specifying the allowed ±strain
            range for each component individually. Keys must be 0–5 (corresponding to E1–E6).
            Example: {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.10, 5: 0.10}

    Returns:
        tuple:
            - filtered_strain (np.ndarray): Array of shape [T', 6] after filtering
            - filtered_stress (np.ndarray): Array of shape [T', 6] after filtering
    """

    # Load the time-series data from the file
    df = pd.read_csv(file_path)

    # Identify relevant columns for strain (E1–E6) and stress (S1–S6)
    strain_cols = [col for col in df.columns if col.startswith("E") and "Solid" not in col]
    stress_cols = [col for col in df.columns if col.startswith("S") and "Solid" not in col]

    # Extract strain and stress arrays from the DataFrame
    strain_seq = df[strain_cols[:6]].values  # Shape: [T, 6]
    stress_seq = df[stress_cols[:6]].values  # Shape: [T, 6]

    # Apply filtering
    if component_thresholds:
        # -----------------------------
        # Selective per-component filtering
        # -----------------------------
        # Start with a boolean mask with all True values (retain all time steps initially)
        mask = np.ones(strain_seq.shape[0], dtype=bool)

        # Loop over each strain component (E1–E6 → index 0–5)
        for i in range(6):
            if i in component_thresholds:
                comp_thresh = component_thresholds[i]
                # Keep rows where strain component i is within ±threshold
                mask &= (strain_seq[:, i] >= -comp_thresh) & (strain_seq[:, i] <= comp_thresh)
            else:
                # If no specific threshold is given for component i, assume it’s unconstrained
                continue

    else:
        # -----------------------------
        # Global filtering on all components
        # -----------------------------
        # Create a boolean array: True where all strain values are within ±strain_threshold
        valid_mask = (strain_seq >= -strain_threshold) & (strain_seq <= strain_threshold)
        mask = valid_mask.all(axis=1)

    # Apply the mask to both strain and stress sequences
    filtered_strain = strain_seq[mask]
    filtered_stress = stress_seq[mask]

    return filtered_strain, filtered_stress


def build_sample_mapping(data_dir):
    """
    Build a mapping from index to file path.
    Assumes files are named IM78552_DATABASE_001.csv to IM78552_DATABASE_1848.csv

    Args:
        data_dir (str): Path to the folder containing time-series files

    Returns:
        dict: index → file_path
    """
    return {
        i - 1: os.path.join(data_dir, f"IM78552_DATABASE_{i:03d}.csv")
        for i in range(1, 1849)
    }


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    T, D = seq.shape
    if T >= target_len:
        # choose indices in [0..T-1] that are evenly spaced
        idx = np.linspace(0, T - 1, num=target_len).round().astype(int)
        return seq[idx]
    else:
        # pad so that data ends at seq[-1]
        pad = np.zeros((target_len - T, D), dtype=seq.dtype)
        return np.vstack([seq, pad])


def load_all_data(input_csv_path,
                  data_dir,
                  max_seq_len=200,
                  strain_threshold=0.025,
                  min_timesteps=20,
                  component_thresholds=None):
    """
    Load and preprocess the full dataset of composite RVE simulations,
    returning padded inputs, targets, and a mask that indicates which
    timesteps are real (True) vs padded (False).

    Returns:
        all_inputs (List[np.ndarray]): [max_seq_len, D_in] arrays
        all_targets (List[np.ndarray]): [max_seq_len, D_out] arrays
        all_masks (List[np.ndarray]): 1D boolean arrays of length max_seq_len
    """
    # 1) Load static metadata → shape (N_cases, 5)
    metadata = load_metadata(input_csv_path)

    # 2) Map each index to its CSV file
    mapping = build_sample_mapping(data_dir)

    all_inputs, all_targets, all_masks = [], [], []

    for idx, file_path in mapping.items():
        # 3) Load & filter by global or per-component thresholds
        strain_seq, stress_seq = load_time_series(
            file_path,
            strain_threshold=strain_threshold,
            component_thresholds=component_thresholds
        )
        # 4) Discard too-short runs
        if strain_seq.shape[0] < min_timesteps:
            continue

        # 5) Once any strain exceeds threshold, truncate at first exceed
        exceed = np.any(np.abs(strain_seq) > strain_threshold, axis=1)
        if exceed.any():
            t_end = np.argmax(exceed)
            strain_seq = strain_seq[: t_end + 1]
            stress_seq = stress_seq[: t_end + 1]

        # 6) Attach static metadata at every timestep
        static_info = np.tile(metadata[idx], (strain_seq.shape[0], 1))  # [T, 5]
        input_seq = np.hstack([strain_seq, static_info])              # [T, 11]

        # 7) Resample/pad to uniform length (end-padding only)
        input_resampled  = resample_sequence(input_seq,  max_seq_len)  # [max_seq_len, 11]
        target_resampled = resample_sequence(stress_seq, max_seq_len)  # [max_seq_len,  6]

        # 8) Build a mask: True for real timesteps (up to original length), False for padding
        real_len = min(strain_seq.shape[0], max_seq_len)
        mask = np.concatenate([
            np.ones(real_len, dtype=bool),
            np.zeros(max_seq_len - real_len, dtype=bool)
        ])  # shape: (max_seq_len,)

        # 9) Collect
        all_inputs .append(input_resampled)
        all_targets.append(target_resampled)
        all_masks  .append(mask)

    return all_inputs, all_targets, all_masks

