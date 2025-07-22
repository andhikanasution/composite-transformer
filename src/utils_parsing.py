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


def pad_sequence(seq, max_len, pad_value=0.0):
    """
    Left-pad (or left-truncate) a 2D sequence to exactly max_len time steps.
    - If seq is longer than max_len: keep *the last* max_len rows.
    - If seq is shorter: pre-pend zeros so that the final entry (idx=-1)
      is always the true last time-step.

    Args:
        seq (np.ndarray): [T, D] original time series
        max_len (int):    desired length
        pad_value (float): scalar to pad with (default=0.0)

    Returns:
        np.ndarray of shape [max_len, D], with true data ending at index -1
    """
    T, D = seq.shape
    if T >= max_len:
        # Truncate from the front so we keep the *most recent* max_len steps
        return seq[-max_len:, :]
    else:
        # Pre-pend padding so that `seq` ends at the very end
        pad = np.full((max_len - T, D), pad_value)
        return np.vstack([pad, seq])



def load_all_data(input_csv_path,
                  data_dir,
                  max_seq_len=1800,
                  strain_threshold=0.05,
                  min_timesteps=20,
                  component_thresholds=None):
    """
    Load and preprocess the full dataset of composite RVE simulations.

    Each simulation includes:
        - Time-series strain and stress data (E1–E6, S1–S6)
        - Static metadata (load angle + lamination parameters)

    Filtering is applied on the strain data:
        - Either a global ±strain_threshold is used (default), or
        - A selective per-component threshold can be provided via a dictionary.

    Sequences with fewer than `min_timesteps` valid points after filtering are excluded.

    Args:
        input_csv_path (str): Path to the metadata CSV file (IM78552_DATABASEInput.csv).
        data_dir (str): Path to the directory containing the individual time-series CSV files.
        max_seq_len (int): Sequence length to pad or truncate each input to (default: 1800).
        strain_threshold (float): Global threshold for filtering all strain components.
                                  Ignored if component_thresholds is provided.
        min_timesteps (int): Minimum number of valid time steps required to keep a sample (default: 20).
        component_thresholds (dict or None): Optional per-component strain limits.
            Format: {0: 0.05, 5: 0.10} to apply ±0.05 on E1 and ±0.10 on E6.

    Returns:
        Tuple:
            - all_inputs (List[np.ndarray]): Padded input sequences of shape [max_seq_len, 11]
            - all_targets (List[np.ndarray]): Padded output sequences of shape [max_seq_len, 6]
    """
    # Load static metadata: θ + lamination parameters
    metadata = load_metadata(input_csv_path)

    # Build mapping from index to each sample file
    mapping = build_sample_mapping(data_dir)

    # Containers for all sequences
    all_inputs, all_targets = [], []

    for idx, path in mapping.items():
        # Load and filter time-series using either global or selective strain threshold
        strain_seq, stress_seq = load_time_series(
            file_path=path,
            strain_threshold=strain_threshold,
            component_thresholds=component_thresholds
        )

        # Skip this sample if it has too few valid timesteps after filtering
        if strain_seq.shape[0] < min_timesteps:
            continue

        # Attach static metadata to each time step
        static_info = np.tile(metadata[idx], (strain_seq.shape[0], 1))  # Shape: [T, 5]
        input_seq = np.hstack([strain_seq, static_info])                # Shape: [T, 11]

        # Pad input and output to fixed length
        input_seq_padded = pad_sequence(input_seq, max_seq_len)        # [max_seq_len, 11]
        target_seq_padded = pad_sequence(stress_seq, max_seq_len)      # [max_seq_len, 6]

        # Add to output lists
        all_inputs.append(input_seq_padded)
        all_targets.append(target_seq_padded)

    return all_inputs, all_targets

