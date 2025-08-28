# Composite Stress Prediction — PatchTST & Informer

Predicting 6-component stress sequences from composite-material strain time series and static layup metadata.
This repo contains **two model families**, each with its **own Python environment**:

* **PatchTST** (Hugging Face Transformers) — point-wise regression.
* **Informer** (local implementation) — three variants:

  1. **Pointwise** (full-sequence mapping)
  2. **Time-marching** with scheduled sampling
  3. **Canonical windowed** (encoder/decoder windows; no padding/masks)

---

## Data layout & assumptions

* **Metadata CSV**: `IM78552_DATABASEInput.csv`

  * Column 7: **θ** (load angle)
  * Columns 8–11: **LP1–LP4** (lamination parameters)
* **Time-series CSVs**: `_CSV/IM78552_DATABASE_001.csv` … `_CSV/IM78552_DATABASE_1848.csv`

  * Columns include **E1–E6** (strain), **S1–S6** (stress), and **time**

The loaders:

* Assemble inputs as `[strains E1..E6, θ, LP1..LP4]` → **11 channels**
* Optionally add lagged stress during training (**+6 channels**) → **17 channels**
* Apply Z-score standardisation (train split computes scalers; val/test reuse them)
* For padded datasets: a boolean mask marks real vs padded timesteps

If your file names or columns differ, adjust:

* `src/utils_parsing.py::build_sample_mapping` (file naming)
* `src/utils_parsing.py::load_time_series` (column selection)

---

## Environments

You’ll get the smoothest experience keeping **separate envs** for PatchTST vs Informer.

> **Python**: 3.10+ recommended
> **PyTorch**: install the build that matches your platform (CUDA/MPS/CPU)

### PatchTST environment

```bash
# create & activate
conda create -n patchtst python=3.10 -y
conda activate patchtst

# install torch for your platform (examples below)
# CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1 (example):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# libraries
pip install "transformers>=4.41" torchinfo psutil scikit-learn tqdm matplotlib pandas numpy
```

### Informer environment

```bash
# create & activate
conda create -n informer python=3.10 -y
conda activate informer

# install torch for your platform
pip install torch --index-url https://download.pytorch.org/whl/cpu
# (or appropriate CUDA/MPS build)

# libraries
pip install psutil scikit-learn tqdm matplotlib pandas numpy
```

> If you prefer, install from the repo root with `PYTHONPATH=. python ...` or `pip install -e .` after adding a minimal `pyproject.toml`.

---

## Configure data paths

All training/eval scripts define **`INPUT_CSV`** and **`DATA_DIR`** near the top.
Edit those lines to point to your local dataset:

```python
INPUT_CSV = "/path/to/data/IM78552_DATABASEInput.csv"
DATA_DIR  = "/path/to/data/_CSV"
```

The scripts will create `models/<run_name>/` automatically.

---

## How the datasets work

* **`CompositeStressDataset`** (`src/dataset.py`): padded sequences up to `max_seq_len` with a mask.
* **`CompositeInformerWindows`** (`src/dataset_windows.py`): **windowed** encoder/decoder tuples (no padding/masks) for canonical Informer.

Scaling:

* Train split computes `input_scaler` and `target_scaler` (Z-score);
* Val/Test splits **must reuse** these saved scalers. Scripts handle this.

---

## Running PatchTST

> Environment: **`patchtst`**

### Train (point-wise)

```bash
conda activate patchtst
python src/patchtst/train_patchtst.py
```

Artifacts (example `models/patchtst_.../`):

* `patchtst_best.pt` — best weights
* `input_scaler.json`, `target_scaler.json` — scalers
* `patchtst_summary.txt` — model summary (torchinfo)
* `resource_log.json`, `mode_config.json`

### Evaluate (point-wise)

```bash
conda activate patchtst
python src/patchtst/evaluate_patchtst.py
```

Artifacts:

* `eval_metrics_pw_*.json`
* `scatter_pw_*.png`

---

## Running Informer

> Environment: **`informer`**

There are **three** variants. All use the local Informer implementation in `src/informer/models/`.

### 1) Informer — Pointwise (full-sequence mapping)

Train:

```bash
conda activate informer
python src/informer/train_informer_pw.py
```

Evaluate:

```bash
python src/informer/evaluate_pointwise.py
```

Artifacts (`models/informer_pointwise_*`):

* `best_pointwise.pt`, `scalers.json`, `resource_log.json`
* `evaluation_metrics.json`, scatter/time-series plots

### 2) Informer — Time-Marching (scheduled sampling)

* Train with **lagged stress channels** added to the encoder
* Validation simulates inference: starts with zero lag → **N\_REFINE** refinement steps

Train:

```bash
conda activate informer
python src/informer/train_informer_tm.py
```

Evaluate:

```bash
python src/informer/evaluate_time_marching.py
```

Artifacts (`models/informer_timemarching_*`):

* `best_tm.pt`, `last_tm.pt`, `scalers.json`
* `history.json` (loss, R²/RMSE, grad-norm, roughness) + curve plots
* `resources.json`, `evaluation_metrics.json`, scatter/time-series/hist plots

### 3) Informer — Canonical Windowed

* Uses `CompositeInformerWindows` (no padding/masks)
* Encoder: `[past y(6); past exo(11)]` = **17** channels
* Decoder: `[last LABEL_LEN y(6); exo(11)]` (future y zeros) = **17** channels

Train:

```bash
conda activate informer
python src/informer/train_informer_tm_canonical.py
```

Evaluate:

```bash
python src/informer/evaluate_time_marching_canonical.py
```

Artifacts (`models/informer_tm_canonical_*`):

* `best_tm.pt`, `scalers.json`, `resources.json`
* `evaluation_metrics.json`, plots

---

## Outputs & logs

Each training script writes to a run-specific directory under `models/`:

* **Weights**: `*.pt`
* **Scalers**: `input_scaler.json`, `target_scaler.json` (or `scalers.json`)
* **History/curves** (time-marching): `history.json`, `curve_*.png`
* **Resource usage**: `resource_log.json` / `resources.json`
* **Evaluation**: `evaluation_metrics.json`, scatter/time-series/histogram `.png`

---

## Reproducibility

* Default `seed=42` and `split_ratio=0.8` in loaders
* PatchTST point-wise: `context_length = MAX_SEQ_LEN`, `prediction_length = 1`
* Informer pointwise: masked losses over full sequence
* Informer time-marching: scheduled sampling linearly anneals teacher-forcing, plus optional zero-lag augmentation

Small nondeterminism may remain due to CUDA/MPS kernels; set additional PyTorch flags if strict determinism is required.

---

## Troubleshooting

* **“Val/test must receive precomputed scalers…”**
  Train first (which saves scalers), then pass/load those scalers for val/test. The provided scripts already do this.

* **GPU/MPS not used**
  Ensure you installed the correct PyTorch build for your hardware; the scripts auto-select **CUDA → MPS → CPU**.

* **OOM / memory pressure**
  Reduce `BATCH_SIZE` and/or `MAX_SEQ_LEN` (PatchTST) or window sizes (Informer). Monitor `resource_log.json`.

* **File naming mismatch**
  Edit `build_sample_mapping` in `src/utils_parsing.py` to match your dataset filenames.

---

## Development notes

* Code style: docstrings, clear section headers, and explicit save paths.
* Plots: stored at the end of evaluations (scatter of final step, per-component time series, histograms).
* Figures/schematics (if needed): see `src/plots.py`, `src/schematics.py`.

