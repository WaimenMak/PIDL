# PIDL – Physics-Informed DL for Traffic State Estimation (A13)

This repo contains a unified PINN/NN model to reconstruct space–time velocity fields on the A13 corridor from sparse sensors. It supports multiple fundamental-diagram (FD) formulations and a pure neural-network baseline, all driven by YAML config files. A companion script runs multi-experiments across different sensor counts and FD types and produces summaries and plots.

## What's here

**Main scripts:**
- `ojits03_a13_pytorch_revised.py` – main training/evaluation script (single run), config-driven
- `run_multi_sensor_experiments.py` – multi-run driver over sensor counts and FD types
- `utils/utils.py` – data loading, sampling, evaluation, and plotting utilities

**Configuration:**
- `configs/config.yaml` – config for single run experiments
- `configs/config_multi.yaml` – config for multi-sensor/multi-FD experiments

**Data:**
- `data/` – velocity tables (e.g., `A13_Velocity_Data_0909-0910.txt`)
- `td_data/` – distance metadata JSON files

**Results:**
- `Results/` – single run outputs
- `Results_FD/` – multi-experiment outputs with runs and summaries
- `runs/` – model checkpoints from single experiments
- `FD_*/` – figures from various experiments

## Requirements

- Python 3.9+ (tested on Linux)
- GPU with CUDA recommended for speed (CPU works too)
- Python packages:
	- torch
	- numpy, pandas
	- pyyaml
	- matplotlib, scipy

Install with:

```bash
pip install torch numpy pandas pyyaml matplotlib scipy
```

## Data layout

- Velocity table (CSV-like text) under `data/`, e.g. `data/A13_Velocity_Data_0909-0910.txt`
- Distance metadata JSON under `td_data/`, e.g. `td_data/2024-09-09.json`

Both paths are set in the configs.

## Single run: train PINN and optional NN baseline

1) Edit `configs/config.yaml` as needed. Key options:

**Data:**
- `data_file: data/A13_Velocity_Data_0909-0910.txt`
- `distance_json: td_data/2024-09-09.json`

**Model:**
- `layers: [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]`
- `fd_name: 'log'` – fundamental diagram for PINN physics
  - Supported: `linear`, `log`, `exp`, `power`, `triangular`
- `f_weight: 1.0` – physics loss weight

**Observations:**
- `sensor_based: True` – use equally spaced sensor columns
- `n_sensors: 5` – number of sensors (when sensor_based=true)
- `N_u: 800` – observation points (when sensor_based=false)
- `N_f: 10000` – physics collocation points

**Training:**
- `epochs: 10000`
- `lr: 0.0001`
- `patience: 2000`
- `log_every: 500`
- `physics_every: 1`
- `seed: 25`
- `run_base: false` – train pure NN baseline for comparison
- `fast: false` – quick smoke test mode

**Output:**
- `out_fig_dir: FD_log` – where plots are saved

2) Run the script:

```bash
python ojits03_a13_pytorch_revised.py --config configs/config.yaml
```

3) Outputs:
- Checkpoints in `runs/a13_exp1/<timestamped folders>`
- Plots in the specified `out_fig_dir`
- Console logs show per-epoch training and final metrics

## Multi-experiments: sweep sensor counts and FD types

Use `run_multi_sensor_experiments.py` with `configs/config_multi.yaml` to automate multiple runs across sensor counts and FD formulations.

1) Edit `configs/config_multi.yaml`. Key options:

**Data:**
- `data_file: data/A13_Velocity_Data_0909-0910.txt`
- `distance_json: td_data/2024-09-09.json`

**Model:**
- `layers: [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]`
- `N_f: 10000`
- `physics_every: 1`

**FD formulations to test:**
- `fd_name_list: ['linear', 'log', 'exp', 'power', 'triangular']`
  - Add `'nn'` to include pure neural network baseline

**Training:**
- `epochs: 10000`
- `lr: 0.0001`
- `log_every: 500`
- `patience: 2000`
- `seed: 25`

**Experiment grid:**
- `num_runs: 5` – independent runs per configuration
- `sensor_list: [3, 5, 7, 10]` – sensor counts to test

**Outputs:**
- `base_run_dir: Results_FD/runs/a13_multi`
- `results_out: Results_FD/summary/a13_multi_results.csv`
- `summary_out: Results_FD/summary/a13_multi_summary.csv`

**Fast mode:**
- `fast: false` – set to true for quick smoke tests

2) Run the multi script:

```bash
python run_multi_sensor_experiments.py --config configs/config_multi.yaml
```

3) Outputs:
- Per-run folders: `Results_FD/runs/a13_multi/NS{n}/run_{i}/{fd_name}/`
  - Contains model checkpoints and metadata
- Results CSV: per-run metrics (sensor_count, run_idx, fd_name, model, error_u, etc.)
- Summary CSV: mean/std statistics per (sensor_count, fd_name, model)
- Plots: multi-model comparisons for each sensor configuration

**Eval mode:** If trained models already exist in the result directories, the script automatically runs in evaluation mode—loading and evaluating existing models instead of retraining. Only models matching the `fd_name_list` and `sensor_list` in the config are evaluated.

## Tips

- **GPU acceleration:** Training benefits from CUDA if available; otherwise runs on CPU
- **Smoke tests:** Set `fast: true` in configs to verify the pipeline quickly
- **Reproducibility:** `seed` controls random sampling and training initialization
- **Custom FD models:** Extend `UnifiedPINN.net_f()` in `ojits03_a13_pytorch_revised.py` and add to configs
