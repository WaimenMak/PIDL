# PIDL – Physics-Informed DL for Traffic State Estimation (A13)

This repo contains a unified PINN/NN model to reconstruct space–time velocity fields on the A13 corridor from sparse sensors. It supports multiple fundamental-diagram (FD) formulations and a pure neural-network baseline, all driven by YAML config files. A companion script runs multi-experiments across different sensor counts and FD types and produces summaries and plots.

## What’s here

- `ojits03_a13_pytorch_revised.py` – main training/evaluation script (single run), config-driven
- `run_multi_sensor_experiments.py` – multi-run driver over sensor counts and FD types
- `utils.py` – data loading, sampling, evaluation, and plotting (including multi-model plots)
- `config.yaml` – config for single run
- `config_multi.yaml` – config for multi-experiments
- `data/`, `td_data/` – inputs: velocity tables and distance metadata
- `runs/`, `Results*/`, `figures*/` – outputs: checkpoints, CSVs, and plots

## Requirements

- Python 3.9+ (tested on Linux)
- Recommended: a GPU with CUDA for speed (CPU works too)
- Python packages (install with pip):
	- torch
	- numpy, pandas
	- pyyaml
	- matplotlib, scipy

Example install:

```bash
pip install torch numpy pandas pyyaml matplotlib scipy
```

## Data layout

- Velocity table (CSV-like text) under `data/`, e.g. `data/A13_Velocity_Data_0909-0910.txt`
- Distance metadata JSON under `td_data/`, e.g. `td_data/2024-09-09.json`

Both paths are set in the configs.

## Single run: train PINN and optional NN baseline

1) Edit `config.yaml` as needed. Key options:

- Data
	- `data_file`: path to velocity table
	- `distance_json`: path to distance metadata
- Model
	- `layers`: network sizes, e.g. `[2, 20, ..., 1]`
	- `fd_name`: fundamental diagram for PINN physics. Supported: `linear`, `log`, `exp`, `power`, `triangular`
	- `f_weight`: physics loss weight (set by config; for baseline NN training we set this to 0 automatically)
- Sampling and observations
	- `sensor_based`: if true, use equally spaced sensor columns; else random point sampling
	- `n_sensors`: number of sensors when `sensor_based=true`
	- `N_u`: number of observation points when `sensor_based=false`
	- `N_f`: number of physics collocation points
- Training
	- `epochs`, `lr`, `patience`, `log_every`, `physics_every`, `seed`
	- `run_base`: if true, also train a pure NN baseline (f_weight=0, fd_name='nn')
	- `fast`: if true, reduce epochs/points for a quick smoke test
- Output
	- `out_fig_dir`: folder where plots for this single run are saved

2) Run the script:

```bash
python ojits03_a13_pytorch_revised.py --config config.yaml
```

3) Outputs

- Checkpoints for PINN (and NN baseline if enabled) in `runs/a13_exp1/<timestamped folders>`
- Plots in `out_fig_dir` from the config (e.g., `fd_lin_test/`)
- Console shows per-epoch training logs and final error metrics

Notes

- Set `run_base: true` to train and plot both PINN and NN. When only PINN is trained (`run_base: false`), plotting adapts automatically.
- `fd_name='triangular'` enables a smooth gate between free-flow plateau and congested-wave PDE.

## Multi-experiments: sweep sensor counts and FD types

Use `run_multi_sensor_experiments.py` with `config_multi.yaml` to automate multiple runs.

1) Edit `config_multi.yaml`. Key options:

- Data: `data_file`, `distance_json`
- Model: `layers`, `N_f`, `physics_every`
- FD sweep: `fd_name_list` – choose any of `linear`, `log`, `exp`, `power`, `triangular`
	- If you want the pure NN baseline in the comparison, include `'nn'` in this list
- Optimization: `epochs`, `lr`, `log_every`, `patience`, `seed`
- Experiment grid: `num_runs` (seeds per setting), `sensor_list` (e.g., `[5,10]`)
- Outputs: `base_run_dir` (per-run folders), `results_out` (per-run CSV), `summary_out` (mean/std CSV)
- `fast: true` for a quick smoke test (reduces epochs and N_f)

2) Run the multi script:

```bash
python run_multi_sensor_experiments.py --config config_multi.yaml
```

3) Outputs

- Per-run folders under `base_run_dir`, grouped by sensor count (e.g. `.../NS5/run_1/linear/`)
- A per-run results CSV at `results_out` with columns like: sensor_count, run_idx, fd_name, model, best_epoch, error_u, etc.
- A summary CSV at `summary_out` with mean/std per (sensor_count, fd_name, model)
- Plots per sensor-count run showing all models side-by-side

## Tips and troubleshooting

- GPU: training benefits from CUDA if available; otherwise it runs on CPU.
- Smoke tests: set `fast: true` in the config(s) to verify the pipeline quickly.
- Reproducibility: `seed` controls random sampling and training initialization.
- Figures: the code creates output directories if missing.
- Adding new FD models: extend `UnifiedPINN.net_f()` and include the new name in configs as needed.

## Citation / attribution

If you use parts of this codebase, please reference this repository and the A13 traffic reconstruction context where appropriate.
