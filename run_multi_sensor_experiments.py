"""
Run multiple experiments varying the number of sensors and collect results.
For each sensor count, run multiple independent seeds, train both PINN and NN,
record per-run metrics, and compute aggregate statistics (mean/std).
All checkpoints are saved into per-run folders.

How It Works:
For each sensor count (e.g., 2, 4, 8, 12 sensors)

For each run/seed (e.g., 5 runs per sensor count)

For each fd_name (e.g., 'linear', 'log', 'exp', 'nn'):

Train a model with that FD formulation
If fd_name == 'nn': train pure neural network (f_weight=0.0)
Otherwise: train PINN with specified FD (f_weight=1.0)
Collect predictions and metrics
Generate plots showing all models side-by-side in one figure

Save results with columns: sensor_count, run_idx, fd_name, model, error_u, etc.

Compute summary statistics grouped by sensor_count, fd_name, and model
"""

from __future__ import annotations

import os
import time
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:
    yaml = None

# Import utilities from utils module
from utils import (
    set_seed,
    EarlyStopConfig,
    load_velocity_table,
    load_distances,
    build_space_time_grid,
    build_index_grid,
    replace_missing_with_mean,
    select_sensor_columns,
    make_collocation,
    evaluate_model,
    plot_all,
)

# Import model from main script
from ojits03_a13_pytorch_revised import UnifiedPINN, build_model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-run experiments over sensor counts (config-driven)")
    p.add_argument('--config', type=str, default='config_multi.yaml', help='Path to multi-experiment YAML config')
    return p.parse_args()


def run_single(
    sensor_count: int,
    run_idx: int,
    base_seed: int,
    vel_df: pd.DataFrame,
    x: np.ndarray,
    t: np.ndarray,
    layers: List[int],
    N_f: int,
    epochs: int,
    lr: float,
    log_every: int,
    patience: int,
    base_run_dir: str,
    physics_every: int,
    fd_name_list: List[str],
) -> List[Dict[str, Any]]:
    """Run one sensor configuration for one seed, training models for each fd_name.
    Returns a list of per-model result dicts.
    """
    set_seed(base_seed + run_idx)

    # Prepare grids and labels
    Exact = np.real(vel_df.T)
    X, T, X_star = build_space_time_grid(x, t)
    idx_grid = build_index_grid(Exact, t)
    u_star = Exact.flatten()[:, None]
    u_star, n_missing, u_mean = replace_missing_with_mean(u_star)

    lb = X_star.min(0).astype(np.float32)
    ub = X_star.max(0).astype(np.float32)

    n_locations = x.shape[0]
    n_timesteps = t.shape[0]

    # Sensor-based selection
    X_u_train, u_train, idx_train, n_valid, sensors, sensor_point_counts = select_sensor_columns(
        u_star, X_star, n_locations, n_timesteps, sensor_count
    )

    # Collocation
    X_f_train = make_collocation(lb, ub, N_f, X_u_train)

    # Run directory for this sensor count + run
    run_root = os.path.join(base_run_dir, f"NS{sensor_count}", f"run_{run_idx}")
    ensure_dir(run_root)

    rows: List[Dict[str, Any]] = []
    model_predictions: List[Dict[str, Any]] = []

    # Train one model per fd_name
    for fd_name in fd_name_list:
        print(f"  Training fd_name={fd_name}...")
        model_dir = os.path.join(run_root, fd_name)
        ensure_dir(model_dir)

        # Determine f_weight: if fd_name is 'nn' or 'NN', use 0.0, else 1.0
        f_weight = 0.0 if fd_name.lower() == 'nn' else 1.0
        model_type = 'NN' if f_weight == 0.0 else 'PINN'

        model = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=f_weight, fd_name=fd_name)
        start = time.time()
        out = model.fit(
            epochs=epochs, lr=lr,
            early_stop=EarlyStopConfig(patience=patience, min_delta=0.0, verbose=True),
            save_dir=model_dir, tag=fd_name,
            log_every=log_every,
            f_subset_per_epoch=min(4000, X_f_train.shape[0]),
            physics_every=physics_every, use_mixed_precision=True,
        )
        train_time = time.time() - start
        error_u, U_pred, _ = evaluate_model(model, X_star, u_star, X, T, Exact)

        # Collect for plotting
        display_name = f"{model_type} ({fd_name})" if model_type == 'PINN' else f"NN ({fd_name})"
        model_predictions.append({
            'name': display_name,
            'U_pred': U_pred,
            'error': error_u,
        })

        # Collect results row
        rows.append({
            'sensor_count': sensor_count,
            'run_idx': run_idx,
            'seed': base_seed + run_idx,
            'fd_name': fd_name,
            'model': model_type,
            'f_weight': f_weight,
            'n_valid': n_valid,
            'best_epoch': out['best_epoch'],
            'best_train': out['best_train'],
            'error_u': error_u,
            'train_time_sec': train_time,
            'checkpoint_path': out['checkpoint_path'],
            'meta_path': out['meta_path'],
            'run_dir': model_dir,
        })

    # Generate plot with all models
    from utils import plot_multi_models
    plot_multi_models(
        Exact=Exact, x=x, t=t, X_u_train=X_u_train, idx_train=idx_train,
        model_results=model_predictions,
        n_valid=n_valid, out_dir=run_root,
        N_u=sensor_count,
    )

    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std over runs for each combination of sensor_count, fd_name, and model.
    """
    grouped = df.groupby(['sensor_count', 'fd_name', 'model'])
    summary = grouped['error_u'].agg(['mean', 'std']).reset_index()
    summary = summary.rename(columns={'mean': 'error_u_mean', 'std': 'error_u_std'})
    # Optionally include best_train summary, too
    bt = grouped['best_train'].agg(['mean', 'std']).reset_index().rename(
        columns={'mean': 'best_train_mean', 'std': 'best_train_std'})
    summary = pd.merge(summary, bt, on=['sensor_count', 'fd_name', 'model'], how='left')
    return summary


def main():
    args = parse_args()

    if yaml is None:
        raise ImportError("PyYAML is required to load the config file. Install via `pip install pyyaml`.")
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    # Pull config values with defaults matching previous CLI
    data_file: str = cfg.get('data_file', 'data/A13_Velocity_Data_0909-0910.txt')
    distance_json: str = cfg.get('distance_json', 'td_data/2024-09-09.json')
    layers: List[int] = cfg.get('layers', [2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
    N_f: int = int(cfg.get('N_f', 10000))
    epochs: int = int(cfg.get('epochs', 10000))
    lr: float = float(cfg.get('lr', 1e-4))
    log_every: int = int(cfg.get('log_every', 200))
    patience: int = int(cfg.get('patience', 2000))
    num_runs: int = int(cfg.get('num_runs', 5))
    sensor_list: List[int] = cfg.get('sensor_list', [])
    fd_name_list: List[str] = cfg.get('fd_name_list', ['linear', 'nn'])
    base_run_dir: str = cfg.get('base_run_dir', 'runs/a13_multi')
    results_out: str = cfg.get('results_out', 'Results/a13_multi_results.csv')
    summary_out: str = cfg.get('summary_out', 'Results/a13_multi_summary.csv')
    seed: int = int(cfg.get('seed', 25))
    fast: bool = bool(cfg.get('fast', False))
    physics_every: int = int(cfg.get('physics_every', 1))

    if not sensor_list:
        raise ValueError("sensor_list must be provided in the multi-experiment config")
    if not fd_name_list:
        raise ValueError("fd_name_list must be provided in the multi-experiment config")

    # Fast mode reductions
    if fast:
        epochs = min(epochs, 2)
        N_f = min(N_f, 2000)
        print('[FAST] Using reduced epochs and collocation points for a smoke test')

    # Ensure result directories exist
    res_dirname = os.path.dirname(results_out)
    sum_dirname = os.path.dirname(summary_out)
    if res_dirname:
        ensure_dir(res_dirname)
    if sum_dirname:
        ensure_dir(sum_dirname)

    # Load data once
    vel_df = load_velocity_table(data_file)
    x = load_distances(distance_json, n_locations_hint=vel_df.shape[0])
    t = np.arange(vel_df.shape[1]).reshape(-1, 1)

    all_rows: List[Dict[str, Any]] = []

    # Iterate sensor counts and runs
    for sensor_count in sensor_list:
        print(f"\n=== Sensor count: {sensor_count} ===")
        for run_idx in range(1, num_runs + 1):
            print(f"-- Run {run_idx}/{num_runs}")
            rows = run_single(
                sensor_count=sensor_count,
                run_idx=run_idx,
                base_seed=seed,
                vel_df=vel_df,
                x=x,
                t=t,
                layers=layers,
                N_f=N_f,
                epochs=epochs,
                lr=lr,
                log_every=log_every,
                patience=patience,
                base_run_dir=base_run_dir,
                physics_every=physics_every,
                fd_name_list=fd_name_list,
            )
            all_rows.extend(rows)

        # After each sensor_count, persist interim
        df_interim = pd.DataFrame(all_rows)
        df_interim.to_csv(results_out, index=False)
        print(f"Saved interim results to {results_out}")

    # Final save of per-run results
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(results_out, index=False)
    print(f"Saved per-run results to {results_out}")

    # Summary
    summary_df = summarize_results(results_df)
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved summary to {summary_out}")


if __name__ == '__main__':
    main()
