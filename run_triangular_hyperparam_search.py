"""
Run hyperparameter search for Triangular FD formulation.

This script tunes the two key hyperparameters of the triangular fundamental diagram:
- w: Wave speed parameter (controls congestion dynamics)
- alpha: Sigmoid steepness (controls transition sharpness between free-flow and congestion)

The script runs a grid search or list-based search over specified parameter combinations,
training models for each combination and recording performance metrics.

Output:
- Individual model checkpoints saved per hyperparameter combination
- CSV with all results (sensor_count, run_idx, w, alpha, error_u, etc.)
- Summary CSV with mean/std aggregated over runs
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
from utils.utils import (
    set_seed,
    infer_speed_limits,
    EarlyStopConfig,
    load_velocity_table,
    load_distances,
    build_space_time_grid,
    build_index_grid,
    replace_missing_with_mean,
    select_sensor_columns,
    make_collocation,
    evaluate_model,
    plot_multi_models,
    plot_training_history,
)

# Import model from main script
from ojits03_a13_pytorch_revised import UnifiedPINN, build_model


def load_trained_model(
    model_dir: str,
    model_tag: str,
    X_u_train: np.ndarray,
    u_train: np.ndarray,
    X_f_train: np.ndarray,
    layers: List[int],
    lb: np.ndarray,
    ub: np.ndarray,
    f_weight: float,
    V_f: float = 100.0,
    speed_limits_df: pd.DataFrame = None,
) -> UnifiedPINN:
    """Load a previously trained triangular FD model from checkpoint."""
    import torch
    import json
    
    ckpt_path = os.path.join(model_dir, f"model_{model_tag}.pt")
    meta_path = os.path.join(model_dir, f"model_{model_tag}_meta.json")
    
    # Load metadata to get w and alpha
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Build model architecture with triangular FD
    model = build_model(
        X_u_train, u_train, X_f_train, layers, lb, ub, 
        f_weight=f_weight, fd_name='triangular', V_f=V_f, 
        speed_limits_df=speed_limits_df
    )
    
    # Load checkpoint
    state = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    
    return model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def check_model_exists(model_dir: str, model_tag: str) -> bool:
    """Check if a trained model already exists."""
    ckpt_path = os.path.join(model_dir, f"model_{model_tag}.pt")
    meta_path = os.path.join(model_dir, f"model_{model_tag}_meta.json")
    return os.path.isfile(ckpt_path) and os.path.isfile(meta_path)


def check_all_models_exist(run_root: str, model_tags: List[str]) -> bool:
    """Check if all models for the given tags exist in the run directory."""
    if not os.path.isdir(run_root):
        return False
    
    for model_tag in model_tags:
        model_dir = os.path.join(run_root, model_tag)
        if not check_model_exists(model_dir, model_tag):
            return False
    
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Triangular FD hyperparameter search (config-driven)")
    p.add_argument('--config', type=str, default='configs/config_triangular_search.yaml', 
                   help='Path to triangular hyperparameter search YAML config')
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
    w_list: List[float],
    alpha_list: List[float],
    f_weight: float = 1.0,
    use_inferred_speed_limits: bool = True,
    V_f: float = 100.0,
    speed_limit_percentile: int = 95,
    valid_speed_limits: tuple = (80, 100),
    use_lbfgs: bool = True,
    lbfgs_epochs: int | None = None,
    plot_loss_history_flag: bool = True,
    loss_plot_log_scale: bool = True,
) -> List[Dict[str, Any]]:
    """Run one sensor configuration for one seed, training models for each (w, alpha) combination."""
    set_seed(base_seed + run_idx)

    # Prepare grids and labels
    Exact = np.flipud(np.real(vel_df.values))
    T, X, X_star = build_space_time_grid(x, t)
    idx_grid = build_index_grid(Exact, t)
    u_star = Exact.flatten()[:, None]
    u_star, n_missing, u_mean = replace_missing_with_mean(u_star)
    
    # Conditionally infer free-flow speed limits per location based on config
    if use_inferred_speed_limits:
        df_free_flow = infer_speed_limits(
            Exact, x, 
            valid_limits=valid_speed_limits, 
            perc=speed_limit_percentile
        )
        print(f"  Using inferred speed limits ({speed_limit_percentile}th percentile, range: {df_free_flow['limit_assigned'].min():.1f}-{df_free_flow['limit_assigned'].max():.1f} km/h)")
    else:
        df_free_flow = None
        print(f"  Using default constant free-flow speed: {V_f:.1f} km/h)")

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

    # Build model configuration list: combination of w and alpha
    model_configs = []
    for w in w_list:
        for alpha in alpha_list:
            model_configs.append({'w': w, 'alpha': alpha})
    
    # Check if all models already exist for this run
    model_tags = [f"tri_w{str(cfg['w']).replace('.', 'd')}_a{str(cfg['alpha']).replace('.', 'd')}" 
                  for cfg in model_configs]
    eval_mode = check_all_models_exist(run_root, model_tags)
    
    if eval_mode:
        print(f"  [EVAL MODE] All models found in {run_root}. Loading and evaluating...")
    else:
        print(f"  [TRAIN MODE] Training models...")

    rows: List[Dict[str, Any]] = []
    model_predictions: List[Dict[str, Any]] = []

    # Train or evaluate one model per configuration
    for cfg in model_configs:
        w = cfg['w']
        alpha = cfg['alpha']
        model_tag = f"tri_w{str(w).replace('.', 'd')}_a{str(alpha).replace('.', 'd')}"
        model_dir = os.path.join(run_root, model_tag)
        ensure_dir(model_dir)

        if eval_mode and check_model_exists(model_dir, model_tag):
            # Load existing model and evaluate
            print(f"  Loading existing model: w={w}, alpha={alpha}...")
            model = load_trained_model(
                model_dir, model_tag, X_u_train, u_train, X_f_train,
                layers, lb, ub, f_weight=f_weight, V_f=V_f, speed_limits_df=df_free_flow
            )
            train_time = 0.0
            error_u, U_pred, _ = evaluate_model(model, X_star, u_star, T, X, Exact)
            
            # Get checkpoint paths
            ckpt_path = os.path.join(model_dir, f"model_{model_tag}.pt")
            meta_path = os.path.join(model_dir, f"model_{model_tag}_meta.json")
            
            # Load metadata
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            best_epoch = meta.get('best_epoch', -1)
            best_train = meta.get('best_train', -1.0)
            history = meta.get('history', {"epoch": [], "train_total": [], "data_loss": [], "phys_loss": []})
            
        else:
            # Train new model with specified w and alpha
            print(f"  Training w={w}, alpha={alpha}...")
            model = build_model(
                X_u_train, u_train, X_f_train, layers, lb, ub, 
                f_weight=f_weight, fd_name='triangular', V_f=V_f, 
                speed_limits_df=df_free_flow,
                tri_w=w, tri_alpha=alpha
            )
            start = time.time()
            out = model.fit(
                epochs=epochs, lr=lr,
                early_stop=EarlyStopConfig(patience=patience, min_delta=0.0, verbose=True),
                save_dir=model_dir, tag=model_tag,
                log_every=log_every,
                f_subset_per_epoch=min(4000, X_f_train.shape[0]),
                physics_every=physics_every, use_mixed_precision=True,
                use_lbfgs=use_lbfgs,
                lbfgs_epochs=lbfgs_epochs,
            )
            train_time = time.time() - start
            error_u, U_pred, _ = evaluate_model(model, X_star, u_star, T, X, Exact)
            
            best_epoch = out['best_epoch']
            best_train = out['best_train']
            ckpt_path = out['checkpoint_path']
            meta_path = out['meta_path']
            history = out['history']
            
            # Save w and alpha to metadata
            import json
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                meta['w'] = w
                meta['alpha'] = alpha
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
        
        # Plot training history
        if plot_loss_history_flag and history['epoch']:
            plot_training_history(
                history=history,
                out_dir=run_root,
                tag=model_tag,
                show_physics=(f_weight > 0.0),
                log_scale=loss_plot_log_scale,
            )

        # Collect for plotting
        display_name = f"Triangular (w={w}, α={alpha})"
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
            'w': w,
            'alpha': alpha,
            'f_weight': f_weight,
            'n_valid': n_valid,
            'best_epoch': best_epoch,
            'best_train': best_train,
            'error_u': error_u,
            'train_time_sec': train_time,
            'checkpoint_path': ckpt_path,
            'meta_path': meta_path,
            'run_dir': model_dir,
        })

    # Plot comparison of all hyperparameter combinations
    plot_multi_models(
        Exact=Exact, x=x, t=t, X_u_train=X_u_train, idx_train=idx_train,
        model_results=model_predictions,
        n_valid=n_valid, out_dir=run_root,
        N_u=sensor_count,
    )

    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std over runs for each combination of sensor_count, w, and alpha."""
    grouped = df.groupby(['sensor_count', 'w', 'alpha'])
    summary = grouped['error_u'].agg(['mean', 'std']).reset_index()
    summary = summary.rename(columns={'mean': 'error_u_mean', 'std': 'error_u_std'})
    bt = grouped['best_train'].agg(['mean', 'std']).reset_index().rename(
        columns={'mean': 'best_train_mean', 'std': 'best_train_std'})
    summary = pd.merge(summary, bt, on=['sensor_count', 'w', 'alpha'], how='left')
    return summary


def main():
    args = parse_args()

    if yaml is None:
        raise ImportError("PyYAML is required to load the config file. Install via `pip install pyyaml`.")
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    # Pull config values
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
    w_list: List[float] = cfg.get('w_list', [20.0])
    alpha_list: List[float] = cfg.get('alpha_list', [20.0])
    f_weight: float = float(cfg.get('f_weight', 1.0))
    base_run_dir: str = cfg.get('base_run_dir', 'runs/triangular_search')
    results_out: str = cfg.get('results_out', 'Results/triangular_search_results.csv')
    summary_out: str = cfg.get('summary_out', 'Results/triangular_search_summary.csv')
    seed: int = int(cfg.get('seed', 25))
    fast: bool = bool(cfg.get('fast', False))
    physics_every: int = int(cfg.get('physics_every', 1))
    
    # Speed limit parameters
    use_inferred_speed_limits: bool = bool(cfg.get('use_inferred_speed_limits', True))
    V_f: float = float(cfg.get('V_f', 100.0))
    speed_limit_percentile: int = int(cfg.get('speed_limit_percentile', 95))
    valid_speed_limits: tuple = tuple(cfg.get('valid_speed_limits', [80, 100]))
    
    # Two-stage optimization parameters
    use_lbfgs: bool = bool(cfg.get('use_lbfgs', True))
    lbfgs_epochs: int | None = cfg.get('lbfgs_epochs', None)
    
    # Plotting parameters
    plot_loss_history_flag: bool = bool(cfg.get('plot_loss_history', True))
    loss_plot_log_scale: bool = bool(cfg.get('loss_plot_log_scale', True))

    if not sensor_list:
        raise ValueError("sensor_list must be provided in the config")
    if not w_list:
        raise ValueError("w_list must be provided in the config")
    if not alpha_list:
        raise ValueError("alpha_list must be provided in the config")

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
    x = np.flipud(x)
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
                w_list=w_list,
                alpha_list=alpha_list,
                f_weight=f_weight,
                use_inferred_speed_limits=use_inferred_speed_limits,
                V_f=V_f,
                speed_limit_percentile=speed_limit_percentile,
                valid_speed_limits=valid_speed_limits,
                use_lbfgs=use_lbfgs,
                lbfgs_epochs=lbfgs_epochs,
                plot_loss_history_flag=plot_loss_history_flag,
                loss_plot_log_scale=loss_plot_log_scale,
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
