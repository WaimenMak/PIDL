"""
Run multiple experiments varying the number of sensors and collect results.
For each sensor count, run multiple independent seeds, train both PINN and NN,
record per-run metrics, and compute aggregate statistics (mean/std).
All checkpoints are saved into per-run folders.

How It Works:
For each sensor count (e.g., 2, 4, 8, 12 sensors)

For each run/seed (e.g., 5 runs per sensor count)

Check if trained models already exist for this run:
- If all models for the specified fd_name_list exist: EVAL MODE
  Load pre-trained models and evaluate them
- If any model is missing: TRAIN MODE
  Train all models from scratch

For each fd_name (e.g., 'linear', 'log', 'exp', 'nn'):

Train or load a model with that FD formulation
If fd_name == 'nn': train pure neural network (f_weight=0.0)
Otherwise: train PINN with specified FD (f_weight=1.0)
Collect predictions and metrics
Generate plots showing all models side-by-side in one figure

Save results with columns: sensor_count, run_idx, fd_name, model, error_u, etc.

Compute summary statistics grouped by sensor_count, fd_name, and model

Note: Only models matching the fd_name_list and sensor_list in the config will be
evaluated when running in eval mode. This allows flexible re-evaluation of specific
model configurations without retraining.
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
    fd_name: str,
    X_u_train: np.ndarray,
    u_train: np.ndarray,
    X_f_train: np.ndarray,
    layers: List[int],
    lb: np.ndarray,
    ub: np.ndarray,
    V_f = 100.0,
    speed_limits_df: pd.DataFrame = None,
) -> UnifiedPINN:
    """Load a previously trained model from checkpoint.
    
    Args:
        model_dir: Directory containing the model checkpoint
        model_tag: The tag used for the checkpoint filename (e.g., 'linear_fw1.0')
        fd_name: The actual fundamental diagram type (e.g., 'linear', 'nn')
        X_u_train, u_train, X_f_train, layers, lb, ub: Model building parameters
        speed_limits_df: Optional DataFrame with location-based speed limits
    
    Returns:
        Loaded UnifiedPINN model
    """
    import torch
    import json
    
    ckpt_path = os.path.join(model_dir, f"model_{model_tag}.pt")
    meta_path = os.path.join(model_dir, f"model_{model_tag}_meta.json")
    
    # Load metadata to get f_weight
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    f_weight = meta.get('f_weight', 1.0)
    
    # Build model architecture with speed limits using actual fd_name
    model = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=f_weight, fd_name=fd_name, V_f=V_f, speed_limits_df=speed_limits_df)
    
    # Load checkpoint
    state = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    
    return model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def check_model_exists(model_dir: str, fd_name: str) -> bool:
    """Check if a trained model already exists in the model directory.
    
    Args:
        model_dir: Directory where the model checkpoint should be saved
        fd_name: The fd_name tag used for the checkpoint filename
    
    Returns:
        True if both checkpoint (.pt) and meta (.json) files exist, False otherwise
    """
    ckpt_path = os.path.join(model_dir, f"model_{fd_name}.pt")
    meta_path = os.path.join(model_dir, f"model_{fd_name}_meta.json")
    return os.path.isfile(ckpt_path) and os.path.isfile(meta_path)


def check_all_models_exist(run_root: str, fd_name_list: List[str]) -> bool:
    """Check if all models for the given fd_name_list exist in the run directory.
    
    Args:
        run_root: Root directory for this run (e.g., 'runs/a13_multi/NS5/run_1')
        fd_name_list: List of fd_names to check for
    
    Returns:
        True if all models exist, False if any are missing
    """
    if not os.path.isdir(run_root):
        return False
    
    for fd_name in fd_name_list:
        model_dir = os.path.join(run_root, fd_name)
        if not check_model_exists(model_dir, fd_name):
            return False
    
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-run experiments over sensor counts (config-driven)")
    p.add_argument('--config', type=str, default='configs/config_multi.yaml', help='Path to multi-experiment YAML config')
    return p.parse_args()


def run_single(
    sensor_count: int,
    N_f: int,
    run_idx: int,
    base_seed: int,
    vel_df: pd.DataFrame,
    x: np.ndarray,
    t: np.ndarray,
    layers: List[int],
    epochs: int,
    lr: float,
    log_every: int,
    patience: int,
    base_run_dir: str,
    physics_every: int,
    fd_name_list: List[str],
    f_weight_list: List[float],
    use_inferred_speed_limits: bool = True,
    V_f: float = 100.0,
    speed_limit_percentile: int = 95,
    valid_speed_limits: tuple = (80, 100),
    use_lbfgs: bool = True,
    lbfgs_epochs: int | None = None,
    plot_loss_history_flag: bool = True,
    loss_plot_log_scale: bool = True,
    noise_level: float = 0.0,
    noise_type: str = "relative_gaussian",
) -> List[Dict[str, Any]]:
    """Run one sensor configuration with one N_f value for one seed, training models for each fd_name.
    Returns a list of per-model result dicts.
    """
    set_seed(base_seed + run_idx)

    # Prepare grids and labels
    # Keep Exact as (n_locations, n_timesteps) - NO transpose
    # Flip vertically to match flipped x coordinates (if x is flipped)
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
    print(f"  - Total available locations: {n_locations}")
    print(f"  - Requested sensors: {sensor_count}")
    print(f"  - Selected sensors: {len(sensors)} (equally spaced)")
    print(f"  - Sensor indices: {sensors}")
    print(f"  - Total observation points: {n_valid}")
    # Optional noise injection on observable sensor readings only
    if noise_level and noise_level > 0.0:
        nl = float(noise_level)
        if noise_type == "relative_gaussian":
            eps = np.random.normal(loc=0.0, scale=nl, size=u_train.shape)
            u_train = u_train * (1.0 + eps)
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")
        print(f"  - Applied noise: type={noise_type}, level={nl*100:.0f}% to {u_train.shape[0]} observations")
    # Collocation
    X_f_train = make_collocation(lb, ub, N_f, X_u_train)

    # Run directory for this sensor count + N_f + run
    run_root = os.path.join(base_run_dir, f"NS{sensor_count}_Nf{N_f}", f"run_{run_idx}")
    ensure_dir(run_root)

    # Build model configuration list: combination of fd_name and f_weight
    # For 'nn', always use f_weight=0.0 once (no duplicates)
    # For PINNs, test each f_weight in the list
    model_configs = []
    for fd_name in fd_name_list:
        if fd_name.lower() == 'nn':
            # Pure NN: always f_weight=0.0, only add once
            model_configs.append({'fd_name': fd_name, 'f_weight': 0.0})
        else:
            # PINN: test each f_weight in the list
            for f_weight in f_weight_list:
                model_configs.append({'fd_name': fd_name, 'f_weight': f_weight})
    
    # Check if all models already exist for this run
    # Use 'd' instead of '.' in tags to avoid file extension issues
    model_tags = [f"{cfg['fd_name']}_fw{str(cfg['f_weight']).replace('.', 'd')}" for cfg in model_configs]
    eval_mode = check_all_models_exist(run_root, model_tags)
    
    if eval_mode:
        print(f"  [EVAL MODE] All models found in {run_root}. Loading and evaluating...")
    else:
        print(f"  [TRAIN MODE] Training models...")

    rows: List[Dict[str, Any]] = []
    model_predictions: List[Dict[str, Any]] = []

    # Train or evaluate one model per configuration
    for cfg in model_configs:
        fd_name = cfg['fd_name']
        f_weight = cfg['f_weight']
        # Use 'd' instead of '.' in tag to avoid file extension issues (e.g., 1.0 -> 1d0)
        model_tag = f"{fd_name}_fw{str(f_weight).replace('.', 'd')}"
        model_dir = os.path.join(run_root, model_tag)
        ensure_dir(model_dir)

        model_type = 'NN' if f_weight == 0.0 else 'PINN'

        if eval_mode and check_model_exists(model_dir, model_tag):
            # Load existing model and evaluate
            print(f"  Loading existing model: fd_name={fd_name}, f_weight={f_weight}...")
            model = load_trained_model(
                model_dir, model_tag, fd_name, X_u_train, u_train, X_f_train,
                layers, lb, ub, V_f=V_f, speed_limits_df=df_free_flow
            )
            error_u, U_pred, _ = evaluate_model(model, X_star, u_star, T, X, Exact)
            
            # Get checkpoint paths
            ckpt_path = os.path.join(model_dir, f"model_{model_tag}.pt")
            meta_path = os.path.join(model_dir, f"model_{model_tag}_meta.json")
            
            # Load metadata for best_epoch, best_train, and train_time
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            best_epoch = meta.get('best_epoch', -1)
            best_train = meta.get('best_train', -1.0)
            train_time = meta.get('train_time_sec', 0.0)
            
            # Get history for plotting if available
            history = meta.get('history', {"epoch": [], "train_total": [], "data_loss": [], "phys_loss": []})
            
        else:
            # Train new model
            print(f"  Training fd_name={fd_name}, f_weight={f_weight}...")
            model = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=f_weight, fd_name=fd_name, V_f=V_f, speed_limits_df=df_free_flow)
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
            
            # Save training time to metadata
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            meta['train_time_sec'] = train_time
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"  Saved train_time={train_time:.2f}s to metadata")
        
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
        if f_weight == 0.0:
            display_name = f"NN ({fd_name})"
        else:
            display_name = f"PINN ({fd_name}, fw={f_weight})"
        
        model_predictions.append({
            'name': display_name,
            'U_pred': U_pred,
            'error': error_u,
        })

        # Collect results row
        rows.append({
            'sensor_count': sensor_count,
            'N_f': N_f,
            'run_idx': run_idx,
            'seed': base_seed + run_idx,
            'fd_name': fd_name,
            'model': model_type,
            'f_weight': f_weight,
            'n_valid': n_valid,
            'noise_level': noise_level,
            'noise_type': noise_type,
            'best_epoch': best_epoch,
            'best_train': best_train,
            'error_u': error_u,
            'train_time_sec': train_time,
            'checkpoint_path': ckpt_path,
            'meta_path': meta_path,
            'run_dir': model_dir,
        })

    plot_multi_models(
        Exact=Exact, x=x, t=t, X_u_train=X_u_train, idx_train=idx_train,
        model_results=model_predictions,
        n_valid=n_valid, out_dir=run_root,
        N_u=sensor_count,
    )

    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std over runs for each combination of sensor_count, N_f, fd_name, model,
    and, when present, noise_level and noise_type.
    """
    group_cols = ['sensor_count', 'N_f', 'fd_name', 'model']
    if 'noise_level' in df.columns:
        group_cols.append('noise_level')
    if 'noise_type' in df.columns:
        group_cols.append('noise_type')
    grouped = df.groupby(group_cols)
    summary = grouped['error_u'].agg(['mean', 'std']).reset_index()
    summary = summary.rename(columns={'mean': 'error_u_mean', 'std': 'error_u_std'})
    # Optionally include best_train summary, too
    bt = grouped['best_train'].agg(['mean', 'std']).reset_index().rename(
        columns={'mean': 'best_train_mean', 'std': 'best_train_std'})
    summary = pd.merge(summary, bt, on=group_cols, how='left')
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
    N_f_list: List[int] = cfg.get('N_f_list', [])
    if not N_f_list:
        # Fallback to single N_f value for backward compatibility
        N_f_list = [int(cfg.get('N_f', 10000))]
    else:
        N_f_list = [int(nf) for nf in N_f_list]
    epochs: int = int(cfg.get('epochs', 10000))
    lr: float = float(cfg.get('lr', 1e-4))
    log_every: int = int(cfg.get('log_every', 200))
    patience: int = int(cfg.get('patience', 2000))
    num_runs: int = int(cfg.get('num_runs', 5))
    sensor_list: List[int] = cfg.get('sensor_list', [])
    fd_name_list: List[str] = cfg.get('fd_name_list', ['linear', 'nn'])
    f_weight_list: List[float] = cfg.get('f_weight_list', [1.0])
    base_run_dir: str = cfg.get('base_run_dir', 'runs/a13_multi')
    results_out: str = cfg.get('results_out', 'Results/a13_multi_results.csv')
    summary_out: str = cfg.get('summary_out', 'Results/a13_multi_summary.csv')
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
    # Noise parameters
    noise_type: str = str(cfg.get('noise_type', 'relative_gaussian'))
    noise_levels: List[float] = [float(v) for v in cfg.get('noise_levels', [])] if 'noise_levels' in cfg else [float(cfg.get('noise_level', 0.0))]

    if not sensor_list:
        raise ValueError("sensor_list must be provided in the multi-experiment config")
    if not fd_name_list:
        raise ValueError("fd_name_list must be provided in the multi-experiment config")
    if not N_f_list:
        raise ValueError("N_f_list must be provided in the multi-experiment config (or use N_f for single value)")

    # Fast mode reductions
    if fast:
        epochs = min(epochs, 2)
        N_f_list = [min(nf, 2000) for nf in N_f_list]
        print('[FAST] Using reduced epochs and collocation points for a smoke test')

    # Load data once
    vel_df = load_velocity_table(data_file)
    x = load_distances(distance_json, n_locations_hint=vel_df.shape[0])
    # Flip x to have start of highway at top (higher km at index 0)
    x = np.flipud(x)
    t = np.arange(vel_df.shape[1]).reshape(-1, 1)

    # Iterate noise levels, then N_f values, sensor counts, and runs
    for nl in noise_levels:
        nl_pct = int(round(nl * 100))
        suffix = f"_Gn_{nl_pct}"
        # Paths with suffix
        base_run_dir_n = f"{base_run_dir}{suffix}"
        def add_suffix_to_csv(path: str) -> str:
            root, ext = os.path.splitext(path)
            if ext.lower() == '.csv':
                return f"{root}{suffix}{ext}"
            return f"{path}{suffix}"
        results_out_n = add_suffix_to_csv(results_out)
        summary_out_n = add_suffix_to_csv(summary_out)

        # Ensure result directories exist for this noise level
        res_dirname = os.path.dirname(results_out_n)
        sum_dirname = os.path.dirname(summary_out_n)
        if res_dirname:
            ensure_dir(res_dirname)
        if sum_dirname:
            ensure_dir(sum_dirname)

        print(f"\n{'='*60}")
        print(f"=== Noise level: {nl_pct}% ({noise_type}) ===")
        print(f"Save suffix: {suffix}")
        print(f"Runs dir: {base_run_dir_n}")
        print(f"Results: {results_out_n}")
        print(f"Summary: {summary_out_n}")
        print(f"{'='*60}")

        all_rows: List[Dict[str, Any]] = []

        # Iterate N_f values, sensor counts, and runs
        for N_f in N_f_list:
            print(f"\n{'-'*60}")
            print(f"--- Collocation points (N_f): {N_f} ---")
            print(f"{'-'*60}")
            for sensor_count in sensor_list:
                print(f"\n=== Sensor count: {sensor_count} ===")
                for run_idx in range(1, num_runs + 1):
                    print(f"-- Run {run_idx}/{num_runs}")
                    rows = run_single(
                        sensor_count=sensor_count,
                        N_f=N_f,
                        run_idx=run_idx,
                        base_seed=seed,
                        vel_df=vel_df,
                        x=x,
                        t=t,
                        layers=layers,
                        epochs=epochs,
                        lr=lr,
                        log_every=log_every,
                        patience=patience,
                        base_run_dir=base_run_dir_n,
                        physics_every=physics_every,
                        fd_name_list=fd_name_list,
                        f_weight_list=f_weight_list,
                        use_inferred_speed_limits=use_inferred_speed_limits,
                        V_f=V_f,
                        speed_limit_percentile=speed_limit_percentile,
                        valid_speed_limits=valid_speed_limits,
                        use_lbfgs=use_lbfgs,
                        lbfgs_epochs=lbfgs_epochs,
                        plot_loss_history_flag=plot_loss_history_flag,
                        loss_plot_log_scale=loss_plot_log_scale,
                        noise_level=nl,
                        noise_type=noise_type,
                    )
                    all_rows.extend(rows)

                # After each sensor_count, persist interim for this noise level
                df_interim = pd.DataFrame(all_rows)
                df_interim.to_csv(results_out_n, index=False)
                print(f"Saved interim results to {results_out_n}")

        # Final save of per-run results for this noise level
        results_df = pd.DataFrame(all_rows)
        results_df.to_csv(results_out_n, index=False)
        print(f"Saved per-run results to {results_out_n}")

        # Summary for this noise level
        summary_df = summarize_results(results_df)
        summary_df.to_csv(summary_out_n, index=False)
        print(f"Saved summary to {summary_out_n}")


if __name__ == '__main__':
    main()
