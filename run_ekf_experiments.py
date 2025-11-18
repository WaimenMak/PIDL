"""
Run Extended Kalman Filter (EKF) experiments for traffic speed estimation.

This script evaluates EKF performance with varying numbers of sensors:
- Load A13 highway data
- For each sensor count configuration:
  - Run multiple independent trials with different sensor placements
  - Apply EKF to estimate full speed field from sparse measurements
  - Compute error metrics (RMSE, MAE, MAPE, R²)
- Generate visualizations comparing true vs estimated fields
- Save aggregate statistics and per-run results
"""

from __future__ import annotations

import os
import time
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import yaml
except ImportError:
    yaml = None

# Import utilities
from utils.utils import (
    set_seed,
    load_velocity_table,
    load_distances,
    replace_missing_with_mean,
    select_sensor_columns,
    build_space_time_grid,
    build_index_grid,
    plot_all,
)

# Import EKF module
from ekf_speed_estimation import run_ekf_estimation


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> Dict[str, float]:
    """
    Compute error metrics between true and predicted values.
    
    Args:
        y_true: True values, shape (T, N) or flattened
        y_pred: Predicted values, shape (T, N) or flattened  
        normalize: If True, normalize data (z-score) before computing metrics (matches PINN)
        
    Returns:
        Dictionary with RMSE, MAE, MAPE, R², error_u (normalized RMSE)
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove any NaN/inf values
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'error_u': np.nan}
    
    # Normalize if requested (z-score normalization like PINN)
    if normalize:
        y_mean = np.mean(y_true_clean)
        y_std = np.std(y_true_clean)
        if y_std > 1e-8:
            y_true_norm = (y_true_clean - y_mean) / y_std
            y_pred_norm = (y_pred_clean - y_mean) / y_std
        else:
            y_true_norm = y_true_clean
            y_pred_norm = y_pred_clean
        # error_u: normalized RMSE (this is what PINN reports)
        error_u = np.sqrt(np.mean((y_true_norm - y_pred_norm) ** 2))
    else:
        error_u = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    
    # RMSE (physical units)
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    # MAPE (avoid division by zero)
    mape_mask = y_true_clean > 1.0
    if np.sum(mape_mask) > 0:
        mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
    else:
        mape = np.nan
    
    # R²
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else np.nan
    
    return {
        'error_u': float(error_u),  # Normalized error (primary metric, matches PINN)
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2)
    }


def run_single_experiment(
    Exact: np.ndarray,        # Speed matrix (n_locations, n_timesteps)
    x: np.ndarray,            # Spatial locations (n_locations, 1)
    t: np.ndarray,            # Time array (n_timesteps, 1)
    X_star: np.ndarray,       # Full grid coordinates
    u_star: np.ndarray,       # Flattened data
    sensor_count: int,
    run_idx: int,
    ekf_params: dict,
    dt: float,
    output_dir: str,
    save_plots: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a single EKF experiment with specified sensor configuration.
    
    Args:
        Exact: Speed matrix [km/h], shape (n_locations, n_timesteps)
        x: Spatial locations [km], shape (n_locations, 1)
        t: Time array [hours], shape (n_timesteps, 1)
        X_star: Full grid coordinates (n_total, 2)
        u_star: Flattened data (n_total, 1)
        sensor_count: Number of sensors
        run_idx: Run index for this configuration
        ekf_params: Dictionary of EKF parameters
        dt: Data time step [hours]
        output_dir: Output directory for results
        save_plots: Whether to save plots
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results
    """
    set_seed(seed + run_idx)
    
    n_locations = x.shape[0]
    n_timesteps = t.shape[0]
    
    # Use the SAME sensor selection method as PINN experiments
    print(f"\n  Run {run_idx}: Selecting {sensor_count} sensors using select_sensor_columns...")
    X_u_train, u_train, idx_train, n_valid, sensors, sensor_point_counts = select_sensor_columns(
        u_star, X_star, n_locations, n_timesteps, sensor_count
    )
    
    print(f"    Selected sensor indices: {sensors}")
    print(f"    Total observation points: {n_valid}")
    for loc_idx, n_pts in sensor_point_counts:
        print(f"      · Sensor {loc_idx}: {n_pts} points")
    
    # Extract sensor indices and flatten x locations
    sensor_indices = np.array(sensors)
    x_locations = x.flatten()
    
    # Transpose Exact to (T, N) for EKF (time first)
    u_true = Exact.T  # Now shape (n_timesteps, n_locations)
    T, N = u_true.shape
    
    print(f"    Total time steps to process: {T}")
    
    # Run EKF with progress tracking
    start_time = time.time()
    log_every = max(1, T // 10)  # Log ~10 times during run
    u_est, u_std, u_sensors = run_ekf_estimation(
        u_true=u_true,
        x_locations=x_locations,
        sensor_indices=sensor_indices,
        dt_data=dt,
        ekf_params=ekf_params.copy(),  # Copy to avoid modifying original dict
        log_every=log_every,
    )
    elapsed = time.time() - start_time
    
    # Compute metrics (normalized like PINN for comparison)
    metrics = compute_metrics(u_true, u_est, normalize=True)
    metrics['time_seconds'] = elapsed
    
    print(f"    Results: error_u={metrics['error_u']:.4f} (normalized), RMSE={metrics['rmse']:.2f} km/h, "
          f"MAE={metrics['mae']:.2f} km/h, Time={elapsed:.1f}s")
    
    # Save plots using plot_all (matching PINN format)
    if save_plots:
        run_dir = os.path.join(output_dir, 'runs', f'ekf_n{sensor_count}_run{run_idx}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Convert EKF results to PINN format for plot_all
        # u_est is (T, N), need to transpose to (N, T) for plot_all which expects (n_locations, n_timesteps)
        U_pred_ekf = u_est.T  # Shape: (n_locations, n_timesteps)
        
        # plot_all expects Exact in (n_locations, n_timesteps) which matches Exact we already have
        plot_all(
            Exact=Exact,
            x=x,
            t=t,
            X_u_train=X_u_train,
            idx_train=idx_train,
            U_pred=U_pred_ekf,
            error_u=metrics['error_u'],
            U_pred2=None,  # No second model
            error_u2=None,
            n_valid=n_valid,
            out_dir=run_dir,
            N_u=n_valid,
            fd_name=f"EKF_{ekf_params.get('fd_name', 'linear')}",
        )
    
    # Prepare result record (matching PINN format)
    result = {
        'sensor_count': sensor_count,
        'run_idx': run_idx,
        'model': 'EKF',  # Model type for comparison with PINN
        'fd_name': ekf_params.get('fd_name', 'linear'),  # FD used in EKF
        'sensor_indices': str(sensors),  # Convert to string for CSV
        'n_observations': n_valid,  # Total observation points (all time steps from sensors)
        **metrics
    }
    
    return result


def main(config_path: str = 'configs/config_ekf.yaml'):
    """
    Main function to run EKF experiments.
    
    Args:
        config_path: Path to configuration file
    """
    print("="*80)
    print("Extended Kalman Filter (EKF) Experiments for Traffic Speed Estimation")
    print("="*80)
    
    # Load configuration
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nLoaded configuration from: {config_path}")
    
    # Extract parameters
    data_file = config['data_file']
    distance_json = config['distance_json']
    output_dir = config['output_dir']
    sensor_list = config['sensor_list']
    runs_per_config = config['runs_per_config']
    base_seed = config['seed']
    save_plots = config.get('save_plots', True)
    
    # Data sampling time step
    dt = config.get('dt_data', 0.25)  # Default 15 minutes if not specified
    
    # EKF parameters
    ekf_params = {
        'dt': config.get('dt', 0.01),  # Internal EKF time step (for stability)
        'V_f': config['V_f'],
        'fd_name': config['fd_name'],
        'process_noise_std': config['process_noise_std'],
        'measurement_noise_std': config['measurement_noise_std'],
        'initial_state_std': config['initial_state_std'],
        'rho_max': config.get('rho_max', 180.0),
        'use_flux_limiter': config.get('use_flux_limiter', True),
        'flux_limiter': config.get('flux_limiter', 'minmod'),
        'cfl_factor': config.get('cfl_factor', 0.5),
    }
    
    print(f"\nEKF Configuration:")
    print(f"  Fundamental Diagram: {ekf_params['fd_name']}")
    print(f"  Free-flow speed V_f: {ekf_params['V_f']} km/h")
    print(f"  Process noise: {ekf_params['process_noise_std']} km/h")
    print(f"  Measurement noise: {ekf_params['measurement_noise_std']} km/h")
    print(f"  Time step dt: {dt} hours")
    print(f"  Flux limiter: {ekf_params['flux_limiter'] if ekf_params['use_flux_limiter'] else 'None'}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'summary'), exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    vel_df = load_velocity_table(data_file)
    print(f"  Data shape: {vel_df.shape}")
    
    # Load distances
    x = load_distances(distance_json, n_locations_hint=vel_df.shape[0])
    # Flip x to have start of highway at top (higher km at index 0) - matching PINN
    x = np.flipud(x)
    t = np.arange(vel_df.shape[1]).reshape(-1, 1)
    
    # Keep Exact as (n_locations, n_timesteps) - matching PINN format
    # Flip vertically to match flipped x coordinates
    Exact = np.flipud(np.real(vel_df.values))
    n_locations, n_timesteps = Exact.shape
    print(f"  Spatial locations: {n_locations}")
    print(f"  Time steps: {n_timesteps}")
    print(f"  Spatial extent: {x.flatten()[0]:.2f} to {x.flatten()[-1]:.2f} km")
    
    # Build grids (matching PINN)
    T, X, X_star = build_space_time_grid(x, t)
    idx_grid = build_index_grid(Exact, t)
    u_star = Exact.flatten()[:, None]
    
    # Replace missing values
    if config.get('replace_missing', True):
        u_star, n_missing, u_mean = replace_missing_with_mean(u_star)
        if n_missing > 0:
            print(f"  Replaced {n_missing} missing values with mean={u_mean:.2f} km/h")
        else:
            print(f"  No missing values found")
    
    # Run experiments
    print(f"\nRunning experiments for sensor counts: {sensor_list}")
    print(f"  Runs per configuration: {runs_per_config}")
    
    all_results = []
    
    for sensor_count in sensor_list:
        print(f"\n{'='*60}")
        print(f"Sensor count: {sensor_count}")
        print(f"{'='*60}")
        
        for run_idx in range(runs_per_config):
            result = run_single_experiment(
                Exact=Exact,
                x=x,
                t=t,
                X_star=X_star,
                u_star=u_star,
                sensor_count=sensor_count,
                run_idx=run_idx,
                ekf_params=ekf_params,
                dt=dt,
                output_dir=output_dir,
                save_plots=save_plots,
                seed=base_seed,
            )
            all_results.append(result)
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, 'summary', 'ekf_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved detailed results to: {results_path}")
    
    # Compute summary statistics (matching PINN format)
    summary_stats = results_df.groupby(['sensor_count', 'fd_name', 'model']).agg({
        'error_u': ['mean', 'std'],  # Primary metric (normalized)
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'mape': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'time_seconds': ['mean', 'std']
    }).reset_index()
    
    summary_stats.columns = ['_'.join(col).strip('_') for col in summary_stats.columns.values]
    summary_path = os.path.join(output_dir, 'summary', 'ekf_summary.csv')
    summary_stats.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to: {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_stats.to_string(index=False))
    
    print("\n" + "="*80)
    print("EKF Experiments Complete!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run EKF experiments for traffic speed estimation')
    parser.add_argument('--config', type=str, default='configs/config_ekf.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(config_path=args.config)
