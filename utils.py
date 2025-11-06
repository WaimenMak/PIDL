"""
Utility functions for A13 highway speed reconstruction project.
Includes data loading, preprocessing, sampling, evaluation, and plotting helpers.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from pyDOE import lhs
import torch


# ----------------------------- General Utilities ---------------------------------
def set_seed(seed: int = 25) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def timestamp_dir(prefix: str) -> str:
    """Generate a timestamped directory name."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


@dataclass
class EarlyStopConfig:
    """Configuration for early stopping during training."""
    patience: int = 2000
    min_delta: float = 0.0
    verbose: bool = True


# ----------------------------- Data Loading -------------------------------
def load_velocity_table(path: str) -> pd.DataFrame:
    """Load A13 velocity text file, drop header row, return DataFrame."""
    vel = pd.read_csv(path, sep=r"\s+", engine="python", header=None)
    vel = vel.iloc[1:]  # drop first non-data row
    return vel


def load_distances(path: str, n_locations_hint: int | None = None) -> np.ndarray:
    """Load distance information from JSON file."""
    with open(path, 'r') as f:
        distance = json.load(f)
    x = np.array(distance['distances']).reshape(-1, 1)
    if n_locations_hint is not None:
        x = x[:n_locations_hint]
    return x


# ----------------------------- Grid Building -------------------------------
def build_space_time_grid(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build space-time grid for training and prediction."""
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(np.float32)
    return X, T, X_star


def build_index_grid(Exact: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build index grid for data sampling."""
    x_idx = np.arange(Exact.shape[0])
    idx_flatten, t_idx = np.meshgrid(x_idx, t)
    return np.hstack((idx_flatten.flatten()[:, None], t_idx.flatten()[:, None]))


# ----------------------------- Data Preprocessing -------------------------------
def replace_missing_with_mean(u_star: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Replace missing/invalid values (<=0) with mean of valid values."""
    valid_mask = u_star > 0
    n_missing = int(np.sum(~valid_mask))
    if n_missing > 0:
        u_mean = float(np.mean(u_star[valid_mask]))
        u_star = u_star.copy()
        u_star[~valid_mask] = u_mean
        return u_star, n_missing, u_mean
    return u_star, 0, float('nan')


# ----------------------------- Data Sampling -------------------------------
def select_random_points(
    X_star: np.ndarray, 
    idx_grid: np.ndarray, 
    u_star: np.ndarray, 
    N_u: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Randomly sample N_u valid training points."""
    valid_train_mask = u_star.flatten() > 0
    valid_indices = np.where(valid_train_mask)[0]
    n_valid = min(N_u, len(valid_indices))
    idx = np.random.choice(valid_indices, n_valid, replace=False)
    X_u_train = X_star[idx, :]
    idx_train = idx_grid[idx, :].astype(int)
    u_train = u_star[idx, :]
    return X_u_train, u_train, idx_train, n_valid


def select_sensor_columns(
    u_star: np.ndarray, 
    X_star: np.ndarray, 
    n_locations: int, 
    n_timesteps: int, 
    n_sensors: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[int], List[Tuple[int, int]]]:
    """
    Select training points from equally-spaced sensor columns.
    Returns all valid time points from each selected sensor location.
    """
    u_star_matrix = u_star.reshape((n_timesteps, n_locations))
    n_sensors_to_select = int(min(n_sensors, n_locations))
    selected_sensors = np.linspace(0, n_locations - 1, n_sensors_to_select, dtype=int)
    selected_sensors = np.unique(selected_sensors).tolist()

    selected_indices: List[int] = []
    selected_idx_grid: List[List[int]] = []
    sensor_point_counts: List[Tuple[int, int]] = []

    for col in selected_sensors:
        valid_rows = np.where(u_star_matrix[:, col] > 0)[0]
        sensor_point_counts.append((col, len(valid_rows)))
        for row in valid_rows:
            flat_idx = col + row * n_locations
            selected_indices.append(flat_idx)
            selected_idx_grid.append([row, col])

    idx = np.array(selected_indices)
    idx_train = np.array(selected_idx_grid)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]
    n_valid = len(idx)
    return X_u_train, u_train, idx_train, n_valid, selected_sensors, sensor_point_counts


def make_collocation(
    lb: np.ndarray, 
    ub: np.ndarray, 
    N_f: int, 
    X_u_train: np.ndarray
) -> np.ndarray:
    """
    Generate physics collocation points using Latin Hypercube Sampling.
    Combines LHS points with supervised training points.
    """
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    return X_f_train.astype(np.float32)


# ----------------------------- Model Evaluation -------------------------------
def evaluate_model(
    model, 
    X_star: np.ndarray, 
    u_star: np.ndarray, 
    X: np.ndarray, 
    T: np.ndarray, 
    Exact: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model predictions and compute error metrics.
    Returns L2 relative error, gridded predictions, and absolute errors.
    """
    u_pred, _ = model.predict(X_star)
    error_u = float(np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    return error_u, U_pred, Error


# ----------------------------- Plotting -------------------------------
def make_observation_matrix(Exact: np.ndarray, idx_train: np.ndarray) -> np.ndarray:
    """Create observation matrix with NaN for unobserved points."""
    Observation = np.full_like(Exact, np.nan)
    for i in range(idx_train.shape[0]):
        x_idx = idx_train[i, 1]
        t_idx = idx_train[i, 0]
        Observation[t_idx, x_idx] = Exact[t_idx, x_idx]
    return Observation


def plot_all(
    Exact: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    X_u_train: np.ndarray,
    idx_train: np.ndarray,
    U_pred: np.ndarray,
    error_u: float,
    U_pred2: np.ndarray,
    error_u2: float,
    n_valid: int,
    out_dir: str,
    N_u: int,
    fd_name: str
) -> None:
    """
    Generate comprehensive visualization with 4 rows:
    - Ground truth
    - Observations
    - PINN predictions
    - NN predictions
    """
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    fig = plt.figure(figsize=(12, 20))

    # Row 0: Ground Truth
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.97, bottom=0.77, left=0.15, right=0.85, wspace=1)

    ax = plt.subplot(gs0[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow_r',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax.set_ylabel('Time $t$ (15 min)', fontsize=18)
    ax.set_xlabel('Location $x$ (km)', fontsize=18)
    ax.set_title('Ground Truth: A13 Highway Speed (km/h)', fontsize=18)

    # Row 1: Observation Data
    gs_obs = gridspec.GridSpec(1, 2)
    gs_obs.update(top=0.72, bottom=0.52, left=0.15, right=0.85, wspace=1)

    ax = plt.subplot(gs_obs[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    Observation = make_observation_matrix(Exact, idx_train)
    cmap = plt.cm.rainbow_r.copy()
    cmap.set_bad(color='white')
    h = ax.imshow(Observation, interpolation='nearest', cmap=cmap,
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'k.', markersize=1.5, clip_on=False, alpha=0.5)
    ax.set_ylabel('Time $t$ (15 min)', fontsize=18)
    ax.set_xlabel('Location $x$ (km)', fontsize=18)
    title_str = f'Observation Data (N={n_valid} points)'
    ax.set_title(title_str, fontsize=18)

    # Row 2: PINN u(t,x)
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.47, bottom=0.27, left=0.15, right=0.85, wspace=1)
    ax = plt.subplot(gs1[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    h = ax.imshow(U_pred, interpolation='nearest', cmap='rainbow_r',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax.set_ylabel('Time $t$ (15 min)', fontsize=18)
    ax.set_xlabel('Location $x$ (km)', fontsize=18)
    ax.set_title(f'PIDL {fd_name} Estimation (Error: {error_u:.4f})', fontsize=18)

    # Row 3: DL u(t,x)
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=0.22, bottom=0.02, left=0.15, right=0.85, wspace=1)
    ax = plt.subplot(gs2[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    h = ax.imshow(U_pred2, interpolation='nearest', cmap='rainbow_r',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax.set_ylabel('Time $t$ (15 min)', fontsize=18)
    ax.set_xlabel('Location $x$ (km)', fontsize=18)
    ax.set_title(f'DL Estimation (Error: {error_u2:.4f})', fontsize=18)

    if out_dir is None:
        plt.show()
    else:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/a13_pidl_dl_pytorch_{N_u}.png')
        plt.show()

    print(f"\nPlots saved to {out_dir}/a13_pidl_dl_pytorch_{N_u}.png")
    print("=" * 60)


def plot_multi_models(
    Exact: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    X_u_train: np.ndarray,
    idx_train: np.ndarray,
    model_results: List[Dict[str, any]],  # List of {'name': str, 'U_pred': ndarray, 'error': float}
    n_valid: int,
    out_dir: str,
    N_u: int,
) -> None:
    """
    Generate comprehensive visualization with flexible number of model rows.
    
    Args:
        Exact: Ground truth velocity field
        x: Spatial coordinates
        t: Time coordinates
        X_u_train: Training point coordinates
        idx_train: Training point indices
        model_results: List of dicts with keys 'name' (model name), 'U_pred' (prediction), 'error' (L2 error)
        n_valid: Number of valid training points
        out_dir: Output directory for plots
        N_u: Number of training points (for filename)
    """
    print("\n" + "=" * 60)
    print("Generating multi-model comparison plots...")
    print("=" * 60)
    
    n_models = len(model_results)
    # Total rows: ground truth + observation + N models
    n_rows = 2 + n_models
    
    # Calculate figure height based on number of rows (each row ~5 inches)
    fig_height = n_rows * 5
    fig = plt.figure(figsize=(12, fig_height))
    
    # Row spacing calculation
    row_height = 1.0 / n_rows
    spacing = 0.02
    
    # Row 0: Ground Truth
    gs0 = gridspec.GridSpec(1, 2)
    top_pos = 1.0 - spacing
    bottom_pos = top_pos - (row_height - 2*spacing)
    gs0.update(top=top_pos, bottom=bottom_pos, left=0.15, right=0.85, wspace=1)
    
    ax = plt.subplot(gs0[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow_r',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax.set_ylabel('Time $t$ (15 min)', fontsize=18)
    ax.set_xlabel('Location $x$ (km)', fontsize=18)
    ax.set_title('Ground Truth: A13 Highway Speed (km/h)', fontsize=18)
    
    # Row 1: Observation Data
    gs_obs = gridspec.GridSpec(1, 2)
    top_pos = bottom_pos - spacing
    bottom_pos = top_pos - (row_height - 2*spacing)
    gs_obs.update(top=top_pos, bottom=bottom_pos, left=0.15, right=0.85, wspace=1)
    
    ax = plt.subplot(gs_obs[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    Observation = make_observation_matrix(Exact, idx_train)
    cmap = plt.cm.rainbow_r.copy()
    cmap.set_bad(color='white')
    h = ax.imshow(Observation, interpolation='nearest', cmap=cmap,
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'k.', markersize=1.5, clip_on=False, alpha=0.5)
    ax.set_ylabel('Time $t$ (15 min)', fontsize=18)
    ax.set_xlabel('Location $x$ (km)', fontsize=18)
    title_str = f'Observation Data (N={n_valid} points)'
    ax.set_title(title_str, fontsize=18)
    
    # Rows 2+: Model predictions
    for i, model_result in enumerate(model_results):
        gs_model = gridspec.GridSpec(1, 2)
        top_pos = bottom_pos - spacing
        bottom_pos = top_pos - (row_height - 2*spacing)
        gs_model.update(top=top_pos, bottom=bottom_pos, left=0.15, right=0.85, wspace=1)
        
        ax = plt.subplot(gs_model[:, :])
        ax.tick_params(axis='both', which='major', labelsize=16)
        h = ax.imshow(model_result['U_pred'], interpolation='nearest', cmap='rainbow_r',
                      extent=[x.min(), x.max(), t.min(), t.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.tick_params(labelsize=16)
        fig.colorbar(h, cax=cax)
        ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
        ax.set_ylabel('Time $t$ (15 min)', fontsize=18)
        ax.set_xlabel('Location $x$ (km)', fontsize=18)
        ax.set_title(f"{model_result['name']} (Error: {model_result['error']:.4f})", fontsize=18)
    
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/a13_multi_model_{N_u}.png', bbox_inches='tight', dpi=150)
        plt.show()
    else:
        plt.show()
    
    print(f"\nPlots saved to {out_dir}/a13_multi_model_{N_u}.png")
    print("=" * 60)

