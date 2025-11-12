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


# def make_collocation(
#     lb: np.ndarray, 
#     ub: np.ndarray, 
#     N_f: int, 
#     X_u_train: np.ndarray
# ) -> np.ndarray:
#     """
#     Generate physics collocation points using Latin Hypercube Sampling.
#     Combines LHS points with supervised training points.
#     """
#     X_f_train = lb + (ub - lb) * lhs(2, N_f)
#     X_f_train = np.vstack((X_f_train, X_u_train))
#     return X_f_train.astype(np.float32)

from scipy.stats import qmc
def make_collocation(
    lb: np.ndarray, 
    ub: np.ndarray, 
    N_f: int, 
    X_u_train: np.ndarray
) -> np.ndarray:
    # Sobol (skip=1024 for better balance), scramble for variance reduction
    print(f"\n  - Using Sobol sampling for collocation points")
    d = 2
    sampler = qmc.Sobol(d=d, scramble=True)
    sampler.fast_forward(1024)
    X_f = sampler.random(N_f)
    X_f = X_f[:N_f]
    X_f = lb + (ub - lb) * X_f
    X_f = np.vstack((X_f, X_u_train))
    return X_f.astype(np.float32)


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
    h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow_r',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', markersize=0.8, clip_on=False)
    ax.set_xlabel('Time $t$ (15 min)', fontsize=18)
    ax.set_ylabel('Location $x$ (km)', fontsize=18)
    ax.set_title('Ground Truth: A13 Highway Speed (km/h)', fontsize=18)

    # Row 1: Observation Data
    gs_obs = gridspec.GridSpec(1, 2)
    gs_obs.update(top=0.72, bottom=0.52, left=0.15, right=0.85, wspace=1)

    ax = plt.subplot(gs_obs[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    Observation = make_observation_matrix(Exact, idx_train)
    cmap = plt.cm.rainbow_r.copy()
    cmap.set_bad(color='white')
    h = ax.imshow(Observation.T, interpolation='nearest', cmap=cmap,
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'k.', markersize=1.5, clip_on=False, alpha=0.5)
    ax.set_xlabel('Time $t$ (15 min)', fontsize=18)
    ax.set_ylabel('Location $x$ (km)', fontsize=18)
    title_str = f'Observation Data (N={n_valid} points)'
    ax.set_title(title_str, fontsize=18)

    # Row 2: PINN u(t,x)
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.47, bottom=0.27, left=0.15, right=0.85, wspace=1)
    ax = plt.subplot(gs1[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow_r',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', markersize=0.8, clip_on=False)
    ax.set_xlabel('Time $t$ (15 min)', fontsize=18)
    ax.set_ylabel('Location $x$ (km)', fontsize=18)
    ax.set_title(f'PIDL {fd_name} Estimation (Error: {error_u:.4f})', fontsize=18)

    # Row 3: DL u(t,x)
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=0.22, bottom=0.02, left=0.15, right=0.85, wspace=1)
    ax = plt.subplot(gs2[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    h = ax.imshow(U_pred2.T, interpolation='nearest', cmap='rainbow_r',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', markersize=0.8, clip_on=False)
    ax.set_xlabel('Time $t$ (15 min)', fontsize=18)
    ax.set_ylabel('Location $x$ (km)', fontsize=18)
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
    Generate comparison plots with clean spacing using constrained_layout.

    Rows: Ground Truth, Observations, and one row per model in model_results.
    """
    print("\n" + "=" * 60)
    print("Generating multi-model comparison plots...")
    print("=" * 60)

    n_models = len(model_results)
    n_rows = 2 + n_models  # GT + Obs + models

    # Use constrained_layout to avoid overlapping titles/labels/colorbars
    fig_height = max(4.5 * n_rows, 10)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_height), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    def add_panel(ax, data, title, cmap_name='rainbow_r', show_points=True, obs=False):
        if obs:
            cmap = plt.cm.get_cmap(cmap_name).copy()
            cmap.set_bad(color='white')
        else:
            cmap = plt.cm.get_cmap(cmap_name)
        # Transpose data to swap axes: time on x-axis, location on y-axis
        im = ax.imshow(
            data.T, interpolation='nearest', cmap=cmap,
            extent=[t.min(), t.max(), x.min(), x.max()],
            origin='lower', aspect='auto'
        )
        ax.tick_params(axis='both', which='major', labelsize=14)
        if show_points:
            mk = 1.5 if obs else 0.8
            # Swap x and y coordinates: plot time on x-axis, location on y-axis
            ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'k.' if obs else 'kx', markersize=mk, alpha=0.6, clip_on=False)
        ax.set_xlabel('Time $t$ (15 min)', fontsize=15, labelpad=6)
        ax.set_ylabel('Location $x$ (km)', fontsize=15)
        ax.set_title(title, fontsize=16, pad=10)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=12)

    # Row 0: Ground truth
    add_panel(axes[0], Exact, 'Ground Truth: A13 Highway Speed (km/h)')

    # Row 1: Observations
    Observation = make_observation_matrix(Exact, idx_train)
    add_panel(
        axes[1], Observation, f'Observation Data (N={n_valid} points)',
        cmap_name='rainbow_r', show_points=True, obs=True
    )

    # Rows 2+: Models
    for i, m in enumerate(model_results):
        add_panel(
            axes[2 + i], m['U_pred'], f"{m['name']} (Error: {m['error']:.4f})",
            cmap_name='rainbow_r', show_points=True, obs=False
        )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/a13_multi_model_{N_u}.png', dpi=150)
        plt.show()
    else:
        plt.show()

    print(f"\nPlots saved to {out_dir}/a13_multi_model_{N_u}.png")
    print("=" * 60)


def plot_training_history(
    history: Dict[str, List[float]],
    out_dir: str,
    tag: str = "training",
    show_physics: bool = True,
    log_scale: bool = False,
) -> None:
    """
    Plot training loss history (total, data, and physics losses).
    
    Args:
        history: Dictionary with keys 'epoch', 'train_total', 'data_loss', 'phys_loss'
        out_dir: Directory to save the plot
        tag: Filename tag for the saved plot
        show_physics: Whether to show physics loss (False for pure NN training)
        log_scale: Whether to use log scale for y-axis
    """
    print("\n" + "=" * 60)
    print("Generating training history plots...")
    print("=" * 60)
    
    epochs = history['epoch']
    train_total = history['train_total']
    data_loss = history['data_loss']
    phys_loss = history['phys_loss']
    
    # Create figure with subplots
    if show_physics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
    
    # Plot 1: Total Loss
    ax = axes[0]
    ax.plot(epochs, train_total, 'b-', linewidth=1.5, label='Total Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    if log_scale:
        ax.set_yscale('log')
    
    # Plot 2: Data Loss
    ax = axes[1]
    ax.plot(epochs, data_loss, 'g-', linewidth=1.5, label='Data Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Data Loss', fontsize=12)
    ax.set_title('Data Loss (MSE)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    if log_scale:
        ax.set_yscale('log')
    
    if show_physics:
        # Plot 3: Physics Loss
        ax = axes[2]
        ax.plot(epochs, phys_loss, 'r-', linewidth=1.5, label='Physics Loss')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Physics Loss', fontsize=12)
        ax.set_title('Physics Residual Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        if log_scale:
            ax.set_yscale('log')
        
        # Plot 4: Combined view (all losses)
        ax = axes[3]
        ax.plot(epochs, train_total, 'b-', linewidth=1.5, label='Total Loss', alpha=0.8)
        ax.plot(epochs, data_loss, 'g-', linewidth=1.5, label='Data Loss', alpha=0.8)
        ax.plot(epochs, phys_loss, 'r-', linewidth=1.5, label='Physics Loss', alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('All Losses Combined', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        if log_scale:
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        filename = f'{out_dir}/loss_history_{tag}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Loss history plot saved to: {filename}")
        plt.close()
    else:
        plt.show()
    
    print("=" * 60)

