"""
Cleaned and modularized training script for A13 highway speed reconstruction
using UnifiedPINN (PINN and pure NN modes). Provides a clear `main()` with
functional decomposition for data loading, sampling, training, evaluation,
and plotting.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from pyDOE import lhs
try:
    import yaml
except ImportError:  # lightweight guidance if PyYAML isn't installed
    yaml = None


# ----------------------------- Utilities ---------------------------------
def set_seed(seed: int = 25) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def timestamp_dir(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


@dataclass
class EarlyStopConfig:
    patience: int = 2000
    min_delta: float = 0.0
    verbose: bool = True

class UnifiedPINN(nn.Module):
    """
    Set f_weight=0.0 to train a standard NN (no physics loss).
    Set f_weight>0.0 to train a Physics-Informed NN (PINN).
    """
    def __init__(
        self,
        X_u,              # ndarray [N_u, 2] -> columns: x, t for supervised points
        u,                # ndarray [N_u, 1] -> labels
        X_f,              # ndarray [N_f, 2] -> physics/collocation points (x, t)
        layers,           # list of layer sizes, e.g. [2, 64, 64, 1]
        lb, ub,           # 2-dim lower/upper bounds (for input normalization)
        normalize_labels: bool = True,
        f_weight: float = 1.0,      # 0.0 => pure NN, >0.0 => PINN
        V_f: float = 110.0,         # free flow speed (km/h)
        t_scale: float = 0.25,      # time scaling in PDE
        device: str = None,
    ):
        super().__init__()

        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.f_weight = float(f_weight)
        self.V_f = float(V_f)
        self.t_scale = float(t_scale)

        # bounds
        self.lb = torch.tensor(lb, dtype=torch.float32, device=self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=self.device)

        # supervised inputs
        X_u = X_u.astype(np.float32)
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=self.device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=self.device)

        # labels (with optional normalization)
        self.normalize_labels = normalize_labels
        u = u.astype(np.float32)
        if self.normalize_labels:
            u_torch = torch.tensor(u, dtype=torch.float32)
            u_mean = torch.mean(u_torch)
            u_std = torch.std(u_torch)
            if u_std < 1e-8:
                u_std = torch.tensor(1.0)
            self.u_mean = u_mean.to(self.device)
            self.u_std = u_std.to(self.device)
            u_norm = (u - self.u_mean.cpu().numpy()) / self.u_std.cpu().numpy()
            self.u = torch.tensor(u_norm, dtype=torch.float32, device=self.device)
            if f_weight>0.0:
                model_type = "PINN"
            else:
                model_type = "NN"
            print(f"[{model_type}] Label normalization: mean={self.u_mean.item():.4f}, std={self.u_std.item():.4f}")
        else:
            self.u = torch.tensor(u, dtype=torch.float32, device=self.device)
            self.u_mean = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            self.u_std = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        # physics points
        X_f = X_f.astype(np.float32)
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=self.device, requires_grad=True)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=self.device, requires_grad=True)

        # model
        self.layers = layers
        self.model = self._initialize_NN(layers).to(self.device)

    # ----- architecture -----
    def _initialize_NN(self, layers):
        mods = []
        for i in range(len(layers)-1):
            mods.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                mods.append(nn.Tanh())
        model = nn.Sequential(*mods)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        return model

    # ----- forward helpers -----
    def _normalize_inputs(self, X):
        # map inputs to [-1, 1]
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def neural_net(self, X):
        return self.model(self._normalize_inputs(X))

    def net_u(self, x, t):
        return self.neural_net(torch.cat([x, t], dim=1))

    @torch.enable_grad()
    def net_f(self, x, t):
        """
        Physics residual based on:
        f = (u_x - 2/V_f * u * u_x - 1/V_f * u_t) * t_scale
        """
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        f = (u_x - (2.0 / self.V_f) * u * u_x - (1.0 / self.V_f) * u_t) * self.t_scale
        # If you prefer the textbook residual, you can use:
        # r = u_t + (self.V_f - 2.0*u) * u_x
        return f

    # ----- training (no validation) -----
    def fit(
        self,
        epochs: int = 5000,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stop: EarlyStopConfig = EarlyStopConfig(),
        save_dir: str = "./checkpoints",
        tag: str = "best",
        log_every: int = 100,
        seed: int = 42,
        # fast-path knobs
        f_subset_per_epoch: int | None = 4000,   # None => use all physics points
        physics_every: int = 1,                  # compute physics every k epochs
        grad_clip_norm: float | None = None,
        use_mixed_precision: bool = True,        # autocast if available
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"model_{tag}.pt")
        meta_path = os.path.join(save_dir, f"model_{tag}_meta.json")

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_train, best_epoch, no_improve = float("inf"), -1, 0
        history = {"epoch": [], "train_total": [], "data_loss": [], "phys_loss": []}

        autocast_ctx = (
            torch.cuda.amp.autocast if (use_mixed_precision and self.device.type == "cuda") else
            torch.cpu.amp.autocast  if (use_mixed_precision and self.device.type == "cpu") else
            None
        )

        # use *all* supervised anchors for training
        x_u_tr, t_u_tr, u_tr = self.x_u, self.t_u, self.u

        for ep in range(1, epochs + 1):
            self.train()
            opt.zero_grad(set_to_none=True)

            # --- physics subset selection for this epoch ---
            use_physics = (self.f_weight > 0.0) and (ep % max(1, physics_every) == 0)
            if use_physics:
                if (f_subset_per_epoch is not None) and (f_subset_per_epoch < self.x_f.shape[0]):
                    idx_f = torch.randint(0, self.x_f.shape[0], (f_subset_per_epoch,), device=self.device)
                    x_f_ep = self.x_f[idx_f].requires_grad_(True)
                    t_f_ep = self.t_f[idx_f].requires_grad_(True)
                else:
                    x_f_ep = self.x_f.requires_grad_(True)
                    t_f_ep = self.t_f.requires_grad_(True)

            # --- forward + losses (full-batch supervised) ---
            if autocast_ctx is not None:
                with autocast_ctx():
                    u_pred = self.net_u(x_u_tr, t_u_tr)
                    data_loss = torch.mean((u_tr - u_pred) ** 2)
                    if use_physics:
                        f_pred = self.net_f(x_f_ep, t_f_ep)
                        phys_loss = torch.mean(f_pred ** 2)
                    else:
                        phys_loss = torch.tensor(0.0, device=self.device)
                    total_loss = data_loss + self.f_weight * phys_loss
            else:
                u_pred = self.net_u(x_u_tr, t_u_tr)
                data_loss = torch.mean((u_tr - u_pred) ** 2)
                if use_physics:
                    f_pred = self.net_f(x_f_ep, t_f_ep)
                    phys_loss = torch.mean(f_pred ** 2)
                else:
                    phys_loss = torch.tensor(0.0, device=self.device)
                total_loss = data_loss + self.f_weight * phys_loss

            total_loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
            opt.step()

            # --- logging/history ---
            train_total = float(total_loss.item())
            if (ep % log_every == 0) or (ep == 1):
                print(f"[Epoch {ep:05d}] train_total={train_total:.4e} | data={data_loss.item():.4e} | phys={phys_loss.item():.4e}")

            history["epoch"].append(ep)
            history["train_total"].append(train_total)
            history["data_loss"].append(float(data_loss.item()))
            history["phys_loss"].append(float(phys_loss.item()))

            # --- early stopping on TRAIN total loss ---
            if train_total < best_train - early_stop.min_delta:
                best_train, best_epoch, no_improve = train_total, ep, 0
                torch.save({
                    "model_state_dict": self.state_dict(),
                    "layers": self.layers,
                    "lb": self.lb.detach().cpu().numpy().tolist(),
                    "ub": self.ub.detach().cpu().numpy().tolist(),
                    "normalize_labels": self.normalize_labels,
                    "u_mean": float(self.u_mean.item()),
                    "u_std": float(self.u_std.item()),
                    "f_weight": self.f_weight,
                    "V_f": self.V_f,
                    "t_scale": self.t_scale,
                }, ckpt_path)
                # save meta every N epochs
                if (ep % log_every == 0) or (ep == 1):
                    with open(meta_path, "w") as f:
                        json.dump({
                            "best_train": best_train,
                            "best_epoch": best_epoch,
                            "epochs_run": ep,
                            "history": history,
                        }, f, indent=2)
            else:
                no_improve += 1
                if early_stop.verbose and (no_improve % early_stop.patience == 0):
                    print(f"  ...no train improvement for {no_improve} epochs (best={best_train:.4e} @ {best_epoch})")
                if no_improve >= early_stop.patience:
                    if early_stop.verbose:
                        print(f"Early stopping at epoch {ep} (best train {best_train:.4e} @ {best_epoch})")
                    break

        # restore best checkpoint
        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(state["model_state_dict"])
        else:
            torch.save({"model_state_dict": self.state_dict()}, ckpt_path)

        return {
            "best_train": best_train,
            "best_epoch": best_epoch,
            "history": history,
            "checkpoint_path": ckpt_path,
            "meta_path": meta_path,
        }

    @torch.no_grad()
    def predict(self, X_star: np.ndarray):
        self.eval()
        X_star = X_star.astype(np.float32)
        x_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=self.device)
        t_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=self.device)
        u_star = self.net_u(x_star, t_star)
        if self.normalize_labels:
            u_star = torch.clip(u_star * self.u_std + self.u_mean, min=torch.tensor([0.0], device=self.device))
        # physics residual for inspection
        x_f_star = x_star.clone().detach().requires_grad_(True)
        t_f_star = t_star.clone().detach().requires_grad_(True)
        f_star = self.net_f(x_f_star, t_f_star)
        return u_star.detach().cpu().numpy(), f_star.detach().cpu().numpy()

    # ----- utility -----
    @staticmethod
    def load_from_checkpoint(ckpt_path: str, X_u, u, X_f, device: str = None):
        """
        Convenience loader: rebuilds model and loads weights, while allowing new datasets.
        """
        state = torch.load(ckpt_path, map_location=device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        model = UnifiedPINN(
            X_u=X_u, u=u, X_f=X_f,
            layers=state["layers"],
            lb=np.array(state["lb"], dtype=np.float32),
            ub=np.array(state["ub"], dtype=np.float32),
            normalize_labels=state["normalize_labels"],
            f_weight=state.get("f_weight", 0.0),
            V_f=state.get("V_f", 110.0),
            t_scale=state.get("t_scale", 0.25),
            device=device,
        )
        model.load_state_dict(state["model_state_dict"])
        # Ensure normalization parameters match checkpoint (important if normalize_labels=True)
        model.u_mean = torch.tensor(state["u_mean"], dtype=torch.float32, device=model.device)
        model.u_std  = torch.tensor(state["u_std"], dtype=torch.float32, device=model.device)
        return model
        

# ----------------------------- Data helpers -------------------------------
def load_velocity_table(path: str) -> pd.DataFrame:
    """Load A13 velocity text file, drop header row, return DataFrame."""
    # Use regex-based separator to avoid FutureWarning from delim_whitespace
    vel = pd.read_csv(path, sep=r"\s+", engine="python", header=None)
    vel = vel.iloc[1:]  # drop first non-data row
    return vel


def load_distances(path: str, n_locations_hint: int | None = None) -> np.ndarray:
    with open(path, 'r') as f:
        distance = json.load(f)
    x = np.array(distance['distances']).reshape(-1, 1)
    if n_locations_hint is not None:
        x = x[:n_locations_hint]
    return x


def build_space_time_grid(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(np.float32)
    return X, T, X_star


def build_index_grid(Exact: np.ndarray, t: np.ndarray) -> np.ndarray:
    x_idx = np.arange(Exact.shape[0])
    idx_flatten, t_idx = np.meshgrid(x_idx, t)
    return np.hstack((idx_flatten.flatten()[:, None], t_idx.flatten()[:, None]))


def replace_missing_with_mean(u_star: np.ndarray) -> Tuple[np.ndarray, int, float]:
    valid_mask = u_star > 0
    n_missing = int(np.sum(~valid_mask))
    if n_missing > 0:
        u_mean = float(np.mean(u_star[valid_mask]))
        u_star = u_star.copy()
        u_star[~valid_mask] = u_mean
        return u_star, n_missing, u_mean
    return u_star, 0, float('nan')


def select_random_points(X_star: np.ndarray, idx_grid: np.ndarray, u_star: np.ndarray, N_u: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    valid_train_mask = u_star.flatten() > 0
    valid_indices = np.where(valid_train_mask)[0]
    n_valid = min(N_u, len(valid_indices))
    idx = np.random.choice(valid_indices, n_valid, replace=False)
    X_u_train = X_star[idx, :]
    idx_train = idx_grid[idx, :].astype(int)
    u_train = u_star[idx, :]
    return X_u_train, u_train, idx_train, n_valid


def select_sensor_columns(u_star: np.ndarray, X_star: np.ndarray, n_locations: int, n_timesteps: int, n_sensors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[int], List[Tuple[int, int]]]:
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


def make_collocation(lb: np.ndarray, ub: np.ndarray, N_f: int, X_u_train: np.ndarray) -> np.ndarray:
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    return X_f_train.astype(np.float32)


def build_model(X_u_train: np.ndarray, u_train: np.ndarray, X_f_train: np.ndarray, layers: List[int], lb: np.ndarray, ub: np.ndarray, f_weight: float) -> UnifiedPINN:
    return UnifiedPINN(
        X_u=X_u_train, u=u_train, X_f=X_f_train, layers=layers,
        lb=lb, ub=ub, normalize_labels=True,
        f_weight=f_weight, V_f=110.0, t_scale=0.25,
    )


def evaluate_model(model: UnifiedPINN, X_star: np.ndarray, u_star: np.ndarray, X: np.ndarray, T: np.ndarray, Exact: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    u_pred, _ = model.predict(X_star)
    error_u = float(np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    return error_u, U_pred, Error


def make_observation_matrix(Exact: np.ndarray, idx_train: np.ndarray) -> np.ndarray:
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
) -> None:
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
    ax.set_title(f'PIDL Estimation (Error: {error_u:.4f})', fontsize=18)

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
        # plt.savefig(f'{out_dir}/a13_pidl_dl_pytorch_{N_u}.pdf')
        plt.savefig(f'{out_dir}/a13_pidl_dl_pytorch_{N_u}.png')
        # plt.savefig(f'{out_dir}/a13_pidl_dl_pytorch_{N_u}.eps')
        plt.show()

    print(f"\nPlots saved to {out_dir}/a13_pidl_dl_pytorch_{N_u}.pdf/eps")
    print("=" * 60)


# ------------------------------ Main flow --------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="UnifiedPINN for A13 velocity reconstruction (config-driven)")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    if yaml is None:
        raise ImportError("PyYAML is required to load the config file. Install via `pip install pyyaml`." )
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    # Extract config values with defaults mirroring previous CLI defaults
    data_file: str = cfg.get('data_file', 'data/A13_Velocity_Data_0909-0910.txt')
    distance_json: str = cfg.get('distance_json', 'td_data/2024-09-09.json')
    layers: List[int] = cfg.get('layers', [2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
    N_u: int = int(cfg.get('N_u', 800))
    N_f: int = int(cfg.get('N_f', 10000))
    sensor_based: bool = bool(cfg.get('sensor_based', False))
    n_sensors: int = int(cfg.get('n_sensors', 5))
    epochs: int = int(cfg.get('epochs', 10000))
    lr: float = float(cfg.get('lr', 1e-4))
    log_every: int = int(cfg.get('log_every', 200))
    physics_every: int = int(cfg.get('physics_every', 1))
    patience: int = int(cfg.get('patience', 2000))
    seed: int = int(cfg.get('seed', 25))
    out_fig_dir: str = cfg.get('out_fig_dir', 'figures_revised2')
    fast: bool = bool(cfg.get('fast', False))
    f_weight: float = float(cfg.get('f_weight', 1.0))

    # Reproducibility
    set_seed(seed)

    # Potential fast mode tweaks (can be toggled from config)
    if fast:
        epochs = min(epochs, 5)
        N_f = min(N_f, 2000)
        N_u = min(N_u, 200)
        print("[FAST] Using reduced epochs and points for a quick smoke test")

    # Load data
    vel = load_velocity_table(data_file)
    print(f"A13 Data Shape: {vel.shape}")
    print(f"Spatial locations: {vel.shape[0]}")
    print(f"Time steps: {vel.shape[1]}")

    x = load_distances(distance_json, n_locations_hint=vel.shape[0])
    t = np.arange(vel.shape[1]).reshape(-1, 1)

    Exact = np.real(vel.T)
    X, T, X_star = build_space_time_grid(x, t)
    idx_grid = build_index_grid(Exact, t)
    u_star = Exact.flatten()[:, None]

    # Fill missing
    u_star, n_missing, u_mean = replace_missing_with_mean(u_star)
    if n_missing > 0:
        print(f"Replaced {n_missing} missing values with mean: {u_mean:.2f}")

    # Bounds
    lb = X_star.min(0).astype(np.float32)
    ub = X_star.max(0).astype(np.float32)

    # Select training points
    n_locations = x.shape[0]
    n_timesteps = t.shape[0]
    if sensor_based:
        print("\n[Data Selection] Using sensor-based column selection (equally distributed)")
        X_u_train, u_train, idx_train, n_valid, sensors, sensor_point_counts = select_sensor_columns(
            u_star, X_star, n_locations, n_timesteps, n_sensors
        )
        print(f"  - Total available locations: {n_locations}")
        print(f"  - Requested sensors: {n_sensors}")
        print(f"  - Selected sensors: {len(sensors)} (equally spaced)")
        print(f"  - Sensor indices: {sensors}")
        print(f"  - Total observation points: {n_valid}")
        for col, n_pts in sensor_point_counts:
            print(f"    · Sensor {col}: {n_pts} points")
        n_sensors_to_select = len(sensors)
    else:
        print("\n[Data Selection] Using random sampling from all valid points")
        X_u_train, u_train, idx_train, n_valid = select_random_points(X_star, idx_grid, u_star, N_u)
        print(f"  - Sampled points: {n_valid}")
        n_sensors_to_select = None

    # Collocation points
    X_f_train = make_collocation(lb, ub, N_f, X_u_train)
    print(f"\nTraining with {n_valid} data points (+ {N_f} collocation points)")

    # Train PINN
    pinn_dir = f"./runs/a13_exp1/" + timestamp_dir("pinn")
    print("\n" + "=" * 60)
    print("Training PINN Model...")
    print("=" * 60)
    model_pinn = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=f_weight)
    start_time = time.time()
    out_pinn = model_pinn.fit(
        epochs=epochs,
        lr=lr,
        early_stop=EarlyStopConfig(patience=patience, min_delta=0.0, verbose=True),
        save_dir=pinn_dir,
        tag="unifiedpinn_pinn_v1",
        log_every=log_every,
        f_subset_per_epoch=min(4000, X_f_train.shape[0]),
        physics_every=physics_every,
        use_mixed_precision=True,
    )
    elapsed = time.time() - start_time
    print(f"Training finished after {out_pinn['best_epoch']} epochs with best train loss {out_pinn['best_train']:.4e}")
    print(f'Training time: {elapsed:.4f} seconds')
    error_u, U_pred, _ = evaluate_model(model_pinn, X_star, u_star, X, T, Exact)
    print(f'PINN Error u: {error_u:.4e}')

    # Train pure NN
    nn_dir = f"./runs/a13_exp1/" + timestamp_dir("nn")
    print("\n" + "=" * 60)
    print("Training Regular NN Model...")
    print("=" * 60)
    model_nn = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=0.0)
    start_time = time.time()
    out_nn = model_nn.fit(
        epochs=epochs,
        lr=lr,
        early_stop=EarlyStopConfig(patience=patience, min_delta=0.0, verbose=True),
        save_dir=nn_dir,
        tag="unifiedpinn_nn_v1",
        log_every=log_every,
        use_mixed_precision=True,
    )
    elapsed = time.time() - start_time
    print(f"Training finished after {out_nn['best_epoch']} epochs with best train loss {out_nn['best_train']:.4e}")
    print(f'Training time: {elapsed:.4f} seconds')
    error_u2, U_pred2, _ = evaluate_model(model_nn, X_star, u_star, X, T, Exact)
    print(f'DL Error u: {error_u2:.4e}')

    # Plot
    plot_all(
        Exact=Exact, x=x, t=t, X_u_train=X_u_train, idx_train=idx_train,
        U_pred=U_pred, error_u=error_u, U_pred2=U_pred2, error_u2=error_u2,
        n_valid=n_valid, out_dir=out_fig_dir,
        N_u=N_u,
    )


if __name__ == "__main__":
    main()

