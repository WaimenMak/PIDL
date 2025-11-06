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
from typing import List

import numpy as np
import torch
import torch.nn as nn
try:
    import yaml
except ImportError:  # lightweight guidance if PyYAML isn't installed
    yaml = None

# Import utilities from utils module
from utils.utils import (
    set_seed,
    timestamp_dir,
    EarlyStopConfig,
    load_velocity_table,
    load_distances,
    build_space_time_grid,
    build_index_grid,
    replace_missing_with_mean,
    select_random_points,
    select_sensor_columns,
    make_collocation,
    evaluate_model,
    plot_all,
    plot_multi_models,
)

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
        # >>> NEW <<<
        fd_name: str = "linear",   # 'linear'| 'log' | 'exp' | 'power' | 'triangular' | 'nn'
    ):
        super().__init__()
        print(f"Initializing PINN with fd_name='{fd_name}', f_weight={f_weight}")
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.f_weight = float(f_weight)
        self.V_f = float(V_f)
        self.t_scale = float(t_scale)
        self.fd_name = fd_name.lower()

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
    
    # NEW -----
    def _denorm_u(self, u_norm):
        # Convert normalized network output to physical units
        return u_norm * self.u_std + self.u_mean

    def _grad(self, y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                retain_graph=True, create_graph=True)[0]
    
    # @torch.enable_grad()
    # def net_f(self, x, t):
    #     """
    #     Physics residual based on:
    #     f = (u_x - 2/V_f * u * u_x - 1/V_f * u_t) * t_scale
    #     """
    #     u = self.net_u(x, t)
    #     u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    #     u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    #     f = (u_x - (2.0 / self.V_f) * u * u_x - (1.0 / self.V_f) * u_t) * self.t_scale
    #     # If you prefer the textbook residual, you can use:
    #     # r = u_t + (self.V_f - 2.0*u) * u_x
    #     return f
    
    @torch.enable_grad()
    def net_f(self, x, t):
        """
        Physics residual based on the chosen fundamental diagram.
        For pure NN (fd_name='nn'), returns zeros since no physics is enforced.
        """
        # If this is a pure NN model, return zero residual
        if self.fd_name.lower() == 'nn' or self.f_weight == 0.0:
            return torch.zeros_like(x)
        
        # network output (normalized) -> physical speed
        u_norm = self.net_u(x, t)
        u = self._denorm_u(u_norm)
                
        # physical derivatives (autograd applies the scale)
        u_t = self._grad(u, t)
        u_x = self._grad(u, x)
        
        fd = self.fd_name
        v_default  = self.V_f
        
        if fd == "linear":
            # Greenshields: r = u_t + (v_f - 2u) u_x
            v_f = v_default
            # Non-dimensional form: (1/v_f) u_t + (1 - 2u/v_f) u_x = 0
            r_nd = (u_t / v_f) + (1.0 - 2.0 * (u / v_f)) * u_x

        elif fd == "log":
            # Greenberg: r = u_t - (v_m - u) u_x
            v_m = v_default
            # Non-dimensional: (1/v_m) u_t + (u/v_m - 1) u_x = 0
            r_nd = (u_t / v_m) + ((u / v_m) - 1.0) * u_x

        elif fd == "exp":
            # Underwood: r = u_t + u (ln(v_f/u) - 1) u_x
            v_f = v_default
            eps = 1e-6
            u_safe = torch.clamp(u, min=eps)
            # Non-dimensional: (1/v_f) u_t + (u/v_f) (ln(v_f/u) - 1) u_x = 0
            r_nd = (u_t / v_f) + ( (u_safe / v_f) * (torch.log(torch.tensor(v_f, device=u.device, dtype=u.dtype)) - torch.log(u_safe) - 1.0) ) * u_x

        elif fd == "power":
            # Pipes–Munjal: r = u_t + ( (n+1)u - n v_f ) u_x
            v_f = v_default
            n   = 3.0 # guess?
            # Non-dimensional: (1/v_f) u_t + ( (n+1) u/v_f - n ) u_x = 0
            r_nd = (u_t / v_f) + ( ((n + 1.0) * (u / v_f)) - n ) * u_x

        elif fd == "triangular":
            # Speed-gated blend:
            #   r = s * (u_t - w u_x) + (1-s) * (u - v_f)
            # Normalize branches:
            #   Congestion: (1/w) u_t - u_x
            #   Free-flow : (u - v_f)/v_f
            v_f  = v_default
            w    = 15.0 # km/h (positive magnitude)
            alpha= 20.0 # gate sharpness

            s = torch.sigmoid(torch.tensor(alpha, device=u.device, dtype=u.dtype) * (v_f - u))

            r_cong_nd = (u_t / w) - u_x        # non-dimensional congestion PDE
            r_free_nd = (u - v_f) / v_f        # non-dimensional free-flow constraint

            r_nd = s * r_cong_nd + (1.0 - s) * r_free_nd

            # Optional: tiny extra penalty if you want to discourage u > v_f
            # r_nd = r_nd + 0.05 * torch.clamp((u - v_f) / v_f, min=0.0)

        else:
            raise ValueError(f"Unknown fd_name='{self.fd_name}'")

        return r_nd * self.t_scale
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
                    "fd_name": self.fd_name,
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
            fd_name=state.get("fd_name", "linear")
        )
        model.load_state_dict(state["model_state_dict"])
        # Ensure normalization parameters match checkpoint (important if normalize_labels=True)
        model.u_mean = torch.tensor(state["u_mean"], dtype=torch.float32, device=model.device)
        model.u_std  = torch.tensor(state["u_std"], dtype=torch.float32, device=model.device)
        return model


# ----------------------------- Model Builder Helper -------------------------------
def build_model(
    X_u_train: np.ndarray, 
    u_train: np.ndarray, 
    X_f_train: np.ndarray, 
    layers: List[int], 
    lb: np.ndarray, 
    ub: np.ndarray, 
    f_weight: float,
    fd_name: str
) -> UnifiedPINN:
    """Convenience function to build a UnifiedPINN model with standard settings."""
    return UnifiedPINN(
        X_u=X_u_train, u=u_train, X_f=X_f_train, layers=layers,
        lb=lb, ub=ub, normalize_labels=True,
        f_weight=f_weight, V_f=110.0, t_scale=0.25, fd_name=fd_name
    )


# ------------------------------ Main flow --------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="UnifiedPINN for A13 velocity reconstruction (config-driven)")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to YAML configuration file')
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
    fd_name: str = cfg.get('fd_name', 'linear')
    run_base: bool = bool(cfg.get('run_base', True))

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
    model_pinn = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=f_weight, fd_name=fd_name)
    start_time = time.time()
    out_pinn = model_pinn.fit(
        epochs=epochs,
        lr=lr,
        early_stop=EarlyStopConfig(patience=patience, min_delta=0.0, verbose=True),
        save_dir=pinn_dir,
        tag=f"pinn_{fd_name}",
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

    # Collect model predictions for plotting
    model_predictions = []
    model_predictions.append({
        'name': f'PINN ({fd_name})',
        'U_pred': U_pred,
        'error': error_u,
    })

    # Train pure NN (baseline) - optional based on run_base config
    if run_base:
        nn_dir = f"./runs/a13_exp1/" + timestamp_dir("nn")
        print("\n" + "=" * 60)
        print("Training Regular NN Model...")
        print("=" * 60)
        model_nn = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=0.0, fd_name='nn')
        start_time = time.time()
        out_nn = model_nn.fit(
            epochs=epochs,
            lr=lr,
            early_stop=EarlyStopConfig(patience=patience, min_delta=0.0, verbose=True),
            save_dir=nn_dir,
            tag="nn",
            log_every=log_every,
            use_mixed_precision=True,
        )
        elapsed = time.time() - start_time
        print(f"Training finished after {out_nn['best_epoch']} epochs with best train loss {out_nn['best_train']:.4e}")
        print(f'Training time: {elapsed:.4f} seconds')
        error_u2, U_pred2, _ = evaluate_model(model_nn, X_star, u_star, X, T, Exact)
        print(f'DL Error u: {error_u2:.4e}')

        # Add baseline NN to predictions
        model_predictions.append({
            'name': 'NN (baseline)',
            'U_pred': U_pred2,
            'error': error_u2,
        })
    else:
        print("\n" + "=" * 60)
        print("Skipping baseline NN training (run_base=False)")
        print("=" * 60)

    # Plot using multi-model plotting function
    if len(model_predictions) == 2:
        # Use classic plot_all for backward compatibility when we have PINN + NN
        plot_all(
            Exact=Exact, x=x, t=t, X_u_train=X_u_train, idx_train=idx_train,
            U_pred=model_predictions[0]['U_pred'], error_u=model_predictions[0]['error'],
            U_pred2=model_predictions[1]['U_pred'], error_u2=model_predictions[1]['error'],
            n_valid=n_valid, out_dir=out_fig_dir,
            N_u=N_u,
        )
    else:
        # Use plot_multi_models for single model or multiple models
        plot_multi_models(
            Exact=Exact, x=x, t=t, X_u_train=X_u_train, idx_train=idx_train,
            model_results=model_predictions,
            n_valid=n_valid, out_dir=out_fig_dir,
            N_u=N_u,
        )


if __name__ == "__main__":
    main()


