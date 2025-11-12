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
import pandas as pd
import torch
import torch.nn as nn
try:
    import yaml
except ImportError:  # lightweight guidance if PyYAML isn't installed
    yaml = None

# Import utilities from utils module
from utils.utils import (
    set_seed,
    infer_speed_limits,
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
    plot_training_history,
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
        V_f: float = 110.0,         # free flow speed (km/h) - used as default if speed_limits_df not provided
        t_scale: float = 0.25,      # time scaling in PDE
        device: str = None,
        # >>> NEW <<<
        fd_name: str = "linear",   # 'linear'| 'log' | 'exp' | 'power' | 'triangular' | 'nn'
        speed_limits_df: pd.DataFrame = None,  # DataFrame with columns ['x', 'limit_assigned'] for location-based speeds
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
        
        # >>> Speed limit handling <<<
        if speed_limits_df is not None:
            # Convert dataframe to numpy arrays and create interpolation lookup
            x_locs = speed_limits_df['x'].values.astype(np.float32)
            v_limits = speed_limits_df['limit_assigned'].values.astype(np.float32)
            
            # Store as tensors for GPU-friendly lookup
            self.x_speed_locs = torch.tensor(x_locs, dtype=torch.float32, device=self.device)
            self.v_speed_limits = torch.tensor(v_limits, dtype=torch.float32, device=self.device)
            self.use_location_speeds = True
            print(f"[Speed Limits] Using location-based free-flow speeds from {len(x_locs)} locations")
            print(f"  Range: {v_limits.min():.1f} - {v_limits.max():.1f} km/h")
        else:
            # Use default constant speed
            self.use_location_speeds = False
            self.default_speed = float(V_f)
            print(f"[Speed Limits] Using constant free-flow speed: {V_f:.1f} km/h")

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
        
        
    def _get_free_flow_speed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get free-flow speed v_f(x) for given locations.
        
        Uses location-based speed limits if available (from speed_limits_df),
        otherwise returns constant default speed.
        
        Args:
            x: Tensor of shape (N, 1) with location coordinates
            
        Returns:
            Tensor of shape (N, 1) with free-flow speeds at each location
        """
        if not self.use_location_speeds:
            # Return constant speed
            return torch.full_like(x, self.default_speed)
        
        # Interpolate speed based on nearest location
        # For each x, find closest location in x_speed_locs and use its limit
        x_expanded = x.expand(-1, len(self.x_speed_locs))  # (N, num_locs)
        locs_expanded = self.x_speed_locs.unsqueeze(0).expand(x.shape[0], -1)  # (N, num_locs)
        
        # Find nearest location for each point
        distances = torch.abs(x_expanded - locs_expanded)
        nearest_idx = torch.argmin(distances, dim=1)  # (N,)
        
        # Lookup speed limit at nearest location
        v_f = self.v_speed_limits[nearest_idx].unsqueeze(1)  # (N, 1)
        
        return v_f

    @torch.enable_grad()
    def net_f(self, x, t):
        # no physics
        if self.fd_name == 'nn' or self.f_weight == 0.0:
            return torch.zeros_like(x)

        # physical speed and its derivatives
        u_norm = self.net_u(x, t)
        u = self._denorm_u(u_norm)
        u_t = self._grad(u, t)
        u_x = self._grad(u, x)

        # Get free-flow speed for all locations
        v_f = self._get_free_flow_speed(x)

        fd = self.fd_name

        if fd == "linear":
            # Greenshields: r_nd = (1/v_f) u_t + (1 - 2u/v_f) u_x
            r_nd = (u_t / v_f) + (1.0 - 2.0 * (u / v_f)) * u_x

        elif fd == "log":
            # Greenberg: r_nd = (1/v_f) u_t + (u/v_f - 1) u_x
            # Note: Using v_f for consistency (previously used separate v_m)
            r_nd = (u_t / v_f) + ((u / v_f) - 1.0) * u_x

        elif fd == "exp":
            # Underwood: r_nd = (1/v_f) u_t + (u/v_f)(ln(v_f/u) - 1) u_x
            eps = 1e-6
            u_safe = torch.clamp(u, min=eps)
            r_nd = (u_t / v_f) + ( (u_safe / v_f) * (torch.log(v_f) - torch.log(u_safe) - 1.0) ) * u_x

        elif fd == "power":
            # Pipes–Munjal: r_nd = (1/v_f) u_t + ((n+1)u/v_f - n) u_x
            n = 3.0
            r_nd = (u_t / v_f) + (((n + 1.0) * (u / v_f)) - n) * u_x

        elif fd == "triangular":
            # r_nd = s(u)*( (1/w)u_t - u_x ) + (1-s(u)) * (u - v_f)/v_f
            w = 15.0
            alpha = 20.0
            s = torch.sigmoid(torch.tensor(alpha, device=u.device, dtype=u.dtype) * (v_f - u))
            r_cong_nd = (u_t / w) - u_x
            r_free_nd = (u - v_f) / v_f
            r_nd = s * r_cong_nd + (1.0 - s) * r_free_nd

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
        # two-stage optimization
        use_lbfgs: bool = False,                 # whether to use L-BFGS after ADAM
        lbfgs_epochs: int | None = None,         # L-BFGS epochs (defaults to epochs//100)
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

        # ----- Two-stage optimization: L-BFGS refinement -----
        if use_lbfgs:
            print("\n" + "=" * 60)
            print("Starting L-BFGS refinement stage...")
            print("=" * 60)
            
            # Determine L-BFGS epochs
            if lbfgs_epochs is None:
                lbfgs_epochs = max(1, epochs // 100)
            
            print(f"Running L-BFGS for {lbfgs_epochs} epochs")
            
            # Create L-BFGS optimizer
            optimizer_lbfgs = torch.optim.LBFGS(
                self.parameters(),
                lr=1.0,  # L-BFGS uses line search, lr=1.0 is standard
                max_iter=20,
                max_eval=25,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=100,
                line_search_fn="strong_wolfe"
            )
            
            # Prepare full physics points for L-BFGS (no subsampling)
            use_physics_lbfgs = self.f_weight > 0.0
            if use_physics_lbfgs:
                x_f_full = self.x_f.requires_grad_(True)
                t_f_full = self.t_f.requires_grad_(True)
            
            # Track L-BFGS history
            lbfgs_best_loss = float("inf")
            lbfgs_history = {"epoch": [], "train_total": [], "data_loss": [], "phys_loss": []}
            
            # Start L-BFGS epoch counting from where ADAM left off
            adam_final_epoch = history["epoch"][-1] if history["epoch"] else epochs
            
            for lbfgs_ep in range(1, lbfgs_epochs + 1):
                self.train()
                
                def closure():
                    optimizer_lbfgs.zero_grad()
                    
                    # Data loss
                    u_pred = self.net_u(x_u_tr, t_u_tr)
                    data_loss = torch.mean((u_tr - u_pred) ** 2)
                    
                    # Physics loss
                    if use_physics_lbfgs:
                        f_pred = self.net_f(x_f_full, t_f_full)
                        phys_loss = torch.mean(f_pred ** 2)
                    else:
                        phys_loss = torch.tensor(0.0, device=self.device)
                    
                    total_loss = data_loss + self.f_weight * phys_loss
                    total_loss.backward()
                    
                    return total_loss
                
                # L-BFGS step
                optimizer_lbfgs.step(closure)
                
                # Evaluate current loss for logging
                with torch.no_grad():
                    u_pred = self.net_u(x_u_tr, t_u_tr)
                    data_loss = torch.mean((u_tr - u_pred) ** 2)
                    
                    if use_physics_lbfgs:
                        f_pred = self.net_f(x_f_full, t_f_full)
                        phys_loss = torch.mean(f_pred ** 2)
                    else:
                        phys_loss = torch.tensor(0.0, device=self.device)
                    
                    total_loss = data_loss + self.f_weight * phys_loss
                    current_loss = float(total_loss.item())
                
                # Logging (using continuous epoch numbers)
                continuous_epoch = adam_final_epoch + lbfgs_ep
                if (lbfgs_ep % max(1, log_every // 10) == 0) or (lbfgs_ep == 1):
                    print(f"[L-BFGS Epoch {continuous_epoch:05d}] train_total={current_loss:.4e} | data={data_loss.item():.4e} | phys={phys_loss.item():.4e}")
                
                lbfgs_history["epoch"].append(continuous_epoch)
                lbfgs_history["train_total"].append(current_loss)
                lbfgs_history["data_loss"].append(float(data_loss.item()))
                lbfgs_history["phys_loss"].append(float(phys_loss.item()))
                
                # Save best L-BFGS model
                if current_loss < lbfgs_best_loss:
                    lbfgs_best_loss = current_loss
                    best_train = current_loss  # Update overall best
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
            
            # Extend history with L-BFGS results
            for key in ["epoch", "train_total", "data_loss", "phys_loss"]:
                history[key].extend(lbfgs_history[key])
            
            # Reload best model from L-BFGS
            if os.path.isfile(ckpt_path):
                state = torch.load(ckpt_path, map_location=self.device)
                self.load_state_dict(state["model_state_dict"])
            
            print(f"\nL-BFGS refinement complete. Best loss: {lbfgs_best_loss:.4e}")

        # Save final meta file with complete history (including L-BFGS if used)
        with open(meta_path, "w") as f:
            json.dump({
                "best_train": best_train,
                "best_epoch": best_epoch,
                "epochs_run": history["epoch"][-1] if history["epoch"] else 0,
                "history": history,
            }, f, indent=2)
        print(f"Saved final training metadata to: {meta_path}")

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
    def load_from_checkpoint(ckpt_path: str, X_u, u, X_f, device: str = None, speed_limits_df: pd.DataFrame = None):
        """
        Convenience loader: rebuilds model and loads weights, while allowing new datasets.
        
        Args:
            ckpt_path: Path to checkpoint file
            X_u, u, X_f: Training data (used for model structure, not for training)
            device: Device to load model on
            speed_limits_df: Optional DataFrame with location-based speed limits
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
            fd_name=state.get("fd_name", "linear"),
            speed_limits_df=speed_limits_df
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
    fd_name: str,
    speed_limits_df: pd.DataFrame = None
) -> UnifiedPINN:
    """Convenience function to build a UnifiedPINN model with standard settings."""
    return UnifiedPINN(
        X_u=X_u_train, u=u_train, X_f=X_f_train, layers=layers,
        lb=lb, ub=ub, normalize_labels=True,
        f_weight=f_weight, V_f=110.0, t_scale=0.25, fd_name=fd_name,
        speed_limits_df=speed_limits_df
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
    # Speed limit parameters
    use_inferred_speed_limits: bool = bool(cfg.get('use_inferred_speed_limits', True))
    V_f: float = float(cfg.get('V_f', 110.0))
    speed_limit_percentile: int = int(cfg.get('speed_limit_percentile', 95))
    valid_speed_limits: tuple = tuple(cfg.get('valid_speed_limits', [80, 100]))
    # Two-stage optimization parameters
    use_lbfgs: bool = bool(cfg.get('use_lbfgs', True))
    lbfgs_epochs: int | None = cfg.get('lbfgs_epochs', None)  # None means epochs//100
    # Plotting parameters
    plot_loss_history_flag: bool = bool(cfg.get('plot_loss_history', True))
    loss_plot_log_scale: bool = bool(cfg.get('loss_plot_log_scale', True))

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
    
    # Conditionally infer free-flow speed per location based on config
    if use_inferred_speed_limits:
        print("\n" + "=" * 60)
        print("Inferring location-based free-flow speed limits from data...")
        print("=" * 60)
        df_free_flow = infer_speed_limits(
            Exact, x, 
            valid_limits=valid_speed_limits, 
            perc=speed_limit_percentile
        )
        print(f"Speed limits inferred using {speed_limit_percentile}th percentile")
        print(f"Valid speed limits: {valid_speed_limits}")
        print(f"Range: {df_free_flow['limit_assigned'].min():.1f} - {df_free_flow['limit_assigned'].max():.1f} km/h")
    else:
        print("\n" + "=" * 60)
        print(f"Using default constant free-flow speed: {V_f:.1f} km/h")
        print("=" * 60)
        df_free_flow = None

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

    # Setup output directory for all results (models, plots, loss history)
    os.makedirs(out_fig_dir, exist_ok=True)
    
    # Train PINN (or load if exists)
    pinn_checkpoint = os.path.join(out_fig_dir, f"model_pinn_{fd_name}.pt")
    pinn_meta_path = os.path.join(out_fig_dir, f"model_pinn_{fd_name}_meta.json")
    
    if os.path.exists(pinn_checkpoint):
        print("\n" + "=" * 60)
        print("PINN Model checkpoint found - Running in EVALUATION mode")
        print(f"Loading from: {pinn_checkpoint}")
        print("=" * 60)
        model_pinn = UnifiedPINN.load_from_checkpoint(
            pinn_checkpoint, X_u_train, u_train, X_f_train, speed_limits_df=df_free_flow
        )
        # Load history if available
        if os.path.exists(pinn_meta_path):
            with open(pinn_meta_path, 'r') as f:
                pinn_meta = json.load(f)
            out_pinn = {
                'best_train': pinn_meta.get('best_train', 0.0),
                'best_epoch': pinn_meta.get('best_epoch', 0),
                'history': pinn_meta.get('history', {"epoch": [], "train_total": [], "data_loss": [], "phys_loss": []}),
            }
        else:
            out_pinn = {'best_train': 0.0, 'best_epoch': 0, 'history': {"epoch": [], "train_total": [], "data_loss": [], "phys_loss": []}}
        elapsed = 0.0
    else:
        print("\n" + "=" * 60)
        print("Training PINN Model...")
        print("=" * 60)
        model_pinn = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=f_weight, fd_name=fd_name, speed_limits_df=df_free_flow)
        start_time = time.time()
        out_pinn = model_pinn.fit(
            epochs=epochs,
            lr=lr,
            early_stop=EarlyStopConfig(patience=patience, min_delta=0.0, verbose=True),
            save_dir=out_fig_dir,
            tag=f"pinn_{fd_name}",
            log_every=log_every,
            f_subset_per_epoch=min(4000, X_f_train.shape[0]),
            physics_every=physics_every,
            use_mixed_precision=True,
            use_lbfgs=use_lbfgs,
            lbfgs_epochs=lbfgs_epochs,
        )
        elapsed = time.time() - start_time
        print(f"Training finished after {out_pinn['best_epoch']} epochs with best train loss {out_pinn['best_train']:.4e}")
        print(f'Training time: {elapsed:.4f} seconds')
    
    # Evaluate PINN
    error_u, U_pred, _ = evaluate_model(model_pinn, X_star, u_star, X, T, Exact)
    print(f'PINN Error u: {error_u:.4e}')
    
    # Plot training history for PINN
    if plot_loss_history_flag:
        plot_training_history(
            history=out_pinn['history'],
            out_dir=out_fig_dir,  # Save to figure output directory
            tag=f"pinn_{fd_name}",
            show_physics=(f_weight > 0.0),
            log_scale=loss_plot_log_scale,
        )

    # Collect model predictions for plotting
    model_predictions = []
    model_predictions.append({
        'name': f'PINN ({fd_name})',
        'U_pred': U_pred,
        'error': error_u,
    })

    # Train pure NN (baseline) - optional based on run_base config
    if run_base:
        nn_checkpoint = os.path.join(out_fig_dir, "model_nn.pt")
        nn_meta_path = os.path.join(out_fig_dir, "model_nn_meta.json")
        
        if os.path.exists(nn_checkpoint):
            print("\n" + "=" * 60)
            print("NN Model checkpoint found - Running in EVALUATION mode")
            print(f"Loading from: {nn_checkpoint}")
            print("=" * 60)
            model_nn = UnifiedPINN.load_from_checkpoint(
                nn_checkpoint, X_u_train, u_train, X_f_train, speed_limits_df=df_free_flow
            )
            # Load history if available
            if os.path.exists(nn_meta_path):
                with open(nn_meta_path, 'r') as f:
                    nn_meta = json.load(f)
                out_nn = {
                    'best_train': nn_meta.get('best_train', 0.0),
                    'best_epoch': nn_meta.get('best_epoch', 0),
                    'history': nn_meta.get('history', {"epoch": [], "train_total": [], "data_loss": [], "phys_loss": []}),
                }
            else:
                out_nn = {'best_train': 0.0, 'best_epoch': 0, 'history': {"epoch": [], "train_total": [], "data_loss": [], "phys_loss": []}}
            elapsed = 0.0
        else:
            print("\n" + "=" * 60)
            print("Training Regular NN Model...")
            print("=" * 60)
            model_nn = build_model(X_u_train, u_train, X_f_train, layers, lb, ub, f_weight=0.0, fd_name='nn', speed_limits_df=df_free_flow)
            model_nn = torch.compile(model_nn, mode="max-autotune")
            start_time = time.time()
            out_nn = model_nn.fit(
                epochs=epochs,
                lr=lr,
                early_stop=EarlyStopConfig(patience=patience, min_delta=0.0, verbose=True),
                save_dir=out_fig_dir,
                tag="nn",
                log_every=log_every,
                use_mixed_precision=True,
                use_lbfgs=use_lbfgs,
                lbfgs_epochs=lbfgs_epochs,
            )
            elapsed = time.time() - start_time
            print(f"Training finished after {out_nn['best_epoch']} epochs with best train loss {out_nn['best_train']:.4e}")
            print(f'Training time: {elapsed:.4f} seconds')
        
        # Evaluate NN
        error_u2, U_pred2, _ = evaluate_model(model_nn, X_star, u_star, X, T, Exact)
        print(f'DL Error u: {error_u2:.4e}')
        
        # Plot training history for NN
        if plot_loss_history_flag:
            plot_training_history(
                history=out_nn['history'],
                out_dir=out_fig_dir,  # Save to figure output directory
                tag="nn",
                show_physics=False,  # No physics loss for pure NN
                log_scale=loss_plot_log_scale,
            )

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
            N_u=N_u, fd_name=fd_name,
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


