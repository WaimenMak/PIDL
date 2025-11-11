"""
Processes A13 highway data from 2024-09-09 to 2024-09-13
"""

import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import pandas as pd
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

se = 25
np.random.seed(se)
torch.manual_seed(se)


# PINN Class
class PhysicsInformedNN(nn.Module):
    def __init__(self, X_u, u, X_f, layers, lb, ub, normalize_labels=True):
        super(PhysicsInformedNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(self.device)

        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32).to(self.device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32).to(self.device)
        
        # Label normalization
        self.normalize_labels = normalize_labels
        if self.normalize_labels:
            self.u_mean = torch.mean(torch.tensor(u, dtype=torch.float32))
            self.u_std = torch.std(torch.tensor(u, dtype=torch.float32))
            # Avoid division by zero
            if self.u_std < 1e-8:
                self.u_std = torch.tensor(1.0)
            # Normalize labels: (u - mean) / std
            u_normalized = (u - self.u_mean.cpu().numpy()) / self.u_std.cpu().numpy()
            self.u = torch.tensor(u_normalized, dtype=torch.float32).to(self.device)
            self.u_mean = self.u_mean.to(self.device)
            self.u_std = self.u_std.to(self.device)
            print(f"[PINN] Label normalization: mean={self.u_mean.item():.2f}, std={self.u_std.item():.2f}")
        else:
            self.u = torch.tensor(u, dtype=torch.float32).to(self.device)
            self.u_mean = torch.tensor(0.0).to(self.device)
            self.u_std = torch.tensor(1.0).to(self.device)

        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device)

        self.layers = layers
        self.model = self.initialize_NN(layers).to(self.device)

        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=20000,
            max_eval=10000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.iter = 0

    def initialize_NN(self, layers):
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())

        model = nn.Sequential(*modules)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        return model

    def neural_net(self, X):
        X_normalized = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(X_normalized)

    def net_u(self, x, t):
        u = self.neural_net(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Traffic flow PDE - adjust parameters for A13 highway if needed
        # f = 0.20 * u_x - 2 * 0.20 / 46.64 * u * u_x - 0.20 / 46.64 * u_t
        V_f = 110  # Free flow speed (km/h)
        t_scale = 1 # 0.25 hours
        f = (u_x - 2/V_f*u*u_x - 1/V_f*u_t)*t_scale
        return f

    def loss_closure(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)

        loss = torch.mean(torch.square(self.u - u_pred)) + torch.mean(torch.square(f_pred))

        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(f'Iter: {self.iter}, Loss: {loss.item():.4e}')
        return loss

    def train_model(self):
        self.model.train()
        self.optimizer.step(self.loss_closure)

    def predict(self, X_star):
        self.model.eval()
        X_star = X_star.astype(np.float32)
        x_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(self.device)
        t_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(self.device)

        u_star = self.net_u(x_star, t_star)
        
        # Denormalize predictions if labels were normalized
        if self.normalize_labels:
            u_star = torch.clip(u_star * self.u_std + self.u_mean, min=torch.tensor([0.0]))

        x_f_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        t_f_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device)
        f_star = self.net_f(x_f_star, t_f_star)

        return u_star.cpu().detach().numpy(), f_star.cpu().detach().numpy()


# Regular NN class
class NN(nn.Module):
    def __init__(self, X_u, u, X_f, layers, lb, ub, normalize_labels=True):
        super(NN, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(self.device)

        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32).to(self.device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32).to(self.device)
        
        # Label normalization
        self.normalize_labels = normalize_labels
        if self.normalize_labels:
            self.u_mean = torch.mean(torch.tensor(u, dtype=torch.float32))
            self.u_std = torch.std(torch.tensor(u, dtype=torch.float32))
            # Avoid division by zero
            if self.u_std < 1e-8:
                self.u_std = torch.tensor(1.0)
            # Normalize labels: (u - mean) / std
            u_normalized = (u - self.u_mean.cpu().numpy()) / self.u_std.cpu().numpy()
            self.u = torch.tensor(u_normalized, dtype=torch.float32).to(self.device)
            self.u_mean = self.u_mean.to(self.device)
            self.u_std = self.u_std.to(self.device)
            print(f"[NN] Label normalization: mean={self.u_mean.item():.2f}, std={self.u_std.item():.2f}")
        else:
            self.u = torch.tensor(u, dtype=torch.float32).to(self.device)
            self.u_mean = torch.tensor(0.0).to(self.device)
            self.u_std = torch.tensor(1.0).to(self.device)
        
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device)

        self.layers = layers
        self.model = self.initialize_NN(layers).to(self.device)
        
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(), 
            max_iter=10000,
            max_eval=10000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.iter = 0

    def initialize_NN(self, layers):
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        model = nn.Sequential(*modules)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        return model

    def neural_net(self, X):
        X_normalized = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(X_normalized)

    def net_u(self, x, t):
        u = self.neural_net(torch.cat([x, t], dim=1))
        return u
    
    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        # f = 0.20 * u_x - 2 * 0.20 / 46.64 * u * u_x - 0.20 / 46.64 * u_t
        V_f = 110  # Free flow speed (km/h)
        t_scale = 4 # 0.25 hours
        f = 0.2*(u_x - 2/V_f*u*u_x - 1/V_f*u_t*4)
        return torch.tensor([0])  # Return zero since we don't use f in loss

    def loss_closure(self):
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        loss = torch.mean(torch.square(self.u - u_pred))
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(f'Iter: {self.iter}, Loss: {loss.item():.4e}')
        return loss

    def train_model(self):
        self.model.train()
        self.optimizer.step(self.loss_closure)

    def predict(self, X_star):
        self.model.eval()
        X_star = X_star.astype(np.float32)
        x_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(self.device)
        t_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(self.device)
        
        u_star = self.net_u(x_star, t_star)
        
        # Denormalize predictions if labels were normalized
        if self.normalize_labels:
            u_star = torch.clip(u_star * self.u_std + self.u_mean, min=torch.tensor([0.0]))
        
        x_f_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        t_f_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device)
        f_star = self.net_f(x_f_star, t_f_star)
        
        return u_star.cpu().detach().numpy(), f_star.cpu().detach().numpy()


if __name__ == "__main__":
    #TODO: Adjust N_u and N_f
    N_u = 800  # Number of points for random sampling (only used when chose_obs_based_on_sensor=False)
    N_f = 10000
    
    # ==== Data Selection Strategy ====
    # False: Random sampling of N_u points from all valid points (scattered observations)
    # True:  Select n_sensors complete columns equally distributed (realistic fixed sensor placement)
    chose_obs_based_on_sensor = True
    n_sensors = 5  # Number of sensor columns to select (only used when chose_obs_based_on_sensor=True)
                   # Sensors will be equally spaced across the highway and take ALL their observations
    
    # ==== Future State Forecasting ====
    # Set to 0 to use all time steps for training
    # Set to > 0 to remove the last n_future_steps from training data (test forecasting ability)
    n_future_steps = 0  # Number of future time steps to exclude from training
    
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    # Use synthetic.mat for spatial-temporal grid structure
    data = scipy.io.loadmat('org_data/synthetic.mat')
    t = data['tScale'].T.flatten()[:, None]
    x = data['xScale'].T.flatten()[:, None]
    
    # Load A13 velocity data
    #TODO: change the dataset with gap
    vel = pd.read_table('data/A13_Velocity_Data_0909-0910.txt', delim_whitespace=True, header=None)
    # discard the first row of the dataframe
    vel = vel.iloc[1:]
    # vel = vel.values
    print(f"A13 Data Shape: {vel.shape}")
    print(f"Spatial locations: {vel.shape[0]}")
    print(f"Time steps: {vel.shape[1]}")
    
    # Adjust spatial-temporal grid to match data dimensions
    # x = x[:vel.shape[0]]  # 30 locations
    # x = np.arange(vel.shape[0]).reshape(-1, 1)
    # use real distance from json file and time labels
    with open('td_data/2024-09-09.json', 'r') as f:
        labels = json.load(f)
    x_locations = np.array(labels['distances']).reshape(-1, 1)[:vel.shape[0]]
    # Reverse x_locations since first row is furthest distance, not start point
    x_locations = np.flip(x_locations, axis=0)
    time_labels_raw = np.array(labels['times'])
    
    # Handle multiple days - create continuous time labels for visualization
    n_timesteps = vel.shape[1]
    n_times_per_day = len(time_labels_raw)
    n_days = n_timesteps // n_times_per_day if n_times_per_day < n_timesteps else 1
    
    if n_days > 1:
        # Create looping time labels for multiple days
        time_labels = np.tile(time_labels_raw, n_days)[:n_timesteps]
        print(f"Multiple days detected: {n_days} days, {n_timesteps} total timesteps")
    else:
        time_labels = time_labels_raw[:n_timesteps]
    
    # Use indices for actual computation (NOT time labels)
    t_time = np.arange(n_timesteps).reshape(-1, 1)
    
    # Swap axes: t_time (x-axis), x_locations (y-axis)
    # vel.shape is (locations, timesteps), we keep it as is
    # Convert velocity data - NO transpose, keep as (locations, timesteps)
    Exact = np.real(vel.values)  # Shape: (n_locations, n_timesteps)
    # Flip Exact vertically to match reversed x_locations (first row should be start point)
    Exact = np.flipud(Exact)
    T, X = np.meshgrid(t_time, x_locations)  # T is time indices (x-axis), X is location (y-axis)

    # X_star: each row is [location, time]
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    # idx_grid: [location_idx, time_idx]
    n_locations = x_locations.shape[0]
    n_timesteps = t_time.shape[0]
    x_idx = np.arange(n_locations)
    t_idx = np.arange(n_timesteps)
    t_idx_mesh, x_idx_mesh = np.meshgrid(t_idx, x_idx)  # Match T, X order
    idx_grid = np.hstack((x_idx_mesh.flatten()[:, None], t_idx_mesh.flatten()[:, None]))
    
    # Flatten Exact in same order as X_star
    u_star = Exact.flatten()[:, None]  # Flattens row-by-row (location first, then time)
    
    # Replace missing values (-1) with mean of valid values
    valid_mask = u_star > 0
    if np.sum(~valid_mask) > 0:
        u_mean = np.mean(u_star[valid_mask])
        u_star[~valid_mask] = u_mean
        print(f"Replaced {np.sum(~valid_mask)} missing values with mean: {u_mean:.2f}")
    
    lb = X_star.min(0).astype(np.float32)
    ub = X_star.max(0).astype(np.float32)

    ############################### Training Data #################################
    n_locations = x_locations.shape[0]
    n_timesteps = t_time.shape[0]
    
    # Calculate training time horizon (exclude future if forecasting test)
    if n_future_steps > 0:
        t_train_max = n_timesteps - n_future_steps
        print(f"\n{'='*60}")
        print(f"FORECASTING MODE: Training on first {t_train_max}/{n_timesteps} time steps")
        print(f"Forecasting horizon: {n_future_steps} time steps ({n_future_steps*15} minutes)")
        print(f"{'='*60}")
    else:
        t_train_max = n_timesteps
        print(f"\n{'='*60}")
        print(f"STANDARD MODE: Using all {n_timesteps} time steps for training")
        print(f"{'='*60}")
    
    if not chose_obs_based_on_sensor:
        # ==== METHOD 1: Random sampling from all valid points (original) ====
        print("\n[Data Selection] Using random sampling from all valid points")
        valid_train_mask = u_star.flatten() > 0
        valid_indices = np.where(valid_train_mask)[0]
        
        # Filter out future time steps if forecasting
        # idx_grid[:, 1] is time_idx
        if n_future_steps > 0:
            time_mask = idx_grid[valid_indices, 1] < t_train_max
            valid_indices = valid_indices[time_mask]
        
        n_valid = min(N_u, len(valid_indices))
        idx = np.random.choice(valid_indices, n_valid, replace=False)
        
        X_u_train = X_star[idx, :]
        idx_train = idx_grid[idx, :].astype(int)
        u_train = u_star[idx, :]
        
        print(f"  - Total valid points (within training horizon): {len(valid_indices)}")
        print(f"  - Sampled points: {n_valid}")
        
    else:
        # ==== METHOD 2: Select complete sensor rows (equally distributed) ====
        print("\n[Data Selection] Using sensor-based selection (equally distributed across locations)")
        # Reshape: u_star is flattened from Exact (n_locations, n_timesteps)
        u_star_matrix = u_star.reshape((n_locations, n_timesteps))  # reshape to [locations, time]
        
        # Select n_sensors equally distributed across spatial domain (locations)
        n_sensors_to_select = min(n_sensors, n_locations)
        selected_sensors = np.linspace(0, n_locations-1, n_sensors_to_select, dtype=int)
        selected_sensors = np.unique(selected_sensors)  # Remove duplicates if any
        selected_sensors = selected_sensors.tolist()
        
        # Collect all valid points from selected sensors (within training time horizon)
        selected_indices = []
        selected_idx_grid = []
        sensor_point_counts = []
        
        for loc_idx in selected_sensors:
            # Get all time points for this location
            valid_times = np.where(u_star_matrix[loc_idx, :] > 0)[0]
            
            # Filter out future time steps if forecasting
            if n_future_steps > 0:
                valid_times = valid_times[valid_times < t_train_max]
            
            n_pts = len(valid_times)
            sensor_point_counts.append((loc_idx, n_pts))
            
            for time_idx in valid_times:
                # Flatten index: row-major order (location, time)
                flat_idx = loc_idx * n_timesteps + time_idx
                selected_indices.append(flat_idx)
                selected_idx_grid.append([loc_idx, time_idx])  # [location_idx, time_idx]
        
        idx = np.array(selected_indices)
        idx_train = np.array(selected_idx_grid)
        X_u_train = X_star[idx, :]
        u_train = u_star[idx, :]
        n_valid = len(idx)
        
        print(f"  - Total available locations: {n_locations}")
        print(f"  - Requested sensors: {n_sensors}")
        print(f"  - Selected sensors: {len(selected_sensors)} (equally spaced)")
        print(f"  - Sensor location indices: {selected_sensors}")
        print(f"  - Total observation points (within training horizon): {n_valid}")
        
        # Show points per selected sensor
        for loc_idx, n_pts in sensor_point_counts:
            print(f"    · Sensor at location {loc_idx}: {n_pts} points")
    
    # Common operations for both methods
    #TODO: add noise to speed, u_train = u_star[idx, :] + noise
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    print(f"\nTraining with {n_valid} data points (+ {N_f} collocation points)")
    ############################### Training Data #################################

    # PINN Model
    print("\n" + "="*60)
    print("Training PINN Model...")
    print("="*60)
    # Enable label normalization to help PINN learn better
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, normalize_labels=True)
    start_time = time.time()
    model.train_model()
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.4f} seconds')
    u_pred, f_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print(f'PINN Error u: {error_u:.4e}')
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)

    # Regular NN Model
    print("\n" + "="*60)
    print("Training Regular NN Model...")
    print("="*60)
    # Enable label normalization to help NN learn better (prevents constant predictions)
    model2 = NN(X_u_train, u_train, X_f_train, layers, lb, ub, normalize_labels=True)
    start_time2 = time.time()
    model2.train_model()
    elapsed2 = time.time() - start_time2
    print(f'Training time: {elapsed2:.4f} seconds')
    u_pred2, f_pred2 = model2.predict(X_star)
    error_u2 = np.linalg.norm(u_star - u_pred2, 2) / np.linalg.norm(u_star, 2)
    print(f'DL Error u: {error_u2:.4e}')
    
    U_pred2 = griddata(X_star, u_pred2.flatten(), (X, T), method='cubic')
    Error2 = np.abs(Exact - U_pred2)

    ################################# Plot #################################
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # Calculate colorbar range from ground truth
    vmin = np.min(Exact)
    vmax = np.max(Exact)
    print(f"Colorbar range: [{vmin:.2f}, {vmax:.2f}] km/h")
    
    # pgf_with_latex = {  # setup matplotlib to use latex for output
    #     "pgf.texsystem": "pdflatex",
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "pgf.preamble": [
    #         r"\usepackage[utf8x]{inputenc}",
    #         r"\usepackage[T1]{fontenc}",
    #     ]
    # }
    # mpl.rcParams.update(pgf_with_latex)
    
    fig = plt.figure(figsize=(12, 20))

    ####### Row 0: Ground Truth ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.97, bottom=0.77, left=0.15, right=0.85, wspace=1)

    ax = plt.subplot(gs0[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    # Extent: [left, right, bottom, top] = [t_min, t_max, x_min, x_max]
    # Use indices for extent
    h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow_r',
                  extent=[0, n_timesteps-1, x_locations.min(), x_locations.max()],
                  origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    # X_u_train[:, 0] is location, X_u_train[:, 1] is time - swap for (x=time, y=location)
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', markersize=0.8, clip_on=False)
    
    # Add forecasting boundary if applicable (vertical line since time is on x-axis)
    if n_future_steps > 0:
        t_boundary_idx = t_train_max - 1  # Last training time index
        ax.axvline(x=t_boundary_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set time tick labels to show actual time values
    # Show ticks at reasonable intervals
    n_ticks = min(10, n_timesteps)
    tick_indices = np.linspace(0, n_timesteps-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(time_labels[i]) for i in tick_indices])
    
    ax.set_xlabel('Time (15 min intervals)', fontsize=18)
    ax.set_ylabel('Location (km)', fontsize=18)
    title = 'Ground Truth: A13 Highway Speed (km/h)'
    if n_future_steps > 0:
        title += f' [Last {n_future_steps} steps unseen]'
    ax.set_title(title, fontsize=18)
    
    ####### Row 1: Observation Data ##################
    gs_obs = gridspec.GridSpec(1, 2)
    gs_obs.update(top=0.72, bottom=0.52, left=0.15, right=0.85, wspace=1)
    
    ax = plt.subplot(gs_obs[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Create observation data matrix (only training points visible, rest is white/NaN)
    Observation = np.full_like(Exact, np.nan)  # Initialize with NaN
    for i in range(len(X_u_train)):
        # idx_train[i, 0] is location_idx, idx_train[i, 1] is time_idx
        loc_idx = idx_train[i, 0]
        time_idx = idx_train[i, 1]
        Observation[loc_idx, time_idx] = Exact[loc_idx, time_idx]
    
    # Create custom colormap with white for NaN values
    cmap = plt.cm.rainbow_r.copy()
    cmap.set_bad(color='white')
    
    h = ax.imshow(Observation, interpolation='nearest', cmap=cmap,
                  extent=[0, n_timesteps-1, x_locations.min(), x_locations.max()],
                  origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)
    
    # X_u_train[:, 1] is time, X_u_train[:, 0] is location
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'k.', markersize=1.5, clip_on=False, alpha=0.5)
    
    # Add vertical line to show forecasting boundary (time is on x-axis)
    if n_future_steps > 0:
        t_boundary_idx = t_train_max - 1
        ax.axvline(x=t_boundary_idx, color='red', linestyle='--', linewidth=2, label='Training/Forecast Boundary')
        ax.legend(loc='upper right', fontsize=12)
        # Add text annotation
        ax.text(t_boundary_idx + 0.02*n_timesteps, 
                x_locations.min() + 0.05*(x_locations.max()-x_locations.min()), 
                'FORECAST ZONE', fontsize=14, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))
    
    # Set time tick labels
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(time_labels[i]) for i in tick_indices])
    
    ax.set_xlabel('Time (15 min intervals)', fontsize=18)
    ax.set_ylabel('Location (km)', fontsize=18)
    if chose_obs_based_on_sensor:
        method_str = f"{n_sensors_to_select} sensors"
        title_str = f'Observation Data (N={n_valid} points from {method_str})'
    else:
        title_str = f'Observation Data (N={n_valid} points, Random sampling)'
    if n_future_steps > 0:
        title_str += f' [Forecasting {n_future_steps} steps]'
    ax.set_title(title_str, fontsize=18)
    
    ####### Row 2: PIDL: u(t,x) ##################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.47, bottom=0.27, left=0.15, right=0.85, wspace=1)

    ax = plt.subplot(gs1[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    h = ax.imshow(U_pred, interpolation='nearest', cmap='rainbow_r',
                  extent=[0, n_timesteps-1, x_locations.min(), x_locations.max()],
                  origin='lower', aspect='auto', vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', markersize=0.8, clip_on=False)
    
    # Add forecasting boundary if applicable (vertical line since time is on x-axis)
    if n_future_steps > 0:
        t_boundary_idx = t_train_max - 1
        ax.axvline(x=t_boundary_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set time tick labels
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(time_labels[i]) for i in tick_indices])
    
    ax.set_xlabel('Time (15 min intervals)', fontsize=18)
    ax.set_ylabel('Location (km)', fontsize=18)
    ax.set_title(f'PIDL Estimation (Error: {error_u:.4f})', fontsize=18)

    ####### Row 3: DL: u(t,x) ##################
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=0.22, bottom=0.02, left=0.15, right=0.85, wspace=1)

    ax = plt.subplot(gs2[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    h = ax.imshow(U_pred2, interpolation='nearest', cmap='rainbow_r',
                  extent=[0, n_timesteps-1, x_locations.min(), x_locations.max()],
                  origin='lower', aspect='auto', vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=16)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', markersize=0.8, clip_on=False)
    
    # Add forecasting boundary if applicable (vertical line since time is on x-axis)
    if n_future_steps > 0:
        t_boundary_idx = t_train_max - 1
        ax.axvline(x=t_boundary_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set time tick labels
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(time_labels[i]) for i in tick_indices])
    
    ax.set_xlabel('Time (15 min intervals)', fontsize=18)
    ax.set_ylabel('Location (km)', fontsize=18)
    ax.set_title(f'DL Estimation (Error: {error_u2:.4f})', fontsize=18)
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Include forecast info in filename if applicable
    if n_future_steps > 0:
        filename_suffix = f'_{N_u}_forecast{n_future_steps}'
    else:
        filename_suffix = f'_{N_u}'
    
    plt.savefig(f'figures/a13_pidl_dl_pytorch{filename_suffix}.pdf')
    plt.savefig(f'figures/a13_pidl_dl_pytorch{filename_suffix}.eps')
    plt.show()
    
    print(f"\nPlots saved to figures/a13_pidl_dl_pytorch{filename_suffix}.pdf/eps")
    
    # Print summary
    if n_future_steps > 0:
        print("\n" + "="*60)
        print("FORECASTING MODE SUMMARY")
        print("="*60)
        print(f"Training data: First {t_train_max} time steps")
        print(f"Forecast horizon: {n_future_steps} time steps ({n_future_steps*15} minutes)")
        print(f"\nPINN Error:  {error_u:.4e}")
        print(f"DL Error:    {error_u2:.4e}")
        print("="*60)
    
    print("="*60)
    ################################# Plot #################################

