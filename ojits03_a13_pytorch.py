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
        t_scale = 0.25 # 0.25 hours
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
    n_sensors = 10  # Number of sensor columns to select (only used when chose_obs_based_on_sensor=True)
                   # Sensors will be equally spaced across the highway and take ALL their observations
    
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
    # use real distance from json file
    with open('td_data/2024-09-09.json', 'r') as f:
        distance = json.load(f)
    x = np.array(distance['distances']).reshape(-1, 1)[:vel.shape[0]]

    # t = t[:vel.shape[1]]  # 480 time steps
    t = np.arange(vel.shape[1]).reshape(-1, 1)

    # Convert velocity data
    Exact = np.real(vel.T)
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    x_idx = np.arange(Exact.shape[0])
    idx_flatten, t_idx = np.meshgrid(x_idx, t)
    idx_grid = np.hstack((idx_flatten.flatten()[:, None], t_idx.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    
    # Replace missing values (-1) with mean of valid values
    valid_mask = u_star > 0
    if np.sum(~valid_mask) > 0:
        u_mean = np.mean(u_star[valid_mask])
        u_star[~valid_mask] = u_mean
        print(f"Replaced {np.sum(~valid_mask)} missing values with mean: {u_mean:.2f}")
    
    lb = X_star.min(0).astype(np.float32)
    ub = X_star.max(0).astype(np.float32)

    ############################### Training Data #################################
    n_locations = x.shape[0]
    n_timesteps = t.shape[0]
    
    if not chose_obs_based_on_sensor:
        # ==== METHOD 1: Random sampling from all valid points (original) ====
        print("\n[Data Selection] Using random sampling from all valid points")
        valid_train_mask = u_star.flatten() > 0
        valid_indices = np.where(valid_train_mask)[0]
        
        n_valid = min(N_u, len(valid_indices))
        idx = np.random.choice(valid_indices, n_valid, replace=False)
        
        X_u_train = X_star[idx, :]
        idx_train = idx_grid[idx, :].astype(int)
        u_train = u_star[idx, :]
        
        print(f"  - Total valid points: {len(valid_indices)}")
        print(f"  - Sampled points: {n_valid}")
        
    else:
        # ==== METHOD 2: Select complete sensor columns (equally distributed) ====
        print("\n[Data Selection] Using sensor-based column selection (equally distributed)")
        u_star_matrix = u_star.reshape((n_timesteps, n_locations))  # reshape to [t, x]
        
        # Select n_sensors equally distributed across spatial domain
        n_sensors_to_select = min(n_sensors, n_locations)
        selected_sensors = np.linspace(0, n_locations-1, n_sensors_to_select, dtype=int)
        selected_sensors = np.unique(selected_sensors)  # Remove duplicates if any
        selected_sensors = selected_sensors.tolist()
        
        # Collect all valid points from selected sensors
        selected_indices = []
        selected_idx_grid = []
        sensor_point_counts = []
        
        for col in selected_sensors:
            valid_rows = np.where(u_star_matrix[:, col] > 0)[0]
            n_pts = len(valid_rows)
            sensor_point_counts.append((col, n_pts))
            
            for row in valid_rows:
                flat_idx = col + row * n_locations  # Convert (row, col) to flat index
                selected_indices.append(flat_idx)
                selected_idx_grid.append([row, col])  # [t_idx, x_idx]
        
        idx = np.array(selected_indices)
        idx_train = np.array(selected_idx_grid)
        X_u_train = X_star[idx, :]
        u_train = u_star[idx, :]
        n_valid = len(idx)
        
        print(f"  - Total available locations: {n_locations}")
        print(f"  - Requested sensors: {n_sensors}")
        print(f"  - Selected sensors: {len(selected_sensors)} (equally spaced)")
        print(f"  - Sensor indices: {selected_sensors}")
        print(f"  - Total observation points: {n_valid}")
        
        # Show points per selected sensor
        for col, n_pts in sensor_point_counts:
            print(f"    · Sensor {col}: {n_pts} points")
    
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
    
    ####### Row 1: Observation Data ##################
    gs_obs = gridspec.GridSpec(1, 2)
    gs_obs.update(top=0.72, bottom=0.52, left=0.15, right=0.85, wspace=1)
    
    ax = plt.subplot(gs_obs[:, :])
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Create observation data matrix (only training points visible, rest is white/NaN)
    Observation = np.full_like(Exact, np.nan)  # Initialize with NaN
    for i, (x_train, t_train) in enumerate(X_u_train):
        # Find closest grid point
        # x_idx = np.argmin(np.abs(x.flatten() - x_train))
        # t_idx = np.argmin(np.abs(t.flatten() - t_train))
        x_idx = idx_train[i, 1]
        t_idx = idx_train[i, 0]
        # Observation[t_idx, x_idx] = u_train[i, 0]
        Observation[t_idx, x_idx] = Exact[t_idx, x_idx]
    
    # Create custom colormap with white for NaN values
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
    if chose_obs_based_on_sensor:
        method_str = f"{n_sensors_to_select} sensors"
        title_str = f'Observation Data (N={n_valid} points from {method_str})'
    else:
        title_str = f'Observation Data (N={n_valid} points, Random sampling)'
    ax.set_title(title_str, fontsize=18)
    
    ####### Row 2: PIDL: u(t,x) ##################
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

    ####### Row 3: DL: u(t,x) ##################
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
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/a13_pidl_dl_pytorch_{N_u}.pdf')
    plt.savefig(f'figures/a13_pidl_dl_pytorch_{N_u}.eps')
    plt.show()
    
    print(f"\nPlots saved to figures/a13_pidl_dl_pytorch_{N_u}.pdf/eps")
    print("="*60)
    ################################# Plot #################################

