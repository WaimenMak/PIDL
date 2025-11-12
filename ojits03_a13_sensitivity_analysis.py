"""
This file: Sensitivity analysis for A13 highway data using PINN and NN models.

Sensitivity Analysis Variables:
- Number of known data points N_u: [10%, 20%, 30%, 40%, 50%] (up for discussion)
- Number of collocation points N_f: [2000, 4000, 6000, 8000, 10000] (up for discussion)
- Noise levels: [0%, 1%, 5%, 10%] (up for discussion)
- Data selection *

Previously completed tasks:
- Processed A13 highway data from 2024-09-09 to 2024-09-13
- Implemented PINN and NN models for sensitivity analysis
"""

import random
from types import MethodType
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
from tqdm import tqdm
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# imported modules for PINN and NN models
from ojits03_a13_pytorch import *

se = 25
np.random.seed(se)
torch.manual_seed(se)

# multi run 
if __name__ == "__main__":
    ## Fixed parameters ##
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    ################ # Load A13 velocity data #################
    vel = pd.read_table('data/A13_Velocity_Data_0909-0910.txt', delim_whitespace=True, header=None)
    # discard the first row of the dataframe
    vel = vel.iloc[1:]
    # vel = vel.values
    print(f"A13 Data Shape: {vel.shape}")
    print(f"Spatial locations: {vel.shape[0]}")
    print(f"Time steps: {vel.shape[1]}")

    # Use synthetic.mat for spatial-temporal grid structure
    data = scipy.io.loadmat('org_data/synthetic.mat')
    t = data['tScale'].T.flatten()[:, None]
    x = data['xScale'].T.flatten()[:, None]

    # Adjust spatial-temporal grid to match data dimensions
    x = x[:vel.shape[0]]  # 30 locations
    # t = t[:vel.shape[1]]  # 480 time steps
    t = np.arange(vel.shape[1]).reshape(-1, 1)

    # Convert velocity data
    Exact = np.real(vel.T)
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Replace missing values (-1) with mean of valid values
    valid_mask = u_star > 0
    if np.sum(~valid_mask) > 0:
        u_mean = np.mean(u_star[valid_mask])
        u_star[~valid_mask] = u_mean
        print(f"Replaced {np.sum(~valid_mask)} missing values with mean: {u_mean:.2f}")
    
    lb = X_star.min(0).astype(np.float32)
    ub = X_star.max(0).astype(np.float32)

    ############### Helper Functions ################
    def run_once(x,t,u_star,
                 chose_obs_based_on_sensors=False,
                 n_sensors = 5,
                 N_u=800, 
                 N_f=10000, 
                 f_weight=1.0,
                 u_obs_noise_type=None, 
                 u_obs_noise_level=None,
                 test_hyper = None):

        print("\n" + "#"*60)
        if chose_obs_based_on_sensors:
            print(f"Using sensor-based observation with {n_sensors} sensors")
        else:
            print("Using random observation points")
            print(f"N_u={N_u}")
        print(f"N_f={N_f}")
        print(f"Noise Level: {u_obs_noise_level} ({u_obs_noise_type} distribution)")
        print("#"*60)
        ############################### Training Data #################################
        n_locations = x.shape[0]
        n_timesteps = t.shape[0]

        if not chose_obs_based_on_sensors:
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
        if u_obs_noise_type is not None and u_obs_noise_level is not None:
            if u_obs_noise_type == 'Gaussian':
                noise = u_obs_noise_level * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])
                u_train = u_train + noise
                print(f"  - Added Gaussian noise with std dev: {u_obs_noise_level * np.std(u_train):.4f}")
                # recommend u_obs_noise_level between 0.01 to 0.1
            elif u_obs_noise_type == 'Gumbel':
                noise = u_obs_noise_level * np.std(u_train) * np.random.gumbel(size=(u_train.shape[0], u_train.shape[1]))
                u_train = u_train + noise
                print(f"  - Added Gumbel noise with std dev: {u_obs_noise_level * np.std(u_train):.4f}")
                # recommend u_obs_noise_level between 0.01 to 0.1
                # Gumbel noise can simulate extreme events better, the noise level times stddev of u_train, means that higher variability data will have larger noise
            else:
                print("  - No noise added to observations")
        ####
        X_f_train = lb + (ub - lb) * lhs(2, N_f)
        X_f_train = np.vstack((X_f_train, X_u_train))
        print(f"\nTraining with {n_valid} data points (+ {N_f} collocation points)")
        ############################### Training Data #################################
        ############################### Training PINN & NN Models #################################

        # PINN Model
        print("\n" + "="*60)
        print("Training PINN Model...")
        print("="*60)
        # Enable label normalization to help PINN learn better
        model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, normalize_labels=True)
        model.f_weight = torch.tensor(f_weight, device=model.x_u.device, dtype=model.x_u.dtype)

        def new_loss_closure(self):
            self.optimizer.zero_grad()

            u_pred = self.net_u(self.x_u, self.t_u)
            f_pred = self.net_f(self.x_f, self.t_f)

            loss = torch.mean(torch.square(self.u - u_pred)) + self.f_weight * torch.mean(torch.square(f_pred))

            loss.backward()
            self.iter += 1
            if self.iter % 100 == 0:
                print(f'Iter: {self.iter}, Loss: {loss.item():.4e}')
            return loss
        
        model.loss_closure = MethodType(new_loss_closure, model)
        
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
        if test_hyper == 'N_f' or test_hyper == 'f_weight':
            # skip NN training when testing N_f or f_weight sensitivity
            print("Skipping NN training for N_f or f_weight sensitivity test")
            error_u2 = None
            Error2 = None
        else:
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

        # sa_dict = {
        #             'N_u': N_u,
        #             'N_f': N_f,
        #             'u_obs_noise': u_obs_noise,
        #             'chose_obs_based_on_sensors': chose_obs_based_on_sensors,
        #             'n_sensors': n_sensors,
        #             'u_obs_noise_type': u_obs_noise_type,
        #             'u_obs_noise_level': u_obs_noise_level
        #         }

        # plot_results(Exact, x, t, n_valid, 
        #              idx_train, X_u_train, U_pred, U_pred2, error_u, error_u2, sa_dict)

        return [error_u, error_u2, Error, Error2]
    
    def run_multiple_runs(x,t,u_star,
                          chose_obs_based_on_sensors=True,
                          params={},
                          n_runs=10):
        PINN_errors = []
        PINN_errors_std = []
        NN_errors = []
        NN_errors_std = []
        for run in range(n_runs):
            setting_seed = se + run*10
            np.random.seed(setting_seed)
            PINN_error, NN_error, full_PINN_ae, full_NN_ae = run_once(x,t,u_star,
                                        chose_obs_based_on_sensors=chose_obs_based_on_sensors,
                                        n_sensors=params['n_sensors'],
                                        N_u=800,
                                        N_f=params['N_f'],
                                        f_weight=params['f_weight'],
                                        u_obs_noise_type=params['u_obs_noise_type'],
                                        u_obs_noise_level=params['u_obs_noise_level'],
                                        test_hyper = params['hyper'])
            PINN_errors.append(PINN_error)
            PINN_errors_std.append(np.std(full_PINN_ae))
            NN_errors.append(NN_error)
            if params['hyper'] != 'N_f' and params['hyper'] != 'f_weight':
                NN_errors_std.append(np.std(full_NN_ae))
            else:
                NN_errors_std.append(0.0)  # dummy value
        return PINN_errors, NN_errors, PINN_errors_std, NN_errors_std

    def plot_results(Exact, x, t, n_valid, 
                     idx_train, X_u_train, U_pred, U_pred2, error_u, error_u2, sa_dict):
        ################################# Plot #################################
        print("\n" + "="*60)
        print("Generating plots...")
        print("="*60)
        
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
            x_idx = idx_train[i, 1]
            t_idx = idx_train[i, 0]
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
        if sa_dict['chose_obs_based_on_sensors']:
            method_str = f"{sa_dict['n_sensors']} sensors"
            title_str = f'Observation Data (N={n_valid} points from {method_str})'
        else:
            title_str = f'Observation Data (N={n_valid} points, Random sampling)'
        ax.set_title(title_str, fontsize=18)

        ####### Row 1: PIDL: u(t,x) ##################
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

        ####### Row 2: DL: u(t,x) ##################
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
        
        if not os.path.exists('figures/sa_figures'):
            os.makedirs('figures/sa_figures')
        plt.savefig(f"figures/sa_figures/{sa_dict['chose_obs_based_on_sensors']}_{sa_dict['N_u']}_{sa_dict['n_sensors']}_{sa_dict['u_obs_noise_type']}_{sa_dict['u_obs_noise_level']}.pdf")
        plt.savefig(f"figures/sa_figures/{sa_dict['chose_obs_based_on_sensors']}_{sa_dict['N_u']}_{sa_dict['n_sensors']}_{sa_dict['u_obs_noise_type']}_{sa_dict['u_obs_noise_level']}.eps")
        plt.show()

        print(f"\nPlots saved to figures/sa_figures/{sa_dict['chose_obs_based_on_sensors']}_{sa_dict['N_u']}_{sa_dict['n_sensors']}_{sa_dict['u_obs_noise_type']}_{sa_dict['u_obs_noise_level']}.pdf/eps")
        print("="*60)

    ########### Sensitivity Analysis Execution ###########

    #  one hyperparameter changes per sensitivity analysis run
    hyper = 'f_weight'  # Choose from 'N_f', 'u_obs_noise_level', 'n_sensors', 'f_weight'

    default_hyperparameters = {
        'n_sensors': 5,
        'N_f': 10000,
        'u_obs_noise_type': 'Gumbel',
        'u_obs_noise_level': 0.0,
        'f_weight': 1.0
    }

    def get_hyperparameter_range(hyper):
        if hyper == 'N_f':
            return [1000, 3000, 6000, 10000] 
        elif hyper == 'u_obs_noise_level':
            return [0.0, 0.05, 0.1, 0.2]
        elif hyper == 'n_sensors':
            return [3, 5, 7, 10, 15]
        elif hyper == 'f_weight':
            return [0.1, 0.5, 1.0, 2.0, 5.0]
        else:
            raise ValueError("Unknown hyperparameter for sensitivity analysis")

    n_runs = 5  # Number of runs per hyperparameter setting
    chose_obs_based_on_sensors = True  # Set to True to use sensor-based selection

    PINN_error_results = {}
    NN_error_results = {}

    # test saving path
    if not os.path.exists('results/sa_results'):
        os.makedirs('results/sa_results')
    #  test save empty dictionary file
    with open(f'results/sa_results/PINN_sa_{hyper}.json', 'w') as f:
        json.dump(PINN_error_results, f, indent=2)
    with open(f'results/sa_results/NN_sa_{hyper}.json', 'w') as f:
        json.dump(NN_error_results, f, indent=2)

    print('we are running sensitivity analysis on:', hyper)

    hyper_range = get_hyperparameter_range(hyper)

    for val in tqdm(hyper_range):
        # set hyperparameter to current value
        params = default_hyperparameters.copy()
        params['hyper'] = hyper  # pass hyperparameter name for logging
        params[hyper] = val
        print('SA VALUE:', val)
        PINN_errors, NN_errors, PINN_errors_std, NN_errors_std = run_multiple_runs(x,t,u_star,
                                                chose_obs_based_on_sensors=chose_obs_based_on_sensors,
                                                params=params,
                                                n_runs=n_runs)
        if hyper == 'N_f' or hyper == 'f_weight':
            # save only PINN results for N_f and f_weight sensitivity
            NN_errors = [-10.0]  # dummy value
            NN_errors_std = [-10.0]  # dummy value
        # mean L2 errors and stddev, mean AE stddev and stddev of stddevs
        PINN_error_results[val] = (np.mean(PINN_errors), np.std(PINN_errors), np.mean(PINN_errors_std), np.std(PINN_errors_std))
        NN_error_results[val] = (np.mean(NN_errors), np.std(NN_errors), np.mean(NN_errors_std), np.std(NN_errors_std))

    # save results to json files
    with open(f'results/sa_results/PINN_sa_{hyper}.json', 'w') as f:
        json.dump(PINN_error_results, f, indent=2)
    with open(f'results/sa_results/NN_sa_{hyper}.json', 'w') as f:
        json.dump(NN_error_results, f, indent=2)

    # plot results
    sa_vals = list(PINN_error_results.keys())
    PINN_means = [PINN_error_results[val][0] for val in sa_vals]
    PINN_stds = [PINN_error_results[val][1] for val in sa_vals]
    NN_means = [NN_error_results[val][0] for val in sa_vals]
    NN_stds = [NN_error_results[val][1] for val in sa_vals]
    plt.figure(figsize=(8,6))
    plt.errorbar(sa_vals, PINN_means, yerr=PINN_stds,
                 label='PINN', marker='o', capsize=5)
    if hyper != 'N_f' and hyper != 'f_weight':  # skip NN plot for N_f and f_weight sensitivity
        plt.errorbar(sa_vals, NN_means, yerr=NN_stds,
                    label='NN', marker='s', capsize=5)
    plt.xlabel(hyper, fontsize=16)
    plt.ylabel('Mean L2 Relative Error', fontsize=16)
    plt.title(f'Sensitivity Analysis on {hyper}', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/sa_results/SA_{hyper}.jpg")
    plt.show()

    