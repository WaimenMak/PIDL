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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# imported modules for PINN and NN models
from ojits03_a13_pytorch import *

se = 25
np.random.seed(se)
torch.manual_seed(se)

# one run 
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
    def run_once(N_u, N_f, u_obs_noise):

        print("\n" + "#"*60)
        print(f"Running sensitivity analysis with N_u={N_u}, N_f={N_f}, noise={u_obs_noise*100:.1f}%")
        print("#"*60)
        ############################### Training Data #################################
        # Use only valid data points for training
        valid_train_mask = u_star.flatten() > 0
        valid_indices = np.where(valid_train_mask)[0]
        
        # Sample from valid points
        n_valid = min(N_u, len(valid_indices))
        idx = np.random.choice(valid_indices, n_valid, replace=False)
        
        X_u_train = X_star[idx, :]
        u_train = u_star[idx, :] + np.random.normal(0, u_obs_noise, size=u_star[idx, :].shape)  # Add noise to training data
        X_f_train = lb + (ub - lb) * lhs(2, N_f)
        X_f_train = np.vstack((X_f_train, X_u_train))
        print(f"\nTraining with {n_valid} data points")
        ############################### Training PINN & NN Models #################################

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

        sa_dict = {
                    'N_u': N_u,
                    'N_f': N_f,
                    'u_obs_noise': u_obs_noise
                }
        
        plot_results(X_u_train, U_pred, error_u, U_pred2, error_u2, sa_dict)

        return [error_u, error_u2]

    def plot_results(X_u_train, U_pred, error_u, U_pred2, error_u2, sa_dict):
        ################################# Plot #################################
        print("\n" + "="*60)
        print("Generating plots...")
        print("="*60)
        
        fig = plt.figure(figsize=(12, 16))

        ####### Row 0: Ground Truth ##################
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=0.96, bottom=0.70, left=0.15, right=0.85, wspace=1)

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
        
        ####### Row 1: PIDL: u(t,x) ##################
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(top=0.60, bottom=0.34, left=0.15, right=0.85, wspace=1)

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
        gs2.update(top=0.30, bottom=0.0, left=0.15, right=0.85, wspace=1)

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
        plt.savefig(f"figures/sa_figures/a13_pidl_dl_{sa_dict['N_u']}_{sa_dict['N_f']}_{sa_dict['u_obs_noise']}.pdf")
        plt.savefig(f"figures/sa_figures/a13_pidl_dl_{sa_dict['N_u']}_{sa_dict['N_f']}_{sa_dict['u_obs_noise']}.eps")
        plt.show()

        print(f"\nPlots saved to figures/sa_figures/a13_pidl_dl_{sa_dict['N_u']}_{sa_dict['N_f']}_{sa_dict['u_obs_noise']}.pdf/eps")
        print("="*60)

    ########### Sensitivity Analysis Execution ###########
    # init: N_u = 800, N_f = 10000, u_obs_noise = 0.0
    N_u_lst = [int(0.1 * u_star.shape[0]), int(0.2 * u_star.shape[0]), int(0.3 * u_star.shape[0])]  # 10%, 20%, 30%
    N_f_lst = [3000, 6000, 10000]  # Example collocation points
    u_obs_noise_lst = [0.0, 0.05, 0.1, 0.2]  # Example: 0%, 5%, 10%, 20% noise

    PINN_error_results = []
    NN_error_results = []

    for N_u in tqdm(N_u_lst):
        for N_f in tqdm(N_f_lst):
            for u_obs_noise in tqdm(u_obs_noise_lst):
                PINN_error, NN_error = run_once(N_u, N_f, u_obs_noise)
                PINN_error_results.append(PINN_error)
                NN_error_results.append(NN_error)
    
    print(PINN_error_results)
    print(NN_error_results)

    # Save results to CSV
    results_df = pd.DataFrame({
        'N_u': np.repeat(N_u_lst, len(N_f_lst) * len(u_obs_noise_lst)),
        'N_f': np.tile(np.repeat(N_f_lst, len(u_obs_noise_lst)), len(N_u_lst)),
        'u_obs_noise': np.tile(u_obs_noise_lst, len(N_u_lst) * len(N_f_lst)),
        'PINN_error': PINN_error_results,
        'NN_error': NN_error_results
    })

    if not os.path.exists('Results'):
        os.makedirs('Results')
    results_df.to_csv('Results/sa_results_a13.csv', index=False)
    print("\nSensitivity analysis results saved to sa_results_a13.csv")

    PINN_ary = np.array(PINN_error_results)
    NN_ary = np.array(NN_error_results)

    # SAVE ARRAYS
    np.save('Results/sa_pinn_errors_a13.npy', PINN_ary)
    np.save('Results/sa_nn_errors_a13.npy', NN_ary)