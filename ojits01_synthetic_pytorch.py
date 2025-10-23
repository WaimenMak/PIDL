"""
@author: Archie Huang
Built upon Dr. Maziar Raissi's PINNs - https://github.com/maziarraissi/PINNs

PyTorch version of ojits01_synthetic.py
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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

se = 1
np.random.seed(se)
torch.manual_seed(se)

# PINN Class
class PhysicsInformedNN(nn.Module):
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        super(PhysicsInformedNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(self.device)

        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32).to(self.device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32).to(self.device)
        self.u = torch.tensor(u, dtype=torch.float32).to(self.device)
        
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device)

        self.layers = layers
        self.model = self.initialize_NN(layers).to(self.device)
        
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(), 
            max_iter=1000,
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

        f = 0.15 * u_x - 0.012 * u * u_x - 0.006 * u_t
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
        
        x_f_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        t_f_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device)
        f_star = self.net_f(x_f_star, t_f_star)
        
        return u_star.cpu().detach().numpy(), f_star.cpu().detach().numpy()

# Regular NN class
class NN(nn.Module):
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        super(NN, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(self.device)

        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32).to(self.device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32).to(self.device)
        self.u = torch.tensor(u, dtype=torch.float32).to(self.device)
        
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
        f = 0.15 * u_x - 0.012 * u * u_x - 0.006 * u_t
        return f

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
        
        x_f_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        t_f_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device)
        f_star = self.net_f(x_f_star, t_f_star)
        
        return u_star.cpu().detach().numpy(), f_star.cpu().detach().numpy()


if __name__ == "__main__":

    N_u = 1000
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    data = scipy.io.loadmat('data/synthetic.mat')
    t = data['tScale'].T.flatten()[:,None]
    x = data['xScale'].T.flatten()[:,None]
    Exact = np.real(data['v'])
    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    lb = X_star.min(0).astype(np.float32)
    ub = X_star.max(0).astype(np.float32)

    ################################# CAV Plot ####################################
    N_cav = 30
    cav = np.real(data['N'])
    idx_cav = np.sort(random.sample(range(int(cav.min()), int(cav.max())), N_cav))
    contr = plt.contour(X, T, cav, idx_cav)
    plt.title('CAV Trajectories', fontsize = 16)
    plt.ylabel('Time $t$ (s)', fontsize = 14)
    plt.xlabel('Location $x$ (m)', fontsize=14)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/cav{}_trajectory.pdf'.format(N_cav))
    plt.savefig('figures/cav{}_trajectory.eps'.format(N_cav))
    # plt.show()
    ################################# CAV Plot ####################################

    ################################# CAV Samples #################################
    X_u_train = contr.collections[0].get_paths()[0].vertices
    for a in range(1, N_cav):
        path = contr.collections[a].get_paths()[0]
        ver = path.vertices
        X_u_train = np.vstack([X_u_train, ver])
    print("Total Trajectory Data Points: ", len(X_u_train), '\n')
    ################################# CAV Samples #################################

    ################################# Filters #####################################
    # RSU communication range and locations
    comm_range = 300
    loca = [0, 1000, 2000, 3000, 4000, 5000]

    mask = ((X_u_train[:, 0] < loca[0] + comm_range) |
            ((loca[1] - comm_range < X_u_train[:, 0]) & (X_u_train[:, 0] < loca[1] + comm_range)) |
            ((loca[2] - comm_range < X_u_train[:, 0]) & (X_u_train[:, 0] < loca[2] + comm_range)) |
            ((loca[3] - comm_range < X_u_train[:, 0]) & (X_u_train[:, 0] < loca[3] + comm_range)) |
            ((loca[4] - comm_range < X_u_train[:, 0]) & (X_u_train[:, 0] < loca[4] + comm_range)) |
            (loca[5] - comm_range < X_u_train[:, 0]))
    X_u_train = X_u_train[mask]

    X_u_train = np.rint(X_u_train)
    X_u_train[:, 0] = np.rint(X_u_train[:, 0] / 5) * 5  # fit for 5-meter bins
    X_u_train = np.unique(X_u_train, axis=0).astype(int)
    print("Total RSU Data Points: ", len(X_u_train), '\n')
    ################################# Filters #####################################

    ############################### Training Data #################################
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.empty([N_u, 1])
    for i in range(0, N_u):
        u_train[i] = Exact[X_u_train[i][1]][int(X_u_train[i][0] / 5)]
    ############################### Training Data #################################

    # PINN Model
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
    start_time = time.time()
    model.train_model()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)
    u_pred, f_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % error_u)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)

    # Regular NN Model
    model2 = NN(X_u_train, u_train, X_f_train, layers, lb, ub)
    start_time2 = time.time()
    model2.train_model()
    elapsed2 = time.time() - start_time2
    print('Training time: %.4f' % elapsed2)
    u_pred2, f_pred2 = model2.predict(X_star)
    error_u2 = np.linalg.norm(u_star-u_pred2,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % error_u2)
    U_pred2 = griddata(X_star, u_pred2.flatten(), (X, T), method='cubic')
    Error2 = np.abs(Exact - U_pred2)

    ################################# Plot #################################
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
        ]
    }
    # mpl.rcParams.update(pgf_with_latex)
    # fig = plt.figure(figsize=(8, 6.5))
    # mpl.rcParams.update(pgf_with_latex)
    fig = plt.figure(figsize=(10, 18))

    ####### Row 0: Ground Truth ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.96, bottom=0.70, left=0.15, right=0.85, wspace=0)

    ax = plt.subplot(gs0[:, :])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_locator(MultipleLocator(60))
    h = ax.imshow(Exact, interpolation='nearest', cmap='RdBu',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=20)
    fig.colorbar(h, cax=cax, ticks=[5, 10, 15, 20])

    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax.set_ylabel('Time $t$ (s)', fontsize=20)
    ax.set_xlabel('Location $x$ (m)', fontsize=20)
    ax.legend(frameon=False, loc='best', fontsize=20)
    ax.set_title('Ground Truth $v(x,t)$ (m/s)', fontsize=20)

    ####### Row 1: PIDL: u(t,x) ##################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.64, bottom=0.38, left=0.15, right=0.85, wspace=0)

    ax = plt.subplot(gs1[:, :])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_locator(MultipleLocator(60))
    h = ax.imshow(U_pred, interpolation='nearest', cmap='RdBu',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=20)
    fig.colorbar(h, cax=cax, ticks=[5, 10, 15, 20])

    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax.set_ylabel('Time $t$ (s)', fontsize=20)
    ax.set_xlabel('Location $x$ (m)', fontsize=20)
    ax.legend(frameon=False, loc='best', fontsize=20)
    ax.set_title('PIDL Estimation $v (x,t)$ (m/s)', fontsize=20)

    ####### Row 2: DL: u(t,x) ##################
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=0.32, bottom=0.06, left=0.15, right=0.85, wspace=0)

    ax = plt.subplot(gs2[:, :])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_locator(MultipleLocator(60))
    h = ax.imshow(U_pred2, interpolation='nearest', cmap='RdBu',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=20)
    fig.colorbar(h, cax=cax, ticks=[5, 10, 15, 20])

    ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax.set_ylabel('Time $t$ (s)', fontsize=20)
    ax.set_xlabel('Location $x$ (m)', fontsize=20)
    ax.legend(frameon=False, loc='best', fontsize=20)
    ax.set_title('DL Estimation $v (x,t)$ (m/s)', fontsize=20)
    plt.savefig('figures/cav{}_pidl_dl_pytorch.pdf'.format(N_cav))
    plt.savefig('figures/cav{}_pidl_dl_pytorch.eps'.format(N_cav))
    plt.show()
    ################################# Plot #################################