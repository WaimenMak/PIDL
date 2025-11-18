"""
Extended Kalman Filter (EKF) for traffic speed estimation.

Implements state estimation for traffic flow using:
- State: speed field u(x,t) at discrete spatial locations
- Dynamics: Linearized traffic flow PDE (LWR equation with fundamental diagram)
- Measurements: Sparse sensor observations

The EKF operates in two steps:
1. Forecast: Propagate state and covariance using linearized traffic model
2. Analysis: Update state and covariance with sensor measurements
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Callable
from scipy import linalg
import warnings


class TrafficEKF:
    """
    Extended Kalman Filter for traffic speed estimation using traffic flow PDEs.
    
    Implements the LWR (Lighthill-Whitham-Richards) equation with various
    fundamental diagrams, linearized for EKF propagation.
    """
    
    def __init__(
        self,
        x_locations: np.ndarray,      # Spatial locations [km]
        dt: float,                     # Time step [hours]
        V_f: float = 100.0,           # Free-flow speed [km/h]
        fd_name: str = 'linear',      # Fundamental diagram type
        process_noise_std: float = 5.0,     # Process noise [km/h]
        measurement_noise_std: float = 2.0,  # Measurement noise [km/h]
        initial_state_std: float = 10.0,     # Initial uncertainty [km/h]
        rho_max: float = 180.0,       # Maximum density [veh/km]
        use_flux_limiter: bool = True,
        flux_limiter: str = 'minmod',
        cfl_factor: float = 0.5,
    ):
        """
        Initialize EKF for traffic speed estimation.
        
        Args:
            x_locations: Array of spatial locations [km], shape (N,)
            dt: Time step [hours]
            V_f: Free-flow speed [km/h]
            fd_name: Fundamental diagram type ('linear', 'log', 'exp')
            process_noise_std: Process noise standard deviation [km/h]
            measurement_noise_std: Measurement noise standard deviation [km/h]
            initial_state_std: Initial state uncertainty [km/h]
            rho_max: Maximum density [veh/km]
            use_flux_limiter: Use flux limiter for numerical stability
            flux_limiter: Type of flux limiter ('minmod', 'superbee', 'vanleer')
            cfl_factor: CFL condition factor for stability check
        """
        self.x_locations = np.asarray(x_locations).flatten()
        self.N = len(self.x_locations)  # Number of spatial locations
        self.dt = dt
        self.V_f = V_f
        self.fd_name = fd_name.lower()
        self.rho_max = rho_max
        self.use_flux_limiter = use_flux_limiter
        self.flux_limiter = flux_limiter
        self.cfl_factor = cfl_factor
        
        # Compute spatial step sizes (non-uniform grid support)
        self.dx = np.diff(self.x_locations)
        if len(self.dx) == 0:
            self.dx = np.array([1.0])  # Single location case
        self.dx_mean = np.mean(self.dx)
        
        # State vector: speeds at all locations
        self.state = np.zeros(self.N)  # u(x) [km/h]
        
        # Covariance matrix
        self.P = np.eye(self.N) * (initial_state_std ** 2)
        
        # Process noise covariance Q
        self.Q = np.eye(self.N) * (process_noise_std ** 2)
        
        # Measurement noise variance
        self.R_var = measurement_noise_std ** 2
        
        # Check CFL condition
        self._check_cfl_condition()
        
    def _check_cfl_condition(self):
        """Check CFL stability condition for explicit time integration."""
        max_wave_speed = 2.0 * self.V_f  # Maximum characteristic speed
        cfl = self.dt * max_wave_speed / self.dx_mean
        if cfl > self.cfl_factor:
            warnings.warn(
                f"CFL condition may be violated: CFL={cfl:.3f} > {self.cfl_factor}. "
                f"Consider reducing dt or increasing dx.",
                UserWarning
            )
    
    def initialize_state(self, u_init: np.ndarray, P_init: Optional[np.ndarray] = None):
        """
        Initialize state and covariance.
        
        Args:
            u_init: Initial speed field, shape (N,)
            P_init: Initial covariance matrix, shape (N, N). If None, uses default.
        """
        self.state = np.asarray(u_init).flatten().copy()
        if P_init is not None:
            self.P = P_init.copy()
    
    def _compute_wave_speed(self, u: np.ndarray) -> np.ndarray:
        """
        Compute characteristic wave speed c(u) = dq/drho for given speeds.
        
        For LWR equation: ∂ρ/∂t + ∂q/∂x = 0
        We work in speed space, so we need: ∂u/∂t + c(u) ∂u/∂x = 0
        where c(u) depends on the fundamental diagram.
        
        Args:
            u: Speed values [km/h], shape (N,)
            
        Returns:
            Wave speeds c(u) [km/h], shape (N,)
        """
        V_f = self.V_f
        u_safe = np.clip(u, 1e-3, V_f - 1e-3)  # Avoid division by zero
        
        if self.fd_name == 'linear':
            # Greenshields: q = u*ρ, ρ = ρ_max(1 - u/V_f)
            # c = dq/dρ = V_f - 2*u
            c = V_f - 2.0 * u_safe
            
        elif self.fd_name == 'log':
            # Greenberg: ρ = ρ_max * ln(V_f/u)
            # c = u - V_f
            c = u_safe - V_f
            
        elif self.fd_name == 'exp':
            # Underwood: u = V_f * exp(-ρ/ρ_max)
            # c = u * (ln(V_f/u) - 1)
            c = u_safe * (np.log(V_f / u_safe) - 1.0)
            
        else:
            raise ValueError(f"Unknown fundamental diagram: {self.fd_name}")
        
        return c
    
    def _flux_limiter_func(self, r: np.ndarray) -> np.ndarray:
        """
        Apply flux limiter function for TVD schemes.
        
        Args:
            r: Ratio of consecutive gradients
            
        Returns:
            Limiter values φ(r)
        """
        if self.flux_limiter == 'minmod':
            return np.maximum(0, np.minimum(1, r))
        elif self.flux_limiter == 'superbee':
            return np.maximum(0, np.maximum(np.minimum(2*r, 1), np.minimum(r, 2)))
        elif self.flux_limiter == 'vanleer':
            return (r + np.abs(r)) / (1 + np.abs(r))
        else:
            return np.ones_like(r)  # No limiting
    
    def _forecast_state(self, u: np.ndarray) -> np.ndarray:
        """
        Forecast state forward one time step using finite difference.
        
        Implements: u^{n+1} = u^n - dt * c(u) * ∂u/∂x
        
        Args:
            u: Current state [km/h], shape (N,)
            
        Returns:
            Forecasted state [km/h], shape (N,)
        """
        u_new = u.copy()
        c = self._compute_wave_speed(u)
        
        # First-order upwind scheme with flux limiting
        for i in range(1, self.N - 1):
            dx_i = self.x_locations[i] - self.x_locations[i-1]
            dx_ip1 = self.x_locations[i+1] - self.x_locations[i]
            
            if c[i] > 0:
                # Use backward difference
                du_dx = (u[i] - u[i-1]) / dx_i
                
                if self.use_flux_limiter and i > 1:
                    dx_im1 = self.x_locations[i-1] - self.x_locations[i-2]
                    du_dx_up = (u[i-1] - u[i-2]) / dx_im1
                    r = du_dx_up / (du_dx + 1e-10)
                    phi = self._flux_limiter_func(r)
                    du_dx = du_dx * (1 + phi) / 2
                    
            else:
                # Use forward difference
                du_dx = (u[i+1] - u[i]) / dx_ip1
                
                if self.use_flux_limiter and i < self.N - 2:
                    dx_ip2 = self.x_locations[i+2] - self.x_locations[i+1]
                    du_dx_down = (u[i+2] - u[i+1]) / dx_ip2
                    r = du_dx_down / (du_dx + 1e-10)
                    phi = self._flux_limiter_func(r)
                    du_dx = du_dx * (1 + phi) / 2
            
            u_new[i] = u[i] - self.dt * c[i] * du_dx
        
        # Boundary conditions: zero gradient (Neumann)
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]
        
        # Physical constraints
        u_new = np.clip(u_new, 1.0, self.V_f)
        
        return u_new
    
    def _compute_jacobian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian F of the forecast model: ∂f/∂u
        
        Linearizes: u^{n+1} = f(u^n)
        F = ∂f/∂u = I - dt * (∂c/∂u * ∂u/∂x + c * ∂²u/∂u∂x)
        
        For simplicity, we use finite difference approximation of F.
        
        Args:
            u: State vector [km/h], shape (N,)
            
        Returns:
            Jacobian matrix F, shape (N, N)
        """
        eps = 1e-5
        F = np.zeros((self.N, self.N))
        
        # Compute Jacobian by finite differences
        for j in range(self.N):
            u_perturb = u.copy()
            u_perturb[j] += eps
            f_plus = self._forecast_state(u_perturb)
            f_center = self._forecast_state(u)
            F[:, j] = (f_plus - f_center) / eps
        
        return F
    
    def forecast(self):
        """
        EKF forecast step: propagate state and covariance forward in time.
        
        Updates:
            self.state: forecasted state
            self.P: forecasted covariance
        """
        # Forecast state
        u_forecast = self._forecast_state(self.state)
        
        # Compute Jacobian
        F = self._compute_jacobian(self.state)
        
        # Forecast covariance: P^- = F * P * F^T + Q
        P_forecast = F @ self.P @ F.T + self.Q
        
        # Ensure symmetry
        P_forecast = 0.5 * (P_forecast + P_forecast.T)
        
        # Update state and covariance
        self.state = u_forecast
        self.P = P_forecast
    
    def analysis(
        self, 
        sensor_indices: np.ndarray, 
        measurements: np.ndarray,
        measurement_noise_std: Optional[float] = None
    ):
        """
        EKF analysis step: update state and covariance with measurements.
        
        Args:
            sensor_indices: Indices of sensors with measurements, shape (M,)
            measurements: Measured speeds [km/h], shape (M,)
            measurement_noise_std: Override default measurement noise std
        """
        M = len(sensor_indices)
        if M == 0:
            return  # No measurements
        
        # Observation operator H (linear): H[i, sensor_indices[i]] = 1
        H = np.zeros((M, self.N))
        H[np.arange(M), sensor_indices] = 1.0
        
        # Measurement noise covariance R
        if measurement_noise_std is not None:
            R = np.eye(M) * (measurement_noise_std ** 2)
        else:
            R = np.eye(M) * self.R_var
        
        # Innovation: y = z - H * x^-
        innovation = measurements - H @ self.state
        
        # Innovation covariance: S = H * P^- * H^T + R
        S = H @ self.P @ H.T + R
        
        # Kalman gain: K = P^- * H^T * S^{-1}
        try:
            K = self.P @ H.T @ linalg.inv(S)
        except linalg.LinAlgError:
            # Singular S, use pseudo-inverse
            K = self.P @ H.T @ linalg.pinv(S)
        
        # Update state: x^+ = x^- + K * y
        self.state = self.state + K @ innovation
        
        # Update covariance: P^+ = (I - K * H) * P^-
        # Using Joseph form for numerical stability (optional)
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        
        # Physical constraints on state
        self.state = np.clip(self.state, 1.0, self.V_f)
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        """
        Predict state n_steps ahead without updating covariance.
        
        Args:
            n_steps: Number of time steps to predict
            
        Returns:
            Predicted state, shape (N,)
        """
        u_pred = self.state.copy()
        for _ in range(n_steps):
            u_pred = self._forecast_state(u_pred)
        return u_pred
    
    def get_state_std(self) -> np.ndarray:
        """
        Get standard deviation of state estimate (uncertainty).
        
        Returns:
            Standard deviations [km/h], shape (N,)
        """
        return np.sqrt(np.diag(self.P))
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state and uncertainty.
        
        Returns:
            state: Speed estimates [km/h], shape (N,)
            std: Standard deviations [km/h], shape (N,)
        """
        return self.state.copy(), self.get_state_std()


def run_ekf_estimation(
    u_true: np.ndarray,           # True speed field [time, space]
    x_locations: np.ndarray,      # Spatial locations
    sensor_indices: np.ndarray,   # Sensor locations (indices)
    dt_data: float,               # Data sampling time step [hours]
    ekf_params: dict,
    log_every: int = 20,          # Print progress every N time steps
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run EKF estimation over time sequence.
    
    Args:
        u_true: True speed field [km/h], shape (T, N)
        x_locations: Spatial locations [km], shape (N,)
        sensor_indices: Indices of sensor locations, shape (M,)
        dt_data: Data sampling time step [hours] (e.g., 0.25 for 15 min)
        ekf_params: Dictionary of EKF parameters (must include 'dt' for internal time step)
        
    Returns:
        u_est: Estimated speed field, shape (T, N)
        u_std: Uncertainty (std dev), shape (T, N)
        u_sensors: Sensor measurements, shape (T, N) with NaN for non-sensors
    """
    T, N = u_true.shape
    M = len(sensor_indices)
    
    # Internal EKF time step
    dt_ekf = ekf_params.pop('dt', dt_data)  # Use internal dt if provided
    n_substeps = max(1, int(np.round(dt_data / dt_ekf)))  # How many EKF steps per data step
    dt_ekf = dt_data / n_substeps  # Adjust to be exact divisor
    
    # Initialize EKF
    ekf = TrafficEKF(
        x_locations=x_locations,
        dt=dt_ekf,
        **ekf_params
    )
    
    # Initialize with spatial mean
    u_init = np.full(N, np.nanmean(u_true[0]))
    ekf.initialize_state(u_init)
    
    # Storage
    u_est = np.zeros((T, N))
    u_std = np.zeros((T, N))
    u_sensors = np.full((T, N), np.nan)
    
    # Run filter over time
    for t in range(T):
        # Run multiple forecast steps between measurements
        for _ in range(n_substeps):
            if t > 0 or _ > 0:  # Skip first step at t=0
                ekf.forecast()
        
        # Get measurements from sensors
        measurements = u_true[t, sensor_indices]
        u_sensors[t, sensor_indices] = measurements
        
        # Analysis step (update with measurements)
        ekf.analysis(sensor_indices, measurements)
        
        # Store results
        state, std = ekf.get_state()
        u_est[t] = state
        u_std[t] = std
        
        # Progress logging (lightweight)
        if log_every > 0 and (t + 1) % log_every == 0:
            # Compute MSE for completed time steps
            mse = np.mean((u_true[:t+1] - u_est[:t+1]) ** 2)
            pct = 100 * (t + 1) / T
            print(f"    Progress: {t+1}/{T} ({pct:.1f}%) | MSE: {mse:.2f} (km/h)²")
    
    return u_est, u_std, u_sensors
