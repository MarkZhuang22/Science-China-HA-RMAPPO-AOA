"""
Centralized Particle Filter (CPF) utilities for bearing-only tracking under
Constant-Acceleration (CA) motion. Implements per-intruder particle predict
and AoA-based update with distance-dependent noise.
"""
from __future__ import annotations
import numpy as np


def wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def aoa_variance(distance: float, sigma0: float, r0: float) -> float:
    """Distance-dependent AoA variance model.
    For r > r0: sigma^2 = sigma0^2 * (r/r0)^2; else sigma^2 = sigma0^2.
    """
    if distance <= r0:
        return sigma0 ** 2
    return sigma0 ** 2 * (distance / max(r0, 1e-8)) ** 2


def F_CA_dt(dt: float) -> np.ndarray:
    """Single-step CA transition matrix (6x6) for a time step dt."""
    T = dt
    F = np.array([
        [1, T, 0.5 * T ** 2, 0, 0, 0],
        [0, 1, T, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, T, 0.5 * T ** 2],
        [0, 0, 0, 0, 1, T],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    return F


def Q_CA_dt(dt: float, sigma_a: float) -> np.ndarray:
    """Single-step CA process noise covariance (6x6) for time step dt and accel std sigma_a."""
    T = dt
    s2 = sigma_a ** 2
    Q_blk = np.array([
        [T ** 4 / 4.0, T ** 3 / 2.0, T ** 2 / 2.0],
        [T ** 3 / 2.0, T ** 2, T],
        [T ** 2 / 2.0, T, 1.0],
    ], dtype=np.float32) * s2
    Q = np.zeros((6, 6), dtype=np.float32)
    Q[0:3, 0:3] = Q_blk
    Q[3:6, 3:6] = Q_blk
    return Q


def predict_particles(X: np.ndarray, F: np.ndarray, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Particle prediction: X <- F X + w, w ~ N(0, Q). X shape: (Np, 6)."""
    Np = X.shape[0]
    # Sample process noise per particle
    w = rng.multivariate_normal(mean=np.zeros(6, dtype=np.float32), cov=Q, size=Np).astype(np.float32)
    X_pred = (X @ F.T).astype(np.float32) + w
    return X_pred


def update_weights_bearing(
    X: np.ndarray,
    w: np.ndarray,
    defender_positions: list[np.ndarray],
    theta_meas: list[float],
    theta_var: list[float],
) -> np.ndarray:
    """AoA update: accumulate log-likelihoods for each assigned defender and update weights.
    - X: particles (Np,6)
    - w: weights (Np,)
    - defender_positions: list of (2,) arrays
    - theta_meas: list of measured bearings (rad)
    - theta_var: list of variances
    Returns updated normalized weights (Np,).
    """
    if len(defender_positions) == 0:
        return w
    px = X[:, 0]
    py = X[:, 3]
    logw = np.log(np.clip(w, 1e-30, 1.0)).astype(np.float32)
    for dpos, th_m, var in zip(defender_positions, theta_meas, theta_var):
        dx = px - dpos[0]
        dy = py - dpos[1]
        th_hat = np.arctan2(dy, dx)
        innov = wrap_angle(th_m - th_hat)
        inv_var = 1.0 / max(var, 1e-12)
        # log-likelihood under N(0,var)
        loglik = -0.5 * (innov ** 2) * inv_var - 0.5 * (np.log(2.0 * np.pi) + np.log(var))
        logw += loglik.astype(np.float32)
    # Normalize weights
    m = np.max(logw)
    w_new = np.exp(logw - m)
    w_new /= np.sum(w_new)
    return w_new.astype(np.float32)


def ess(weights: np.ndarray) -> float:
    return 1.0 / float(np.sum(np.square(weights)))


def resample_systematic(X: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Systematic resampling."""
    N = len(w)
    positions = (rng.random() + np.arange(N)) / N
    cumulative_sum = np.cumsum(w)
    cumulative_sum[-1] = 1.0
    idx = np.searchsorted(cumulative_sum, positions)
    X_new = X[idx]
    w_new = np.full(N, 1.0 / N, dtype=np.float32)
    return X_new, w_new


def weighted_mean_and_cov(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted mean (6,) and covariance (6,6)."""
    mu = np.sum(X * w[:, None], axis=0)
    Xm = X - mu
    C = (Xm * w[:, None]).T @ Xm
    return mu.astype(np.float32), C.astype(np.float32)

