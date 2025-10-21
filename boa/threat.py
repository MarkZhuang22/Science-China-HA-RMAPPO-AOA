"""
Threat metric utilities: look-ahead of CA state to compute earliest
ellipse–circle intersection time and threat weights.
"""
from __future__ import annotations
import numpy as np


def F_CA_tau(tau: float) -> np.ndarray:
    """CA transition matrix F(τ) for continuous look-ahead τ seconds (6x6)."""
    T = tau
    F = np.array([
        [1, T, 0.5 * T ** 2, 0, 0, 0],
        [0, 1, T, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, T, 0.5 * T ** 2],
        [0, 0, 0, 0, 1, T],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    return F


def Q_CA_tau(tau: float, sigma_a: float) -> np.ndarray:
    """CA process noise covariance Q(τ) for look-ahead τ and acceleration std sigma_a."""
    T = tau
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


def J_p() -> np.ndarray:
    """Selector for position components (2x6)."""
    return np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0]], dtype=np.float32)


def predict_pos_mean(mu_x: np.ndarray, tau: float) -> np.ndarray:
    """Predict position mean at look-ahead τ using CA kinematics."""
    px, vx, ax, py, vy, ay = mu_x
    T = tau
    mu = np.array([
        px + vx * T + 0.5 * ax * T ** 2,
        py + vy * T + 0.5 * ay * T ** 2,
    ], dtype=np.float32)
    return mu


def predict_pos_cov(Px: np.ndarray, tau: float, sigma_a: float) -> np.ndarray:
    """Predict 2x2 position covariance at look-ahead τ from full 6x6 Px."""
    F = F_CA_tau(tau)
    Q = Q_CA_tau(tau, sigma_a)
    P_tau = F @ Px @ F.T + Q
    J = J_p()
    Pp = J @ P_tau @ J.T
    return Pp.astype(np.float32)


def earliest_collision_time(
    mu_x: np.ndarray,
    Px: np.ndarray,
    zone_center: np.ndarray,
    zone_radius: float,
    sigma_a: float,
    tau_max: float,
    tau_step: float,
) -> float:
    """Earliest τ in (0, tau_max] such that ||p_a - μ_p(τ)|| - r_zone - ρ(τ) <= 0,
    with ρ(τ) = 3 sqrt(λ_max(Pp(τ))). Uses scanning to bracket and bisection refine.
    Returns tau_max if no root is found.
    """
    def d_at(t: float) -> float:
        mu = predict_pos_mean(mu_x, t)
        Pp = predict_pos_cov(Px, t, sigma_a)
        # largest eigenvalue of 2x2 cov
        lam_max = float(np.max(np.linalg.eigvalsh(Pp)))
        rho = 3.0 * np.sqrt(max(lam_max, 0.0))
        return float(np.linalg.norm(zone_center - mu) - zone_radius - rho)

    # scan to find bracket
    t_prev = 0.0
    d_prev = d_at(t_prev)
    t = tau_step
    while t <= tau_max + 1e-8:
        d_cur = d_at(t)
        if d_prev > 0.0 and d_cur <= 0.0:
            # bisection
            lo, hi = t_prev, t
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                dm = d_at(mid)
                if dm <= 0.0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        t_prev, d_prev = t, d_cur
        t += tau_step
    return tau_max


def threat_weight(tau: float, lam: float) -> float:
    return float(np.exp(-lam * tau))

