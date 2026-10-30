"""Constant-velocity Kalman filter over map-grid metres, with innovation gating.

State: position, velocity, and the bias of map fixes. Consecutive map fixes
share most of their error (attitude or camera mount error times height, and
NAIP's own registration error), so averaging many of them does not shrink
the error the way independent fixes would. Modelling that offset as a slowly
varying state keeps the reported accuracy honest. Odometry and ground updates
measure velocity, which the bias does not affect.
"""

from __future__ import annotations

import math

import numpy as np

CHI2_2DOF_999 = 13.8


class PositionFilter:
    def __init__(self, accel_sigma: float = 4.0, fix_bias_m: float = 3.0, fix_bias_tau_s: float = 30.0):
        """
        Args:
            accel_sigma: m/s^2, process noise of the constant-velocity model.
            fix_bias_m: 1-sigma of the error shared by consecutive map fixes.
            fix_bias_tau_s: s over which that error changes.
        """
        self.q = accel_sigma ** 2
        self.bias_sigma, self.bias_tau = fix_bias_m, fix_bias_tau_s
        self.x = None              # [x, y, vx, vy, bx, by] grid metres
        self.P = None
        self.t = None
        self.rejects = 0

    @property
    def initialised(self) -> bool:
        return self.x is not None

    def reset(self, x: float, y: float, sigma: float, t: float, v=(0.0, 0.0), v_sigma: float = 10.0):
        self.x = np.array([x, y, v[0], v[1], 0.0, 0.0], float)
        self.P = np.diag([sigma**2, sigma**2, v_sigma**2, v_sigma**2, self.bias_sigma**2, self.bias_sigma**2])
        self.t = t
        self.rejects = 0

    def predict(self, t: float):
        dt = t - self.t
        if dt <= 0:
            return
        a = math.exp(-dt / self.bias_tau)
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[4, 4] = F[5, 5] = a
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt], [0, 0], [0, 0]])
        Q = G @ G.T * self.q
        Q[4, 4] = Q[5, 5] = self.bias_sigma**2 * (1 - a * a)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, x: float, y: float, sigma: float, t: float, gate: bool = True) -> bool:
        """Map fix (position plus fix bias); returns False if gated out as an outlier."""
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = H[0, 4] = H[1, 5] = 1
        if not self._update(np.array([x, y]), H, sigma, t, gate):
            self.rejects += 1
            return False
        self.rejects = 0
        return True

    def update_velocity(self, vx: float, vy: float, sigma: float, t: float, gate: bool = True) -> bool:
        """Velocity measurement (odometry, or zero on the ground); False if gated out."""
        H = np.zeros((2, 6))
        H[0, 2] = H[1, 3] = 1
        return self._update(np.array([vx, vy]), H, sigma, t, gate)

    def _update(self, z: np.ndarray, H: np.ndarray, sigma: float, t: float, gate: bool) -> bool:
        self.predict(t)
        innov = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * sigma**2
        if gate and innov @ np.linalg.solve(S, innov) > CHI2_2DOF_999:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(6) - K @ H) @ self.P
        return True

    def pos_sigma(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))

    def vel_sigma(self) -> float:
        return float(np.sqrt(max(self.P[2, 2], self.P[3, 3])))
