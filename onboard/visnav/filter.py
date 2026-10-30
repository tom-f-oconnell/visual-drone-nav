"""Constant-velocity Kalman filter over map-grid metres, with innovation gating."""

from __future__ import annotations

import numpy as np

CHI2_2DOF_999 = 13.8


class PositionFilter:
    def __init__(self, accel_sigma: float = 4.0):
        self.q = accel_sigma ** 2
        self.x = None              # [x, y, vx, vy] grid metres
        self.P = None
        self.t = None
        self.rejects = 0

    @property
    def initialised(self) -> bool:
        return self.x is not None

    def reset(self, x: float, y: float, sigma: float, t: float, v=(0.0, 0.0), v_sigma: float = 10.0):
        self.x = np.array([x, y, v[0], v[1]], float)
        self.P = np.diag([sigma**2, sigma**2, v_sigma**2, v_sigma**2])
        self.t = t
        self.rejects = 0

    def predict(self, t: float):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q
        self.t = t

    def update(self, x: float, y: float, sigma: float, t: float, gate: bool = True) -> bool:
        self.predict(t)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1
        R = np.eye(2) * sigma**2
        innov = np.array([x, y]) - H @ self.x
        S = H @ self.P @ H.T + R
        if gate and innov @ np.linalg.solve(S, innov) > CHI2_2DOF_999:
            self.rejects += 1
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ H) @ self.P
        self.rejects = 0
        return True

    def pos_sigma(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))

    def vel_sigma(self) -> float:
        return float(np.sqrt(max(self.P[2, 2], self.P[3, 3])))
