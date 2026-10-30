"""Frame-to-frame visual odometry from the downward camera.

Each frame is warped to a north-up patch of the ground (attitude and height
remove rotation and scale), and the shift between consecutive patches is found
by phase correlation. That measures horizontal motion at any height where the
camera sees texture, so it covers takeoff, landing and the gaps between map
fixes, where matching against NAIP cannot work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .ortho import CameraModel, ground_patch


@dataclass
class Motion:
    d_north: float   # metres the point below the camera moved since the previous frame
    d_east: float
    dt: float
    sigma: float     # 1-sigma of each component, metres
    response: float  # phase correlation peak, 0 to 1


class FrameOdometry:
    def __init__(self, cam: CameraModel, size: int = 256, min_response: float = 0.1, max_dt: float = 1.0,
                 px_sigma: float = 0.3, height_rel_sigma: float = 0.05, att_sigma_deg: float = 0.15):
        """
        Args:
            size: patch side in pixels. The patch covers about half the
                shorter side of the camera footprint, whatever the height.
            min_response: phase correlation peaks below this are rejected.
            max_dt: frames further apart than this (s) are not compared.
            px_sigma: shift precision in patch pixels.
            height_rel_sigma: relative error of height and heading together,
                which scale and rotate the measured displacement.
            att_sigma_deg: change in attitude error between consecutive
                frames, which moves the computed point below the camera.
        """
        self.cam, self.size = cam, size
        self.min_response, self.max_dt = min_response, max_dt
        self.px_sigma, self.height_rel_sigma = px_sigma, height_rel_sigma
        self.att_sigma = math.radians(att_sigma_deg)
        self.window = cv2.createHanningWindow((size, size), cv2.CV_32F)
        self.prev = None

    def reset(self):
        self.prev = None

    def gsd(self, h: float) -> float:
        """Patch metres per pixel at height h."""
        short_side = 2 * h * min(self.cam.width / 2 / self.cam.fx, self.cam.height / 2 / self.cam.fy)
        return 0.5 * short_side / self.size

    def _patch(self, gray, roll, pitch, yaw, h, gsd):
        patch, mask, centre = ground_patch(gray, self.cam, roll, pitch, yaw, h, gsd, self.size)
        f = patch.astype(np.float32)
        f -= cv2.GaussianBlur(f, (0, 0), self.size / 32)  # drop lighting gradients
        f[~mask] = 0
        return f * self.window, mask, centre

    def step(self, frame: np.ndarray, t: float, roll: float, pitch: float, yaw: float, h: float) -> Motion | None:
        """Motion since the previous frame, or None for the first frame or a weak match."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, (gray, t, roll, pitch, yaw, h)
        if prev is None or not 0 < t - prev[1] <= self.max_dt:
            return None
        gsd = self.gsd(min(prev[5], h))
        a, mask_a, c0 = self._patch(prev[0], *prev[2:], gsd)
        b, mask_b, c1 = self._patch(gray, roll, pitch, yaw, h, gsd)
        if mask_a.mean() < 0.9 or mask_b.mean() < 0.9:
            return None
        (dx, dy), response = cv2.phaseCorrelate(a, b)
        if response < self.min_response:
            return None
        # Ground content moves opposite to the camera; then remove the change in
        # where the patch was centred (attitude changes move it).
        d_north = dy * gsd - (c1[0] - c0[0])
        d_east = -dx * gsd - (c1[1] - c0[1])
        dist = math.hypot(d_north, d_east)
        sigma = math.sqrt((self.px_sigma * gsd) ** 2 + (dist * self.height_rel_sigma) ** 2
                          + 2 * (h * self.att_sigma) ** 2)
        return Motion(d_north, d_east, t - prev[1], sigma, response)
