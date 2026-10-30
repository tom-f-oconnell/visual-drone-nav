"""Camera model and orthorectification of a downward camera frame onto the map grid.

Frames: NED world (x north, y east, z down), ArduPilot body (x forward, y right,
z down), OpenCV camera (x right, y down, z optical axis). The vehicle is at
height h above a flat ground plane, z = 0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=list)
    # Mount: camera looks straight down; image top points toward the vehicle nose
    # when mount_yaw_deg = 0. Rotate if the camera is mounted turned.
    mount_yaw_deg: float = 0.0
    mount_pitch_deg: float = 0.0  # forward tilt of the optical axis from nadir

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float, **kw) -> "CameraModel":
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(width, height, f, f, width / 2, height / 2, **kw)

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraModel":
        """Load an OpenCV calibration: {width, height, fx, fy, cx, cy, dist, mount_*}."""
        return cls(**json.loads(Path(path).read_text()))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def R_body_cam(self) -> np.ndarray:
        """Rotation taking camera-frame vectors into the body frame."""
        nadir = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        return rot_z(math.radians(self.mount_yaw_deg)) @ rot_y(math.radians(self.mount_pitch_deg)) @ nadir

    def undistort(self, img: np.ndarray) -> np.ndarray:
        if not any(self.dist):
            return img
        return cv2.undistort(img, self.K, np.array(self.dist))


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_ned_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler (radians, as in MAVLink ATTITUDE) to body->NED rotation."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def ground_homography(cam: CameraModel, roll, pitch, yaw, h: float) -> np.ndarray:
    """3x3 H with [N, E, 1] ~ H @ [u, v, 1]: image pixel to ground offset from nadir (metres)."""
    M = R_ned_body(roll, pitch, yaw) @ cam.R_body_cam @ np.linalg.inv(cam.K)
    return np.diag([h, h, 1.0]) @ M


@dataclass
class Ortho:
    image: np.ndarray       # north-up image on the map pixel grid
    mask: np.ndarray        # valid pixels
    nadir: tuple[float, float]  # pixel in `image` directly below the camera
    H: np.ndarray           # camera pixel -> ortho pixel


def orthorectify(img: np.ndarray, cam: CameraModel, roll, pitch, yaw, h: float,
                 J: np.ndarray, max_size: int = 1400, max_tilt_deg: float = 40.0) -> Ortho | None:
    """Warp a camera frame so it has the map's orientation and scale.

    J is MapRaster.ned_to_px_jacobian at the prior position, mapping metres
    north/east to map pixels. Returns None when the view is too oblique.
    """
    G = ground_homography(cam, roll, pitch, yaw, h)
    corners = np.array([[0, 0, 1], [cam.width, 0, 1], [cam.width, cam.height, 1], [0, cam.height, 1.0]]).T
    rays = R_ned_body(roll, pitch, yaw) @ cam.R_body_cam @ np.linalg.inv(cam.K) @ corners
    rays /= np.linalg.norm(rays, axis=0)
    if rays[2].min() < math.cos(math.radians(max_tilt_deg + 35)):
        return None  # a corner ray is too close to the horizon for a usable footprint

    A = np.eye(3)
    A[:2, :2] = J
    pts = A @ G @ corners
    pts = pts[:2] / pts[2]
    # The nadir maps to (0, 0) here. Keep at most max_size pixels around it so
    # high altitude or wide lenses do not produce huge warps.
    half = max_size / 2
    lo = np.maximum(pts.min(axis=1), -half)
    hi = np.minimum(pts.max(axis=1), half)
    size = hi - lo
    A[0, 2], A[1, 2] = -lo[0], -lo[1]
    H = A @ G
    w, hgt = int(math.ceil(size[0])), int(math.ceil(size[1]))
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warped = cv2.warpPerspective(gray, H, (w, hgt), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full(gray.shape, 255, np.uint8), H, (w, hgt), flags=cv2.INTER_NEAREST) > 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return Ortho(warped, mask, (-lo[0], -lo[1]), H)
