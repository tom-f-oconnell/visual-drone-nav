"""Glue: camera frame + attitude + height -> map fix -> filtered position."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .filter import PositionFilter
from .geo import MapRaster
from .matcher import Fix, MapMatcher
from .ortho import CameraModel, orthorectify

log = logging.getLogger(__name__)


@dataclass
class NavState:
    lat: float
    lon: float
    vn: float
    ve: float
    sigma_m: float
    vel_sigma: float
    last_fix_age: float


class Navigator:
    def __init__(self, map_: MapRaster, cam: CameraModel, matcher: MapMatcher,
                 start_lat: float, start_lon: float, start_sigma_m: float = 150.0,
                 max_search_m: float = 400.0, min_search_m: float = 40.0):
        self.map, self.cam, self.matcher = map_, cam, matcher
        self.start = (start_lat, start_lon)
        self.start_sigma = start_sigma_m
        self.max_search_m, self.min_search_m = max_search_m, min_search_m
        self.kf = PositionFilter()
        self.last_fix_t: float | None = None
        self.last_fix: Fix | None = None

    def prior(self, t: float):
        """Best guess of (lat, lon, sigma_m) at time t."""
        if not self.kf.initialised:
            return (*self.start, self.start_sigma)
        self.kf.predict(t)
        lat, lon = self.map.grid_to_ll(self.kf.x[0], self.kf.x[1])
        return lat, lon, self.kf.pos_sigma()

    def process(self, frame: np.ndarray, t: float, roll: float, pitch: float, yaw: float,
                h_agl: float) -> Fix | None:
        lat, lon, sigma = self.prior(t)
        search_m = float(np.clip(3 * sigma, self.min_search_m, self.max_search_m))
        J = self.map.ned_to_px_jacobian(lat, lon)
        ortho = orthorectify(self.cam.undistort(frame), self.cam, roll, pitch, yaw, h_agl, J)
        if ortho is None:
            return None
        col, row = self.map.ll_to_px(lat, lon)
        fix = self.matcher.locate(ortho, col, row, int(search_m / self.map.gsd))
        if fix is None:
            return None

        x, y = self.map.px_to_grid(fix.col, fix.row)
        if not self.kf.initialised:
            self.kf.reset(x, y, fix.sigma_m, t)
            accepted = True
        else:
            accepted = self.kf.update(x, y, fix.sigma_m, t)
            if not accepted and self.kf.rejects >= 3 and fix.inliers >= 2 * self.matcher.min_inliers:
                log.warning("re-locking on strong fixes after %d rejects", self.kf.rejects)
                self.kf.reset(x, y, fix.sigma_m, t)
                accepted = True
        if accepted:
            self.last_fix_t, self.last_fix = t, fix
            return fix
        return None

    def state(self, t: float) -> NavState | None:
        if not self.kf.initialised:
            return None
        lat, lon, sigma = self.prior(t)
        vn, ve = self.map.grid_to_ned_rotation(lat, lon) @ self.kf.x[2:]
        age = t - self.last_fix_t if self.last_fix_t is not None else float("inf")
        return NavState(lat, lon, float(vn), float(ve), sigma, self.kf.vel_sigma(), age)
