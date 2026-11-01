"""Glue: camera frame + attitude + height -> map fix and odometry -> filtered position.

The filter starts at the known start position (the vehicle is placed at the
mission home), so there is a position to report, and the autopilot can arm,
before any map fix. On the ground it is held still; in the air, frame-to-frame
odometry carries it through takeoff, landing and gaps between map fixes; above
min_map_alt, matches against the NAIP map correct the accumulated drift. With no
matcher it runs on odometry alone, and the map's pixels are never looked at.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from .filter import PositionFilter
from .geo import MapRaster
from .matcher import Fix
from .odometry import FrameOdometry
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
    last_fix_age: float     # s since the last map fix
    last_update_age: float  # s since the last map fix, odometry or ground update
    on_ground: bool


class Navigator:
    def __init__(self, map_: MapRaster, cam: CameraModel, matcher,
                 start_lat: float, start_lon: float, start_sigma_m: float = 150.0,
                 max_search_m: float = 400.0, min_search_m: float = 40.0,
                 min_map_alt: float = 25.0, ground_alt: float = 1.0, accel_sigma: float = 4.0,
                 odometry_drift: float = 0.05, odometry: FrameOdometry | None = None):
        """
        Args:
            matcher: has locate(ortho, prior_col, prior_row, search_radius_px) -> Fix | None,
                or None for odometry alone.
            start_sigma_m: 1-sigma of the start position; small when the
                vehicle is placed at a known point, large to test acquisition.
            min_map_alt: m AGL below which frames are not matched to the map
                (the footprint is too small), only used for odometry.
            ground_alt: m AGL below which the vehicle is taken to be on the
                ground and still.
            accel_sigma: m/s^2, how hard the vehicle may accelerate between
                updates in the air (on the ground it cannot).
            odometry_drift: 1-sigma position error per metre flown on odometry.
                Heading and scale errors persist from frame to frame, so this
                grows with distance instead of averaging out; without it the
                search radius stays small while the real error grows past it.
        """
        self.map, self.cam, self.matcher = map_, cam, matcher
        self.start = (start_lat, start_lon)
        self.start_sigma = start_sigma_m
        self.max_search_m, self.min_search_m = max_search_m, min_search_m
        self.min_map_alt, self.ground_alt = min_map_alt, ground_alt
        self.accel_sigma = accel_sigma
        self.odometry_drift = odometry_drift
        self.odometry_dist = 0.0  # m flown since the last map fix
        self.odometry = odometry or FrameOdometry(cam)
        self.kf = PositionFilter(accel_sigma)
        self.yaw_corr = 0.0  # rad added to the autopilot's yaw, learnt from map fixes (compass bias)
        self.last_fix_t: float | None = None
        self.last_update_t: float | None = None
        self.last_fix: Fix | None = None
        self.on_ground = False
        # Filter access only, so state() (e.g. the GPS sender thread) never waits for map matching.
        self._lock = threading.RLock()

    def _ensure_started(self, t: float):
        if not self.kf.initialised:
            x, y = self.map.ll_to_grid(*self.start)
            self.kf.reset(x, y, self.start_sigma, t)

    def prior(self, t: float):
        """Best guess of (lat, lon, sigma_m) at time t."""
        with self._lock:
            self._ensure_started(t)
            self.kf.predict(t)
            x, y, sigma = self.kf.x[0], self.kf.x[1], self.kf.pos_sigma()
        lat, lon = self.map.grid_to_ll(x, y)
        return lat, lon, sigma

    def process(self, frame: np.ndarray | None, t: float, roll: float, pitch: float, yaw: float,
                h_agl: float) -> Fix | None:
        """Use one frame; returns the map fix if one was found and accepted."""
        lat, lon, sigma = self.prior(t)
        yaw += self.yaw_corr
        self.on_ground = h_agl < self.ground_alt
        with self._lock:
            # Also keeps the speed accuracy sent to ArduPilot small before takeoff (EKF3 flags >1.5 m/s as a glitch).
            self.kf.q = (0.1 if self.on_ground else self.accel_sigma) ** 2
            if self.on_ground:
                self.kf.update_velocity(0.0, 0.0, 0.05, t, gate=False)
                self.last_update_t = t
        if self.on_ground:
            self.odometry.reset()
            return None
        if frame is None:
            return None
        frame = self.cam.undistort(frame)
        motion = self.odometry.step(frame, t, roll, pitch, yaw, h_agl)
        if motion is not None:
            ne_to_grid = np.linalg.inv(self.map.grid_to_ned_rotation(lat, lon))
            v = ne_to_grid @ np.array([motion.d_north, motion.d_east]) / motion.dt
            dist = self.odometry_dist + float(np.hypot(motion.d_north, motion.d_east))
            with self._lock:
                self.kf.add_position_variance(self.odometry_drift ** 2 * (dist ** 2 - self.odometry_dist ** 2))
                self.odometry_dist = dist
                if self.kf.update_velocity(v[0], v[1], motion.sigma / motion.dt, t):
                    self.last_update_t = t
        if self.matcher is None or h_agl < self.min_map_alt:
            return None

        lat, lon, sigma = self.prior(t)
        search_m = float(np.clip(3 * sigma, self.min_search_m, self.max_search_m))
        J = self.map.ned_to_px_jacobian(lat, lon)
        ortho = orthorectify(frame, self.cam, roll, pitch, yaw, h_agl, J)
        if ortho is None:
            return None
        col, row = self.map.ll_to_px(lat, lon)
        fix = self.matcher.locate(ortho, col, row, int(search_m / self.map.gsd))
        if fix is None:
            return None

        x, y = self.map.px_to_grid(fix.col, fix.row)
        with self._lock:
            accepted = self.kf.update(x, y, fix.sigma_m, t)
            if not accepted and self.kf.rejects >= 3 and fix.strong:
                log.warning("re-locking on strong fixes after %d rejects", self.kf.rejects)
                self.kf.reset(x, y, fix.sigma_m, t, v=self.kf.x[2:4], v_sigma=self.kf.vel_sigma())
                accepted = True
            if not accepted:
                return None
            self.last_fix_t = self.last_update_t = t
            self.odometry_dist = 0.0
        self.last_fix = fix
        # The fix's rotation is the remaining heading error; low-pass it so one bad fix does little.
        self.yaw_corr += 0.2 * np.clip(np.radians(fix.heading_err_deg), -0.1, 0.1)
        return fix

    def state(self, t: float) -> NavState | None:
        with self._lock:
            if not self.kf.initialised:
                return None
            lat, lon, sigma = self.prior(t)
            v, vel_sigma = self.kf.x[2:4].copy(), self.kf.vel_sigma()
            fix_age = t - self.last_fix_t if self.last_fix_t is not None else float("inf")
            update_age = t - self.last_update_t if self.last_update_t is not None else float("inf")
        vn, ve = self.map.grid_to_ned_rotation(lat, lon) @ v
        return NavState(lat, lon, float(vn), float(ve), sigma, vel_sigma, fix_age, update_age, self.on_ground)
