"""Closed-loop multirotor flight simulator with GPS denied.

The simulated vehicle flies the bundle's mission the way ArduPilot would with
visnav attached:

    truth dynamics --> camera render --> visnav Navigator --> "GPS_INPUT"
          ^                                                       |
          |                                                       v
     attitude/thrust <-- position controller <-- autopilot EKF (IMU + GPS fusion)

The controller only ever sees the EKF estimate, which dead-reckons on a noisy,
biased IMU and is corrected by vision fixes. So navigation errors feed back
into where the vehicle actually goes, as they would in flight. This is a
simplified stand-in for ArduPilot SITL, not a replacement: there are no motor
models, no ArduPilot code, and the EKF is a 2-state Kalman filter per axis.

Coordinates: local ENU metres on the map grid, origin at the map centre.
"""

from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .appearance import Appearance, CameraLook
from .geo import MapRaster
from .navigator import Navigator
from .ortho import CameraModel
from .sim import SensorErrors, render, warm_up

G = 9.81
ODOMETRY_ONLY_SIGMA = 5.0  # m, position accuracy reported to the autopilot on odometry alone, see FlightSim.run


@dataclass
class SimConfig:
    physics_hz: float = 50.0
    camera_hz: float = 4.0
    gps_rate_hz: float = 5.0
    max_speed: float = 10.0
    max_accel: float = 3.5
    climb_rate: float = 3.0
    land_rate: float = 1.5
    accept_radius: float = 6.0
    wind: tuple[float, float] = (3.0, -1.5)   # m/s east, north
    gust_sigma: float = 1.5
    drag: float = 0.35                         # 1/s, linear drag against air
    accel_noise: float = 0.15                  # m/s^2 white noise on IMU
    accel_bias: float = 0.08                   # m/s^2 constant bias per axis
    att_noise_deg: float = 1.0                 # slowly varying AHRS errors (see sim.SensorErrors)
    yaw_noise_deg: float = 3.0
    alt_noise_frac: float = 0.01
    compass_bias_deg: float = 0.0              # lasting errors, which only map fixes can correct
    height_bias_pct: float = 0.0
    min_vision_alt: float = 25.0               # map matching only above this; odometry below
    max_update_age: float = 4.0
    lost_sigma: float = 60.0
    thumb_every_s: float = 1.0
    record_hz: float = 5.0
    max_time: float = 1800.0
    seed: int = 0
    appearance: Appearance = Appearance()      # default: frames show the imagery as it is


class AxisEKF:
    """Per-axis Kalman filter (position, velocity, accelerometer bias), like EKF3's horizontal states.

    Estimating the bias while vision fixes arrive is what keeps dead reckoning
    short-term accurate once fixes stop (e.g. low on the landing descent).
    """

    def __init__(self, p: float, accel_sigma: float, bias_walk: float = 0.002):
        self.x = np.array([p, 0.0, 0.0])
        self.P = np.diag([1.0, 0.1, 0.1 ** 2])
        self.q = accel_sigma ** 2
        self.qb = bias_walk ** 2

    def predict(self, a: float, dt: float):
        F = np.array([[1, dt, -0.5 * dt * dt], [0, 1, -dt], [0, 0, 1]])
        self.x = F @ self.x + np.array([0.5 * dt * dt, dt, 0.0]) * a
        Gm = np.array([0.5 * dt * dt, dt, 0.0])
        self.P = F @ self.P @ F.T + np.outer(Gm, Gm) * self.q + np.diag([0, 0, self.qb * dt])

    def update_pos(self, z: float, sigma: float):
        S = self.P[0, 0] + sigma ** 2
        K = self.P[:, 0] / S
        self.x = self.x + K * (z - self.x[0])
        self.P = self.P - np.outer(K, self.P[0, :])


@dataclass
class Truth:
    p: np.ndarray            # E, N, U (metres)
    v: np.ndarray
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0         # radians from north, clockwise


@dataclass
class FlightSim:
    map: MapRaster
    cam: CameraModel
    nav: Navigator
    meta: dict
    cfg: SimConfig = field(default_factory=SimConfig)
    world: MapRaster | None = None  # imagery camera frames are rendered from (default: map)

    def __post_init__(self):
        if self.world is None:
            self.world = self.map
        warm_up()
        h, w = self.map.shape
        self.origin = np.array(self.map.px_to_grid(w / 2, h / 2))
        self.rng = np.random.default_rng(self.cfg.seed)
        self.look = CameraLook(self.cfg.appearance, [self.cfg.seed, 1])
        m = self.meta["mission"]
        self.home = self._ll_to_local(self.meta["home"]["lat"], self.meta["home"]["lon"])
        wps = [(*self._ll_to_local(w["lat"], w["lon"]), w["alt"] or m["default_alt"]) for w in m["waypoints"]]
        self.plan = [("takeoff", self.home[0], self.home[1], m["takeoff_alt"])]
        self.plan += [("wp", x, y, a) for x, y, a in wps]
        if m["end_action"] == "rtl":
            self.plan += [("wp", self.home[0], self.home[1], wps[-1][2]), ("land", self.home[0], self.home[1], 0.0)]
        elif m["end_action"] == "land":
            self.plan += [("land", wps[-1][0], wps[-1][1], 0.0)]
        self.max_speed = min(self.cfg.max_speed, m.get("speed", self.cfg.max_speed))
        self.truth = Truth(np.array([self.home[0], self.home[1], 0.0]), np.zeros(3))
        self.ekf = [AxisEKF(self.home[0], self.cfg.accel_noise * 3), AxisEKF(self.home[1], self.cfg.accel_noise * 3)]
        self.bias = self.rng.normal(0, self.cfg.accel_bias, 2)
        self.sensors = SensorErrors(self.rng, self.cfg.att_noise_deg, self.cfg.yaw_noise_deg,
                                    h_frac=self.cfg.alt_noise_frac, yaw_bias_deg=self.cfg.compass_bias_deg,
                                    h_bias_frac=self.cfg.height_bias_pct / 100)
        self.gust = np.zeros(2)
        self.t = 0.0
        self.leg = 0
        self.landed = False

    @property
    def odometry_only(self) -> bool:
        return self.nav is not None and self.nav.matcher is None

    # Frames -----------------------------------------------------------------
    def _ll_to_local(self, lat, lon):
        x, y = self.map.ll_to_grid(lat, lon)
        return np.array([x, y]) - self.origin

    def _local_to_ll(self, x, y):
        return self.map.grid_to_ll(x + self.origin[0], y + self.origin[1])

    def scene(self) -> dict:
        """Static data for the viewer: map extent, waypoints, home."""
        h, w = self.map.shape
        x0, y0 = np.array(self.map.px_to_grid(0, 0)) - self.origin
        x1, y1 = np.array(self.map.px_to_grid(w, h)) - self.origin
        return {
            "extent": [float(x0), float(y1), float(x1), float(y0)],  # west, south, east, north
            "home": [float(self.home[0]), float(self.home[1])],
            "plan": [{"kind": k, "x": float(x), "y": float(y), "alt": float(a)} for k, x, y, a in self.plan],
            "name": self.meta["mission"]["name"],
            "navigation": "odometry" if self.odometry_only else "map",
            "camera": {"hfov_deg": math.degrees(2 * math.atan(self.cam.width / 2 / self.cam.fx)),
                       "aspect": self.cam.height / self.cam.width},
            "gsd": self.map.gsd,
        }

    # Guidance and control -----------------------------------------------------
    def _target(self):
        return self.plan[min(self.leg, len(self.plan) - 1)]

    def _control(self, est_p: np.ndarray, est_v: np.ndarray, dt: float):
        kind, tx, ty, talt = self._target()
        tr = self.truth
        # Vertical: baro-held altitude, first-order climb (ArduPilot alt hold is tight).
        if kind == "land":
            vz_cmd = -self.cfg.land_rate if tr.p[2] > 0.05 else 0.0
        else:
            vz_cmd = float(np.clip(1.2 * (talt - tr.p[2]), -self.cfg.climb_rate, self.cfg.climb_rate))
        # Horizontal: hold position during takeoff, otherwise go to the target.
        err = np.array([tx, ty]) - est_p
        dist = float(np.linalg.norm(err))
        if kind == "takeoff" or (kind == "land" and dist < 2):
            v_cmd = 0.8 * err
        else:
            speed = min(self.max_speed, math.sqrt(2 * 0.6 * self.cfg.max_accel * dist) + 0.5)
            v_cmd = err / max(dist, 1e-6) * speed
        a_cmd = 1.5 * (v_cmd - est_v)
        n = np.linalg.norm(a_cmd)
        if n > self.cfg.max_accel:
            a_cmd *= self.cfg.max_accel / n
        # Feed-forward drag so cruise speed is reachable against the air.
        a_cmd = a_cmd + self.cfg.drag * est_v

        # Yaw toward the direction of travel (WP_YAW_BEHAVIOR = 1).
        if kind == "wp" and dist > 10:
            want = math.atan2(err[0], err[1])
            dyaw = (want - tr.yaw + math.pi) % (2 * math.pi) - math.pi
            tr.yaw += float(np.clip(dyaw, -math.radians(60) * dt, math.radians(60) * dt))

        # Tilt needed for a_cmd, in the body frame, then attitude lag.
        cy, sy = math.cos(tr.yaw), math.sin(tr.yaw)
        a_fwd = a_cmd[1] * cy + a_cmd[0] * sy
        a_right = -a_cmd[1] * sy + a_cmd[0] * cy
        pitch_cmd = -math.atan2(a_fwd, G)
        roll_cmd = math.atan2(a_right * math.cos(pitch_cmd), G)
        k = dt / 0.15
        tr.pitch += (pitch_cmd - tr.pitch) * k
        tr.roll += (roll_cmd - tr.roll) * k

        # Advance: tilt gives horizontal acceleration, drag against the wind.
        a_fwd_act = -G * math.tan(tr.pitch)
        a_right_act = G * math.tan(tr.roll) / math.cos(tr.pitch)
        a = np.array([a_fwd_act * sy + a_right_act * cy, a_fwd_act * cy - a_right_act * sy])
        if tr.p[2] < 0.2 and vz_cmd <= 0:
            a[:] = 0
            tr.v[:2] = 0
        else:
            self.gust += -self.gust * dt / 5 + self.rng.normal(0, self.cfg.gust_sigma * math.sqrt(2 * dt / 5), 2)
            air = tr.v[:2] - (np.array(self.cfg.wind) + self.gust)
            a = a - self.cfg.drag * air
        tr.v[:2] += a * dt
        tr.v[2] += (vz_cmd - tr.v[2]) * min(dt / 0.4, 1)
        tr.p += tr.v * dt
        tr.p[2] = max(tr.p[2], 0.0)

        # Leg completion.
        if kind == "takeoff" and tr.p[2] > talt - 1:
            self.leg += 1
        elif kind == "wp" and dist < self.cfg.accept_radius:
            self.leg += 1
        elif kind == "land" and tr.p[2] <= 0.05:
            self.landed = True
        return a

    # Main loop ----------------------------------------------------------------
    def run(self, on_record=None, on_progress=None):
        cfg = self.cfg
        dt = 1 / cfg.physics_hz
        next_cam = next_gps = next_rec = next_thumb = 0.0
        records, fixes, thumb = [], 0, None
        frames = map_frames = off_imagery = 0
        nav_time = 0.0
        last_fix = None
        while not self.landed and self.leg < len(self.plan) and self.t < cfg.max_time:
            est_p = np.array([self.ekf[0].x[0], self.ekf[1].x[0]])
            est_v = np.array([self.ekf[0].x[1], self.ekf[1].x[1]])
            a = self._control(est_p, est_v, dt)
            self.t += dt
            imu = a + self.bias + self.rng.normal(0, cfg.accel_noise, 2)
            for i in range(2):
                self.ekf[i].predict(imu[i], dt)

            tr = self.truth
            if self.t >= next_cam:
                next_cam += 1 / cfg.camera_hz
                img = None
                if tr.p[2] > 0.3:  # on the ground the camera sees only the landing gear's shadow
                    lat, lon = self._local_to_ll(tr.p[0], tr.p[1])
                    raw = render(self.world, self.cam, lat, lon, tr.p[2], tr.roll, tr.pitch, tr.yaw)
                    off_imagery += int(np.mean(raw[..., 1] == 0) > 0.5)
                    img = self.look(raw, self.t)
                    frames += 1
                    map_frames += int(tr.p[2] >= cfg.min_vision_alt)
                roll, pitch, yaw, h = self.sensors(self.t, tr.roll, tr.pitch, tr.yaw, tr.p[2])
                nav_start = time.perf_counter()
                fix = self.nav.process(img, self.t, roll, pitch, yaw, max(h, 0.0))
                nav_time += time.perf_counter() - nav_start
                if fix:
                    fixes += 1
                    last_fix = self._ll_to_local(fix.lat, fix.lon).tolist() + [round(fix.score, 1)]
                if img is not None and self.t >= next_thumb:
                        next_thumb += cfg.thumb_every_s
                        small = cv2.resize(img, (192, int(192 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
                        thumb = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()

            if self.t >= next_gps:
                next_gps += 1 / cfg.gps_rate_hz
                st = self.nav.state(self.t)
                # Nothing corrects odometry alone, so its reported uncertainty only grows, and an autopilot told that
                # would soon stop following it and fly on its IMU. To measure odometry rather than that, keep trusting it.
                alone = self.odometry_only
                if st and st.last_update_age < cfg.max_update_age and (alone or st.sigma_m < cfg.lost_sigma):
                    # Position only, as ArduPilot is set up (EK3_SRC1_VELXY 0): velocity comes from the IMU.
                    x, y = self._ll_to_local(st.lat, st.lon)
                    sigma = min(st.sigma_m, ODOMETRY_ONLY_SIGMA) if alone else st.sigma_m
                    for i, z in enumerate((x, y)):
                        self.ekf[i].update_pos(z, max(sigma, 1.0))

            if self.t >= next_rec:
                next_rec += 1 / cfg.record_hz
                st = self.nav.state(self.t)
                rec = {
                    "t": round(self.t, 2),
                    "p": [round(float(v), 2) for v in tr.p],
                    "att": [round(tr.roll, 4), round(tr.pitch, 4), round(tr.yaw, 4)],
                    "ekf": [round(float(self.ekf[0].x[0]), 2), round(float(self.ekf[1].x[0]), 2)],
                    "sig": round(float(math.sqrt(max(self.ekf[0].P[0, 0], self.ekf[1].P[0, 0]))), 2),
                    "vis": bool(st and st.last_fix_age < 1.5),
                    "fix": last_fix,
                    "leg": self.leg,
                }
                if thumb:
                    rec["img"] = thumb
                    thumb = None
                last_fix = None
                records.append(rec)
                if on_record:
                    on_record(rec)
            if on_progress and int(self.t * cfg.physics_hz) % int(10 * cfg.physics_hz) == 0:
                on_progress(self.t)

        def err(low, high):
            return np.array([math.hypot(r["p"][0] - r["ekf"][0], r["p"][1] - r["ekf"][1]) for r in records
                             if low < r["p"][2] <= high])

        errs, low = err(0.3, np.inf), err(0.3, cfg.min_vision_alt)
        path = np.array([r["p"][:2] for r in records])
        distance = float(np.hypot(*np.diff(path, axis=0).T).sum()) if len(path) > 1 else 0.0
        final = math.hypot(records[-1]["p"][0] - records[-1]["ekf"][0], records[-1]["p"][1] - records[-1]["ekf"][1]) if records else 0.0
        summary = {
            "navigation": "odometry" if self.odometry_only else "map",
            "duration_s": round(self.t, 1), "landed": self.landed,
            "mission_complete": self.landed or self.leg >= len(self.plan), "camera_frames": frames,
            "vision_fixes": fixes, "fix_rate": round(fixes / max(map_frames, 1), 3),
            "ekf_err_median_m": round(float(np.median(errs)), 2) if len(errs) else None,
            "ekf_err_p95_m": round(float(np.percentile(errs, 95)), 2) if len(errs) else None,
            "ekf_err_max_m": round(float(errs.max()), 2) if len(errs) else None,
            "ekf_err_below_map_alt_max_m": round(float(low.max()), 2) if len(low) else None,
            "ekf_err_final_m": round(final, 2),
            "distance_m": round(distance, 1),
            "final_err_pct_of_distance": round(100 * final / distance, 2) if distance else None,
            "nav_ms_per_frame": round(1000 * nav_time / max(frames, 1), 1),
            "frames_off_imagery": off_imagery,
            "landing_error_m": round(float(np.linalg.norm(self.truth.p[:2] - np.array(self.plan[-1][1:3]))), 2),
            "appearance": cfg.appearance.changes(),
            "compass_bias_deg": cfg.compass_bias_deg,
            "height_bias_pct": cfg.height_bias_pct,
        }
        return records, summary


def map_texture(map_: MapRaster, max_px: int = 4096, quality: int = 82) -> bytes:
    """JPEG of the map for the 3D viewer's ground plane."""
    h, w = map_.shape
    s = min(1.0, max_px / max(h, w))
    img = cv2.cvtColor(map_.rgb, cv2.COLOR_RGB2BGR)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
