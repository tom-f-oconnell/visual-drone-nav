"""Synthetic camera frames rendered from the map, for bench testing without a drone.

`render` produces what the downward camera would see at a given pose by
projecting the map through the inverse of the orthorectification. Photometric
changes (gamma, colour cast, blur, noise) stand in for season, lighting and
sensor differences between the NAIP flight and today.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from .geo import MapRaster
from .ortho import CameraModel, ground_homography


def render(map_: MapRaster, cam: CameraModel, lat, lon, h, roll, pitch, yaw,
           rng: np.random.Generator | None = None, photometric: bool = True) -> np.ndarray:
    J = map_.ned_to_px_jacobian(lat, lon)
    col, row = map_.ll_to_px(lat, lon)
    A = np.array([[J[0, 0], J[0, 1], col], [J[1, 0], J[1, 1], row], [0, 0, 1.0]])
    H = A @ ground_homography(cam, roll, pitch, yaw, h)
    bgr = cv2.cvtColor(map_.rgb, cv2.COLOR_RGB2BGR)
    img = cv2.warpPerspective(bgr, H, (cam.width, cam.height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    if photometric:
        rng = rng or np.random.default_rng()
        f = img.astype(np.float32) / 255
        f = f ** rng.uniform(0.75, 1.3)
        f *= rng.uniform(0.85, 1.15, size=3)
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.5))
        f += rng.normal(0, 0.02, f.shape)
        img = np.clip(f * 255, 0, 255).astype(np.uint8)
    return img


@dataclass
class Pose:
    t: float
    lat: float
    lon: float
    h: float
    roll: float
    pitch: float
    yaw: float


def fly_route(map_: MapRaster, route_ll: list[tuple[float, float, float]], speed: float,
              rate_hz: float, rng: np.random.Generator):
    """Yield poses along waypoints (lat, lon, alt) with attitude wobble like a multirotor."""
    to_grid = map_.ll_to_grid
    pts = [np.array(to_grid(lat, lon)) for lat, lon, _ in route_ll]
    t = 0.0
    for (a, b), (_, _, ha), (_, _, hb) in zip(zip(pts, pts[1:]), route_ll, route_ll[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        n = max(int(length / speed * rate_hz), 1)
        heading_grid = math.atan2(seg[0], seg[1])  # from grid north, clockwise
        for i in range(n):
            s = i / n
            x, y = a + seg * s
            lat, lon = map_.grid_to_ll(x, y)
            yield Pose(
                t, lat, lon, ha + (hb - ha) * s,
                roll=math.radians(rng.normal(0, 3)),
                pitch=math.radians(-6 + rng.normal(0, 3)),  # nose down while cruising
                yaw=heading_grid + math.radians(rng.normal(0, 2)),
            )
            t += 1 / rate_hz


def synthetic_map(path: str, lat: float, lon: float, size_m: float = 2000, gsd: float = 0.6, seed: int = 0):
    """Write a procedural aerial-looking RGB GeoTIFF (fields, roads, buildings, trees).

    Only for tests in places without NAIP access. Real NAIP is far richer.
    """
    rng = np.random.default_rng(seed)
    n = int(size_m / gsd)
    img = np.zeros((n, n, 3), np.float32)
    # Fields: Voronoi-ish patches with per-field colour and row texture.
    m = n // 8
    seeds = rng.uniform(0, m, (60, 2))
    sy, sx = np.mgrid[0:m, 0:m]
    small = np.argmin(np.stack([(sx - a) ** 2 + (sy - b) ** 2 for a, b in seeds]), axis=0)
    lab = cv2.resize(small.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[0:n, 0:n]
    palette = rng.uniform([60, 80, 40], [170, 160, 110], (60, 3))
    img[:] = palette[lab]
    ang = rng.uniform(0, math.pi, 60)[lab]
    period = rng.uniform(2, 6, 60)[lab]
    amp = rng.uniform(4, 14, 60)[lab]
    img += (np.sin((xx * np.cos(ang) + yy * np.sin(ang)) / period) * amp)[..., None]
    noise = cv2.resize(rng.normal(0, 1, (n // 16, n // 16)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
    img += noise[..., None] * 12
    # Roads.
    for _ in range(14):
        p0, p1 = rng.uniform(-n * 0.2, n * 1.2, (2, 2)).astype(int)
        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (120, 120, 118), int(rng.integers(5, 14)))
    # Buildings and trees.
    for _ in range(900):
        cx, cy = rng.uniform(0, n, 2)
        if rng.random() < 0.5:
            w, h = rng.uniform(8, 40, 2)
            box = cv2.boxPoints(((cx, cy), (w, h), rng.uniform(0, 90))).astype(np.int32)
            cv2.fillPoly(img, [box], tuple(float(v) for v in rng.uniform(140, 230, 3)))
        else:
            cv2.circle(img, (int(cx), int(cy)), int(rng.uniform(3, 10)), (40, 70, 35), -1)
    rgb = np.clip(img, 0, 255).astype(np.uint8)

    epsg = 32600 + int((lon + 180) // 6) + 1
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    transform = from_origin(x - size_m / 2, y + size_m / 2, gsd, gsd)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8",
                       crs=f"EPSG:{epsg}", transform=transform, nodata=0, compress="deflate") as dst:
        dst.write(np.moveaxis(rgb, -1, 0))
    return path
