"""The reference map: a NAIP GeoTIFF from a mission bundle, held in memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Geod, Transformer

_GEOD = Geod(ellps="WGS84")


@dataclass
class MapRaster:
    rgb: np.ndarray        # H x W x 3 uint8 (RGB order)
    gray: np.ndarray       # H x W uint8, contrast normalised
    valid: np.ndarray      # H x W bool
    transform: rasterio.Affine
    crs: str

    @classmethod
    def load(cls, path: str | Path) -> "MapRaster":
        with rasterio.open(path) as src:
            data = src.read([1, 2, 3])
            transform, crs = src.transform, src.crs.to_string()
        rgb = np.ascontiguousarray(np.moveaxis(data, 0, -1))
        valid = rgb.any(axis=2)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(gray)
        return cls(rgb, gray, valid, transform, crs)

    def __post_init__(self):
        self._to_map = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._to_ll = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self._inv = ~self.transform

    @property
    def gsd(self) -> float:
        return abs(self.transform.a)

    @property
    def shape(self):
        return self.gray.shape

    # Coordinate conversions. "grid" = map CRS metres (UTM easting, northing).
    def ll_to_grid(self, lat, lon):
        return self._to_map.transform(lon, lat)

    def grid_to_ll(self, x, y):
        lon, lat = self._to_ll.transform(x, y)
        return lat, lon

    def grid_to_px(self, x, y):
        t = self._inv
        return t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f

    def px_to_grid(self, col, row):
        t = self.transform
        return t.a * col + t.b * row + t.c, t.d * col + t.e * row + t.f

    def ll_to_px(self, lat, lon):
        return self.grid_to_px(*self.ll_to_grid(lat, lon))

    def px_to_ll(self, col, row):
        return self.grid_to_ll(*self.px_to_grid(col, row))

    def ned_to_px_jacobian(self, lat: float, lon: float) -> np.ndarray:
        """2x2 matrix J with [dcol, drow] = J @ [dNorth, dEast] (metres, true north).

        Captures grid convergence and the UTM scale factor, so camera geometry
        computed in a true north frame lands correctly on the map grid.
        """
        c0 = np.array(self.ll_to_px(lat, lon))
        lon_n, lat_n, _ = _GEOD.fwd(lon, lat, 0.0, 10.0)
        lon_e, lat_e, _ = _GEOD.fwd(lon, lat, 90.0, 10.0)
        dn = (np.array(self.ll_to_px(lat_n, lon_n)) - c0) / 10.0
        de = (np.array(self.ll_to_px(lat_e, lon_e)) - c0) / 10.0
        return np.column_stack([dn, de])

    def grid_to_ned_rotation(self, lat: float, lon: float) -> np.ndarray:
        """2x2 matrix mapping a grid vector [dx, dy] (east, north) to true [dN, dE]."""
        j = self.ned_to_px_jacobian(lat, lon)            # NE -> px
        px_to_grid = np.array([[self.gsd, 0.0], [0.0, -self.gsd]])
        ne_to_grid = px_to_grid @ j                       # NE -> grid (x, y)
        return np.linalg.inv(ne_to_grid)

    def crop(self, col: float, row: float, radius: int):
        """Square crop of the gray map around a pixel; returns (img, mask, (x0, y0))."""
        h, w = self.gray.shape
        x0, y0 = max(int(col) - radius, 0), max(int(row) - radius, 0)
        x1, y1 = min(int(col) + radius, w), min(int(row) + radius, h)
        if x1 <= x0 or y1 <= y0:
            return None, None, (x0, y0)
        return self.gray[y0:y1, x0:x1], self.valid[y0:y1, x0:x1], (x0, y0)
