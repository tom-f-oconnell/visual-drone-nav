"""Flight plans: waypoint model, ArduPilot mission export and mission bundles.

A mission bundle is what gets copied to the Jetson. It contains

* ``mission.waypoints``  QGC WPL 110 mission, loadable by Mission Planner,
  QGroundControl, MAVProxy or ``visnav-upload``.
* ``mission.json``       waypoints, camera assumptions and map metadata.
* ``map.tif``            NAIP mosaic of the route corridor in one UTM zone,
                         pixels outside the corridor set to 0 (nodata).
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import shapely
from pydantic import BaseModel, Field, field_validator
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import LineString, Point, mapping

from .naip import Cell, NaipCache, cells_for_bbox, utm_epsg

# MAVLink command ids used in the exported mission.
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_FRAME_GLOBAL = 0
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3


class Waypoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt: float | None = Field(None, description="metres above home; mission default if null")
    hold_s: float = 0.0


class Camera(BaseModel):
    """Downward camera on the vehicle, used for footprint checks."""

    hfov_deg: float = 90.0
    width_px: int = 1280
    height_px: int = 800


class Mission(BaseModel):
    name: str = "mission"
    waypoints: list[Waypoint] = Field(default_factory=list)
    default_alt: float = Field(120.0, gt=0, description="m above home")
    takeoff_alt: float = Field(30.0, gt=0)
    speed: float = Field(10.0, gt=0, description="m/s")
    end_action: str = Field("rtl", pattern="^(rtl|land|none)$")
    corridor_m: float = Field(400.0, gt=0, description="NAIP half-width around the route")
    camera: Camera = Field(default_factory=Camera)

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        v = "".join(c if c.isalnum() or c in "-_" else "_" for c in v.strip())
        return v or "mission"

    def alt_of(self, wp: Waypoint) -> float:
        return wp.alt if wp.alt is not None else self.default_alt

    # Geometry --------------------------------------------------------------
    @property
    def epsg(self) -> int:
        wp = self.waypoints[0]
        return utm_epsg(wp.lat, wp.lon)

    def route_utm(self):
        """Route as a shapely geometry in the mission UTM zone, plus the transformers.

        With end_action "rtl" the route closes back to home (the first
        waypoint), because the vehicle flies that leg too and needs imagery there.
        """
        fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        wps = self.waypoints + (self.waypoints[:1] if self.end_action == "rtl" and len(self.waypoints) > 2 else [])
        pts = [fwd.transform(w.lon, w.lat) for w in wps]
        geom = Point(pts[0]) if len(pts) == 1 else LineString(pts)
        return geom, fwd, inv

    def corridor(self):
        """Corridor polygon in UTM metres and in lon/lat."""
        geom, _, inv = self.route_utm()
        poly = geom.buffer(self.corridor_m)
        return poly, shapely.transform(poly, lambda c: np.column_stack(inv.transform(c[:, 0], c[:, 1])))

    def length_m(self) -> float:
        geom, _, _ = self.route_utm()
        return geom.length

    def footprint_m(self, alt: float | None = None) -> float:
        """Ground width covered by the camera at altitude (flat terrain, nadir)."""
        alt = self.default_alt if alt is None else alt
        return 2 * alt * math.tan(math.radians(self.camera.hfov_deg) / 2)

    def warnings(self, gsd: float) -> list[str]:
        out = []
        if len(self.waypoints) < 2:
            out.append("Add at least two waypoints.")
            return out
        for i, wp in enumerate(self.waypoints):
            fp = self.footprint_m(self.alt_of(wp))
            if fp / gsd < 250:
                out.append(
                    f"WP{i + 1}: camera footprint {fp:.0f} m is only {fp / gsd:.0f} map pixels wide; "
                    "matching needs roughly 250+ (fly higher or use a wider lens)."
                )
            if self.alt_of(wp) > 122:
                out.append(f"WP{i + 1}: {self.alt_of(wp):.0f} m is above the 400 ft (122 m) Part 107 limit.")
        if self.corridor_m < self.footprint_m() / 2 + 100:
            out.append("Corridor is narrow for the camera footprint; drift outside it cannot be corrected.")
        return out

    # Export ----------------------------------------------------------------
    def to_wpl(self) -> str:
        """QGC WPL 110 text. Item 0 is home (overwritten by the autopilot)."""
        rows = []

        def add(cmd, frame, p1=0.0, p2=0.0, p3=0.0, p4=0.0, lat=0.0, lon=0.0, alt=0.0):
            seq = len(rows)
            current = 1 if seq == 0 else 0
            rows.append(
                f"{seq}\t{current}\t{frame}\t{cmd}\t{p1:.6f}\t{p2:.6f}\t{p3:.6f}\t{p4:.6f}\t"
                f"{lat:.8f}\t{lon:.8f}\t{alt:.6f}\t1"
            )

        home = self.waypoints[0]
        add(MAV_CMD_NAV_WAYPOINT, MAV_FRAME_GLOBAL, lat=home.lat, lon=home.lon, alt=0)
        add(MAV_CMD_NAV_TAKEOFF, MAV_FRAME_GLOBAL_RELATIVE_ALT, alt=self.takeoff_alt)
        add(MAV_CMD_DO_CHANGE_SPEED, MAV_FRAME_GLOBAL_RELATIVE_ALT, p1=1, p2=self.speed, p3=-1)
        for wp in self.waypoints:
            add(MAV_CMD_NAV_WAYPOINT, MAV_FRAME_GLOBAL_RELATIVE_ALT, p1=wp.hold_s,
                lat=wp.lat, lon=wp.lon, alt=self.alt_of(wp))
        if self.end_action == "rtl":
            add(MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_FRAME_GLOBAL_RELATIVE_ALT)
        elif self.end_action == "land":
            last = self.waypoints[-1]
            add(MAV_CMD_NAV_LAND, MAV_FRAME_GLOBAL_RELATIVE_ALT, lat=last.lat, lon=last.lon)
        return "QGC WPL 110\n" + "\n".join(rows) + "\n"

    def required_cells(self) -> list[Cell]:
        _, poly_ll = self.corridor()
        cells = cells_for_bbox(*poly_ll.bounds)
        return [c for c in cells if poly_ll.intersects(_cell_box(c))]


def _cell_box(cell: Cell):
    from shapely.geometry import box

    return box(*cell.bbox)


def build_map(mission: Mission, cache: NaipCache, out_path: Path) -> dict:
    """Mosaic cached cells covering the mission corridor into one UTM GeoTIFF."""
    poly_utm, _ = mission.corridor()
    gsd = cache.gsd
    l, b, r, t = poly_utm.bounds
    l, b = math.floor(l / gsd) * gsd, math.floor(b / gsd) * gsd
    r, t = math.ceil(r / gsd) * gsd, math.ceil(t / gsd) * gsd
    width, height = int(round((r - l) / gsd)), int(round((t - b) / gsd))
    transform = from_origin(l, t, gsd, gsd)
    crs = f"EPSG:{mission.epsg}"

    rgb = np.zeros((3, height, width), np.uint8)
    missing = []
    for cell in mission.required_cells():
        path = cache.path(cell)
        if not path.exists():
            missing.append(cell.key)
            continue
        with rasterio.open(path) as src, WarpedVRT(
            src, crs=crs, transform=transform, width=width, height=height,
            resampling=Resampling.bilinear, nodata=0,
        ) as vrt:
            data = vrt.read([1, 2, 3])
        valid = data.any(axis=0)
        rgb[:, valid] = data[:, valid]

    outside = geometry_mask([mapping(poly_utm)], (height, width), transform, invert=False)
    rgb[:, outside] = 0
    coverage = float(rgb.any(axis=0)[~outside].mean()) if (~outside).any() else 0.0

    with rasterio.open(
        out_path, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8",
        crs=crs, transform=transform, nodata=0, tiled=True, blockxsize=512, blockysize=512,
        compress="deflate", predictor=2,
    ) as dst:
        dst.write(rgb)
    return {
        "file": out_path.name, "crs": crs, "epsg": mission.epsg, "gsd": gsd,
        "width": width, "height": height, "transform": list(transform)[:6],
        "coverage": round(coverage, 4), "missing_cells": missing,
    }


def build_bundle(mission: Mission, cache: NaipCache, workdir: Path) -> bytes:
    """Write the bundle files into workdir and return them zipped."""
    workdir.mkdir(parents=True, exist_ok=True)
    map_meta = build_map(mission, cache, workdir / "map.tif")
    wpl = mission.to_wpl()
    (workdir / "mission.waypoints").write_text(wpl)
    _, fwd, _ = mission.route_utm()
    meta = {
        "format": "naipnav-bundle/1",
        "mission": mission.model_dump(),
        "home": {"lat": mission.waypoints[0].lat, "lon": mission.waypoints[0].lon},
        "route_utm": [list(fwd.transform(w.lon, w.lat)) + [mission.alt_of(w)] for w in mission.waypoints],
        "length_m": round(mission.length_m(), 1),
        "map": map_meta,
        "warnings": mission.warnings(cache.gsd),
    }
    (workdir / "mission.json").write_text(json.dumps(meta, indent=2))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("mission.json", "mission.waypoints", "map.tif"):
            zf.write(workdir / name, f"{mission.name}/{name}")
    return buf.getvalue()
