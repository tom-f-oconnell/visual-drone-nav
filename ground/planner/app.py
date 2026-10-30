"""FastAPI backend for the flight planner.

Run with:  uvicorn planner.app:app --reload   (from the ground/ directory)

Environment variables:
  NAIPNAV_DATA     data directory (default ./data): cache, missions, bundles
  NAIPNAV_GSD      cache resolution in metres (default 0.6)
  NAIPNAV_SOURCE   "pc" (Planetary Computer, default) or a path to local NAIP GeoTIFFs
  NAIPNAV_YEARS    optional comma separated NAIP years to allow, e.g. "2021,2022"
  NAIPNAV_FETCH_THREADS  cells downloaded in parallel (default 6)
"""

from __future__ import annotations

import math
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from shapely.geometry import mapping

from .mission import Mission, build_bundle
from .naip import Cell, LocalSource, NaipCache, StacSource, cells_for_bbox

DATA = Path(os.environ.get("NAIPNAV_DATA", "data")).resolve()
GSD = float(os.environ.get("NAIPNAV_GSD", "0.6"))
SOURCE = os.environ.get("NAIPNAV_SOURCE", "pc")
YEARS = [int(y) for y in os.environ.get("NAIPNAV_YEARS", "").split(",") if y.strip()]
FETCH_THREADS = int(os.environ.get("NAIPNAV_FETCH_THREADS", "6"))

STATIC = Path(__file__).parent / "static"
MISSIONS = DATA / "missions"
BUNDLES = DATA / "bundles"
for d in (MISSIONS, BUNDLES):
    d.mkdir(parents=True, exist_ok=True)

source = StacSource(years=YEARS or None) if SOURCE == "pc" else LocalSource(SOURCE)
cache = NaipCache(DATA / "naip", source, gsd=GSD)

app = FastAPI(title="NAIP flight planner")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_pool = ThreadPoolExecutor(max_workers=4)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {"gsd": GSD, "source": source.name, "data_dir": str(DATA)}


# Missions -----------------------------------------------------------------
@app.get("/api/missions")
def list_missions():
    return sorted(p.stem for p in MISSIONS.glob("*.json"))


@app.get("/api/missions/{name}")
def get_mission(name: str) -> Mission:
    p = MISSIONS / f"{Mission(name=name).name}.json"
    if not p.exists():
        raise HTTPException(404, "no such mission")
    return Mission.model_validate_json(p.read_text())


@app.put("/api/missions/{name}")
def save_mission(name: str, mission: Mission):
    mission.name = Mission(name=name).name
    (MISSIONS / f"{mission.name}.json").write_text(mission.model_dump_json(indent=2))
    return {"saved": mission.name}


@app.delete("/api/missions/{name}")
def delete_mission(name: str):
    (MISSIONS / f"{Mission(name=name).name}.json").unlink(missing_ok=True)
    return {"deleted": name}


@app.post("/api/plan/check")
def check_plan(mission: Mission):
    """Stats, warnings and the corridor polygon for the current plan."""
    if not mission.waypoints:
        return {"warnings": ["Click the map to add waypoints."], "cells": 0, "cached": 0}
    _, poly_ll = mission.corridor()
    cells = mission.required_cells()
    cached = sum(cache.has(c) for c in cells)
    return {
        "length_m": round(mission.length_m(), 1),
        "est_time_s": round(mission.length_m() / mission.speed),
        "footprint_m": round(mission.footprint_m(), 1),
        "cells": len(cells),
        "cached": cached,
        "est_download_mb": round((len(cells) - cached) * 1.1 * 0.85e6 / GSD**2 * 3 / 1e6 * 0.5, 1),
        "corridor": mapping(poly_ll),
        "warnings": mission.warnings(GSD),
    }


@app.get("/api/missions/{name}/mission.waypoints", response_class=PlainTextResponse)
def export_wpl(name: str):
    return get_mission(name).to_wpl()


@app.post("/api/export")
def export_bundle(mission: Mission):
    if len(mission.waypoints) < 2:
        raise HTTPException(400, "need at least two waypoints")
    missing = [c.key for c in mission.required_cells() if not cache.has(c)]
    if missing:
        raise HTTPException(409, f"{len(missing)} NAIP cells not cached yet; run 'Cache NAIP' first")
    data = build_bundle(mission, cache, BUNDLES / mission.name)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{mission.name}.zip"'})


# NAIP cache ----------------------------------------------------------------
def _fetch_cell(job: dict, cell: Cell):
    if job["cancel"]:
        return
    try:
        path = cache.fetch(cell)
        with _jobs_lock:
            job["done"] += 1
            if path is None:
                job["no_coverage"].append(cell.key)
    except Exception as exc:  # noqa: BLE001  keep going, report per cell
        with _jobs_lock:
            job["done"] += 1
            job["errors"].append(f"{cell.key}: {exc}")


def _run_job(job_id: str, cells: list[Cell]):
    job = _jobs[job_id]
    # Fetches are latency bound (NAIP on Planetary Computer is stored in West
    # Europe), so several cells in flight at once is much faster.
    with ThreadPoolExecutor(max_workers=FETCH_THREADS) as pool:
        list(pool.map(lambda c: _fetch_cell(job, c), cells))
    job["state"] = "cancelled" if job["cancel"] else "finished"


@app.post("/api/cache/jobs")
def start_cache_job(mission: Mission):
    if not mission.waypoints:
        raise HTTPException(400, "no waypoints")
    cells = [c for c in mission.required_cells() if not cache.has(c)]
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"id": job_id, "state": "running", "total": len(cells), "done": 0,
                     "errors": [], "no_coverage": [], "cancel": False}
    _pool.submit(_run_job, job_id, cells)
    return _jobs[job_id]


@app.get("/api/cache/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404)
    return _jobs[job_id]


@app.delete("/api/cache/jobs/{job_id}")
def cancel_job(job_id: str):
    if job_id in _jobs:
        _jobs[job_id]["cancel"] = True
    return {"cancelled": job_id}


@app.get("/api/cache/cells")
def cached_cells():
    feats = []
    for meta in cache.cached_cells():
        w, s, e, n = meta["bbox"]
        feats.append({
            "type": "Feature",
            "properties": {"key": meta["cell"], "years": sorted({x["year"] for x in meta["scenes"]}),
                           "coverage": meta["coverage"]},
            "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
        })
    return {"type": "FeatureCollection", "features": feats}


# Tiles of cached imagery, so the planner can show exactly what the drone will match against.
_EMPTY_PNG = cv2.imencode(".png", np.zeros((1, 1, 4), np.uint8))[1].tobytes()
_R = 6378137.0


def _tile_bounds_3857(z: int, x: int, y: int):
    size = 2 * math.pi * _R / 2**z
    left = -math.pi * _R + x * size
    top = math.pi * _R - y * size
    return left, top - size, left + size, top


def _tile_bounds_ll(z: int, x: int, y: int):
    n = 2**z
    lon = lambda i: i / n * 360 - 180  # noqa: E731
    lat = lambda j: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * j / n))))  # noqa: E731
    return lon(x), lat(y + 1), lon(x + 1), lat(y)


@app.get("/api/tiles/{z}/{x}/{y}.png")
def tile(z: int, x: int, y: int):
    if z < 12:
        return Response(_EMPTY_PNG, media_type="image/png", headers={"Cache-Control": "no-store"})
    cells = [c for c in cells_for_bbox(*_tile_bounds_ll(z, x, y)) if cache.has(c)]
    if not cells:
        return Response(_EMPTY_PNG, media_type="image/png", headers={"Cache-Control": "no-store"})
    transform = from_bounds(*_tile_bounds_3857(z, x, y), 256, 256)
    rgba = np.zeros((256, 256, 4), np.uint8)
    for cell in cells:
        with rasterio.open(cache.path(cell)) as src, WarpedVRT(
            src, crs="EPSG:3857", transform=transform, width=256, height=256,
            resampling=Resampling.bilinear, nodata=0,
        ) as vrt:
            data = vrt.read([1, 2, 3])
        valid = data.any(axis=0)
        rgba[valid, :3] = data[:, valid].T
        rgba[valid, 3] = 255
    bgra = rgba[:, :, [2, 1, 0, 3]]
    return Response(cv2.imencode(".png", bgra)[1].tobytes(), media_type="image/png",
                    headers={"Cache-Control": "max-age=60"})


@app.get("/api/cache/cells/{key}")
def cell_meta(key: str):
    iy, ix = (int(v) for v in key.split("_"))
    meta = cache.meta(Cell(iy, ix))
    if meta is None:
        raise HTTPException(404)
    return meta
