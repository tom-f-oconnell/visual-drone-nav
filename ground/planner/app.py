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

import asyncio
import math
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from shapely.geometry import mapping

from .mission import Mission, build_bundle
from .naip import Cell, LocalSource, NaipCache, StacSource, cells_for_bbox
from .simruns import ACTIVE_STATES, SimRequest, SimRuns, TooManyRuns, simulator_available

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


def _source(years: list[int] | None = None):
    return StacSource(years=years) if SOURCE == "pc" else LocalSource(SOURCE, years=years)


source = _source(YEARS or None)
cache = NaipCache(DATA / "naip", source, gsd=GSD)
sim_runs = SimRuns(DATA / "sim", cache, lambda year: NaipCache(DATA / "naip", _source([year]), gsd=GSD), FETCH_THREADS)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    sim_runs.stop_all()


app = FastAPI(title="NAIP flight planner", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_pool = ThreadPoolExecutor(max_workers=4)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {"gsd": GSD, "source": source.name, "data_dir": str(DATA), "simulator": simulator_available()}


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
        "cell_keys": [c.key for c in cells],
        "cached": cached,
        "est_download_mb": round((len(cells) - cached) * 1.1 * 0.85e6 / GSD**2 * 3 / 1e6 * 0.5, 1),
        "corridor": mapping(poly_ll),
        "warnings": mission.warnings(GSD),
    }


def _cached_years(cells: list[Cell]) -> set[int]:
    return {s["year"] for c in cells for s in (cache.meta(c) or {"scenes": []})["scenes"] if s["year"]}


def _naip_label(years) -> str:
    return "NAIP " + (", ".join(str(y) for y in sorted(years, reverse=True)) or "year unknown")


@app.post("/api/plan/imagery")
def plan_imagery(mission: Mission):
    """Short label for which NAIP years the route's imagery is, or would be, from.

    Only asks the imagery source if some cells are not cached. Cells are filled
    newest first, so its answer for those is the newest year near the route.
    """
    default = _naip_label(YEARS) if YEARS else "Newest NAIP available"
    if not mission.waypoints:
        return {"label": default}
    cells = mission.required_cells()
    cached = _cached_years(cells)
    if all(cache.has(c) for c in cells):
        return {"label": _naip_label(cached)}
    try:
        years = [scene.year for scene in source.scenes(mission.corridor()[1].bounds)]
    except Exception:  # noqa: BLE001  e.g. offline: say what is known
        years = None
    newest = max(years or [0])
    if not newest:
        label = f"{_naip_label(cached)} cached" if cached else "No NAIP here" if years == [] else default
    elif cached - {newest}:
        label = f"{_naip_label(cached)} cached, {newest} for the rest"
    else:
        label = _naip_label({newest})
    return {"label": label}


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
    route_cells = mission.required_cells()
    cells = [c for c in route_cells if not cache.has(c)]
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"id": job_id, "state": "running", "route_cells": len(route_cells), "total": len(cells), "done": 0,
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


# Simulation -----------------------------------------------------------------
def _run_dir(run_id: str) -> Path:
    d = sim_runs.dir(run_id)
    if d is None:
        raise HTTPException(404, "no such simulation")
    return d


@app.post("/api/sim/years")
def sim_years(mission: Mission):
    """NAIP years over the corridor that are not in the map, to render the camera from."""
    if not mission.waypoints:
        return {"years": []}
    _, poly_ll = mission.corridor()
    all_years = {scene.year for scene in _source().scenes(poly_ll.bounds) if scene.year}
    return {"years": sorted(all_years - _cached_years(mission.required_cells()), reverse=True)}


@app.get("/api/sim/appearance/random")
def random_sim_appearance():
    """One realistic stand-in for lighting, season and sensor differences."""
    if not simulator_available():
        raise HTTPException(503, "the simulator is not installed: pip install -e onboard --no-deps")
    from dataclasses import asdict

    import numpy as np
    from visnav.appearance import Appearance

    return asdict(Appearance.random(np.random.default_rng()))


@app.get("/api/sim/runs")
def list_sim_runs():
    return sim_runs.list()


def _check_can_simulate(req: SimRequest):
    if not simulator_available():
        raise HTTPException(503, "the simulator is not installed: pip install -e onboard --no-deps")
    if len(req.mission.waypoints) < 2:
        raise HTTPException(400, "need at least two waypoints")
    from visnav.appearance import Appearance

    try:
        Appearance.from_dict(req.options.appearance)
    except ValueError as exc:
        raise HTTPException(422, f"camera appearance: {exc}") from exc
    missing = [c.key for c in req.mission.required_cells() if not cache.has(c)]
    if missing:
        raise HTTPException(409, f"{len(missing)} NAIP cells not cached yet; run 'Cache NAIP' first")


@app.post("/api/sim/runs")
def start_sim_run(req: SimRequest):
    _check_can_simulate(req)
    try:
        return sim_runs.start(req.mission, req.options)
    except TooManyRuns as exc:
        raise HTTPException(429, str(exc)) from exc


@app.post("/api/sim/compare")
def start_sim_compare(req: SimRequest):
    """Fly the route twice with the same options: with map matching and on odometry alone."""
    _check_can_simulate(req)
    try:
        return sim_runs.compare(req.mission, req.options)
    except TooManyRuns as exc:
        raise HTTPException(429, str(exc)) from exc


@app.get("/api/sim/compare/{a}/{b}")
def sim_compare(a: str, b: str):
    _run_dir(a)
    _run_dir(b)
    return sim_runs.comparison(a, b)


@app.get("/api/sim/runs/{run_id}")
def sim_run_info(run_id: str):
    _run_dir(run_id)
    return sim_runs.info(run_id)


@app.post("/api/sim/runs/{run_id}/stop")
def stop_sim_run(run_id: str):
    _run_dir(run_id)
    sim_runs.stop(run_id)
    return {"stopped": run_id}


@app.delete("/api/sim/runs/{run_id}")
def delete_sim_run(run_id: str):
    _run_dir(run_id)
    sim_runs.delete(run_id)
    return {"deleted": run_id}


# The 3D viewer of `visnav sim3d`, whose page uses relative URLs, so each run has its own copy of the routes.
@app.get("/sim/{run_id}/")
def sim_viewer(run_id: str):
    _run_dir(run_id)
    if not simulator_available():
        raise HTTPException(503, "the simulator is not installed: pip install -e onboard --no-deps")
    from visnav.viewer import viewer_html

    return Response(viewer_html("three.min.js"), media_type="text/html; charset=utf-8")


@app.get("/sim/{run_id}/three.min.js")
def sim_viewer_three(run_id: str):
    _run_dir(run_id)
    from visnav.viewer import STATIC as VIEWER_STATIC

    return FileResponse(VIEWER_STATIC / "three.min.js", media_type="application/javascript")


def _run_file(run_id: str, name: str) -> Path:
    path = _run_dir(run_id) / name
    if not path.exists():  # the simulator writes these once it is up
        raise HTTPException(404, f"no {name} yet")
    return path


@app.get("/sim/{run_id}/scene.json")
def sim_scene(run_id: str):
    return FileResponse(_run_file(run_id, "scene.json"), media_type="application/json")


@app.get("/sim/{run_id}/map.jpg")
def sim_map(run_id: str):
    return FileResponse(_run_file(run_id, "map.jpg"), media_type="image/jpeg")


@app.get("/sim/{run_id}/events")
async def sim_events(run_id: str):
    """Server-sent events: every record so far, then new ones as the simulator appends them."""
    path = _run_dir(run_id) / "records.jsonl"

    async def stream():
        pos, partial = 0, b""
        while True:
            # Checked before reading, so nothing written just before the run ended is missed.
            over = sim_runs.state(run_id) not in ACTIVE_STATES
            if path.exists():
                with open(path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read(1 << 20)
                pos += len(chunk)
                *lines, partial = (partial + chunk).split(b"\n")
                for line in lines:
                    yield b"data: " + line + b"\n\n"
                if len(chunk) == 1 << 20:
                    continue
            if over:
                yield b"event: done\ndata: {}\n\n"
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


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
