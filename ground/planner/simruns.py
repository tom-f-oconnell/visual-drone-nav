"""Flight simulations started from the website.

Each run is a `visnav sim3d` subprocess, so a heavy simulation cannot stall the
planner, writing into its own directory under <data>/sim:

  run.json       what was asked for, including the whole plan, so it can be flown again
  scene.json     static data for the viewer, written once the simulator is up
  map.jpg        the viewer's ground plane
  records.jsonl  the flight so far, one JSON record per line
  summary.json   error statistics, written last
  series.json    error and path along the flight, for comparisons, written at the end
  error.txt      why it failed, if it did

Any two runs of the same waypoints can be compared, whatever else differed, from
what is stored. Two can also be started together, as the same flight with map
matching and on odometry alone or by flying a comparison again, each naming the
other as its partner in run.json.

The viewer follows records.jsonl, so a finished run replays the same way as a
live one, and runs survive a restart of the planner. The temporary bundle (the
map and camera GeoTIFFs) is deleted after each run, because it can be rebuilt
from the imagery caches.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .clouds import overlay
from .imagery import CellCache, Library
from .mission import Mission, build_map, write_bundle

log = logging.getLogger(__name__)
MAX_RUNNING = 3
ACTIVE_STATES = ("preparing", "fetching", "running")
RUN_ID = re.compile(r"[0-9a-f]{8}")
TAIL_BYTES = 1 << 18  # longer than any one record, thumbnail included
CLOUD_BASE_M = 750.0  # by default a camera below this sees no clouds, as it is under them
SERIES_POINTS = 600  # of a flight, for the comparison page's charts


def simulator_available() -> bool:
    """The onboard package is optional for the planner, e.g. when only exporting bundles."""
    return importlib.util.find_spec("visnav") is not None


class TooManyRuns(Exception):
    pass


class SimOptions(BaseModel):
    seed: int = Field(0, ge=0)
    wind_speed: float = Field(3.0, ge=0, le=30, description="m/s")
    wind_from_deg: float = Field(300.0, ge=0, le=360, description="direction the wind blows from")
    gust: float = Field(1.5, ge=0, le=15, description="m/s")
    realtime: float = Field(0.0, ge=0, le=100, description="multiple of real time, 0 for as fast as possible")
    world_imagery: Literal["naip", "sentinel2"] = Field("naip", description="what world_year means")
    world_year: int | Literal["newest"] | None = Field(None, description="render the camera from this year, or the newest "
                                                       "imagery, instead of from the map")
    clouds: Literal["auto", "real", "shadows", "none"] = Field("auto", description="clouds from a Sentinel-2 pass: auto shows them "
                                                               "only while above CLOUD_BASE_M; real at every height, hiding and "
                                                               "dimming the ground; shadows only dimming it; or none")
    pass_date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="the pass; by default the last, or within "
                                  "world_year one the seed picks out of all of them, however cloudy")
    appearance: dict[str, float] = Field(default_factory=dict, description="visnav.appearance.Appearance fields to change")
    fog: dict[str, float] = Field(default_factory=dict, description="visnav.fog.Fog fields, or empty for no fog")
    navigation: Literal["map", "odometry"] = Field("map", description="odometry alone never looks at the saved imagery")
    compass_bias_deg: float = Field(0.0, ge=-45, le=45, description="heading error lasting the whole flight")
    height_bias_pct: float = Field(0.0, ge=-50, le=50, description="height error lasting the whole flight")


class SimRequest(BaseModel):
    mission: Mission
    options: SimOptions = SimOptions()


def route_of(scene: dict) -> str:
    """Identifies a flight's waypoints, whatever its altitudes, way of ending and map.

    Taken from the plan's shape measured from home, because a flight's local
    coordinates start at the centre of its own map, and older runs kept no
    latitudes and longitudes. So the same shape elsewhere would match too, which
    clicking on a map does not produce to within 10 cm.
    """
    hx, hy = scene["home"]
    points = [[round((p["x"] - hx) * 10), round((p["y"] - hy) * 10)] for p in scene["plan"] if p["kind"] == "wp"]
    if len(points) > 1 and points[-1] == [0, 0]:  # the leg of a return to launch
        points.pop()
    return hashlib.sha1(json.dumps(points).encode()).hexdigest()[:12]


def _listed(values: dict) -> str:
    return ", ".join(f"{name.replace('_', ' ')} {value:g}" for name, value in values.items())


def _camera_imagery(options: dict) -> str:
    year = options["world_year"]
    if year is None:
        return "the map's own"
    if options["world_imagery"] == "sentinel2":
        return "last Sentinel-2 pass" if year == "newest" else f"Sentinel-2 {year}"
    return "newest NAIP" if year == "newest" else f"NAIP {year}"


def _sky(world: dict | None) -> str:
    if not world or world.get("clouds", "none") == "none" or not world.get("pass_date"):
        return "none"
    if world["clouds"] == "real":
        seen = f", seen above {world['base_m']:g} m" if world.get("base_m") else ""
        return f"pass of {world['pass_date']}, {world.get('cloud_pct', 0)}% of the route{seen}"
    return f"shadows of the pass of {world['pass_date']}, {world.get('shadow_pct', 0)}% of the route"


def flight_settings(run: dict, scene: dict | None) -> dict[str, str | None]:
    """What can set flights of the same waypoints apart, as text, most telling first.

    Args:
        run: the run's info, with its options complete.
        scene: its scene, if the simulator has written it.

    Returns:
        By setting, its value, or None if this run did not record it.
    """
    o, mission = run["options"], run.get("mission")
    plan = scene["plan"] if scene else []
    alts = [p["alt"] for p in plan if p["kind"] == "wp"]
    alts = [a for i, a in enumerate(alts) if not i or alts[i - 1] != a]
    end = None
    if plan:
        at_home = math.hypot(plan[-1]["x"] - scene["home"][0], plan[-1]["y"] - scene["home"][1]) < 0.1
        end = "stay at the last waypoint" if plan[-1]["kind"] != "land" else "return to launch" if at_home else "land at the last waypoint"
    imagery = {"naip": "NAIP", "sentinel2": "Sentinel-2"}[run["imagery"]]
    camera = mission["camera"] if mission else None
    return {
        "Navigation": "odometry only" if o["navigation"] == "odometry" else "with map matching",
        "Map imagery": (imagery + (f" as of {run['imagery_date']}" if run.get("imagery_date") else "")
                        + (f", {scene['gsd']:g} m" if scene else "")),
        "Camera imagery": _camera_imagery(o),
        "Clouds": _sky(run["world"]),
        "Fog": _listed(o["fog"]) or "none",
        "Camera appearance": _listed(o["appearance"]) or "as the imagery",
        "Cruise altitude": ", ".join(f"{a:g}" for a in alts) + " m" if alts else None,
        "Takeoff altitude": f"{plan[0]['alt']:g} m" if plan else None,
        "Speed": f"{mission['speed']:g} m/s" if mission else None,
        "Corridor half-width": f"{mission['corridor_m']:g} m" if mission else None,
        "Camera field of view": f"{round(scene['camera']['hfov_deg'], 1):g}°" if scene else None,
        "Camera resolution": f"{camera['width_px']} × {camera['height_px']} px" if camera else None,
        "End of flight": end,
        "Wind": f"{o['wind_speed']:g} m/s from {o['wind_from_deg']:g}°",
        "Gusts": f"{o['gust']:g} m/s",
        "Compass bias": f"{o['compass_bias_deg']:g}°",
        "Height bias": f"{o['height_bias_pct']:g}%",
        "Seed": str(o["seed"]),
    }


class SimRuns:
    """Starts, tracks and lists simulation runs.

    Args:
        root: directory holding one subdirectory per run.
        library: the imagery caches.
        fetch_threads: cells downloaded in parallel for the camera's imagery.
    """

    def __init__(self, root: Path, library: Library, fetch_threads: int = 6):
        self.root, self.library = root, library
        self.fetch_threads = fetch_threads
        self.root.mkdir(parents=True, exist_ok=True)
        self._live: dict[str, dict] = {}  # in-memory state of runs this process is executing
        self._lock = threading.Lock()
        self._download_lock = threading.Lock()  # the two runs of a comparison need the same cells
        self.clear_worlds: dict[str, tuple[Path, float]] = {}  # by run, from _build_world to its command line
        self._scenes: dict[str, dict] = {}  # by run, see _scene

    def dir(self, run_id: str) -> Path | None:
        """The run's directory, or None if there is no such run."""
        if not RUN_ID.fullmatch(run_id) or not (self.root / run_id / "run.json").exists():
            return None
        return self.root / run_id

    def _start(self, runs: list[tuple[str, SimRequest, str | None]]):
        threads = [threading.Thread(target=self._execute, args=(run_id, req.mission, req.options), daemon=True)
                   for run_id, req, _ in runs]
        with self._lock:
            if sum(1 for live in self._live.values() if not live["cancel"]) + len(runs) > MAX_RUNNING:
                raise TooManyRuns(f"only {MAX_RUNNING} simulations can run at once")
            for (run_id, _, _), thread in zip(runs, threads):
                self._live[run_id] = {"state": "preparing", "proc": None, "cancel": False, "cells": None, "thread": thread}
        for (run_id, req, partner), thread in zip(runs, threads):
            d = self.root / run_id
            d.mkdir()
            (d / "run.json").write_text(json.dumps({
                "id": run_id, "name": req.mission.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "options": req.options.model_dump(), "partner": partner, "imagery": req.mission.imagery,
                "imagery_date": req.mission.imagery_date, "mission": req.mission.model_dump(),
            }))
            thread.start()

    def start(self, mission: Mission, options: SimOptions) -> dict:
        """Start a run in the background. Raises TooManyRuns if too many are running already."""
        run_id = uuid.uuid4().hex[:8]
        self._start([(run_id, SimRequest(mission=mission, options=options), None)])
        return self.info(run_id)

    def start_pair(self, first: SimRequest, second: SimRequest) -> list[dict]:
        """Start two flights as partners, or neither. Raises TooManyRuns as `start` does."""
        a, b = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
        self._start([(a, first, b), (b, second, a)])
        return [self.info(a), self.info(b)]

    def compare(self, mission: Mission, options: SimOptions) -> list[dict]:
        """Start the same flight with map matching and on odometry alone. Raises TooManyRuns as `start` does."""
        return self.start_pair(SimRequest(mission=mission, options=options.model_copy(update={"navigation": "map"})),
                               SimRequest(mission=mission, options=options.model_copy(update={"navigation": "odometry"})))

    def rerun_requests(self, a: str, b: str, new_seed: bool = False) -> list[SimRequest] | None:
        """What to start to fly two runs again.

        Args:
            a, b: the runs.
            new_seed: give both one new seed, instead of each keeping its own.

        Returns:
            The two requests, or None if a run is from before plans were kept with each run.
        """
        runs = [json.loads((self.root / i / "run.json").read_text()) for i in (a, b)]
        if not all("mission" in run for run in runs):
            return None
        requests = [SimRequest.model_validate({"mission": run["mission"], "options": run["options"]}) for run in runs]
        if new_seed:
            seed = random.randrange(100000)
            for req in requests:
                req.options.seed = seed
        return requests

    def _execute(self, run_id: str, mission: Mission, options: SimOptions):
        d, live = self.root / run_id, self._live[run_id]
        bundle = d / "bundle"
        try:
            write_bundle(mission, self.plan_cache(mission), bundle)
            cmd = [sys.executable, "-m", "visnav.main", "sim3d", "--bundle", str(bundle), "--port", "0", "--exit",
                   "--run-dir", str(d), "--seed", str(options.seed), "--gust", str(options.gust),
                   "--realtime", str(options.realtime)]
            # Wind "from" a direction blows toward the opposite one.
            speed, frm = options.wind_speed, math.radians(options.wind_from_deg)
            cmd += ["--wind-e", f"{-speed * math.sin(frm):.3f}", "--wind-n", f"{-speed * math.cos(frm):.3f}"]
            cmd += ["--compass-bias", str(options.compass_bias_deg), "--height-bias", str(options.height_bias_pct)]
            if options.navigation == "odometry":
                cmd += ["--matcher", "none"]
            if options.appearance:
                cmd += ["--appearance", ",".join(f"{k}={v}" for k, v in options.appearance.items())]
            if options.fog:
                cmd += ["--fog", ",".join(f"{k}={v}" for k, v in options.fog.items())]
            world = self._build_world(mission, options, bundle, live, d)
            if world:
                cmd += ["--world", str(world)]
            below = self.clear_worlds.pop(run_id, None)
            if below:
                cmd += ["--clear-world", str(below[0]), "--cloud-base", str(below[1])]
            with self._lock:
                if live["cancel"]:
                    return
                with open(d / "sim.log", "w") as out:
                    live["proc"] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
                live["state"] = "running"
            code = live["proc"].wait()
            # A negative code is a signal: stopped by us, or Ctrl-C reaching the whole process group.
            if code > 0 and not (d / "summary.json").exists():
                tail = (d / "sim.log").read_text().strip().splitlines()[-1:]
                (d / "error.txt").write_text(tail[0] if tail else f"simulator exited with code {code}")
        except Exception as exc:  # noqa: BLE001  shown to the user, the planner carries on
            log.exception("simulation %s failed", run_id)
            if not live["cancel"]:
                (d / "error.txt").write_text(str(exc))
        finally:
            shutil.rmtree(bundle, ignore_errors=True)
            with self._lock:
                self._live.pop(run_id, None)
            try:
                self.series(run_id)  # stores it
            except Exception:  # noqa: BLE001  it can be made again from the records
                log.exception("no series for simulation %s", run_id)

    def plan_cache(self, mission: Mission) -> CellCache:
        """The cache a plan's map is built from."""
        return self.library.clear(mission.imagery_date) if mission.imagery == "sentinel2" else self.library.naip()

    def _download(self, mission: Mission, cache: CellCache, what: str, live: dict):
        """Download the corridor's cells that are not cached."""
        def fetch(cell):
            if not live["cancel"]:
                cache.fetch(cell)
            with self._lock:
                live["cells"]["done"] += 1

        with self._download_lock:
            cells = [c for c in mission.required_cells() if not cache.has(c)]
            live.update(state="fetching", fetching=what, cells={"done": 0, "total": len(cells)})
            with ThreadPoolExecutor(max_workers=self.fetch_threads) as pool:
                list(pool.map(fetch, cells))

    def choose_pass(self, mission: Mission, options: SimOptions) -> str | None:
        """The Sentinel-2 pass for a flight: the one asked for, the last, or one of a year's by the seed.

        All of a year's passes count, however cloudy, so that over many seeds
        flights meet clouds as often as that year had them.
        """
        if options.pass_date:
            return options.pass_date
        bbox = mission.corridor()[1].bounds
        s2 = self.library.sentinel2
        if isinstance(options.world_year, int):
            passes = s2.passes(bbox, f"{options.world_year}-01-01", f"{options.world_year}-12-31")
            if not passes:
                return None
            dates = sorted(p["date"] for p in passes)
            return dates[int(np.random.default_rng([options.seed, options.world_year]).integers(len(dates)))]
        passes = s2.passes(bbox, limit=1)
        return passes[0]["date"] if passes else None

    def _build_world(self, mission: Mission, options: SimOptions, bundle: Path, live: dict, d: Path) -> Path | None:
        """Mosaic what the camera sees, with its clouds, downloading the cells that are not cached.

        Returns:
            The imagery to render the camera from, or None for the map itself.
        """
        sentinel = options.world_imagery == "sentinel2" and options.world_year is not None
        clouds, base = options.clouds, 0.0
        if clouds == "auto":
            clouds, base = ("real", CLOUD_BASE_M) if mission.max_alt() >= CLOUD_BASE_M else ("none", 0.0)
        info = {"choice": options.clouds, "clouds": clouds, "base_m": base, "pass_date": None}
        date = None
        if sentinel or clouds != "none":
            try:
                date = self.choose_pass(mission, options)
            except Exception as exc:  # noqa: BLE001  e.g. offline: fly without, and say so
                log.warning("no Sentinel-2 pass: %s", exc)
                info["note"] = f"Could not reach Sentinel-2 ({exc}), so no clouds."
            if date is None and "note" not in info:
                info["note"] = "No Sentinel-2 pass for that date, so no clouds."
            if sentinel and date is None:
                raise ValueError(info["note"])

        path, pass_tif = bundle / "world.tif", bundle / "pass.tif"
        if date:
            pass_cache = self.library.on_date(date)
            self._download(mission, pass_cache, f"Sentinel-2 pass of {date}", live)
            if live["cancel"]:
                return None
            build_map(mission, pass_cache, pass_tif, scl=True)
            info["pass_date"] = date

        if sentinel and clouds == "real":
            coverage = build_map(mission, self.library.on_date(date), path)["coverage"]
        elif sentinel:
            ground = self.library.clear(date)
            self._download(mission, ground, f"clear Sentinel-2 as of {date}", live)
            coverage = build_map(mission, ground, path)["coverage"]
        elif options.world_year:
            year = options.world_year
            ground = self.library.naip() if year == "newest" else self.library.naip(year)
            self._download(mission, ground, f"NAIP {'' if year == 'newest' else year}".strip(), live)
            coverage = build_map(mission, ground, path)["coverage"]
        elif date:
            coverage = build_map(mission, self.plan_cache(mission), path)["coverage"]
        else:
            (d / "world.json").write_text(json.dumps(info))
            return None
        if live["cancel"]:
            return None
        if coverage < 0.5:
            raise ValueError(f"The camera's imagery covers only {100 * coverage:.0f}% of the corridor")
        if date and base:
            # What the camera sees below the clouds, on the climb and the descent.
            clear = bundle / "world_clear.tif"
            if sentinel:
                ground = self.library.clear(date)
                self._download(mission, ground, f"clear Sentinel-2 as of {date}", live)
                build_map(mission, ground, clear)
            else:
                shutil.copy(path, clear)
            self.clear_worlds[d.name] = (clear, base)
        if date:
            info.update(overlay(path, pass_tif, clouds=clouds == "real", shadows=clouds != "none" and not (sentinel and clouds == "real")))
        (d / "world.json").write_text(json.dumps(info))
        return path

    def _last_record(self, d: Path) -> dict | None:
        path = d / "records.jsonl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            f.seek(max(path.stat().st_size - TAIL_BYTES, 0))
            lines = f.read().split(b"\n")
        # The last piece is empty, or a record still being written.
        return json.loads(lines[-2]) if len(lines) >= 2 else None

    def state(self, run_id: str) -> str:
        d = self.root / run_id
        if (d / "summary.json").exists():
            return "finished"
        live = self._live.get(run_id)
        if live is not None:
            return live["state"]
        return "failed" if (d / "error.txt").exists() else "stopped"

    def _scene(self, run_id: str) -> dict | None:
        """What is used here of scene.json, kept in memory: it never changes, and clouds can make the file large."""
        if run_id not in self._scenes:
            path = self.root / run_id / "scene.json"
            if not path.exists():
                return None
            scene = json.loads(path.read_text())
            self._scenes[run_id] = {name: scene[name] for name in ("home", "plan", "camera", "gsd")}
        return self._scenes[run_id]

    def info(self, run_id: str) -> dict:
        """What the website shows of a run."""
        d = self.root / run_id
        out = json.loads((d / "run.json").read_text())
        # Runs from before an option existed flew as its default does now.
        out["options"] = {**SimOptions().model_dump(), **out["options"]}
        out.setdefault("imagery", "naip")
        out["state"] = self.state(run_id)
        out["viewable"] = (d / "scene.json").exists()
        out["world"] = json.loads((d / "world.json").read_text()) if (d / "world.json").exists() else None
        out["summary"] = json.loads((d / "summary.json").read_text()) if out["state"] == "finished" else None
        out["error"] = (d / "error.txt").read_text() if out["state"] == "failed" else None
        live = self._live.get(run_id)
        out["cells"] = dict(live["cells"]) if live and live["cells"] else None
        out["fetching"] = live.get("fetching") if live else None
        out["progress"] = None
        scene = self._scene(run_id)
        if out["state"] == "running" and scene:
            rec = self._last_record(d)
            if rec:
                out["progress"] = {"t": rec["t"], "leg": rec["leg"], "legs": len(scene["plan"])}
        out["route"] = route_of(scene) if scene else None
        out["settings"] = flight_settings(out, scene)
        return out

    def series(self, run_id: str) -> dict | None:
        """The flight so far, thinned, for charts: distance flown (km), position error and true path (m from home).

        Once a run is over this is stored, so it needs its records only once. It
        is checked against their size, because a run that another planner process
        is flying looks over from here.
        """
        d = self.root / run_id
        path, stored, scene = d / "records.jsonl", d / "series.json", self._scene(run_id)
        size = path.stat().st_size if path.exists() else None
        if stored.exists():
            kept = json.loads(stored.read_text())
            if size is None or kept["records_bytes"] == size:
                return kept
        if size is None or scene is None:
            return None
        records = [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1]]  # the last may be half written
        if not records:
            return None
        km, flown = [], 0.0
        for prev, r in zip(records[:1] + records, records):
            flown += math.hypot(r["p"][0] - prev["p"][0], r["p"][1] - prev["p"][1])
            km.append(flown / 1000)
        keep = list(range(0, len(records), max(1, len(records) // SERIES_POINTS)))
        keep += [] if keep[-1] == len(records) - 1 else [len(records) - 1]
        out = {
            "km": [round(km[i], 4) for i in keep],
            "err": [round(math.hypot(records[i]["p"][0] - records[i]["ekf"][0], records[i]["p"][1] - records[i]["ekf"][1]), 2) for i in keep],
            "x": [round(records[i]["p"][0] - scene["home"][0], 2) for i in keep],
            "y": [round(records[i]["p"][1] - scene["home"][1], 2) for i in keep],
            "records_bytes": size,
        }
        if self.state(run_id) not in ACTIVE_STATES:
            tmp = stored.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text(json.dumps(out))
            tmp.replace(stored)
        return out

    def comparison(self, a: str, b: str) -> dict:
        """What the comparison page shows of two runs, with paths and plans measured from home."""
        runs = []
        for run_id in (a, b):
            run, series, scene = self.info(run_id), self.series(run_id), self._scene(run_id)
            run["series"] = series
            run["plan"] = [[p["x"] - scene["home"][0], p["y"] - scene["home"][1]] for p in scene["plan"]] if scene else []
            if run["summary"] and series:  # older summaries lack these
                distance, final = 1000 * series["km"][-1], series["err"][-1]
                run["summary"].setdefault("navigation", run["options"]["navigation"])
                run["summary"].setdefault("distance_m", round(distance, 1))
                run["summary"].setdefault("ekf_err_final_m", final)
                run["summary"].setdefault("final_err_pct_of_distance", round(100 * final / distance, 2) if distance else None)
            runs.append(run)
        return {"runs": runs, "active": any(r["state"] in ACTIVE_STATES for r in runs)}

    def list(self) -> list[dict]:
        """All runs, newest first."""
        runs = [self.info(p.parent.name) for p in self.root.glob("*/run.json")]
        return sorted(runs, key=lambda r: r["created"], reverse=True)

    def stop(self, run_id: str):
        with self._lock:
            live = self._live.get(run_id)
            if live is None:
                return
            live["cancel"] = True
            if live["proc"] is not None:
                live["proc"].terminate()

    def delete(self, run_id: str):
        self.stop(run_id)
        live = self._live.get(run_id)
        if live is not None:
            live["thread"].join(timeout=60)  # it may still be writing to the directory
        shutil.rmtree(self.root / run_id, ignore_errors=True)
        self._scenes.pop(run_id, None)

    def stop_all(self):
        for run_id in list(self._live):
            self.stop(run_id)
