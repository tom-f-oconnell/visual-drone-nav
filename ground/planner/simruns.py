"""Flight simulations started from the website.

Each run is a `visnav sim3d` subprocess, so a heavy simulation cannot stall the
planner, writing into its own directory under <data>/sim:

  run.json       what was asked for
  scene.json     static data for the viewer, written once the simulator is up
  map.jpg        the viewer's ground plane
  records.jsonl  the flight so far, one JSON record per line
  summary.json   error statistics, written last
  error.txt      why it failed, if it did

Two runs can be started as a comparison: the same flight with map matching and on
odometry alone, each naming the other as its partner in run.json.

The viewer follows records.jsonl, so a finished run replays the same way as a
live one, and runs survive a restart of the planner. The temporary bundle (the
map and camera GeoTIFFs) is deleted after each run, because it can be rebuilt
from the imagery caches.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
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

    def dir(self, run_id: str) -> Path | None:
        """The run's directory, or None if there is no such run."""
        if not RUN_ID.fullmatch(run_id) or not (self.root / run_id / "run.json").exists():
            return None
        return self.root / run_id

    def _start(self, mission: Mission, runs: list[tuple[str, SimOptions, str | None]]):
        threads = [threading.Thread(target=self._execute, args=(run_id, mission, options), daemon=True)
                   for run_id, options, _ in runs]
        with self._lock:
            if sum(1 for live in self._live.values() if not live["cancel"]) + len(runs) > MAX_RUNNING:
                raise TooManyRuns(f"only {MAX_RUNNING} simulations can run at once")
            for (run_id, _, _), thread in zip(runs, threads):
                self._live[run_id] = {"state": "preparing", "proc": None, "cancel": False, "cells": None, "thread": thread}
        for (run_id, options, partner), thread in zip(runs, threads):
            d = self.root / run_id
            d.mkdir()
            (d / "run.json").write_text(json.dumps({
                "id": run_id, "name": mission.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "options": options.model_dump(), "partner": partner, "imagery": mission.imagery,
                "imagery_date": mission.imagery_date,
            }))
            thread.start()

    def start(self, mission: Mission, options: SimOptions) -> dict:
        """Start a run in the background. Raises TooManyRuns if too many are running already."""
        run_id = uuid.uuid4().hex[:8]
        self._start(mission, [(run_id, options, None)])
        return self.info(run_id)

    def compare(self, mission: Mission, options: SimOptions) -> list[dict]:
        """Start the same flight with map matching and on odometry alone. Raises TooManyRuns as `start` does."""
        a, b = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
        self._start(mission, [(a, options.model_copy(update={"navigation": "map"}), b),
                              (b, options.model_copy(update={"navigation": "odometry"}), a)])
        return [self.info(a), self.info(b)]

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

    def info(self, run_id: str) -> dict:
        """What the website shows of a run."""
        d = self.root / run_id
        out = json.loads((d / "run.json").read_text())
        out["state"] = self.state(run_id)
        out["viewable"] = (d / "scene.json").exists()
        out["world"] = json.loads((d / "world.json").read_text()) if (d / "world.json").exists() else None
        out["summary"] = json.loads((d / "summary.json").read_text()) if out["state"] == "finished" else None
        out["error"] = (d / "error.txt").read_text() if out["state"] == "failed" else None
        live = self._live.get(run_id)
        out["cells"] = dict(live["cells"]) if live and live["cells"] else None
        out["fetching"] = live.get("fetching") if live else None
        out["progress"] = None
        if out["state"] == "running" and out["viewable"]:
            rec = self._last_record(d)
            if rec:
                legs = len(json.loads((d / "scene.json").read_text())["plan"])
                out["progress"] = {"t": rec["t"], "leg": rec["leg"], "legs": legs}
        return out

    def series(self, run_id: str) -> dict | None:
        """The flight so far, thinned, for charts: distance flown (km), position error and true path (m)."""
        path = self.root / run_id / "records.jsonl"
        if not path.exists():
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
        return {
            "km": [round(km[i], 4) for i in keep],
            "err": [round(math.hypot(records[i]["p"][0] - records[i]["ekf"][0], records[i]["p"][1] - records[i]["ekf"][1]), 2) for i in keep],
            "x": [records[i]["p"][0] for i in keep],
            "y": [records[i]["p"][1] for i in keep],
        }

    def comparison(self, a: str, b: str) -> dict:
        """What the comparison page shows of two runs."""
        runs = [{**self.info(i), "series": self.series(i)} for i in (a, b)]
        scenes = [self.root / i / "scene.json" for i in (a, b) if (self.root / i / "scene.json").exists()]
        plan = json.loads(scenes[0].read_text())["plan"] if scenes else []
        return {"runs": runs, "plan": [[p["x"], p["y"]] for p in plan], "active": any(r["state"] in ACTIVE_STATES for r in runs)}

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

    def stop_all(self):
        for run_id in list(self._live):
            self.stop(run_id)
