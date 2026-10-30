"""Flight simulations started from the website.

Each run is a `visnav sim3d` subprocess, so a heavy simulation cannot stall the
planner, writing into its own directory under <data>/sim:

  run.json       what was asked for
  scene.json     static data for the viewer, written once the simulator is up
  map.jpg        the viewer's ground plane
  records.jsonl  the flight so far, one JSON record per line
  summary.json   error statistics, written last
  error.txt      why it failed, if it did

The viewer follows records.jsonl, so a finished run replays the same way as a
live one, and runs survive a restart of the planner. The temporary bundle (the
map and camera GeoTIFFs) is deleted after each run, because it can be rebuilt
from the NAIP cache.
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
from typing import Callable

from pydantic import BaseModel, Field

from .mission import Mission, build_map, write_bundle
from .naip import NaipCache

log = logging.getLogger(__name__)
MAX_RUNNING = 3
ACTIVE_STATES = ("preparing", "fetching", "running")
RUN_ID = re.compile(r"[0-9a-f]{8}")
TAIL_BYTES = 1 << 18  # longer than any one record, thumbnail included


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
    world_year: int | None = Field(None, description="render the camera from this NAIP year instead of the map")


class SimRequest(BaseModel):
    mission: Mission
    options: SimOptions = SimOptions()


class SimRuns:
    """Starts, tracks and lists simulation runs.

    Args:
        root: directory holding one subdirectory per run.
        cache: NAIP cache the map is built from.
        year_cache: returns the NAIP cache of one year, for world_year.
        fetch_threads: cells downloaded in parallel for world_year.
    """

    def __init__(self, root: Path, cache: NaipCache, year_cache: Callable[[int], NaipCache], fetch_threads: int = 6):
        self.root, self.cache, self.year_cache = root, cache, year_cache
        self.fetch_threads = fetch_threads
        self.root.mkdir(parents=True, exist_ok=True)
        self._live: dict[str, dict] = {}  # in-memory state of runs this process is executing
        self._lock = threading.Lock()

    def dir(self, run_id: str) -> Path | None:
        """The run's directory, or None if there is no such run."""
        if not RUN_ID.fullmatch(run_id) or not (self.root / run_id / "run.json").exists():
            return None
        return self.root / run_id

    def start(self, mission: Mission, options: SimOptions) -> dict:
        """Start a run in the background. Raises TooManyRuns if too many are running already."""
        run_id = uuid.uuid4().hex[:8]
        thread = threading.Thread(target=self._execute, args=(run_id, mission, options), daemon=True)
        with self._lock:
            if sum(1 for live in self._live.values() if not live["cancel"]) >= MAX_RUNNING:
                raise TooManyRuns(f"{MAX_RUNNING} simulations are already running")
            self._live[run_id] = {"state": "preparing", "proc": None, "cancel": False, "cells": None, "thread": thread}
        d = self.root / run_id
        d.mkdir()
        (d / "run.json").write_text(json.dumps({
            "id": run_id, "name": mission.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "options": options.model_dump(),
        }))
        thread.start()
        return self.info(run_id)

    def _execute(self, run_id: str, mission: Mission, options: SimOptions):
        d, live = self.root / run_id, self._live[run_id]
        bundle = d / "bundle"
        try:
            write_bundle(mission, self.cache, bundle)
            cmd = [sys.executable, "-m", "visnav.main", "sim3d", "--bundle", str(bundle), "--port", "0", "--exit",
                   "--run-dir", str(d), "--seed", str(options.seed), "--gust", str(options.gust),
                   "--realtime", str(options.realtime)]
            # Wind "from" a direction blows toward the opposite one.
            speed, frm = options.wind_speed, math.radians(options.wind_from_deg)
            cmd += ["--wind-e", f"{-speed * math.sin(frm):.3f}", "--wind-n", f"{-speed * math.cos(frm):.3f}"]
            if options.world_year:
                cmd += ["--world", str(self._build_world(mission, options.world_year, bundle, live))]
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

    def _build_world(self, mission: Mission, year: int, bundle: Path, live: dict) -> Path:
        """Mosaic the corridor from one NAIP year, downloading the cells that are not cached."""
        cache = self.year_cache(year)
        cells = [c for c in mission.required_cells() if not cache.has(c)]
        live.update(state="fetching", cells={"done": 0, "total": len(cells)})

        def fetch(cell):
            if not live["cancel"]:
                cache.fetch(cell)
            with self._lock:
                live["cells"]["done"] += 1

        with ThreadPoolExecutor(max_workers=self.fetch_threads) as pool:
            list(pool.map(fetch, cells))
        if live["cancel"]:
            return bundle
        path = bundle / "world.tif"
        coverage = build_map(mission, cache, path)["coverage"]
        if coverage < 0.5:
            raise ValueError(f"NAIP {year} covers only {100 * coverage:.0f}% of the corridor")
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
        out["summary"] = json.loads((d / "summary.json").read_text()) if out["state"] == "finished" else None
        out["error"] = (d / "error.txt").read_text() if out["state"] == "failed" else None
        live = self._live.get(run_id)
        out["cells"] = dict(live["cells"]) if live and live["cells"] else None
        out["progress"] = None
        if out["state"] == "running" and out["viewable"]:
            rec = self._last_record(d)
            if rec:
                legs = len(json.loads((d / "scene.json").read_text())["plan"])
                out["progress"] = {"t": rec["t"], "leg": rec["leg"], "legs": legs}
        return out

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
