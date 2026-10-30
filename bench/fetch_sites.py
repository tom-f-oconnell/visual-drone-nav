"""Download NAIP for the benchmark sites and build a map bundle and a world per site.

Usage: fetch_sites.py DATA_DIR [SITE ...] [--parallel 4]

Writes, per site, DATA_DIR/<site>/map/ (a mission bundle: map.tif from the
older NAIP year at 0.6 m, mission.json, mission.waypoints) and
DATA_DIR/<site>/world.tif (the newer year at its finest GSD) plus world.json
listing the scenes used. Cells are cached under DATA_DIR/naip as the planner
does, so reruns only build the mosaics.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from planner.mission import Camera, Mission, build_bundle, build_map
from planner.naip import NaipCache, StacSource

from sites import BY_NAME, SITES, Site

MAP_GSD = 0.6
MAP_CORRIDOR_M = 400    # wide enough for a 150 m search around a prior that is itself off the route
WORLD_CORRIDOR_M = 250  # camera footprints only


def mission_for(site: Site, corridor_m: float) -> Mission:
    # "land" so the corridor covers only the benchmark's two legs, not a return leg home.
    return Mission(
        name=site.name, waypoints=site.waypoints(), default_alt=120, takeoff_alt=30, speed=10,
        end_action="land", corridor_m=corridor_m, camera=Camera(hfov_deg=90, width_px=960, height_px=600),
    )


def fetch(mission: Mission, cache: NaipCache, threads: int) -> list[dict]:
    cells = mission.required_cells()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(cache.fetch, cells))
    return [cache.meta(c) for c in cells if cache.has(c)]


def scenes_of(metas: list[dict]) -> list[dict]:
    seen = {}
    for m in metas:
        for s in m["scenes"]:
            seen[s["id"]] = s
    return sorted(seen.values(), key=lambda s: s["id"])


def build_site(site: Site, data: Path, threads: int = 6):
    out = data / site.name
    out.mkdir(parents=True, exist_ok=True)
    map_cache = NaipCache(data / "naip", StacSource(years=[site.map_year]), MAP_GSD)
    world_cache = NaipCache(data / "naip", StacSource(years=[site.world_year]), site.world_gsd)

    m = mission_for(site, MAP_CORRIDOR_M)
    map_cells = fetch(m, map_cache, threads)
    build_bundle(m, map_cache, out / "map")

    w = mission_for(site, WORLD_CORRIDOR_M)
    world_cells = fetch(w, world_cache, threads)
    world_meta = build_map(w, world_cache, out / "world.tif")
    world_meta["scenes"] = scenes_of(world_cells)
    (out / "world.json").write_text(json.dumps(world_meta, indent=2))

    map_scenes = scenes_of(map_cells)
    print(f"{site.name}: map {[s['id'] for s in map_scenes]}  world {[s['id'] for s in world_meta['scenes']]}"
          f"  world coverage {world_meta['coverage']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("sites", nargs="*", help="default: all")
    ap.add_argument("--parallel", type=int, default=4, help="sites fetched at once")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    sites = [BY_NAME[n] for n in args.sites] if args.sites else SITES
    with ThreadPoolExecutor(args.parallel) as ex:
        for f in [ex.submit(build_site, s, args.data) for s in sites]:
            f.result()


if __name__ == "__main__":
    main()
