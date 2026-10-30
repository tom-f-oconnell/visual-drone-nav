"""Cross-year matching benchmark: correct, wrong and missing fixes per matcher.

Usage: match_bench.py DATA_DIR OUT_DIR [--sites ...] [--matchers SPEC ...] [--frames 80]

For each site built by fetch_sites.py, frames are rendered from the world (a
later NAIP year) at poses along the route, orthorectified with the navigator's
code from a noisy attitude and height, and located by each matcher against the
map, with the prior at the true position plus a random offset, for each search
radius. Every (site, matcher) pair runs as its own process and renders the same
frames from the same seeds. One CSV row per frame and radius goes to
OUT_DIR/<site>__<matcher>.csv; report.py turns them into tables.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pyproj import Geod

from sites import BY_NAME, SITES

GEOD = Geod(ellps="WGS84")
CORRECT_M = 10.0
DEFAULT_MATCHERS = ["orb", "corr", "lg:features=disk", "lg:features=aliked"]
FIELDS = ["site", "frame", "radius_m", "matcher", "fixed", "err_m", "err_n_m", "err_e_m", "score", "ms",
          "offset_m", "yaw_err_deg", "est_heading_deg", "diag"]


def tag(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", spec)


def route_poses(site: str, data: Path, n: int, seed: int):
    """The site's map, camera and n (index, pose) pairs evenly spaced along the route."""
    from visnav.geo import MapRaster
    from visnav.ortho import CameraModel
    from visnav.sim import fly_route

    meta = json.loads((data / site / "map" / "mission.json").read_text())
    map_ = MapRaster.load(data / site / "map" / "map.tif")
    c = meta["mission"]["camera"]
    cam = CameraModel.from_hfov(c["width_px"], c["height_px"], c["hfov_deg"])
    route = [(w["lat"], w["lon"], w["alt"] or meta["mission"]["default_alt"]) for w in meta["mission"]["waypoints"]]
    poses = list(fly_route(map_, route, meta["mission"]["speed"], 1.0, np.random.default_rng(seed)))
    idx = np.unique(np.linspace(0, len(poses) - 1, n).round().astype(int))
    return map_, cam, [(int(i), poses[i]) for i in idx]


def frames(site: str, data: Path, n: int, seed: int, att_noise: float, yaw_noise: float, alt_noise: float,
           same_year: bool = False, detail: float = 0.0):
    """Yield (index, pose, ortho, heading error, rng, map) for n frames evenly spaced along the route.

    The navigator is given roll/pitch with att_noise, heading with yaw_noise
    (degrees, 1 sigma) and height with alt_noise (fraction), as `visnav sim` does.
    With same_year, frames are rendered from the map itself instead of the world.
    detail is the contrast of render's procedural sub-NAIP ground texture, where
    the installed visnav has it.
    """
    from visnav.geo import MapRaster
    from visnav.ortho import orthorectify
    from visnav.sim import render

    map_, cam, poses = route_poses(site, data, n, seed)
    world = map_ if same_year else MapRaster.load(data / site / "world.tif")
    key = zlib.crc32(site.encode())
    kw = {"detail": detail} if "detail" in inspect.signature(render).parameters else {}
    for i, p in poses:
        rng = np.random.default_rng([seed, key, i])
        img = render(world, cam, p.lat, p.lon, p.h, p.roll, p.pitch, p.yaw, rng, **kw)
        yaw_err = rng.normal(0, yaw_noise)
        roll = p.roll + math.radians(rng.normal(0, att_noise))
        pitch = p.pitch + math.radians(rng.normal(0, att_noise))
        h = p.h * (1 + rng.normal(0, alt_noise))
        J = map_.ned_to_px_jacobian(p.lat, p.lon)
        ortho = orthorectify(cam.undistort(img), cam, roll, pitch, p.yaw + math.radians(yaw_err), h, J)
        yield i, p, ortho, yaw_err, rng, map_


def run_job(site: str, spec: str, data: Path, out: Path, args) -> str:
    import cv2

    cv2.setNumThreads(args.threads)
    import matchers

    if spec.startswith("lg"):
        import torch

        torch.set_num_threads(args.threads)
        if torch.get_num_interop_threads() != 1:  # settable once per process, and workers are reused
            torch.set_num_interop_threads(1)
    rows, matcher, setup_s = [], None, 0.0
    for i, p, ortho, yaw_err, rng, map_ in frames(site, data, args.frames, args.seed, args.att_noise,
                                                    args.yaw_noise, args.alt_noise, args.same_year, args.detail):
        if matcher is None:
            t0 = time.perf_counter()
            matcher = matchers.make(spec, map_)
            setup_s = time.perf_counter() - t0
        col, row = map_.ll_to_px(p.lat, p.lon)
        for radius in args.radii:
            off = rng.normal(0, radius / 3, 2)
            off *= min(1.0, radius / max(float(np.hypot(*off)), 1e-9))
            rec = {"site": site, "frame": i, "radius_m": radius, "matcher": spec, "fixed": 0, "err_m": "",
                   "err_n_m": "", "err_e_m": "", "score": "", "ms": "", "offset_m": round(float(np.hypot(*off)), 1),
                   "yaw_err_deg": round(yaw_err, 2), "est_heading_deg": "", "diag": ""}
            if ortho is not None:
                t0 = time.perf_counter()
                fix = matcher.locate(ortho, col + off[1] / map_.gsd, row - off[0] / map_.gsd,
                                     int(radius / map_.gsd))
                rec["ms"] = round(1000 * (time.perf_counter() - t0), 1)
                if fix is not None:
                    az, _, dist = GEOD.inv(p.lon, p.lat, fix.lon, fix.lat)
                    rec.update(fixed=1, err_m=round(dist, 2), err_n_m=round(dist * math.cos(math.radians(az)), 2),
                               err_e_m=round(dist * math.sin(math.radians(az)), 2),
                               score=round(fix.score, 3), est_heading_deg=round(fix.heading_err_deg, 2),
                               diag=json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                for k, v in fix.diag.items()}))
            rows.append(rec)
    path = out / f"{site}__{tag(spec)}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / f"{site}__{tag(spec)}.json").write_text(json.dumps({"setup_s": round(setup_s, 2), "spec": spec,
                                                                "threads": args.threads}))
    fixed = [r for r in rows if r["fixed"]]
    ok = sum(1 for r in fixed if r["err_m"] < CORRECT_M)
    return f"{site:11s} {spec:24s} n={len(rows)} correct={ok} wrong={len(fixed) - ok} (unthresholded)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--sites", nargs="*", default=[s.name for s in SITES])
    ap.add_argument("--matchers", nargs="*", default=DEFAULT_MATCHERS,
                    help='matcher specs, e.g. orb "corr:down=4,angles=-3/0/3" "lg:features=aliked"')
    ap.add_argument("--frames", type=int, default=80, help="frames per site")
    ap.add_argument("--radii", type=float, nargs="*", default=[40.0, 150.0], help="search radii, m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--att-noise", type=float, default=1.0, help="deg roll/pitch error")
    ap.add_argument("--yaw-noise", type=float, default=3.0, help="deg heading error")
    ap.add_argument("--alt-noise", type=float, default=0.03, help="fractional height error")
    ap.add_argument("--same-year", action="store_true", help="render frames from the map, not the world")
    ap.add_argument("--detail", type=float, default=0.0,
                    help="contrast of visnav.sim.render's procedural ground texture (0: NAIP only)")
    ap.add_argument("--jobs", type=int, default=12, help="processes")
    ap.add_argument("--threads", type=int, default=1, help="threads per process")
    args = ap.parse_args(argv)
    # Workers inherit these, so BLAS/OpenMP pools stay at --threads.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    unknown = [s for s in args.sites if s not in BY_NAME]
    if unknown:
        ap.error(f"unknown sites {unknown}")
    jobs = [(s, m) for m in args.matchers for s in args.sites]
    # Slow (learned) matchers first so they do not finish last on their own.
    jobs.sort(key=lambda j: not j[1].startswith("lg"))
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_job, s, m, args.data, args.out, args) for s, m in jobs]
        for f in futs:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
