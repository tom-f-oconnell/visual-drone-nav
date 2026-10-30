"""Local registration offset between each site's map and world, at every benchmark frame.

Usage: registration.py DATA_DIR [SITE ...] [--frames 80]

NAIP years are orthorectified separately, so the same ground feature can sit
several metres apart in the map and the world (tree and building lean, DEM
differences, seamlines between quarter quads). A perfect matcher then reports
the camera where the map shows it, not where the world was rendered, so a
"wrong" fix 10 to 15 m off may be the map's error rather than the matcher's.
This measures that offset over a 300 m window around each frame position
(gradient-orientation correlation of the map and of the world resampled to the
map grid) and writes DATA_DIR/<site>/registration.csv. Offsets with a PSR
under about 8 are unreliable, and so are offsets along orchard rows or other
straight repeated structure, which constrain only one direction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from match_bench import route_poses
from sites import BY_NAME, SITES
from visnav.matcher import orientation_field, parabolic_2d

WINDOW_M = 300.0
MAX_OFFSET_M = 20.0


def world_on_map_grid(path, map_) -> np.ndarray:
    """World imagery as grey levels on the map's pixel grid, 0 where it has no data."""
    with rasterio.open(path) as src:
        g = cv2.cvtColor(np.ascontiguousarray(np.moveaxis(src.read([1, 2, 3]), 0, -1)), cv2.COLOR_RGB2GRAY)
        out = np.zeros(map_.gray.shape, np.uint8)
        reproject(g, out, src_transform=src.transform, src_crs=src.crs, dst_transform=map_.transform,
                  dst_crs=map_.crs, resampling=Resampling.average, src_nodata=0, dst_nodata=0)
    return out


def offset_at(map_gray: np.ndarray, world_gray: np.ndarray, col: float, row: float, gsd: float):
    """(north, east) metres to add to a world position to get the map position of the same ground, and PSR."""
    h = int(WINDOW_M / gsd / 2)
    c, r = int(col), int(row)
    a = map_gray[r - h:r + h, c - h:c + h]
    b = world_gray[r - h:r + h, c - h:c + h]
    if a.shape != (2 * h, 2 * h) or (b == 0).mean() > 0.05 or (a == 0).mean() > 0.05:
        return np.nan, np.nan, 0.0
    fa, fb = orientation_field(a, 2.0, 2.0), orientation_field(b, 2.0, 2.0)
    R = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(fa) * np.conj(np.fft.fft2(fb)))))
    m = int(MAX_OFFSET_M / gsd)
    win = R[h - m:h + m + 1, h - m:h + m + 1]
    y, x = np.unravel_index(int(np.argmax(win)), win.shape)
    # Sidelobe statistics over every shift, not just the plausible ones, so the PSR is comparable between sites.
    side = np.ones(R.shape, bool)
    side[h - m + y - 3:h - m + y + 4, h - m + x - 3:h - m + x + 4] = False
    psr = float((win[y, x] - R[side].mean()) / (R[side].std() + 1e-9))
    dx, dy = parabolic_2d(win, y, x)
    return -(y + dy - m) * gsd, (x + dx - m) * gsd, psr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("sites", nargs="*")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    for site in [BY_NAME[n] for n in args.sites] or SITES:
        map_, _, poses = route_poses(site.name, args.data, args.frames, args.seed)
        world = world_on_map_grid(args.data / site.name / "world.tif", map_)
        rows = []
        for i, p in poses:
            n, e, psr = offset_at(map_.gray, world, *map_.ll_to_px(p.lat, p.lon), map_.gsd)
            rows.append({"frame": i, "reg_n_m": round(n, 2), "reg_e_m": round(e, 2), "psr": round(psr, 1)})
        with open(args.data / site.name / "registration.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        good = [r for r in rows if r["psr"] >= 8]
        mag = [float(np.hypot(r["reg_n_m"], r["reg_e_m"])) for r in good]
        print(f"{site.name:11s} measured at {len(good)}/{len(rows)} frames; offset median "
              f"{np.median(mag) if mag else float('nan'):.1f} m, max {max(mag) if mag else float('nan'):.1f} m")


if __name__ == "__main__":
    main()
