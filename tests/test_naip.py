"""NAIP cache: overview reads, alignment of cached cells and year-keyed cache directories."""

import math

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from planner.naip import Cell, LocalSource, NaipCache, StacSource


def checker(xs, ys):
    """1 on alternate 6 m squares of the UTM grid, else 0."""
    return ((np.floor(xs / 6)[None, :] + np.floor(ys / 6)[:, None]) % 2).astype(np.uint8)


def test_fine_scene_read_from_overview_and_aligned(tmp_path):
    cell = Cell.containing(40.005, -105.295)
    crs, transform, width, height = cell.grid(0.6)
    l, t = transform.c, transform.f
    # A 0.3 m scene with margin around the cell. Band 1 is a checkerboard, to
    # check the cached cell lands exactly on the UTM grid. Band 2 has one pixel
    # stripes, which only the 0.6 m overview (built with nearest) makes constant.
    x0, y0 = (l // 6 - 2) * 6, (t // 6 + 2) * 6
    nx = 4 * math.ceil((width * 0.6 + 24) / 1.2)
    ny = 4 * math.ceil((height * 0.6 + 24) / 1.2)
    xs, ys = x0 + 0.3 * (np.arange(nx) + 0.5), y0 - 0.3 * (np.arange(ny) + 0.5)
    stripes = np.where(np.arange(nx) % 2, 200, 100).astype(np.uint8)
    rgb = np.stack([50 + 150 * checker(xs, ys), np.broadcast_to(stripes, (ny, nx)),
                    np.full((ny, nx), 80, np.uint8)])
    (tmp_path / "src").mkdir()
    with rasterio.open(tmp_path / "src" / "m_test_20230901.tif", "w", driver="GTiff", width=nx, height=ny,
                       count=3, dtype="uint8", crs=crs, transform=from_origin(x0, y0, 0.3, 0.3), nodata=0,
                       tiled=True) as dst:
        dst.write(rgb)
        dst.build_overviews([2, 4], Resampling.nearest)

    cache = NaipCache(tmp_path / "cache", LocalSource(tmp_path / "src"), gsd=0.6)
    with rasterio.open(cache.fetch(cell)) as src:
        got = src.read()
    gx, gy = l + 0.6 * (np.arange(width) + 0.5), t - 0.6 * (np.arange(height) + 0.5)
    assert np.array_equal(got[0], 50 + 150 * checker(gx, gy))
    assert np.unique(got[1]).size == 1
    assert cache.meta(cell)["coverage"] == 1.0


def test_cache_dir_keyed_by_years(tmp_path):
    assert NaipCache(tmp_path, StacSource(), 0.6).dir.name == "gsd_0.60"
    assert NaipCache(tmp_path, StacSource(years=[2023, 2021]), 0.6).dir.name == "gsd_0.60_y2021-2023"
