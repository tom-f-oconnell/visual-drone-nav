"""Aerial and satellite imagery download and on-disk cache.

Imagery is cached on a fixed grid of 0.01 x 0.01 degree cells (about 1.1 km x
0.85 km at mid US latitudes). Each cell is stored once as an RGB GeoTIFF in
the UTM zone of its centre, at a fixed ground sample distance, so cells are
reused by every mission that touches them.

Three imagery sources are supported:

* ``StacSource``: NAIP from the Microsoft Planetary Computer STAC API (default).
  The COGs are read with windowed HTTP range requests, so only the pixels
  covering a cell are downloaded, never the whole 7.5 minute quarter quad. Scenes
  finer than the cache GSD (e.g. 0.3 m NAIP from 2022 on) are read from a COG
  overview, so they cost no more than 0.6 m scenes.
* ``Sentinel2Source``: Sentinel-2 true colour from the same API, worldwide at
  10 m, read the same way. For maps, the newest clear scenes as of some date, with
  each scene's cloud classification keeping clouds out of the cell. For simulated
  flights, one pass as it was, clouds and all (``PassSource``).
* ``LocalSource``: a directory of GeoTIFFs you downloaded yourself (for example
  NAIP from USGS EarthExplorer), useful offline or on an air-gapped laptop.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

log = logging.getLogger(__name__)

CELL_DEG = 0.01
CACHED_CELL = re.compile(r"(-?\d+)_(-?\d+)\.tif")  # not the temporary name a cell has while being written
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# GDAL settings that make windowed COG reads over HTTP efficient.
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE="TRUE",
)


def utm_epsg(lat: float, lon: float) -> int:
    zone = int(math.floor((lon + 180) / 6)) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass(frozen=True)
class Cell:
    """One cache cell, identified by integer indices of its south-west corner."""

    iy: int
    ix: int

    @classmethod
    def containing(cls, lat: float, lon: float) -> "Cell":
        return cls(math.floor(lat / CELL_DEG), math.floor(lon / CELL_DEG))

    @property
    def key(self) -> str:
        return f"{self.iy}_{self.ix}"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in degrees."""
        s, w = self.iy * CELL_DEG, self.ix * CELL_DEG
        return (w, s, w + CELL_DEG, s + CELL_DEG)

    @property
    def epsg(self) -> int:
        w, s, e, n = self.bbox
        return utm_epsg((s + n) / 2, (w + e) / 2)

    def grid(self, gsd: float):
        """Pixel grid of this cell: (crs, transform, width, height), snapped to gsd."""
        crs = f"EPSG:{self.epsg}"
        l, b, r, t = transform_bounds("EPSG:4326", crs, *self.bbox, densify_pts=21)
        l, b = math.floor(l / gsd) * gsd, math.floor(b / gsd) * gsd
        r, t = math.ceil(r / gsd) * gsd, math.ceil(t / gsd) * gsd
        width, height = int(round((r - l) / gsd)), int(round((t - b) / gsd))
        return crs, from_origin(l, t, gsd, gsd), width, height


def cells_for_bbox(west: float, south: float, east: float, north: float) -> list[Cell]:
    a, b = Cell.containing(south, west), Cell.containing(north, east)
    return [Cell(iy, ix) for iy in range(a.iy, b.iy + 1) for ix in range(a.ix, b.ix + 1)]


def neighbouring_groups(cells: Iterable[Cell]) -> list[set[Cell]]:
    """Splits cells into groups joined by a shared edge or corner."""
    unseen, groups = set(cells), []
    while unseen:
        group, frontier = set(), [unseen.pop()]
        while frontier:
            cell = frontier.pop()
            group.add(cell)
            around = {Cell(cell.iy + dy, cell.ix + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
            frontier.extend(around & unseen)
            unseen -= around
        groups.append(group)
    return groups


@dataclass
class SceneRef:
    href: str
    year: int
    id: str = ""
    date: str = ""
    mask_href: str = ""  # Sentinel-2 scene classification, to keep clouds out
    scl_href: str = ""   # the same, to store with the cell


class Source(Protocol):
    name: str

    def scenes(self, bbox: tuple[float, float, float, float]) -> list[SceneRef]:
        """Scenes intersecting bbox (lon/lat), newest first."""


class StacSource:
    """NAIP from a STAC API. Defaults to Microsoft Planetary Computer."""

    name = "planetary-computer"

    def __init__(self, url: str = PC_STAC_URL, collection: str = "naip", asset: str = "image",
                 years: Iterable[int] | None = None):
        self.url, self.collection, self.asset = url, collection, asset
        self.years = set(years) if years else None
        self._local = threading.local()  # one client per thread, cells are fetched in parallel

    def _open(self):
        if getattr(self._local, "client", None) is None:
            import pystac_client

            modifier = None
            if "planetarycomputer" in self.url:
                import planetary_computer

                modifier = planetary_computer.sign_inplace
            self._local.client = pystac_client.Client.open(self.url, modifier=modifier)
        return self._local.client

    def scenes(self, bbox):
        search = self._open().search(collections=[self.collection], bbox=list(bbox), max_items=200)
        out = []
        for item in search.items():
            year = int(str(item.properties.get("naip:year") or item.datetime.year))
            if self.years and year not in self.years:
                continue
            asset = item.assets.get(self.asset)
            if asset is not None:
                out.append(SceneRef(asset.href, year, item.id))
        out.sort(key=lambda s: s.year, reverse=True)
        return out


class Sentinel2Source(StacSource):
    """Sentinel-2 L2A true colour from a STAC API.

    ``scenes`` is what a clear map is made from: the newest scenes, as of a date
    if given, that are not too cloudy. ``passes`` and ``pass_scenes`` are everything
    that was flown over, however cloudy.

    Args:
        url: STAC API.
        max_cloud: scenes cloudier than this percentage are not used for clear maps.
        max_scenes: how many scenes to look through for each cell. Sentinel-2
            passes every few days, so this reaches back a few months.
        as_of: newest date to use for clear maps (YYYY-MM-DD), by default today.
    """

    name = "sentinel-2"

    def __init__(self, url: str = PC_STAC_URL, max_cloud: float = 20.0, max_scenes: int = 16, as_of: str | None = None):
        super().__init__(url, collection="sentinel-2-l2a", asset="visual")
        self.max_cloud, self.max_scenes, self.as_of = max_cloud, max_scenes, as_of

    @property
    def tag(self) -> str:
        return f"asof{self.as_of}" if self.as_of else ""

    def at(self, as_of: str | None) -> "Sentinel2Source":
        return Sentinel2Source(self.url, self.max_cloud, self.max_scenes, as_of)

    def _ref(self, item, **kw) -> SceneRef:
        return SceneRef(item.assets[self.asset].href, item.datetime.year, item.id, item.datetime.date().isoformat(), **kw)

    def scenes(self, bbox):
        when = {"datetime": f"2015-01-01/{self.as_of}"} if self.as_of else {}
        search = self._open().search(collections=[self.collection], bbox=list(bbox), max_items=self.max_scenes,
                                     query={"eo:cloud_cover": {"lt": self.max_cloud}}, sortby="-properties.datetime", **when)
        return [self._ref(item, mask_href=item.assets["SCL"].href) for item in search.items()]

    def passes(self, bbox, start: str | None = None, end: str | None = None, limit: int = 500) -> list[dict]:
        """Every date with a pass over bbox, newest first, with its cloud percentage over the whole 110 km tile."""
        when = {"datetime": f"{start or '2015-01-01'}/{end or '2100-01-01'}"} if start or end else {}
        search = self._open().search(collections=[self.collection], bbox=list(bbox), max_items=limit, sortby="-properties.datetime", **when)
        dates: dict[str, list[float]] = {}
        for item in search.items():
            dates.setdefault(item.datetime.date().isoformat(), []).append(float(item.properties.get("eo:cloud_cover", 0.0)))
        return [{"date": d, "cloud_pct": round(sum(c) / len(c))} for d, c in dates.items()]

    def first_year(self, bbox) -> int | None:
        items = list(self._open().search(collections=[self.collection], bbox=list(bbox), max_items=1, sortby="properties.datetime").items())
        return items[0].datetime.year if items else None

    def pass_scenes(self, bbox, date: str) -> list[SceneRef]:
        """Scenes of one date first, then the nearest in time, for ground that pass did not reach."""
        day = datetime.date.fromisoformat(date)
        span = datetime.timedelta(days=15)
        search = self._open().search(collections=[self.collection], bbox=list(bbox), max_items=40,
                                     datetime=f"{day - span}/{day + span}", sortby="-properties.datetime")
        refs = [self._ref(item, scl_href=item.assets["SCL"].href) for item in search.items()]
        return sorted(refs, key=lambda r: abs((datetime.date.fromisoformat(r.date) - day).days))


class PassSource:
    """One Sentinel-2 pass as it was, clouds and all, with its scene classification, for simulated flights."""

    name = "sentinel-2-pass"

    def __init__(self, source, date: str):
        self.source, self.date, self.tag = source, date, f"pass{date}"

    def scenes(self, bbox):
        return self.source.pass_scenes(bbox, self.date)


class LocalSource:
    """GeoTIFFs in a local directory (searched recursively)."""

    name = "local"

    def __init__(self, root: str | os.PathLike, years: Iterable[int] | None = None):
        self.root = Path(root)
        self.years = set(years) if years else None
        self._index: list[tuple[SceneRef, tuple[float, float, float, float]]] | None = None

    def _build_index(self):
        idx = []
        for p in sorted(self.root.rglob("*.tif")):
            try:
                with rasterio.open(p) as src:
                    bb = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    year = int(src.tags().get("YEAR", 0)) or _year_from_name(p.name)
            except rasterio.errors.RasterioIOError:
                continue
            idx.append((SceneRef(str(p), year, p.stem), bb))
        self._index = idx

    def scenes(self, bbox):
        if self._index is None:
            self._build_index()
        w, s, e, n = bbox
        hits = [ref for ref, (l, b, r, t) in self._index if l < e and r > w and b < n and t > s
                and (not self.years or ref.year in self.years)]
        hits.sort(key=lambda r: r.year, reverse=True)
        return hits


def _year_from_name(name: str) -> int:
    # NAIP file names end in _YYYYMMDD.tif, e.g. m_3910501_ne_13_060_20210620.tif
    digits = name.rsplit("_", 1)[-1][:4]
    return int(digits) if digits.isdigit() else 0


class LocalSentinel2Source:
    """Sentinel-2 as GeoTIFFs in a directory: *_YYYYMMDD.tif, each with an optional classification *_YYYYMMDD_scl.tif."""

    name = "local-sentinel-2"

    def __init__(self, root: str | os.PathLike, as_of: str | None = None):
        self.root, self.as_of = Path(root), as_of

    @property
    def tag(self) -> str:
        return f"asof{self.as_of}" if self.as_of else ""

    def at(self, as_of: str | None) -> "LocalSentinel2Source":
        return LocalSentinel2Source(self.root, as_of)

    def _files(self, bbox) -> list[SceneRef]:
        """Scenes over bbox, newest first, with mask_href set where there is a classification."""
        w, s, e, n = bbox
        refs = []
        for p in sorted(self.root.rglob("*.tif")):
            if p.stem.endswith("_scl"):
                continue
            digits = p.stem.rsplit("_", 1)[-1]
            with rasterio.open(p) as src:
                l, b, r, t = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            if l < e and r > w and b < n and t > s:
                scl = p.with_name(f"{p.stem}_scl.tif")
                refs.append(SceneRef(str(p), int(digits[:4]), p.stem, f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}", str(scl) if scl.exists() else ""))
        return sorted(refs, key=lambda r: r.date, reverse=True)

    def scenes(self, bbox):
        return [r for r in self._files(bbox) if not self.as_of or r.date <= self.as_of]

    def passes(self, bbox, start: str | None = None, end: str | None = None, limit: int = 500) -> list[dict]:
        out = []
        for r in self._files(bbox):
            if (start and r.date < start) or (end and r.date > end):
                continue
            cloud = 0.0
            if r.mask_href:
                with rasterio.open(r.mask_href) as src:
                    cloud = 100 * float(np.isin(src.read(1), (8, 9, 10)).mean())
            out.append({"date": r.date, "cloud_pct": round(cloud)})
        return out[:limit]

    def first_year(self, bbox) -> int | None:
        files = self._files(bbox)
        return files[-1].year if files else None

    def pass_scenes(self, bbox, date: str) -> list[SceneRef]:
        day = datetime.date.fromisoformat(date)
        refs = [SceneRef(r.href, r.year, r.id, r.date, scl_href=r.mask_href) for r in self._files(bbox)]
        return sorted(refs, key=lambda r: abs((datetime.date.fromisoformat(r.date) - day).days))


def read_scene(href: str, crs: str, transform, width: int, height: int) -> np.ndarray | None:
    """Read the RGB of one scene reprojected onto a pixel grid, 0 where it has no data.

    One windowed read, then an in-memory reproject. Over HTTP this is several
    times faster than a WarpedVRT, which fetches blocks in many small serial
    requests. Reads from the coarsest overview that is still at least as fine
    as the grid, which also avoids aliasing when downsampling.

    Returns:
        3 x height x width uint8 array, or None if the scene misses the grid.
    """
    gsd = abs(transform.a)
    with rasterio.open(href) as src:
        level = None
        for i, factor in enumerate(src.overviews(1)):
            if src.res[0] * factor <= gsd * 1.01:
                level = i
    with rasterio.open(href, **({} if level is None else {"overview_level": level})) as src:
        bounds = transform_bounds(crs, src.crs, *array_bounds(height, width, transform), densify_pts=21)
        win = from_bounds(*bounds, src.transform)
        # Pad so bilinear resampling has neighbours at the edges.
        win = Window(math.floor(win.col_off) - 2, math.floor(win.row_off) - 2,
                     math.ceil(win.width) + 4, math.ceil(win.height) + 4)
        try:
            win = win.intersection(Window(0, 0, src.width, src.height))
        except WindowError:
            return None
        data = src.read([1, 2, 3], window=win)
        src_transform, src_crs = src.window_transform(win), src.crs
    out = np.zeros((3, height, width), np.uint8)
    reproject(data, out, src_transform=src_transform, src_crs=src_crs, dst_transform=transform,
              dst_crs=crs, resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
    return out


# Sentinel-2 scene classes not to use: no data, defective, cloud shadow, clouds, thin cirrus, snow.
S2_UNCLEAR = (0, 1, 3, 8, 9, 10, 11)


def read_clear(href: str, crs: str, transform, width: int, height: int) -> np.ndarray:
    """Pixels of a grid that a Sentinel-2 scene classification (SCL) calls clear."""
    return ~np.isin(read_scl(href, crs, transform, width, height), S2_UNCLEAR)


def read_scl(href: str, crs: str, transform, width: int, height: int) -> np.ndarray:
    """A Sentinel-2 scene classification on a grid, 0 where it has no data."""
    with rasterio.open(href) as src:
        bounds = transform_bounds(crs, src.crs, *array_bounds(height, width, transform), densify_pts=21)
        win = from_bounds(*bounds, src.transform)
        win = Window(math.floor(win.col_off) - 1, math.floor(win.row_off) - 1, math.ceil(win.width) + 2, math.ceil(win.height) + 2)
        try:
            win = win.intersection(Window(0, 0, src.width, src.height))
        except WindowError:
            return np.zeros((height, width), np.uint8)
        data = src.read(1, window=win)
        src_transform, src_crs = src.window_transform(win), src.crs
    out = np.zeros((height, width), np.uint8)
    reproject(data, out, src_transform=src_transform, src_crs=src_crs, dst_transform=transform, dst_crs=crs,
              resampling=Resampling.nearest)
    return out


class CellCache:
    """Cell cache on disk: <root>/gsd_<gsd>[_y<years>][_<tag>]/<key>.tif plus a JSON sidecar.

    Sources limited to some NAIP years get their own directory, so changing
    NAIPNAV_YEARS never silently reuses cells cached from other years. So do
    Sentinel-2 dates, which is what lets several be stored at once.
    """

    def __init__(self, root: str | os.PathLike, source: Source, gsd: float = 0.6):
        self.root = Path(root)
        self.source = source
        self.gsd = gsd
        years = getattr(source, "years", None)
        suffix = "_y" + "-".join(str(y) for y in sorted(years)) if years else ""
        tag = getattr(source, "tag", "")
        suffix += f"_{tag}" if tag else ""
        self.dir = self.root / f"gsd_{gsd:.2f}{suffix}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, cell: Cell) -> Path:
        return self.dir / f"{cell.key}.tif"

    def has(self, cell: Cell) -> bool:
        return self.path(cell).exists()

    def meta(self, cell: Cell) -> dict | None:
        p = self.path(cell).with_suffix(".json")
        return json.loads(p.read_text()) if p.exists() else None

    def fetch(self, cell: Cell, force: bool = False) -> Path | None:
        """Download one cell if it is not cached. Returns None if NAIP has no coverage."""
        dst = self.path(cell)
        if dst.exists() and not force:
            return dst
        crs, transform, width, height = cell.grid(self.gsd)
        rgb = np.zeros((3, height, width), np.uint8)
        filled = np.zeros((height, width), bool)
        used, first, scl = [], None, None
        with rasterio.Env(**GDAL_ENV):
            for scene in self.source.scenes(cell.bbox):
                if filled.all():
                    break
                try:
                    data = read_scene(scene.href, crs, transform, width, height)
                except rasterio.errors.RasterioIOError as exc:
                    log.warning("skipping %s: %s", scene.href, exc)
                    continue
                if data is None:
                    continue
                first = (scene, data) if first is None and data.any() else first
                valid = data.any(axis=0) & ~filled
                if scene.mask_href:
                    valid &= read_clear(scene.mask_href, crs, transform, width, height)
                if not valid.any():
                    continue
                rgb[:, valid] = data[:, valid]
                if scene.scl_href:
                    scl = np.zeros((height, width), np.uint8) if scl is None else scl
                    scl[valid] = read_scl(scene.scl_href, crs, transform, width, height)[valid]
                filled |= valid
                used.append({"id": scene.id, "year": scene.year, "date": scene.date})
            # Clouds or snow in every scene looked at: better seen than a hole in the map.
            if first is not None and not filled.all():
                scene, data = first
                valid = data.any(axis=0) & ~filled
                if valid.any():
                    rgb[:, valid] = data[:, valid]
                    filled |= valid
                    if scene.id not in {u["id"] for u in used}:
                        used.append({"id": scene.id, "year": scene.year, "date": scene.date})
        if not used:
            return None
        tmp = dst.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp.tif")
        with rasterio.open(
            tmp, "w", driver="GTiff", width=width, height=height, count=3 if scl is None else 4, dtype="uint8",
            crs=crs, transform=transform, nodata=0, tiled=True, blockxsize=256, blockysize=256,
            compress="deflate", predictor=2,
        ) as out:
            out.write(rgb, [1, 2, 3])
            if scl is not None:
                out.write(scl, 4)
            out.update_tags(SOURCE=self.source.name, SCENES=",".join(u["id"] for u in used))
        tmp.replace(dst)
        dst.with_suffix(".json").write_text(json.dumps({
            "cell": cell.key, "bbox": cell.bbox, "crs": crs, "gsd": self.gsd,
            "source": self.source.name, "scenes": used, "coverage": float(filled.mean()),
        }))
        return dst

    def cached_cells(self) -> list[dict]:
        out = []
        for p in self.dir.glob("*.json"):
            out.append(json.loads(p.read_text()))
        return out


class Library:
    """Every imagery cache under a data directory.

    Args:
        data: the data directory.
        naip_source: returns a NAIP source, limited to some years if given any.
        sentinel2: Sentinel-2 source, for clear imagery and single passes.
        naip_years: years NAIP is limited to by default, if any.
        naip_gsd: NAIP cache resolution, m.
        s2_gsd: Sentinel-2 cache resolution, m.
    """

    def __init__(self, data: Path, naip_source: Callable, sentinel2, naip_years: list[int] | None = None,
                 naip_gsd: float = 0.6, s2_gsd: float = 10.0):
        self.data, self.naip_source, self.sentinel2 = Path(data), naip_source, sentinel2
        self.naip_gsd, self.s2_gsd = naip_gsd, s2_gsd
        self.default_naip = CellCache(self.data / "naip", naip_source(naip_years), naip_gsd)
        self._caches: dict[tuple, CellCache] = {}
        self._lock = threading.Lock()

    def _memo(self, key: tuple, build: Callable[[], CellCache]) -> CellCache:
        with self._lock:
            if key not in self._caches:
                self._caches[key] = build()
            return self._caches[key]

    def naip(self, year: int | None = None) -> CellCache:
        """The default NAIP cache, or one year's."""
        if year is None:
            return self.default_naip
        return self._memo(("naip", year), lambda: CellCache(self.data / "naip", self.naip_source([year]), self.naip_gsd))

    def clear(self, as_of: str | None = None) -> CellCache:
        """Sentinel-2 made clear of clouds, newest or as of a date. Each date is stored separately."""
        return self._memo(("clear", as_of), lambda: CellCache(self.data / "sentinel2", self.sentinel2.at(as_of), self.s2_gsd))

    def on_date(self, date: str) -> CellCache:
        """One Sentinel-2 pass as it was, clouds and all."""
        return self._memo(("pass", date), lambda: CellCache(self.data / "sentinel2", PassSource(self.sentinel2, date), self.s2_gsd))

    def stored(self, cells: list[Cell]) -> dict:
        """Sentinel-2 dates stored for all these cells: clear as-of dates and passes as they were, newest first."""
        out = {"clear": [], "passes": []}
        for d in sorted((self.data / "sentinel2").glob("gsd_*"), reverse=True):
            if cells and all((d / f"{c.key}.tif").exists() for c in cells):
                if "_pass" in d.name:
                    out["passes"].append(d.name.rsplit("_pass", 1)[1])
                else:
                    out["clear"].append(d.name.rsplit("_asof", 1)[1] if "_asof" in d.name else "newest")
        return out

    def regions(self, as_of: str | None = None) -> list[dict]:
        """Groups of neighbouring cached cells of any imagery, the most recently downloaded first.

        Args:
            as_of: the date of the Sentinel-2 imagery the map shows, if not the newest.

        Returns:
            For each group: bbox (west, south, east, north), center (lat, lon of the cell
            nearest the middle, as a bent route may have nothing at the middle itself), cells,
            downloaded (time of its newest file), and for each of naip and sentinel2 how many
            of its cells have any of that imagery (cells) and how many have what the map
            shows (on_map), rather than only another year, date or a single pass.
        """
        shown = {"naip": self.default_naip.dir, "sentinel2": self.clear(as_of).dir}
        found, on_map, newest = {k: set() for k in shown}, {k: set() for k in shown}, {}
        for imagery, shown_dir in shown.items():
            for path in (self.data / imagery).glob("gsd_*/*.tif"):
                key = CACHED_CELL.fullmatch(path.name)
                if not key:
                    continue
                cell = Cell(int(key[1]), int(key[2]))
                found[imagery].add(cell)
                if path.parent == shown_dir:
                    on_map[imagery].add(cell)
                newest[cell] = max(newest.get(cell, 0.0), path.stat().st_mtime)
        regions = []
        for group in neighbouring_groups(newest):
            south, north = min(c.iy for c in group), max(c.iy for c in group) + 1
            west, east = min(c.ix for c in group), max(c.ix for c in group) + 1
            middle = min(group, key=lambda c: (2 * c.iy + 1 - south - north) ** 2 + (2 * c.ix + 1 - west - east) ** 2)
            w, s, e, n = middle.bbox
            regions.append({
                "bbox": [round(v * CELL_DEG, 2) for v in (west, south, east, north)],
                "center": [round((s + n) / 2, 3), round((w + e) / 2, 3)],
                "cells": len(group),
                "downloaded": max(newest[c] for c in group),
                **{k: {"cells": len(found[k] & group), "on_map": len(on_map[k] & group)} for k in shown},
            })
        return sorted(regions, key=lambda r: r["downloaded"], reverse=True)
