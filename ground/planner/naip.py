"""NAIP download and on-disk cache.

Imagery is cached on a fixed grid of 0.01 x 0.01 degree cells (about 1.1 km x
0.85 km at mid US latitudes). Each cell is stored once as an RGB GeoTIFF in
the UTM zone of its centre, at a fixed ground sample distance, so cells are
reused by every mission that touches them.

Two imagery sources are supported:

* ``StacSource``: Microsoft Planetary Computer STAC API (default). The COGs
  are read with windowed HTTP range requests, so only the pixels covering a
  cell are downloaded, never the whole 7.5 minute quarter quad.
* ``LocalSource``: a directory of NAIP GeoTIFFs you downloaded yourself (for
  example from USGS EarthExplorer), useful offline or on an air-gapped laptop.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

log = logging.getLogger(__name__)

CELL_DEG = 0.01
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


@dataclass
class SceneRef:
    href: str
    year: int
    id: str = ""


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
        self._client = None

    def _open(self):
        if self._client is None:
            import pystac_client

            modifier = None
            if "planetarycomputer" in self.url:
                import planetary_computer

                modifier = planetary_computer.sign_inplace
            self._client = pystac_client.Client.open(self.url, modifier=modifier)
        return self._client

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


class LocalSource:
    """NAIP GeoTIFFs in a local directory (searched recursively)."""

    name = "local"

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
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
        hits = [ref for ref, (l, b, r, t) in self._index if l < e and r > w and b < n and t > s]
        hits.sort(key=lambda r: r.year, reverse=True)
        return hits


def _year_from_name(name: str) -> int:
    # NAIP file names end in _YYYYMMDD.tif, e.g. m_3910501_ne_13_060_20210620.tif
    digits = name.rsplit("_", 1)[-1][:4]
    return int(digits) if digits.isdigit() else 0


class NaipCache:
    """Cell cache on disk: <root>/<gsd>/<key>.tif plus a JSON sidecar."""

    def __init__(self, root: str | os.PathLike, source: Source, gsd: float = 0.6):
        self.root = Path(root)
        self.source = source
        self.gsd = gsd
        self.dir = self.root / f"gsd_{gsd:.2f}"
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
        used = []
        with rasterio.Env(**GDAL_ENV):
            for scene in self.source.scenes(cell.bbox):
                if filled.all():
                    break
                try:
                    with rasterio.open(scene.href) as src, WarpedVRT(
                        src, crs=crs, transform=transform, width=width, height=height,
                        resampling=Resampling.bilinear, nodata=0,
                    ) as vrt:
                        data = vrt.read([1, 2, 3])
                except rasterio.errors.RasterioIOError as exc:
                    log.warning("skipping %s: %s", scene.href, exc)
                    continue
                valid = data.any(axis=0) & ~filled
                if not valid.any():
                    continue
                rgb[:, valid] = data[:, valid]
                filled |= valid
                used.append({"id": scene.id, "year": scene.year})
        if not used:
            return None
        tmp = dst.with_suffix(".tmp.tif")
        with rasterio.open(
            tmp, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8",
            crs=crs, transform=transform, nodata=0, tiled=True, blockxsize=256, blockysize=256,
            compress="deflate", predictor=2,
        ) as out:
            out.write(rgb)
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
