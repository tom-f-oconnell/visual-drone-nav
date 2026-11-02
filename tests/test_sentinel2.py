"""Sentinel-2 alongside NAIP: clouds kept out of the cache, separate caches, plans on either, matching by height."""

import importlib
import io
import json
import math
import os
import time
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from planner.imagery import Cell, CellCache, SceneRef, Sentinel2Source, utm_epsg
from planner.mission import Mission
from visnav.geo import MapRaster
from visnav.matcher import CorrMatcher
from visnav.ortho import CameraModel, orthorectify
from visnav.sim import render, synthetic_map

LAT, LON = 40.0, -105.3
CAM = CameraModel.from_hfov(1280, 800, 90)


def write(path, data, crs, transform):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[-1], height=data.shape[-2], count=1 if data.ndim == 2 else 3,
                       dtype="uint8", crs=crs, transform=transform) as dst:
        dst.write(data, 1) if data.ndim == 2 else dst.write(data)
    return str(path)


def downsampled(naip_path, path, gsd=10.0):
    """A coarse copy of a map, as Sentinel-2 sees the same ground."""
    with rasterio.open(naip_path) as src:
        k = src.res[0] / gsd
        data = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        small = cv2.resize(data, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        return write(path, np.moveaxis(small, -1, 0), src.crs, from_origin(src.transform.c, src.transform.f, gsd, gsd))


class TwoPasses:
    """The newest pass has a cloud over part of the cell, and an older pass is clear."""

    name = "test"

    def __init__(self, tmp_path, cell, second_class=4):
        crs, transform, width, height = cell.grid(10.0)
        cloudy = np.zeros((height, width), np.uint8) + 4
        cloudy[:height // 2] = 9
        new = write(tmp_path / "new.tif", np.full((3, height, width), 200, np.uint8), crs, transform)
        old = write(tmp_path / "old.tif", np.full((3, height, width), 50, np.uint8), crs, transform)
        self.scenes_ = [SceneRef(new, 2026, "new", "2026-10-24", write(tmp_path / "new_scl.tif", cloudy, crs, transform)),
                        SceneRef(old, 2026, "old", "2026-09-30", write(tmp_path / "old_scl.tif", np.full((height, width), second_class, np.uint8), crs, transform))]

    def scenes(self, bbox):
        return self.scenes_


def test_clouds_are_filled_from_an_older_pass(tmp_path):
    cell = Cell.containing(LAT, LON)
    cache = CellCache(tmp_path / "c", TwoPasses(tmp_path, cell), gsd=10.0)
    with rasterio.open(cache.fetch(cell)) as src:
        data = src.read(1)
    h = data.shape[0]
    assert (data[: h // 2] == 50).all() and (data[h // 2:] == 200).all()
    meta = cache.meta(cell)
    assert meta["coverage"] == 1.0 and [s["date"] for s in meta["scenes"]] == ["2026-10-24", "2026-09-30"]


def test_clouds_everywhere_are_better_seen_than_a_hole(tmp_path):
    cell = Cell.containing(LAT, LON)
    cache = CellCache(tmp_path / "c", TwoPasses(tmp_path, cell, second_class=9), gsd=10.0)
    with rasterio.open(cache.fetch(cell)) as src:
        data = src.read(1)
    h = data.shape[0]
    assert (data[h // 2:] == 200).all() and (data[: h // 2] == 200).all()  # the newest pass, clouds and all
    assert cache.meta(cell)["coverage"] == 1.0


def test_source_asks_for_recent_clear_scenes():
    asked = {}
    href = lambda s: SimpleNamespace(href=s)  # noqa: E731
    item = SimpleNamespace(assets={"visual": href("v.tif"), "SCL": href("s.tif")}, id="S2", datetime=SimpleNamespace(
        year=2026, date=lambda: SimpleNamespace(isoformat=lambda: "2026-10-24")))

    def search(**kw):
        asked.update(kw)
        return SimpleNamespace(items=lambda: [item])

    source = Sentinel2Source(max_cloud=15.0, max_scenes=7)
    source._open = lambda: SimpleNamespace(search=search)
    scene = source.scenes((-105.3, 40.0, -105.29, 40.01))[0]
    assert asked["collections"] == ["sentinel-2-l2a"] and asked["max_items"] == 7
    assert asked["query"] == {"eo:cloud_cover": {"lt": 15.0}} and asked["sortby"] == "-properties.datetime"
    assert (scene.href, scene.mask_href, scene.date) == ("v.tif", "s.tif", "2026-10-24")


def test_planner_warns_when_the_camera_is_too_low_for_the_map():
    waypoints = [{"lat": LAT, "lon": LON}, {"lat": LAT + 0.01, "lon": LON}]
    low = Mission(default_alt=120, waypoints=waypoints)
    assert not any("too small" in w for w in low.warnings(0.6))
    assert any("too small" in w and "10 m pixels" in w for w in low.warnings(10.0))
    assert not any("too small" in w for w in Mission(default_alt=300, waypoints=waypoints).warnings(10.0))
    assert any("too small" in w for w in Mission(default_alt=50, waypoints=waypoints).warnings(0.6))


@pytest.fixture(scope="module")
def maps(tmp_path_factory):
    d = tmp_path_factory.mktemp("s2maps")
    naip = synthetic_map(str(d / "m_synthetic_20220701.tif"), LAT, LON, size_m=3600, gsd=0.6, seed=8)
    return MapRaster.load(naip), MapRaster.load(downsampled(naip, d / "s2.tif"))


@pytest.mark.parametrize("h, found", [(120.0, False), (300.0, True), (500.0, True)])
def test_matching_a_coarse_map_works_only_from_high_enough(maps, h, found):
    world, coarse = maps
    matcher = CorrMatcher(coarse)
    assert matcher.down == 1 and matcher.sigma_m == 10.0 and CorrMatcher(world).down == 4
    img = render(world, CAM, LAT, LON, h, 0.0, 0.0, 0.0)
    ortho = orthorectify(img, CAM, 0.0, 0.0, 0.0, h, coarse.ned_to_px_jacobian(LAT, LON))
    col, row = coarse.ll_to_px(LAT + 0.0003, LON - 0.0004)
    fix = matcher.locate(ortho, col, row, 10)
    assert (fix is not None) == found
    if found:
        assert math.hypot((fix.lat - LAT) * 111000, (fix.lon - LON) * 85000) < 25


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("sources")
    (d / "naip").mkdir()
    (d / "s2").mkdir()
    naip = synthetic_map(str(d / "naip" / "m_synthetic_20220701.tif"), LAT, LON, size_m=3000, gsd=0.6, seed=8)
    downsampled(naip, d / "s2" / "s2_20261024.tif")
    os.environ.update(NAIPNAV_DATA=str(tmp_path_factory.mktemp("data")), NAIPNAV_SOURCE=str(d / "naip"), NAIPNAV_S2_SOURCE=str(d / "s2"))
    import planner.app

    importlib.reload(planner.app)
    return TestClient(planner.app.app)


def plan(imagery="sentinel2", alt=300, corridor=700):
    return {"name": f"plan_{imagery}", "default_alt": alt, "takeoff_alt": 30, "speed": 10, "corridor_m": corridor, "end_action": "land",
            "imagery": imagery, "camera": {"hfov_deg": 90, "width_px": 640, "height_px": 400},
            "waypoints": [{"lat": LAT - 0.002, "lon": LON}, {"lat": LAT + 0.002, "lon": LON}]}


def cache_route(client, m):
    job = client.post("/api/cache/jobs", json=m).json()
    for _ in range(240):
        job = client.get(f"/api/cache/jobs/{job['id']}").json()
        if job["state"] != "running":
            break
        time.sleep(0.25)
    assert job["state"] == "finished" and not job["errors"], job
    return job


def test_each_imagery_has_its_own_cache_tiles_and_bundle(client):
    import planner.app

    m = plan()
    assert client.post("/api/plan/check", json=m).json()["cached"] == 0
    job = cache_route(client, m)
    assert job["total"] == job["route_cells"] > 0
    assert client.post("/api/plan/check", json=m).json()["cached"] == job["total"]

    # Caching one leaves the other alone, and saved plans with no imagery mean NAIP.
    assert client.post("/api/plan/check", json=plan("naip")).json()["cached"] == 0
    assert list((planner.app.DATA / "naip").rglob("*.tif")) == []
    legacy = {k: v for k, v in plan("naip").items() if k != "imagery"}
    assert Mission.model_validate(legacy).imagery == "naip"

    assert client.get("/api/cache/cells", params={"imagery": "sentinel2"}).json()["features"]
    assert client.get("/api/cache/cells").json()["features"] == []
    assert client.get("/api/cache/cells", params={"imagery": "landsat"}).status_code == 422

    n, z = 2 ** 15, 15
    x, y = int((LON + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(LAT))) / math.pi) / 2 * n)
    def tile(imagery):
        r = client.get(f"/api/tiles/{z}/{x}/{y}.png", params={"imagery": imagery})
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert tile("sentinel2").shape[0] == 256 and tile("sentinel2")[..., 3].mean() > 100
    assert tile("naip").shape[0] == 1  # nothing cached

    r = client.post("/api/export", json=m)
    assert r.status_code == 200
    z_ = zipfile.ZipFile(io.BytesIO(r.content))
    meta = json.loads(z_.read("plan_sentinel2/mission.json"))
    assert meta["map"]["gsd"] == 10.0 and meta["mission"]["imagery"] == "sentinel2"
    assert client.post("/api/export", json=plan("naip")).status_code == 409


def test_labels_and_where_naip_is_missing(client):
    assert client.post("/api/plan/imagery", json={**plan(), "waypoints": []}).json() == {"label": "Newest clear Sentinel-2", "naip_here": None}
    here = client.post("/api/plan/imagery", json=plan()).json()
    assert here["naip_here"] is True and here["label"].startswith("Sentinel-2") or here["label"] == "Newest clear Sentinel-2"
    away = {**plan("naip"), "waypoints": [{"lat": -33.0, "lon": 151.0}, {"lat": -33.01, "lon": 151.0}]}
    r = client.post("/api/plan/imagery", json=away).json()
    assert r == {"label": "No NAIP here", "naip_here": False}


def test_flying_a_sentinel2_map_with_a_camera_seeing_naip(client):
    m = plan()
    cache_route(client, m)
    assert client.post("/api/sim/years", json=m).json()["years"] == [2022]  # every NAIP year is another imagery
    run_id = client.post("/api/sim/runs", json={"mission": m, "options": {"world_year": "newest", "seed": 1}}).json()["id"]
    for _ in range(600):
        run = client.get(f"/api/sim/runs/{run_id}").json()
        if run["state"] in ("finished", "failed", "stopped"):
            break
        time.sleep(0.5)
    assert run["state"] == "finished", run
    assert run["imagery"] == "sentinel2" and run["options"]["world_year"] == "newest"
    s = run["summary"]
    assert s["landed"] and s["vision_fixes"] > 10 and s["ekf_err_median_m"] < 25, s  # most of a flight this short is too low to match
    assert client.post("/api/sim/runs", json={"mission": m, "options": {"world_year": "last"}}).status_code == 422
