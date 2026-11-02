"""Sentinel-2 alongside NAIP: clouds kept out of the cache, separate caches, plans on either, matching by height."""

import importlib
import io
import json
import math
import os
import time
import zipfile
from dataclasses import replace as dataclass_replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from planner.imagery import Cell, CellCache, SceneRef, Sentinel2Source, utm_epsg
from planner.mission import Mission
from visnav.flightsim import FlightSim, SimConfig
from visnav.geo import MapRaster
from visnav.matcher import CorrMatcher
from visnav.navigator import Navigator
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

    # The same waypoints with a NAIP map: another map's size, resolution and centre, but the same frame to compare in.
    naip = plan("naip", alt=120)
    cache_route(client, naip)
    other = client.post("/api/sim/runs", json={"mission": naip, "options": {"seed": 1}}).json()["id"]
    for _ in range(600):
        other_run = client.get(f"/api/sim/runs/{other}").json()
        if other_run["state"] in ("finished", "failed", "stopped"):
            break
        time.sleep(0.5)
    assert other_run["state"] == "finished", other_run
    assert other_run["route"] == run["route"]
    data = client.get(f"/api/sim/compare/{run_id}/{other}").json()
    assert [r["settings"]["Map imagery"] for r in data["runs"]] == ["Sentinel-2, 10 m", "NAIP, 0.6 m"]
    assert [r["settings"]["Cruise altitude"] for r in data["runs"]] == ["300 m", "120 m"]
    plans = [r["plan"] for r in data["runs"]]
    assert plans[0][-1] == pytest.approx(plans[1][-1], abs=0.1)
    assert all(math.hypot(r["series"]["x"][0], r["series"]["y"][0]) < 1 for r in data["runs"])


# Several dates, and passes kept as they were, for flights under real clouds.
CLOUDY, OLD_CLEAR, NEWEST = "20240703", "20240601", "20261024"


@pytest.fixture(scope="module")
def dated(tmp_path_factory):
    """NAIP, and Sentinel-2 of the same ground on three dates: the middle one half cloudy."""
    d = tmp_path_factory.mktemp("dated")
    (d / "naip").mkdir()
    (d / "s2").mkdir()
    naip = synthetic_map(str(d / "naip" / "m_synthetic_20240701.tif"), LAT, LON, size_m=3000, gsd=0.6, seed=8)
    for date in (CLOUDY, OLD_CLEAR, NEWEST):
        downsampled(naip, d / "s2" / f"s2_{date}.tif")
    with rasterio.open(d / "s2" / f"s2_{CLOUDY}.tif") as src:
        data, crs, transform = src.read(), src.crs, src.transform
    h = data.shape[1]
    # The northern half is cloud, white in the imagery, with a band of shadow south of it.
    data[:, : h // 2] = 240
    write(d / "s2" / f"s2_{CLOUDY}.tif", data, crs, transform)
    scl = np.full((h, data.shape[2]), 4, np.uint8)
    scl[: h // 2] = 9
    scl[h // 2: h // 2 + 12] = 3
    write(d / "s2" / f"s2_{CLOUDY}_scl.tif", scl, crs, transform)
    for date in (OLD_CLEAR, NEWEST):
        write(d / "s2" / f"s2_{date}_scl.tif", np.full((h, data.shape[2]), 4, np.uint8), crs, transform)
    return d


@pytest.fixture(scope="module")
def dated_client(dated, tmp_path_factory):
    os.environ.update(NAIPNAV_DATA=str(tmp_path_factory.mktemp("data")), NAIPNAV_SOURCE=str(dated / "naip"), NAIPNAV_S2_SOURCE=str(dated / "s2"))
    import planner.app

    importlib.reload(planner.app)
    return TestClient(planner.app.app)


def route_map(client, m):
    r = client.post("/api/export", json=m)
    assert r.status_code == 200, r.text
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with rasterio.open(io.BytesIO(z.read(f"{m['name']}/map.tif"))) as src:
        return src.read()


def test_several_dates_are_stored_side_by_side_and_maps_stay_clear(dated_client):
    client = dated_client
    newest, as_of = plan(), {**plan(), "name": "as_of", "imagery_date": "2024-07-10"}
    job = cache_route(client, newest)
    assert job["kept_pass"] == "2026-10-24"
    assert client.post("/api/plan/check", json=as_of).json()["cached"] == 0  # another date, cached separately
    assert cache_route(client, as_of)["kept_pass"] == "2024-07-03"

    stored = client.post("/api/sentinel2/passes", json=newest).json()["stored"]
    assert set(stored["clear"]) == {"newest", "2024-07-10"} and set(stored["passes"]) == {"2026-10-24", "2024-07-03"}

    # The map as of the cloudy pass has no cloud: those pixels come from the pass before.
    assert (route_map(client, as_of) == 240).all(axis=0).mean() == 0
    label = client.post("/api/plan/imagery", json=as_of).json()["label"]
    assert label == "Sentinel-2 Jun 2024 to Jul 2024"

    n, z = 2 ** 15, 15
    x, y = int((LON + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(LAT))) / math.pi) / 2 * n)
    assert client.get(f"/api/tiles/{z}/{x}/{y}.png", params={"imagery": "sentinel2", "date": "2024-07-10"}).status_code == 200
    assert client.get(f"/api/tiles/{z}/{x}/{y}.png", params={"imagery": "sentinel2", "date": "soon"}).status_code == 422
    assert client.post("/api/plan/check", json={**newest, "imagery_date": "soon"}).status_code == 422


def test_passes_keep_their_clouds_and_a_year_counts_all_of_them(dated_client):
    import planner.app

    client = dated_client
    m = plan()
    listed = client.post("/api/sentinel2/passes", params={"year": 2024}, json=m).json()
    assert [p["date"] for p in listed["passes"]] == ["2024-07-03", "2024-06-01"]
    assert listed["passes"][0]["cloud_pct"] == 50 and listed["passes"][1]["cloud_pct"] == 0
    assert listed["years"] == [int(time.strftime("%Y")) - i for i in range(int(time.strftime("%Y")) - 2024 + 1)]

    from planner.simruns import SimOptions

    runs = planner.app.sim_runs
    mission = Mission.model_validate(m)
    picks = {runs.choose_pass(mission, SimOptions(world_year=2024, seed=s)) for s in range(12)}
    assert picks == {"2024-07-03", "2024-06-01"}  # cloudy or not
    assert runs.choose_pass(mission, SimOptions()) == "2026-10-24"
    assert runs.choose_pass(mission, SimOptions(pass_date="2024-06-01")) == "2024-06-01"
    assert runs.choose_pass(mission, SimOptions(world_year=2012)) is None

    cache = planner.app.library.on_date("2024-07-03")
    with rasterio.open(cache.fetch(Cell.containing(LAT + 0.005, LON))) as src:  # in the cloud
        assert src.count == 4 and (src.read(1) == 240).mean() > 0.9 and (src.read(4) == 9).mean() > 0.9


def test_clouds_hide_and_shadows_dim(tmp_path):
    from planner.clouds import SHADOW_GAIN, overlay

    crs, gsd = "EPSG:32613", 10.0
    transform = from_origin(480000, 4430000, gsd, gsd)
    scl = np.full((40, 40), 4, np.uint8)
    scl[:10] = 9
    scl[20:30] = 3
    pass_rgb = np.full((3, 40, 40), 100, np.uint8)
    pass_rgb[:, :10] = 240
    pass_tif = tmp_path / "pass.tif"
    with rasterio.open(pass_tif, "w", driver="GTiff", width=40, height=40, count=4, dtype="uint8", crs=crs, transform=transform) as dst:
        dst.write(pass_rgb, [1, 2, 3])
        dst.write(scl, 4)

    def world(name):
        path = tmp_path / name
        k = 4
        fine = from_origin(480000, 4430000, gsd / k, gsd / k)
        with rasterio.open(path, "w", driver="GTiff", width=160, height=160, count=3, dtype="uint8", crs=crs, transform=fine, nodata=0) as dst:
            dst.write(np.full((3, 160, 160), 80, np.uint8))
        return path

    real = world("real.tif")
    stats = overlay(real, pass_tif)
    assert stats == {"cloud_pct": pytest.approx(25, abs=2), "shadow_pct": pytest.approx(25, abs=2)}
    with rasterio.open(real) as src:
        assert src.count == 4
        out, alpha = src.read(1), src.read(4)
    assert out[10, 80] == pytest.approx(240, abs=2) and alpha[10, 80] == 255  # the cloud hides the ground
    assert out[100, 80] == pytest.approx(80 * SHADOW_GAIN, abs=2) and alpha[100, 80] == 0  # its shadow dims it
    assert out[70, 80] == 80 and out[150, 80] == 80

    below = world("below.tif")
    overlay(below, pass_tif, clouds=False)
    with rasterio.open(below) as src:
        out, alpha = src.read(1), src.read(4)
    assert out[10, 80] == 80 and alpha.max() == 0 and out[100, 80] == pytest.approx(80 * SHADOW_GAIN, abs=2)


def test_no_ground_texture_is_seen_through_a_cloud(maps):
    world, _ = maps
    clear = render(world, CAM, LAT, LON, 40.0, 0.0, 0.0, 0.0)
    plain = render(world, CAM, LAT, LON, 40.0, 0.0, 0.0, 0.0, detail=0.0)
    assert (clear != plain).mean() > 0.3
    hidden = dataclass_replace(world, cloud=np.full(world.gray.shape, 255, np.uint8))
    assert np.array_equal(render(hidden, CAM, LAT, LON, 40.0, 0.0, 0.0, 0.0), plain)


def fly(client, m, **options):
    run_id = client.post("/api/sim/runs", json={"mission": m, "options": {"seed": 1, **options}}).json()["id"]
    for _ in range(600):
        run = client.get(f"/api/sim/runs/{run_id}").json()
        if run["state"] in ("finished", "failed", "stopped"):
            break
        time.sleep(0.5)
    assert run["state"] == "finished", run
    return run


def test_clouds_follow_the_altitude_unless_overridden(dated_client):
    client = dated_client
    m = {**plan("naip", alt=120), "name": "naip_clouds"}
    cache_route(client, m)
    cloudy = {"world_year": 2024, "pass_date": "2024-07-03"}

    # Below the cloud base, none by default, even with a pass asked for, and nothing is laid over the imagery.
    low = fly(client, m, **cloudy)
    assert low["world"]["choice"] == "auto" and low["world"]["clouds"] == "none" and low["world"]["pass_date"] is None
    assert low["options"]["clouds"] == "auto"

    # Overridden, at every height.
    included = fly(client, m, clouds="real", **cloudy)
    assert included["world"]["pass_date"] == "2024-07-03" and included["world"]["base_m"] == 0.0
    assert included["world"]["cloud_pct"] > 20 and included["summary"]["landed"]
    assert low["summary"]["vision_fixes"] > included["summary"]["vision_fixes"]

    below = fly(client, m, clouds="shadows", **cloudy)
    assert below["world"]["shadow_pct"] > 0 and below["world"]["clouds"] == "shadows"

    # Going above it, they are there by default, only while above it.
    high = {**plan("naip", alt=120), "name": "naip_high", "waypoints": [{"lat": LAT - 0.002, "lon": LON, "alt": 760}, {"lat": LAT + 0.002, "lon": LON}]}
    assert Mission.model_validate(high).max_alt() == 760
    import planner.app

    def takes_off(**options):
        """How a run is set up, without flying all the way up there and back."""
        run_id = client.post("/api/sim/runs", json={"mission": high, "options": {"seed": 1, **options}}).json()["id"]
        for _ in range(400):
            run = client.get(f"/api/sim/runs/{run_id}").json()
            if run["viewable"] or run["state"] in ("finished", "failed", "stopped"):
                break
            time.sleep(0.5)
        assert run["viewable"], run
        client.post(f"/api/sim/runs/{run_id}/stop")
        return run, json.loads((planner.app.sim_runs.root / run_id / "scene.json").read_text())

    run, scene = takes_off(**cloudy)
    assert run["world"]["clouds"] == "real" and run["world"]["base_m"] == 750.0 and run["world"]["cloud_pct"] > 20
    assert scene["clouds"]["base_m"] == 750.0 and scene["clouds"]["image"].startswith("data:image/png")
    run, scene = takes_off(clouds="none", **cloudy)
    assert run["world"]["pass_date"] is None and scene["clouds"] is None

    # Sentinel-2 as the camera: the pass itself when clouds are included, clear imagery as of it when not.
    s2 = plan()
    cache_route(client, s2)
    raw = fly(client, s2, world_imagery="sentinel2", clouds="real", **cloudy)
    assert raw["world"]["cloud_pct"] > 20 and raw["summary"]["landed"]
    clear = fly(client, s2, world_imagery="sentinel2", **cloudy)
    assert clear["world"]["clouds"] == "none" and clear["world"]["pass_date"] == "2024-07-03"


def test_camera_is_under_the_clouds_below_the_base(maps):
    world, _ = maps
    clouded = dataclass_replace(world, rgb=np.full_like(world.rgb, 250), cloud=np.full(world.gray.shape, 255, np.uint8))
    meta = {"home": {"lat": LAT, "lon": LON}, "mission": {"default_alt": 800, "takeoff_alt": 30, "speed": 10, "end_action": "land",
                                                          "waypoints": [{"lat": LAT + 0.001, "lon": LON, "alt": None}], "name": "t"}}
    nav = Navigator(world, CAM, None, LAT, LON)
    sim = FlightSim(world, CAM, nav, meta, SimConfig(cloud_base_m=750.0), world=clouded, clear_world=world)
    assert sim._camera_world(749.0) is world and sim._camera_world(750.0) is clouded
    assert sim.scene()["clouds"]["base_m"] == 750.0
    # Without a clear world, or a base, the same imagery at every height.
    assert FlightSim(world, CAM, nav, meta, SimConfig(), world=clouded)._camera_world(10.0) is clouded
    assert FlightSim(world, CAM, nav, meta, SimConfig())._cloud_layer() is None
