"""Planner -> bundle -> onboard navigation, using a synthetic stand-in for NAIP.

The sandbox this was written in cannot reach NAIP servers, so a procedural
GeoTIFF is served through LocalSource. Point NAIPNAV_TEST_NAIP at a directory
of real NAIP GeoTIFFs (and set NAIPNAV_TEST_LAT/LON inside them) to run the
same test against real imagery.
"""

import importlib
import math
import io
import os
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

LAT, LON = float(os.environ.get("NAIPNAV_TEST_LAT", 40.0)), float(os.environ.get("NAIPNAV_TEST_LON", -105.3))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    naip_dir = os.environ.get("NAIPNAV_TEST_NAIP")
    if not naip_dir:
        from visnav.sim import synthetic_map

        naip_dir = tmp_path_factory.mktemp("naip")
        synthetic_map(str(naip_dir / "m_synthetic_20220701.tif"), LAT, LON, size_m=2400, gsd=0.6, seed=3)
    os.environ["NAIPNAV_DATA"] = str(tmp_path_factory.mktemp("data"))
    os.environ["NAIPNAV_SOURCE"] = str(naip_dir)
    import planner.app

    importlib.reload(planner.app)
    return TestClient(planner.app.app)


def mission(name="e2e"):
    d = 0.004
    return {
        "name": name, "default_alt": 120, "takeoff_alt": 30, "speed": 10, "corridor_m": 250,
        "end_action": "rtl", "camera": {"hfov_deg": 90, "width_px": 960, "height_px": 600},
        "waypoints": [
            {"lat": LAT - d, "lon": LON - d}, {"lat": LAT + d, "lon": LON - d / 2},
            {"lat": LAT + d / 2, "lon": LON + d},
        ],
    }


def test_wpl_export():
    from planner.mission import Mission

    lines = Mission.model_validate(mission()).to_wpl().splitlines()
    assert lines[0] == "QGC WPL 110"
    cmds = [int(ln.split("\t")[3]) for ln in lines[1:]]
    assert cmds == [16, 22, 178, 16, 16, 16, 20]
    assert [int(ln.split("\t")[0]) for ln in lines[1:]] == list(range(7))


def test_corridor_covers_return_leg():
    from shapely.geometry import Point

    from planner.mission import Mission

    m = Mission.model_validate(mission())
    poly, _ = m.corridor()
    _, fwd, _ = m.route_utm()
    home, last = m.waypoints[0], m.waypoints[-1]
    assert poly.contains(Point(fwd.transform((home.lon + last.lon) / 2, (home.lat + last.lat) / 2)))


def test_cache_and_export_then_fly(client, tmp_path):
    m = mission()
    chk = client.post("/api/plan/check", json=m).json()
    assert chk["cells"] > 0 and chk["cached"] == 0
    assert client.post("/api/export", json=m).status_code == 409  # nothing cached yet

    job = client.post("/api/cache/jobs", json=m).json()
    for _ in range(120):
        job = client.get(f"/api/cache/jobs/{job['id']}").json()
        if job["state"] != "running":
            break
        time.sleep(0.5)
    assert job["state"] == "finished" and not job["errors"], job
    assert client.post("/api/plan/check", json=m).json()["cached"] == chk["cells"]

    # A z18 tile over the route renders real pixels.
    import math
    z = 17
    x = int((LON + 180) / 360 * 2**z)
    y = int((1 - math.asinh(math.tan(math.radians(LAT))) / math.pi) / 2 * 2**z)
    tile = client.get(f"/api/tiles/{z}/{x}/{y}.png")
    assert tile.status_code == 200 and len(tile.content) > 5000

    r = client.post("/api/export", json=m)
    assert r.status_code == 200
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(tmp_path)
    bundle = tmp_path / "e2e"
    import json
    meta = json.loads((bundle / "mission.json").read_text())
    assert meta["map"]["coverage"] > 0.99

    from visnav.main import main
    # Typical EKF3 roll/pitch error; at 120 m each degree moves fixes ~2 m (see README).
    report = main(["sim", "--bundle", str(bundle), "--rate", "1", "--seed", "1", "--att-noise", "0.5",
                   "--appearance", "random"])
    assert report["fix_rate"] > 0.8, report
    assert report["err_median_m"] < 3.0, report
    assert report["err_p95_m"] < 8.0, report


def test_closed_loop_flight_sim(client, tmp_path):
    """The simulated vehicle takes off, flies a short mission and lands on vision alone."""
    import json

    from visnav.appearance import Appearance
    from visnav.flightsim import FlightSim, SimConfig, map_texture
    from visnav.main import main as visnav_main  # noqa: F401  (CLI import smoke test)
    from visnav.matcher import CorrMatcher
    from visnav.navigator import Navigator
    from visnav.geo import MapRaster
    from visnav.ortho import CameraModel
    from visnav.viewer import export_html

    m = mission("short")
    m["waypoints"] = m["waypoints"][:2]
    job = client.post("/api/cache/jobs", json=m).json()
    for _ in range(120):
        if client.get(f"/api/cache/jobs/{job['id']}").json()["state"] != "running":
            break
        time.sleep(0.5)
    r = client.post("/api/export", json=m)
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(tmp_path)
    bundle = tmp_path / "short"
    meta = json.loads((bundle / "mission.json").read_text())
    map_ = MapRaster.load(bundle / "map.tif")
    cam = CameraModel.from_hfov(960, 600, 90)
    nav = Navigator(map_, cam, CorrMatcher(map_), meta["home"]["lat"], meta["home"]["lon"], start_sigma_m=5.0)
    sim = FlightSim(map_, cam, nav, meta, SimConfig(camera_hz=2.0, seed=5, att_noise_deg=0.5, appearance=Appearance.parse("random", 5)))
    records, summary = sim.run()
    assert summary["landed"], summary
    assert summary["fix_rate"] > 0.7, summary
    assert summary["ekf_err_median_m"] < 3.0, summary
    # Below map-matching height only odometry runs: takeoff and landing.
    assert summary["ekf_err_below_map_alt_max_m"] < 5.0, summary
    assert summary["landing_error_m"] < 5.0, summary
    # Vehicle actually reached the far waypoint (it flew on the estimate, not truth).
    far = sim.plan[2]
    assert min(math.hypot(r["p"][0] - far[1], r["p"][1] - far[2]) for r in records) < 15

    out = export_html(tmp_path / "flight.html", sim.scene(), records, map_texture(map_, 1024), summary)
    html = out.read_text()
    assert "window.NAVSIM" in html and "three.min.js" in html and "__THREE_SRC__" not in html
