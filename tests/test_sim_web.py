"""Running and viewing simulations from the planner website, and what it says about the imagery."""

import importlib
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

LAT, LON = 40.0, -105.3


def mission(name="sim", end_action="land"):
    d = 0.002
    return {
        "name": name, "default_alt": 80, "takeoff_alt": 30, "speed": 10, "corridor_m": 250, "end_action": end_action,
        "camera": {"hfov_deg": 90, "width_px": 640, "height_px": 400},
        "waypoints": [{"lat": LAT - d, "lon": LON}, {"lat": LAT + d, "lon": LON}],
    }


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from visnav.sim import synthetic_map

    naip_dir = tmp_path_factory.mktemp("naip")
    for year in (2020, 2022):
        synthetic_map(str(naip_dir / f"m_synthetic_{year}0701.tif"), LAT, LON, size_m=2400, gsd=0.6, seed=3)
    os.environ["NAIPNAV_DATA"] = str(tmp_path_factory.mktemp("data"))
    os.environ["NAIPNAV_SOURCE"] = str(naip_dir)
    import planner.app

    importlib.reload(planner.app)
    c = TestClient(planner.app.app)
    job = c.post("/api/cache/jobs", json=mission()).json()
    for _ in range(120):
        job = c.get(f"/api/cache/jobs/{job['id']}").json()
        if job["state"] != "running":
            break
        time.sleep(0.25)
    assert job["state"] == "finished" and not job["errors"], job
    return c


def wait_for(client, run_id, states, seconds=300):
    for _ in range(int(seconds * 2)):
        run = client.get(f"/api/sim/runs/{run_id}").json()
        if run["state"] in states:
            return run
        time.sleep(0.5)
    raise AssertionError(f"still {run['state']}: {run}")


def events(client, run_id):
    return client.get(f"/sim/{run_id}/events").text.split("\n\n")


def test_run_view_and_replay(client):
    assert client.get("/api/config").json()["simulator"] is True
    r = client.post("/api/sim/runs", json={"mission": mission(), "options": {"seed": 2, "wind_speed": 2.0}})
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]
    assert r.json()["state"] in ("preparing", "running")

    run = wait_for(client, run_id, ("finished", "failed", "stopped"))
    assert run["state"] == "finished", run
    assert run["summary"]["landed"] and run["summary"]["landing_error_m"] < 15, run
    assert run["viewable"]

    # The viewer page and everything it asks for, by the relative URLs it uses.
    page = client.get(f"/sim/{run_id}/")
    assert page.status_code == 200 and 'src="three.min.js"' in page.text and "__THREE_SRC__" not in page.text
    assert client.get(f"/sim/{run_id}/three.min.js").status_code == 200
    assert client.get(f"/sim/{run_id}/scene.json").json()["name"] == "sim"
    assert client.get(f"/sim/{run_id}/map.jpg").content[:2] == b"\xff\xd8"

    # A finished run replays in full, then says it is done.
    chunks = events(client, run_id)
    records = [json.loads(c[len("data: "):]) for c in chunks if c.startswith("data: ")]
    assert records and records[-1]["p"][2] < 1.0
    assert sum(c.startswith("event: done") for c in chunks) == 1

    # The bundle's GeoTIFFs are gone, what the viewer needs stays.
    import planner.app

    assert not (planner.app.DATA / "sim" / run_id / "bundle").exists()
    assert any(r["id"] == run_id for r in client.get("/api/sim/runs").json())

    assert client.delete(f"/api/sim/runs/{run_id}").status_code == 200
    assert client.get(f"/api/sim/runs/{run_id}").status_code == 404


def test_stop_a_run(client):
    run_id = client.post("/api/sim/runs", json={"mission": mission("stopped")}).json()["id"]
    assert client.get(f"/sim/{run_id}/scene.json").status_code == 404  # still building the bundle
    wait_for(client, run_id, ("running",))
    assert client.post(f"/api/sim/runs/{run_id}/stop").status_code == 200
    run = wait_for(client, run_id, ("stopped", "failed", "finished"), seconds=30)
    assert run["state"] == "stopped", run
    assert sum(c.startswith("event: done") for c in events(client, run_id)) == 1


def test_camera_from_another_naip_year(client):
    m = mission("other_year")
    assert client.post("/api/sim/years", json=m).json()["years"] == [2020]  # the map is from 2022
    run_id = client.post("/api/sim/runs", json={"mission": m, "options": {"world_year": 2020}}).json()["id"]
    run = wait_for(client, run_id, ("finished", "failed", "stopped"))
    assert run["state"] == "finished", run
    import planner.app

    assert (planner.app.DATA / "naip" / "gsd_0.60_y2020").exists()  # kept for next time


def test_mission_without_landing_ends_at_last_waypoint(client):
    run_id = client.post("/api/sim/runs", json={"mission": mission("no_land", "none")}).json()["id"]
    run = wait_for(client, run_id, ("finished", "failed", "stopped"))
    assert run["state"] == "finished", run
    assert run["summary"]["mission_complete"] and not run["summary"]["landed"], run
    assert run["summary"]["duration_s"] < 200, run


def test_imagery_label_names_the_naip_year(client, monkeypatch):
    import planner.app
    from planner.mission import Mission

    def label(m):
        return client.post("/api/plan/imagery", json=m).json()["label"]

    longer = {**mission(), "waypoints": [*mission()["waypoints"], {"lat": LAT + 0.009, "lon": LON}]}
    outside = {**mission(), "waypoints": [{"lat": 44.0, "lon": -100.0}, {"lat": 44.01, "lon": -100.0}]}
    assert label({**mission(), "waypoints": []}) == "Newest NAIP available"
    assert label(mission()) == "NAIP 2022"  # cached
    assert label(longer) == "NAIP 2022"  # partly cached, the source's newest for the rest
    assert label(outside) == "No NAIP here"

    # Cells cached from an older year stay as they are: say so.
    sidecars = [planner.app.cache.path(c).with_suffix(".json") for c in Mission.model_validate(mission()).required_cells()]
    originals = [p.read_text() for p in sidecars]
    try:
        for p, text in zip(sidecars, originals):
            p.write_text(json.dumps({**json.loads(text), "scenes": [{"id": "old", "year": 2019}]}))
        assert label(mission()) == "NAIP 2019"
        assert label(longer) == "NAIP 2019 cached, 2022 for the rest"

        def offline(bbox):
            raise OSError("no network")

        monkeypatch.setattr(planner.app.source, "scenes", offline)
        assert label(longer) == "NAIP 2019 cached"
        assert label(outside) == "Newest NAIP available"
    finally:
        for p, text in zip(sidecars, originals):
            p.write_text(text)


def test_refusals(client):
    assert client.post("/api/sim/runs", json={"mission": {**mission(), "waypoints": []}}).status_code == 400
    far = {**mission(), "waypoints": [{"lat": 44.0, "lon": -100.0}, {"lat": 44.01, "lon": -100.0}]}
    assert client.post("/api/sim/runs", json={"mission": far}).status_code == 409
    bad = {"mission": mission(), "options": {"wind_speed": -1}}
    assert client.post("/api/sim/runs", json=bad).status_code == 422
    for path in ("/api/sim/runs/00000000", "/sim/00000000/", "/sim/00000000/events", "/api/sim/runs/..%2F..%2Fetc"):
        assert client.get(path).status_code == 404, path
