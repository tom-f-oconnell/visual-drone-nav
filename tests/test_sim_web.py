"""Running and viewing simulations from the planner website, and what it says about the imagery."""

import importlib
import json
import os
import time
from html.parser import HTMLParser

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


def test_camera_appearance(client):
    """Unchanged unless asked; Randomize's values and hand-set ones reach the simulator, which reports them."""
    from visnav.appearance import REALISTIC

    values = client.get("/api/sim/appearance/random").json()
    assert set(values) == set(REALISTIC)
    assert all(REALISTIC[k][0] <= v <= REALISTIC[k][1] for k, v in values.items())
    values["gamma"], values["blue"] = 1.1, 1.0  # the summary lists changes, and a random value can round to its default
    run_id = client.post("/api/sim/runs", json={"mission": mission("look"), "options": {"appearance": values}}).json()["id"]
    run = wait_for(client, run_id, ("finished", "failed", "stopped"))
    assert run["state"] == "finished", run
    assert run["options"]["appearance"] == values
    assert run["summary"]["appearance"] == {k: v for k, v in values.items() if k != "blue"}


def test_compare_with_and_without_map_matching(client):
    """Both flights get the same settings, name each other, and the comparison page has what it needs."""
    options = {"seed": 3, "compass_bias_deg": 3.0, "navigation": "odometry"}
    a, b = client.post("/api/sim/compare", json={"mission": mission("cmp"), "options": options}).json()
    assert (a["partner"], b["partner"]) == (b["id"], a["id"])
    assert [a["options"]["navigation"], b["options"]["navigation"]] == ["map", "odometry"]
    assert a["options"]["seed"] == b["options"]["seed"] == 3

    # Another comparison does not fit alongside these, and neither flight of it starts.
    before = len(client.get("/api/sim/runs").json())
    assert client.post("/api/sim/compare", json={"mission": mission("cmp2")}).status_code == 429
    assert len(client.get("/api/sim/runs").json()) == before

    runs = [wait_for(client, r["id"], ("finished", "failed", "stopped")) for r in (a, b)]
    assert all(r["state"] == "finished" for r in runs), runs
    with_map, alone = runs[0]["summary"], runs[1]["summary"]
    assert with_map["navigation"] == "map" and alone["navigation"] == "odometry"
    assert with_map["vision_fixes"] > 0 and alone["vision_fixes"] == 0
    assert alone["compass_bias_deg"] == with_map["compass_bias_deg"] == 3.0

    data = client.get(f"/api/sim/compare/{a['id']}/{b['id']}").json()
    assert not data["active"] and len(data["plan"]) >= 3
    for run in data["runs"]:
        s = run["series"]
        assert len(s["km"]) == len(s["err"]) == len(s["x"]) == len(s["y"]) > 10
        assert s["km"] == sorted(s["km"]) and s["km"][-1] == pytest.approx(run["summary"]["distance_m"] / 1000, abs=0.01)
    page = client.get(f"/static/compare.html?a={a['id']}&b={b['id']}")
    assert page.status_code == 200 and "Odometry only" in page.text
    assert client.get(f"/api/sim/compare/{a['id']}/00000000").status_code == 404


def test_every_field_explains_itself_on_hover(client):
    """Each input and select has a title of its own or from the label around it."""

    class Fields(HTMLParser):
        def __init__(self):
            super().__init__()
            self.labels, self.missing, self.count = [], [], 0

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "label":
                self.labels.append(attrs.get("title"))
            elif tag in ("input", "select"):
                self.count += 1
                if not attrs.get("title") and not any(self.labels):
                    self.missing.append(attrs.get("id") or attrs.get("data-look"))

        def handle_endtag(self, tag):
            if tag == "label":
                self.labels.pop()

    fields = Fields()
    fields.feed(client.get("/").text)
    assert fields.count > 20 and not fields.missing, fields.missing


def test_cache_job_says_how_much_of_the_route_it_had_to_fetch(client):
    m = mission()
    cells = client.post("/api/plan/check", json=m).json()
    assert cells["cached"] == cells["cells"] == len(cells["cell_keys"]) > 1
    job = client.post("/api/cache/jobs", json=m).json()
    assert job["total"] == 0 and job["route_cells"] == cells["cells"]


def test_refusals(client):
    assert client.post("/api/sim/runs", json={"mission": {**mission(), "waypoints": []}}).status_code == 400
    far = {**mission(), "waypoints": [{"lat": 44.0, "lon": -100.0}, {"lat": 44.01, "lon": -100.0}]}
    assert client.post("/api/sim/runs", json={"mission": far}).status_code == 409
    for options in ({"wind_speed": -1}, {"navigation": "nonsense"}, {"compass_bias_deg": 90}):
        assert client.post("/api/sim/runs", json={"mission": mission(), "options": options}).status_code == 422
    for look in ({"gamma": 0}, {"red": 99}, {"nonsense": 1}):
        r = client.post("/api/sim/runs", json={"mission": mission(), "options": {"appearance": look}})
        assert r.status_code == 422 and "camera appearance" in r.json()["detail"], r.text
    for path in ("/api/sim/runs/00000000", "/sim/00000000/", "/sim/00000000/events", "/api/sim/runs/..%2F..%2Fetc"):
        assert client.get(path).status_code == 404, path
