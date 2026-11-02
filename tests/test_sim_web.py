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
    os.environ["NAIPNAV_S2_SOURCE"] = str(tmp_path_factory.mktemp("no_s2"))  # no passes, so no clouds
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

    for _ in range(120):  # removed just after the simulator exits, which is just after it says it finished
        if not (planner.app.DATA / "sim" / run_id / "bundle").exists():
            break
        time.sleep(0.5)
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
    from visnav.appearance import REALISTIC, Appearance

    values = client.get("/api/sim/appearance/random").json()
    assert set(values) == set(REALISTIC)
    assert all(REALISTIC[k][0] <= v <= REALISTIC[k][1] for k, v in values.items())
    values["blue"] = 1.0  # the summary lists changes, and a random value can round to its default
    run_id = client.post("/api/sim/runs", json={"mission": mission("look"), "options": {"appearance": values}}).json()["id"]
    run = wait_for(client, run_id, ("finished", "failed", "stopped"))
    assert run["state"] == "finished", run
    assert run["options"]["appearance"] == values
    assert run["summary"]["appearance"] == Appearance.from_dict(values).changes() and "blue" not in run["summary"]["appearance"]


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
    assert not data["active"]
    for run in data["runs"]:
        s = run["series"]
        assert len(run["plan"]) >= 3 and run["plan"][0] == [0, 0]
        assert len(s["km"]) == len(s["err"]) == len(s["x"]) == len(s["y"]) > 10
        assert s["km"] == sorted(s["km"]) and s["km"][-1] == pytest.approx(run["summary"]["distance_m"] / 1000, abs=0.01)
    assert [r["settings"]["Navigation"] for r in data["runs"]] == ["with map matching", "odometry only"]
    assert client.get(f"/static/compare.html?a={a['id']}&b={b['id']}").status_code == 200
    assert client.get(f"/api/sim/compare/{a['id']}/00000000").status_code == 404


def test_compare_any_two_flights_of_the_same_waypoints_and_fly_them_again(client):
    """Altitude and wind differ. The results come from what is stored, and the same waypoints are required."""
    import planner.app

    low = {**mission("low"), "default_alt": 70}
    first = client.post("/api/sim/runs", json={"mission": mission("cmp_any"), "options": {"seed": 4, "wind_speed": 1.0}}).json()["id"]
    second = client.post("/api/sim/runs", json={"mission": low, "options": {"seed": 5, "wind_speed": 6.0}}).json()["id"]
    a, b = (wait_for(client, i, ("finished", "failed", "stopped")) for i in (first, second))
    assert a["state"] == b["state"] == "finished", (a, b)
    assert a["route"] == b["route"] and a["partner"] is None
    assert a["mission"]["default_alt"] == 80 and b["mission"]["default_alt"] == 70

    # Results do not need the flights' records, which are by far their largest part.
    data = client.get(f"/api/sim/compare/{first}/{second}").json()
    for run_id in (first, second):
        (planner.app.DATA / "sim" / run_id / "records.jsonl").unlink()
    assert client.get(f"/api/sim/compare/{first}/{second}").json()["runs"][0]["series"] == data["runs"][0]["series"]
    settings = [r["settings"] for r in data["runs"]]
    assert [s["Wind"] for s in settings] == ["1 m/s from 300°", "6 m/s from 300°"]
    assert [s["Cruise altitude"] for s in settings] == ["80 m", "70 m"]
    assert settings[0]["Corridor half-width"] == settings[1]["Corridor half-width"] == "250 m"

    # Another route is not comparable, even with the same name.
    other = {**mission("cmp_any"), "waypoints": [{"lat": LAT - 0.002, "lon": LON}, {"lat": LAT + 0.002, "lon": LON + 0.001}]}
    elsewhere = client.post("/api/sim/runs", json={"mission": other}).json()["id"]
    assert wait_for(client, elsewhere, ("finished", "failed", "stopped"))["route"] != a["route"]
    assert client.get(f"/api/sim/compare/{first}/{elsewhere}").status_code == 409
    assert client.post(f"/api/sim/compare/{first}/{elsewhere}/rerun").status_code == 409

    # Flown again, each keeps what it was asked for, as new runs that name each other.
    again = client.post(f"/api/sim/compare/{first}/{second}/rerun").json()
    assert (again[0]["partner"], again[1]["partner"]) == (again[1]["id"], again[0]["id"])
    assert [r["options"]["seed"] for r in again] == [4, 5] and [r["options"]["wind_speed"] for r in again] == [1.0, 6.0]
    assert again[1]["mission"]["default_alt"] == 70
    assert client.post(f"/api/sim/compare/{first}/{second}/rerun").status_code == 429  # the limit on running at once
    for run in again:
        client.post(f"/api/sim/runs/{run['id']}/stop")
    for run in again:
        wait_for(client, run["id"], ("finished", "failed", "stopped"), seconds=30)
    seeded = client.post(f"/api/sim/compare/{first}/{second}/rerun", params={"new_seed": True}).json()
    assert seeded[0]["options"]["seed"] == seeded[1]["options"]["seed"]
    for run in seeded:
        client.post(f"/api/sim/runs/{run['id']}/stop")
    for run in seeded:
        wait_for(client, run["id"], ("finished", "failed", "stopped"), seconds=30)
    assert first in {r["id"] for r in client.get("/api/sim/runs").json()}  # the originals stay


def fake_run(root, run_id, home=(0.0, 0.0), points=((0.0, 0.0), (100.0, 0.0), (100.0, 250.0)), end="land", alt=120.0,
             options=None, mission=True, summary=True, track=(10.0, 3.0)):
    """A stored run as the simulator leaves one, on a map whose centre puts home at `home`."""
    d = root / run_id
    d.mkdir()
    at = [(home[0] + x, home[1] + y) for x, y in points]
    plan = [{"kind": "takeoff", "x": home[0], "y": home[1], "alt": 30.0}] + [{"kind": "wp", "x": x, "y": y, "alt": alt} for x, y in at]
    plan += {"land": [{"kind": "land", "x": at[-1][0], "y": at[-1][1], "alt": 0.0}],
             "rtl": [{"kind": "wp", "x": home[0], "y": home[1], "alt": alt}, {"kind": "land", "x": home[0], "y": home[1], "alt": 0.0}],
             "none": []}[end]
    (d / "scene.json").write_text(json.dumps({"home": list(home), "plan": plan, "camera": {"hfov_deg": 89.99999, "aspect": 0.625}, "gsd": 0.6, "name": "m"}))
    run = {"id": run_id, "name": "m", "created": "2026-11-02T10:00:00", "options": options or {}}
    if mission:
        run["mission"] = {"speed": 10.0, "corridor_m": 400.0, "camera": {"width_px": 1280, "height_px": 800}}
    (d / "run.json").write_text(json.dumps(run))
    with open(d / "records.jsonl", "w") as f:
        for i in range(5):
            x, y = home[0] + i * track[0], home[1] + i * track[1]
            f.write(json.dumps({"t": i * 0.2, "p": [x, y, 50.0], "ekf": [x + 3.0, y + 4.0], "leg": 1}) + "\n")
    if summary:
        (d / "summary.json").write_text(json.dumps({"duration_s": 1.0, "landed": True, "ekf_err_median_m": 5.0}))
    return d


@pytest.fixture
def runs(tmp_path):
    from planner.simruns import SimRuns

    return SimRuns(tmp_path, None)


def test_same_route_on_any_map_at_any_altitude_with_any_ending(runs):
    fake_run(runs.root, "00000001")
    fake_run(runs.root, "00000002", home=(-3601.5, 692.55), alt=300.0)  # another map's centre
    fake_run(runs.root, "00000003", end="rtl")
    fake_run(runs.root, "00000004", end="none")
    fake_run(runs.root, "00000005", points=((0.0, 0.0), (100.0, 0.0), (100.0, 251.0)))
    fake_run(runs.root, "00000006", points=((0.0, 0.0), (100.0, 0.0)))
    routes = [runs.info(f"0000000{i}")["route"] for i in range(1, 7)]
    assert len(set(routes[:4])) == 1 and len(set(routes)) == 3


def test_paths_are_measured_from_home_so_different_maps_line_up(runs):
    fake_run(runs.root, "00000001")
    fake_run(runs.root, "00000002", home=(-3601.5, 692.55))
    a, b = runs.comparison("00000001", "00000002")["runs"]
    assert a["series"]["x"] == b["series"]["x"] == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert a["series"]["y"] == b["series"]["y"] == [0.0, 3.0, 6.0, 9.0, 12.0]
    assert a["plan"] == b["plan"] and a["plan"][0] == [0.0, 0.0] and [100.0, 250.0] in a["plan"]
    assert a["series"]["err"] == [5.0] * 5


def test_series_is_stored_once_and_follows_the_records(runs):
    d = fake_run(runs.root, "00000001")
    first = runs.series("00000001")
    assert json.loads((d / "series.json").read_text()) == first
    (d / "records.jsonl").unlink()
    assert runs.series("00000001") == first
    (d / "records.jsonl").write_text(json.dumps({"p": [1.0, 0.0, 0.0], "ekf": [1.0, 0.0]}) + "\n")
    assert runs.series("00000001")["x"] == [1.0]  # flown further since, e.g. by another planner process


def test_series_of_a_run_in_flight_is_not_stored(runs):
    d = fake_run(runs.root, "00000001", summary=False)
    runs._live["00000001"] = {"state": "running", "cells": None}
    assert len(runs.series("00000001")["km"]) == 5
    assert not (d / "series.json").exists()


@pytest.mark.parametrize("options, name", [
    ({"navigation": "odometry"}, "Navigation"),
    ({"world_year": 2020}, "Camera imagery"),
    ({"world_imagery": "sentinel2", "world_year": 2024}, "Camera imagery"),
    ({"fog": {"visibility": 400.0}}, "Fog"),
    ({"appearance": {"gamma": 1.2}}, "Camera appearance"),
    ({"wind_speed": 8.0}, "Wind"),
    ({"wind_from_deg": 10.0}, "Wind"),
    ({"gust": 4.0}, "Gusts"),
    ({"compass_bias_deg": 2.0}, "Compass bias"),
    ({"height_bias_pct": 5.0}, "Height bias"),
    ({"seed": 7}, "Seed"),
])
def test_each_option_shows_as_a_difference(runs, options, name):
    fake_run(runs.root, "00000001")
    fake_run(runs.root, "00000002", options=options)
    a, b = (runs.info(i)["settings"] for i in ("00000001", "00000002"))
    assert [k for k in a if a[k] != b[k]] == [name]


def test_each_part_of_the_plan_shows_as_a_difference(runs):
    def changed(**fake):
        fake_run(runs.root, "00000001")
        fake_run(runs.root, "00000002", **fake)
        a, b = (runs.info(i)["settings"] for i in ("00000001", "00000002"))
        for i in ("00000001", "00000002"):
            runs.delete(i)
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    assert changed(alt=300.0) == {"Cruise altitude": ("120 m", "300 m")}
    assert changed(end="rtl") == {"End of flight": ("land at the last waypoint", "return to launch")}
    assert changed(end="none") == {"End of flight": ("land at the last waypoint", "stay at the last waypoint")}

    # And what only run.json has, or what the world turned out to be.
    a, b = fake_run(runs.root, "00000001"), fake_run(runs.root, "00000002")
    for d, speed, corridor, gsd, imagery in ((a, 10.0, 400.0, 0.6, "naip"), (b, 5.0, 700.0, 10.0, "sentinel2")):
        run = json.loads((d / "run.json").read_text())
        run["mission"].update(speed=speed, corridor_m=corridor, camera={"width_px": 640, "height_px": 400})
        (d / "run.json").write_text(json.dumps({**run, "imagery": imagery, "imagery_date": "2026-01-02" if gsd == 10.0 else None}))
        scene = json.loads((d / "scene.json").read_text())
        (d / "scene.json").write_text(json.dumps({**scene, "gsd": gsd, "camera": {"hfov_deg": 90 if gsd == 0.6 else 70.0}}))
    (b / "world.json").write_text(json.dumps({"clouds": "real", "pass_date": "2026-10-29", "cloud_pct": 40.5, "base_m": 750.0}))
    s = runs.info("00000002")["settings"]
    assert s["Map imagery"] == "Sentinel-2 as of 2026-01-02, 10 m" and runs.info("00000001")["settings"]["Map imagery"] == "NAIP, 0.6 m"
    assert s["Speed"] == "5 m/s" and s["Corridor half-width"] == "700 m" and s["Camera field of view"] == "70°"
    assert s["Clouds"] == "pass of 2026-10-29, 40.5% of the route, seen above 750 m"


def test_a_run_from_before_plans_were_kept(runs):
    """Compares using what it stored, says what it did not, and cannot be flown again."""
    fake_run(runs.root, "00000001", mission=False, options={"seed": 0, "world_year": 2012})
    fake_run(runs.root, "00000002")
    old, new = runs.comparison("00000001", "00000002")["runs"]
    assert old["route"] == new["route"]
    assert old["settings"]["Speed"] is None and new["settings"]["Speed"] == "10 m/s"
    assert old["settings"]["Camera field of view"] == "90°" and old["settings"]["Cruise altitude"] == "120 m"
    assert old["options"]["navigation"] == "map" and old["options"]["clouds"] == "auto"  # as they flew
    assert old["summary"]["distance_m"] == pytest.approx(41.8, abs=0.5) and old["summary"]["ekf_err_final_m"] == 5.0
    assert old["summary"]["final_err_pct_of_distance"] == pytest.approx(12, abs=0.5) and old["summary"]["navigation"] == "map"
    assert runs.rerun_requests("00000001", "00000002") is None


def test_stopped_flight_can_be_compared(runs):
    fake_run(runs.root, "00000001", summary=False)
    fake_run(runs.root, "00000002")
    stopped, finished = runs.comparison("00000001", "00000002")["runs"]
    assert stopped["state"] == "stopped" and stopped["summary"] is None and stopped["series"]["km"]
    assert finished["state"] == "finished"


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
    problem = client.post("/api/sim/runs", json={"mission": mission(), "options": {"wind_speed": -1}}).json()["detail"][0]
    assert problem["loc"] == ["body", "options", "wind_speed"] and problem["msg"]  # what the page's errorText shows
    for fog in ({"visibility": 1}, {"nonsense": 1}):
        r = client.post("/api/sim/runs", json={"mission": mission(), "options": {"fog": fog}})
        assert r.status_code == 422 and "fog" in r.json()["detail"], r.text
    for look in ({"gamma": 0}, {"red": 99}, {"nonsense": 1}):
        r = client.post("/api/sim/runs", json={"mission": mission(), "options": {"appearance": look}})
        assert r.status_code == 422 and "camera appearance" in r.json()["detail"], r.text
    for path in ("/api/sim/runs/00000000", "/sim/00000000/", "/sim/00000000/events", "/api/sim/runs/..%2F..%2Fetc"):
        assert client.get(path).status_code == 404, path
