"""Visual odometry alone against odometry with map matching, flown in the same simulator."""

import dataclasses

import numpy as np
import pytest

from visnav.flightsim import FlightSim, SimConfig
from visnav.geo import MapRaster
from visnav.main import main
from visnav.matcher import CorrMatcher
from visnav.navigator import Navigator
from visnav.ortho import CameraModel
from visnav.sim import SensorErrors, synthetic_map

LAT, LON = 40.0, -105.3
CAM = CameraModel.from_hfov(640, 400, 90)


def meta(far_north_deg=0.009):
    return {"home": {"lat": LAT - far_north_deg / 2, "lon": LON}, "mission": {
        "default_alt": 100, "takeoff_alt": 30, "speed": 10, "end_action": "land",
        "waypoints": [{"lat": LAT + far_north_deg / 2, "lon": LON, "alt": None}]}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    path = tmp_path_factory.mktemp("compare") / "m_synthetic_20220701.tif"
    return MapRaster.load(synthetic_map(str(path), LAT, LON, size_m=2000, gsd=0.6, seed=6))


def fly(world, matcher, map_=None, **cfg):
    map_ = map_ if map_ is not None else world
    m = meta()
    nav = Navigator(map_, CAM, matcher, m["home"]["lat"], m["home"]["lon"], start_sigma_m=5.0)
    return FlightSim(map_, CAM, nav, m, SimConfig(seed=2, camera_hz=2.0, **cfg), world=world).run()


def test_odometry_only_never_looks_at_the_saved_imagery(world):
    """Blanking the map's pixels changes nothing, as only its georeferencing is used."""
    blank = dataclasses.replace(world, rgb=np.zeros_like(world.rgb), gray=np.zeros_like(world.gray),
                                valid=np.zeros_like(world.valid))
    records, summary = fly(world, None, max_time=40.0)
    blank_records, blank_summary = fly(world, None, map_=blank, max_time=40.0)
    assert [r["ekf"] for r in records] == [r["ekf"] for r in blank_records]
    assert summary["navigation"] == blank_summary["navigation"] == "odometry"
    assert summary["vision_fixes"] == 0 and summary["distance_m"] > 100


def test_only_map_matching_holds_against_a_compass_error(world):
    """5 degrees off pushes odometry about 9% of the distance sideways; map fixes keep it bounded."""
    _, alone = fly(world, None, compass_bias_deg=5.0)
    _, with_map = fly(world, CorrMatcher(world), compass_bias_deg=5.0)
    assert alone["landed"] and with_map["landed"]
    assert alone["final_err_pct_of_distance"] > 5, alone
    assert with_map["ekf_err_final_m"] < alone["ekf_err_final_m"] / 3, (with_map, alone)
    assert with_map["compass_bias_deg"] == alone["compass_bias_deg"] == 5.0


def test_lasting_sensor_errors():
    rng = np.random.default_rng(0)
    sensors = SensorErrors(rng, att_deg=0.0, yaw_deg=0.0, white_deg=0.0, h_min=0.0, h_frac=0.0,
                           yaw_bias_deg=10.0, h_bias_frac=0.05)
    for t in (0.0, 100.0, 1000.0):
        _, _, yaw, h = sensors(t, 0.0, 0.0, 0.0, 100.0)
        assert np.degrees(yaw) == pytest.approx(10.0)
        assert h == pytest.approx(105.0)


def test_flying_on_odometry_alone_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="simulators"):
        main(["run", "--bundle", str(tmp_path), "--matcher", "none"])
