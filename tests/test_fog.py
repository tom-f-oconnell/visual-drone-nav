"""Fog in simulated frames: how much of the ground it hides, by height, angle and place."""

import math

import numpy as np
import pytest

from visnav.flightsim import FlightSim, SimConfig
from visnav.fog import VISIBILITY_CONTRAST, Fog, FogLayer
from visnav.geo import MapRaster
from visnav.matcher import CorrMatcher
from visnav.navigator import Navigator
from visnav.ortho import CameraModel
from visnav.sim import render, synthetic_map

LAT, LON = 40.0, -105.3
CAM = CameraModel.from_hfov(640, 400, 90)
EVEN = Fog(visibility=400.0, top=200.0, patchiness=0.0)


def centre(T):
    return float(T[T.shape[0] // 2, T.shape[1] // 2])


def see(fog, h, roll=0.0, pitch=0.0, t=0.0, east=0.0, north=0.0, **kw):
    return FogLayer(fog, CAM, **kw).transmission(t, east, north, h, roll, pitch, 0.0)


def test_visibility_is_where_contrast_falls_to_2_percent():
    assert centre(see(Fog(visibility=100.0, top=1000.0, patchiness=0.0), 100.0)) == pytest.approx(0.02, abs=0.001)
    assert VISIBILITY_CONTRAST == pytest.approx(-math.log(0.02), abs=0.001)


@pytest.mark.parametrize("h", [10.0, 60.0, 120.0, 200.0])
def test_inside_the_layer_the_camera_looks_through_its_own_height(h):
    assert centre(see(EVEN, h)) == pytest.approx(math.exp(-VISIBILITY_CONTRAST * h / 400.0), abs=0.002)


@pytest.mark.parametrize("h", [200.0, 500.0, 5000.0])
def test_above_the_layer_only_its_depth_counts_so_it_works_at_any_height(h):
    assert centre(see(EVEN, h)) == pytest.approx(math.exp(-VISIBILITY_CONTRAST * 200.0 / 400.0), abs=0.002)


def test_on_the_ground_there_is_no_fog_between_camera_and_ground():
    assert see(EVEN, 0.0).min() == pytest.approx(1.0)


def test_frame_edges_look_through_more_fog_and_tilting_moves_that():
    T = see(EVEN, 120.0)
    assert T[0, 0] < T[0, 320] < centre(T)
    tilted = see(EVEN, 120.0, pitch=math.radians(15))
    assert not np.isclose(tilted[200, 320], centre(T), atol=0.005)


def test_patches_stay_on_the_ground_drift_with_the_wind_and_follow_the_seed():
    fog = Fog(visibility=400.0, top=200.0, patchiness=0.8, patch=100.0)
    assert np.ptp(see(fog, 120.0)) > 0.2  # varies across a frame
    assert np.array_equal(see(fog, 120.0, seed=3), see(fog, 120.0, seed=3))
    assert not np.array_equal(see(fog, 120.0, seed=3), see(fog, 120.0, seed=4))

    # Moving the camera 50 m east moves what was seen 50 m to the west in the frame.
    px = int(50 / (2 * 120 / 640))
    a, b = see(fog, 120.0), see(fog, 120.0, east=50.0)
    assert np.abs(a[150:250, 300:400] - b[150:250, 300 - px:400 - px]).max() < 0.05 + np.abs(a[150:250, 300:400] - a[150:250, 310:410]).max()

    # After 10 s of 5 m/s wind toward the east the pattern has moved 50 m east with it, as if the camera moved 50 m west.
    drifting = FogLayer(fog, CAM, wind=(5.0, 0.0))
    assert np.array_equal(drifting.transmission(10.0, 0.0, 0.0, 120.0, 0.0, 0.0, 0.0),
                          FogLayer(fog, CAM).transmission(0.0, -50.0, 0.0, 120.0, 0.0, 0.0, 0.0))


def test_a_frame_fades_towards_the_fog_light():
    img = np.zeros((400, 640, 3), np.uint8)
    img[::2] = 200
    seen = FogLayer(Fog(visibility=30.0, top=1000.0, patchiness=0.0, brightness=0.8), CAM)(img, 0.0, 0.0, 0.0, 120.0, 0.0, 0.0, 0.0)
    assert seen[200, 320].mean() == pytest.approx(0.8 * 255, abs=1)  # nothing of the ground left
    assert FogLayer(Fog(visibility=1e5, top=1.0, patchiness=0.0), CAM)(img, 0.0, 0.0, 0.0, 120.0, 0.0, 0.0, 0.0)[200, 320, 0] == pytest.approx(200, abs=1)


def test_parse_and_limits():
    assert Fog.parse("") is None and Fog.parse("on") == Fog()
    assert Fog.parse("visibility=300, top=150") == Fog(visibility=300.0, top=150.0)
    for bad in ("visibility", "visibility=x", "colour=1", "visibility=1", "patchiness=2"):
        with pytest.raises(ValueError):
            Fog.parse(bad)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    path = tmp_path_factory.mktemp("fog") / "m_synthetic_20220701.tif"
    return MapRaster.load(synthetic_map(str(path), LAT, LON, size_m=1500, gsd=0.6, seed=4))


def flight(world, fog, seconds=40.0):
    meta = {"home": {"lat": LAT, "lon": LON}, "mission": {"default_alt": 80, "takeoff_alt": 30, "speed": 10, "end_action": "land", "name": "t",
                                                          "waypoints": [{"lat": LAT + 0.002, "lon": LON, "alt": None}]}}
    sim = FlightSim(world, CAM, Navigator(world, CAM, CorrMatcher(world), LAT, LON), meta, SimConfig(seed=1, max_time=seconds, fog=fog))
    return sim, *sim.run()


def test_flights_are_clear_by_default_and_foggy_when_asked(world):
    sim, _, summary = flight(world, None, 15.0)
    assert sim.scene()["fog"] is None and summary["fog"] is None and sim.fog is None

    fog = Fog(visibility=60.0, top=1000.0, patchiness=0.0)
    sim, records, summary = flight(world, fog, 25.0)
    assert sim.scene()["fog"] == fog.as_dict() and summary["fog"] == fog.as_dict()
    # In fog this thick, at height, the matcher finds nothing. Low down it still sees the ground for odometry.
    assert summary["vision_fixes"] == 0 and records[-1]["p"][2] > 25


def test_render_is_unchanged_without_fog(world):
    a = render(world, CAM, LAT, LON, 100.0, 0.0, 0.0, 0.0)
    assert np.array_equal(a, render(world, CAM, LAT, LON, 100.0, 0.0, 0.0, 0.0))
