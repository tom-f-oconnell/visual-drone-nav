"""The simulated camera shows the imagery as it is unless asked, and never jumps between frames."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from visnav.appearance import REALISTIC, Appearance, CameraLook
from visnav.camera import SitlCamera
from visnav.flightsim import FlightSim, SimConfig
from visnav.geo import MapRaster
from visnav.matcher import CorrMatcher
from visnav.navigator import Navigator
from visnav.ortho import CameraModel
from visnav.sim import render, synthetic_map

LAT, LON = 40.0, -105.3
CAM = CameraModel.from_hfov(640, 400, 90)
GREY = np.full((60, 80, 3), 128, np.uint8)


def means(frames):
    return np.array([f.mean((0, 1)) for f in frames])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    path = tmp_path_factory.mktemp("look") / "m_synthetic_20220701.tif"
    return MapRaster.load(synthetic_map(str(path), LAT, LON, size_m=1500, gsd=0.6, seed=4))


def test_default_leaves_frames_untouched(world):
    a = render(world, CAM, LAT, LON, 120.0, 0.0, 0.0, 0.0)
    assert np.array_equal(a, render(world, CAM, LAT, LON, 120.0, 0.0, 0.0, 0.0))  # nothing random in rendering
    assert CameraLook()(a, 3.0) is a
    assert CameraLook(Appearance(drift_tau_s=30))(a, 3.0) is a  # how fast nothing drifts changes nothing


def test_gains_apply_to_their_own_channels():
    blue, green, red = means([CameraLook(Appearance(red=1.5, green=1.2, blue=0.5))(GREY)])[0]
    assert (blue, green, red) == pytest.approx((64, 154, 192), abs=1)


def test_offsets_are_the_same_on_every_frame():
    look = CameraLook(Appearance(gamma=1.2, red=0.9, green=1.05, blue=1.1, blur=0.8))
    first = look(GREY, 0.0)
    for t in (0.25, 60.0, 3600.0):
        assert np.array_equal(look(GREY, t), first)


def test_noise_changes_pixels_but_not_a_frames_colour():
    look = CameraLook(Appearance(noise=0.02), seed=1)
    frames = [look(GREY, i * 0.25) for i in range(20)]
    assert not np.array_equal(frames[0], frames[1])
    assert np.abs(means(frames) - 128).max() < 0.5


def test_drift_is_slow_starts_at_the_offsets_and_reaches_its_size():
    tau, hz, sd = 60.0, 4.0, 0.05
    a = Appearance(red=1.1, brightness_drift=0.0, color_drift=sd, drift_tau_s=tau)
    look = CameraLook(a, seed=2)
    wanted = np.array([128, 128, 128 * 1.1])
    m = means([look(GREY, i / hz) for i in range(int(400 * tau * hz))])
    assert m[0] == pytest.approx(wanted, abs=0.6)

    ratio = np.log(m[:, 2] / m[:, 1])
    assert np.median(np.abs(np.diff(ratio))) < 0.01  # the old per-frame colour cast moved this 12%
    # Red over green wanders by two channels' drift, around the offset.
    assert ratio.std() == pytest.approx(sd * 2 ** 0.5, rel=0.2)
    assert ratio.mean() == pytest.approx(np.log(1.1), abs=0.01)


def test_brightness_drift_keeps_the_colour_balance():
    look = CameraLook(Appearance(brightness_drift=0.1, drift_tau_s=10.0), seed=3)
    m = means([look(GREY, float(i)) for i in range(200)])
    assert m[:, 0].std() > 5
    assert np.abs(m[:, 2] - m[:, 0]).max() < 0.01


def test_same_seed_same_look():
    a = Appearance.parse("random", 7)
    assert a == Appearance.parse("random", 7) != Appearance.parse("random", 8)
    frames = [[CameraLook(a, 5)(GREY, t) for t in (0.0, 1.0)] for _ in range(2)]
    assert np.array_equal(frames[0], frames[1])


def test_random_is_realistic():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = Appearance.random(rng)
        assert all(low <= getattr(a, k) <= high for k, (low, high) in REALISTIC.items())


def test_parse():
    assert Appearance.parse("") == Appearance()
    assert Appearance.parse("gamma=1.2, red=0.9,drift_tau_s=30") == Appearance(gamma=1.2, red=0.9, drift_tau_s=30.0)
    assert Appearance(gamma=1.2, drift_tau_s=30.0).changes() == {"gamma": 1.2, "drift_tau_s": 30.0}
    for bad in ("gamma", "gamma=x", "hue=1", "gamma=0", "noise=-0.1", "gamma=nan", "blur=1000"):
        with pytest.raises(ValueError):
            Appearance.parse(bad)


def test_sitl_camera_applies_the_look(world):
    fix = SimpleNamespace(lat=int(LAT * 1e7), lng=int(LON * 1e7))
    ap = SimpleNamespace(simstate=(5.0, fix), height_agl=lambda: 120.0,
                         attitude_at=lambda t: SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0))
    plain, t = SitlCamera(ap, world, CAM).latest()
    warm, _ = SitlCamera(ap, world, CAM, look=CameraLook(Appearance(red=1.3))).latest()
    assert t == 5.0
    assert np.array_equal(plain[..., :2], warm[..., :2])
    assert warm[..., 2].mean() / plain[..., 2].mean() == pytest.approx(1.3, abs=0.02)


def flight(world, appearance, seconds=12.0):
    meta = {"home": {"lat": LAT, "lon": LON}, "mission": {
        "default_alt": 80, "takeoff_alt": 30, "speed": 10, "end_action": "land",
        "waypoints": [{"lat": LAT + 0.002, "lon": LON, "alt": None}]}}
    nav = Navigator(world, CAM, CorrMatcher(world), LAT, LON)
    return FlightSim(world, CAM, nav, meta, SimConfig(seed=1, max_time=seconds, appearance=appearance)).run()


def test_a_look_changes_the_camera_and_nothing_else_in_the_flight(world):
    import base64

    import cv2

    def thumbs(records):
        return [cv2.imdecode(np.frombuffer(base64.b64decode(r["img"]), np.uint8), cv2.IMREAD_COLOR)
                for r in records if "img" in r]

    plain, plain_summary = flight(world, Appearance())
    warm, warm_summary = flight(world, Appearance(red=1.3))
    assert plain_summary["appearance"] == {} and warm_summary["appearance"] == {"red": 1.3}
    json.dumps(warm_summary)  # written to summary.json

    # The vehicle's own randomness is untouched: the same wind and IMU up to where vision first differs.
    assert [r["p"] for r in plain[:10]] == [r["p"] for r in warm[:10]]
    ratio = means(thumbs(warm))[:, 2] / means(thumbs(plain))[:, 2]
    assert ratio.min() > 1.15 and ratio.max() < 1.4
