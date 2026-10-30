"""Visual odometry, the navigator before any map fix, and the fix-bias model."""

import math

import numpy as np
import pytest
from pyproj import Geod

from visnav.filter import PositionFilter
from visnav.geo import MapRaster
from visnav.matcher import MapMatcher
from visnav.navigator import Navigator
from visnav.odometry import FrameOdometry
from visnav.ortho import CameraModel
from visnav.sim import render, synthetic_map

LAT, LON = 40.0, -105.3
GEOD = Geod(ellps="WGS84")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    path = tmp_path_factory.mktemp("odo") / "m_synthetic_20220701.tif"
    return MapRaster.load(synthetic_map(str(path), LAT, LON, size_m=600, gsd=0.6, seed=4))


@pytest.mark.parametrize("h", [2.0, 10.0, 40.0, 120.0])
def test_odometry_measures_known_move(world, h):
    cam = CameraModel.from_hfov(960, 600, 90)
    rng = np.random.default_rng(int(h))
    for _ in range(3):
        odo = FrameOdometry(cam)
        dn, de = rng.normal(0, 0.02 * h + 0.1, 2)
        att0 = np.radians([rng.normal(0, 5), rng.normal(0, 5), rng.uniform(0, 360)])
        att1 = att0 + np.radians(rng.normal(0, 2, 3))
        lon1, lat1, _ = GEOD.fwd(LON, LAT, math.degrees(math.atan2(de, dn)), math.hypot(dn, de))
        odo.step(render(world, cam, LAT, LON, h, *att0, rng), 0.0, *att0, h)
        m = odo.step(render(world, cam, lat1, lon1, h, *att1, rng), 0.25, *att1, h)
        assert m is not None
        assert math.hypot(m.d_north - dn, m.d_east - de) < max(m.sigma, 0.01), (m, dn, de)


def test_navigator_reports_start_on_ground(world):
    cam = CameraModel.from_hfov(960, 600, 90)
    nav = Navigator(world, cam, MapMatcher(world), LAT, LON, start_sigma_m=5.0)
    for i in range(20):
        assert nav.process(None, i * 0.25, 0.0, 0.0, 0.0, 0.05) is None
    st = nav.state(5.0)
    assert GEOD.inv(LON, LAT, st.lon, st.lat)[2] < 0.01
    assert st.sigma_m < 6 and abs(st.vn) < 0.01 and abs(st.ve) < 0.01 and st.last_update_age < 0.5


def test_sigma_grows_with_distance_on_odometry_alone(world):
    """Heading error persists, so without map fixes uncertainty must keep up with the distance flown."""

    class NoFix:
        def locate(self, ortho, prior_col, prior_row, search_radius_px):
            return None

    cam = CameraModel.from_hfov(960, 600, 90)
    rng = np.random.default_rng(0)
    nav = Navigator(world, cam, NoFix(), LAT, LON, start_sigma_m=5.0, odometry_drift=0.05)
    step_m, steps = 2.5, 96  # 240 m at 10 m/s, 4 Hz
    for i in range(steps + 1):
        lon, lat, _ = GEOD.fwd(LON, LAT, 0.0, i * step_m - steps * step_m / 2)
        nav.process(render(world, cam, lat, lon, 120.0, 0.0, 0.0, 0.0, rng), i * 0.25, 0.0, 0.0, 0.0, 120.0)
    assert nav.odometry_dist == pytest.approx(steps * step_m, rel=0.1)
    assert nav.state(steps * 0.25).sigma_m > 0.05 * steps * step_m * 0.9


def test_correlated_fixes_do_not_shrink_sigma_below_bias():
    kf = PositionFilter(fix_bias_m=3.0)
    kf.reset(0.0, 0.0, 5.0, 0.0, v_sigma=0.1)
    for i in range(1, 200):
        kf.update_velocity(0.0, 0.0, 0.05, i * 0.25)
        kf.update(2.5, 0.0, 2.0, i * 0.25)  # the same offset every time, as an attitude bias gives
    assert kf.pos_sigma() > 1.0
