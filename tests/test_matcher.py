"""Correlation matcher, matcher chains and the navigator's re-lock rule."""

import math

import numpy as np
import pytest
from pyproj import Geod

from visnav.geo import MapRaster
from visnav.matcher import ChainMatcher, CorrMatcher, FeatureBackend, Fix, MapMatcher
from visnav.navigator import Navigator
from visnav.ortho import CameraModel, orthorectify
from visnav.sim import render, synthetic_map

LAT, LON = 40.0, -105.3
H = 120.0
GEOD = Geod(ellps="WGS84")
CAM = CameraModel.from_hfov(960, 600, 90)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    path = tmp_path_factory.mktemp("match") / "m_synthetic_20220701.tif"
    return MapRaster.load(synthetic_map(str(path), LAT, LON, size_m=1500, gsd=0.6, seed=9))


def frame_at(world, lat, lon, yaw_err_deg=0.0, seed=0, h=H):
    """Ortho of a nadir frame at (lat, lon), oriented using a heading wrong by yaw_err_deg."""
    img = render(world, CAM, lat, lon, h, 0.0, 0.0, 0.0, np.random.default_rng(seed))
    J = world.ned_to_px_jacobian(lat, lon)
    return orthorectify(CAM.undistort(img), CAM, 0.0, 0.0, math.radians(yaw_err_deg), h, J)


def offset(lat, lon, north, east):
    lon1, lat1, _ = GEOD.fwd(lon, lat, math.degrees(math.atan2(east, north)), math.hypot(north, east))
    return lat1, lon1


def test_corr_finds_position_from_offset_prior(world):
    fix = CorrMatcher(world).locate(frame_at(world, LAT, LON), *world.ll_to_px(*offset(LAT, LON, 20, -15)),
                                    int(40 / world.gsd))
    assert fix is not None and fix.strong
    assert GEOD.inv(LON, LAT, fix.lon, fix.lat)[2] < 2.0, fix


@pytest.mark.parametrize("yaw_err", [-3.0, 3.0])
def test_corr_heading_error_agrees_with_features(world, yaw_err):
    ortho = frame_at(world, LAT, LON, yaw_err)
    prior = world.ll_to_px(LAT, LON)
    radius = int(40 / world.gsd)
    corr = CorrMatcher(world).locate(ortho, *prior, radius)
    feat = MapMatcher(world, FeatureBackend("orb")).locate(ortho, *prior, radius)
    assert feat is not None and corr is not None
    assert abs(feat.heading_err_deg + yaw_err) < 0.5, feat  # true heading = told heading + heading_err_deg
    assert abs(corr.heading_err_deg + yaw_err) < 1.5, corr


def test_corr_rejects_peak_outside_or_on_rim_of_search_disc(world):
    """With the true position outside the disc, the highest shift is on the rim, which is no peak."""
    ortho = frame_at(world, LAT, LON)
    prior = world.ll_to_px(*offset(LAT, LON, 80, 0))
    assert CorrMatcher(world).locate(ortho, *prior, int(40 / world.gsd)) is None


def test_corr_search_area_may_overhang_map(world):
    """Wide searches near the map's edge still work; the outside just holds no candidates."""
    north_m = world.gray.shape[0] * world.gsd / 2 - 200  # 200 m inside the top edge
    lat, lon = offset(LAT, LON, north_m, 0)
    fix = CorrMatcher(world).locate(frame_at(world, lat, lon), *world.ll_to_px(lat, lon), int(350 / world.gsd))
    assert fix is not None
    assert GEOD.inv(lon, lat, fix.lon, fix.lat)[2] < 2.0, fix


def test_required_score_rises_with_search_radius(world):
    m = CorrMatcher(world, min_score=2.6)
    px = [int(r / world.gsd) for r in (20, 40, 150, 400)]
    assert m.required_score(px[0]) == m.required_score(px[1]) == pytest.approx(2.6, abs=0.01)
    assert m.required_score(px[1]) < m.required_score(px[2]) < m.required_score(px[3])


def test_corr_does_not_match_small_footprints(world):
    assert CorrMatcher(world).locate(frame_at(world, LAT, LON, h=40.0), *world.ll_to_px(LAT, LON),
                                     int(40 / world.gsd)) is None


def make_fix(col=0.0, row=0.0, strong=False):
    return Fix(col, row, 0.0, 0.0, 0, 0, 0.0, 1.0, 0.0, 3.0, score=3.0, strong=strong)


class Fixed:
    """Matcher returning a set fix, counting its calls."""

    def __init__(self, fix):
        self.fix, self.calls = fix, 0

    def locate(self, ortho, prior_col, prior_row, search_radius_px):
        self.calls += 1
        return self.fix


def test_chain_order():
    weak, strong = make_fix(1.0), make_fix(2.0, strong=True)
    a, b = Fixed(strong), Fixed(weak)
    assert ChainMatcher(a, b).locate(None, 0, 0, 1) is strong and b.calls == 0
    assert ChainMatcher(Fixed(weak), Fixed(strong)).locate(None, 0, 0, 1) is strong
    assert ChainMatcher(Fixed(None), Fixed(weak)).locate(None, 0, 0, 1) is weak
    assert ChainMatcher(Fixed(weak), Fixed(make_fix(3.0))).locate(None, 0, 0, 1) is weak
    assert ChainMatcher(Fixed(None), Fixed(None)).locate(None, 0, 0, 1) is None


@pytest.mark.parametrize("strong", [False, True])
def test_only_strong_fixes_relock_the_filter(world, strong):
    """Far from the filter's estimate, fixes are rejected; three rejects and a strong fix move the filter."""
    far = world.ll_to_px(*offset(LAT, LON, 200, 0))
    nav = Navigator(world, CAM, Fixed(make_fix(*far, strong=strong)), LAT, LON, start_sigma_m=5.0)
    img = render(world, CAM, LAT, LON, H, 0.0, 0.0, 0.0, np.random.default_rng(0))
    for i in range(8):
        nav.process(img, i * 0.25, 0.0, 0.0, 0.0, H)
    err = GEOD.inv(LON, LAT, nav.state(2.0).lon, nav.state(2.0).lat)[2]
    assert (err > 150) == strong, err
