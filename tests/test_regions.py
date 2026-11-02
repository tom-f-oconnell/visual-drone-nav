"""Grouping cached imagery into regions, for the planner's list and where its map opens."""

import importlib
import os

import pytest
from fastapi.testclient import TestClient

from planner.imagery import Cell, Library, LocalSentinel2Source, LocalSource, neighbouring_groups

NAIP, CLEAR = "naip/gsd_0.60", "sentinel2/gsd_10.00"


@pytest.fixture
def library(tmp_path):
    return Library(tmp_path, lambda years=None: LocalSource(tmp_path, years), LocalSentinel2Source(tmp_path))


def put(library, directory, key, downloaded=1000.0):
    path = library.data / directory / f"{key}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, (downloaded, downloaded))
    return path


def test_groups_join_at_edges_and_corners_only():
    first, beside, diagonal = Cell(0, 0), Cell(0, 1), Cell(1, 2)
    alone = Cell(1, 4)  # one empty column away from the diagonal one
    groups = neighbouring_groups([first, beside, diagonal, alone])
    assert sorted(groups, key=len) == [{alone}, {first, beside, diagonal}]


def test_one_region_for_touching_cells_of_either_imagery(library):
    assert library.regions() == []
    put(library, NAIP, "4000_-10500")
    put(library, NAIP, "4000_-10499")
    put(library, CLEAR, "4000_-10500")
    put(library, CLEAR, "3999_-10500")

    [region] = library.regions()
    assert region["cells"] == 3
    assert region["bbox"] == [-105.0, 39.99, -104.98, 40.01]
    assert region["naip"] == {"cells": 2, "on_map": 2}
    assert region["sentinel2"] == {"cells": 2, "on_map": 2}


def test_newest_first(library):
    put(library, NAIP, "4000_-10500", downloaded=2000.0)
    put(library, NAIP, "4000_-10499", downloaded=1000.0)
    put(library, CLEAR, "4800_235", downloaded=1500.0)
    put(library, CLEAR, "-3400_1510", downloaded=500.0)

    regions = library.regions()
    assert [r["downloaded"] for r in regions] == [2000.0, 1500.0, 500.0]
    assert [r["center"][0] for r in regions[1:]] == [48.005, -33.995]
    assert regions[1]["naip"] == {"cells": 0, "on_map": 0}

    # Its older cell gets replaced: the region is the same, now the newest by more.
    put(library, NAIP, "4000_-10499", downloaded=3000.0)
    assert [r["downloaded"] for r in library.regions()] == [3000.0, 1500.0, 500.0]


def test_center_is_a_cached_cell_even_on_a_bent_route(library):
    for ix in range(10):
        put(library, NAIP, f"4000_{-10500 + ix}")
    for iy in range(1, 10):
        put(library, NAIP, f"{4000 + iy}_-10500")

    [region] = library.regions()
    lat, lon = region["center"]
    assert Cell.containing(lat, lon).key in {p.stem for p in library.default_naip.dir.glob("*.tif")}


def test_what_the_map_cannot_show_is_told_apart(library):
    put(library, "naip/gsd_0.60_y2020", "4000_-10500")
    put(library, "sentinel2/gsd_10.00_pass2026-10-29", "4000_-10500")
    put(library, "sentinel2/gsd_10.00_asof2026-01-01", "4000_-10500")

    [region] = library.regions()
    assert region["naip"] == {"cells": 1, "on_map": 0}
    assert region["sentinel2"] == {"cells": 1, "on_map": 0}

    [region] = library.regions("2026-01-01")  # the plan's date: now the map's Sentinel-2 is that one
    assert region["sentinel2"] == {"cells": 1, "on_map": 1}


def test_cell_still_being_written_is_not_listed(library):
    put(library, NAIP, "4000_-10500.1a2b3c4d.tmp")
    assert library.regions() == []


def test_api(tmp_path, monkeypatch):
    monkeypatch.setenv("NAIPNAV_DATA", str(tmp_path))
    monkeypatch.setenv("NAIPNAV_SOURCE", str(tmp_path))
    monkeypatch.setenv("NAIPNAV_S2_SOURCE", str(tmp_path))
    import planner.app

    importlib.reload(planner.app)
    client = TestClient(planner.app.app)
    assert 'id="regions"' in client.get("/").text
    assert client.get("/api/cache/regions").json() == []

    put(planner.app.library, NAIP, "4000_-10500")
    [region] = client.get("/api/cache/regions").json()
    assert region["naip"]["on_map"] == 1 and region["sentinel2"]["cells"] == 0
    assert client.get("/api/cache/regions", params={"date": "2026-01-01"}).status_code == 200
    assert client.get("/api/cache/regions", params={"date": "nonsense"}).status_code == 422
