"""Benchmark sites: one short route per terrain type, with two NAIP years each.

The map (what the navigator matches against) is the older year at the bundle
GSD of 0.6 m. The world (what the simulated camera sees) is a later year at the
finest GSD that year was flown at, so the benchmark measures matching across
NAIP years, as in a real flight, rather than against the map's own pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

ROUTE_D = 0.004  # degrees; the same three-waypoint pattern as tests/test_end_to_end.py


@dataclass(frozen=True)
class Site:
    name: str
    lat: float
    lon: float
    map_year: int
    world_year: int
    world_gsd: float
    terrain: str

    def waypoints(self) -> list[dict]:
        """About 1.6 km in two legs around (lat, lon)."""
        d = ROUTE_D
        return [
            {"lat": self.lat - d, "lon": self.lon - d}, {"lat": self.lat + d, "lon": self.lon - d / 2},
            {"lat": self.lat + d / 2, "lon": self.lon + d},
        ]


SITES = [
    Site("foothills", 40.00, -105.30, 2021, 2023, 0.3, "CO foothills: conifer, grass, rock, roads, some houses"),
    Site("farmsuburb", 40.10, -105.10, 2021, 2023, 0.3, "CO plains: irrigated fields, new suburbs"),
    Site("iowa", 42.00, -93.70, 2021, 2023, 0.3, "IA row crops, farmsteads, gravel roads"),
    Site("kansas", 37.70, -101.05, 2021, 2023, 0.3, "KS Great Plains: center-pivot fields"),
    Site("frisco", 33.15, -96.82, 2020, 2022, 0.6, "TX town centre and suburbs"),
    Site("tucson", 32.30, -111.05, 2021, 2023, 0.3, "AZ Sonoran desert, sparse houses, washes"),
    Site("cascades", 44.30, -121.60, 2020, 2022, 0.3, "OR ponderosa pine forest, forest roads"),
    Site("georgia", 31.30, -82.50, 2021, 2023, 0.6, "GA pine plantations, clearcuts, swamp"),
    Site("vermont", 44.26, -72.58, 2021, 2023, 0.3, "VT small town, river, mixed forest"),
    Site("marsh", 32.80, -79.82, 2021, 2023, 0.6, "SC tidal salt marsh and creeks"),
    Site("orchard", 36.70, -119.60, 2020, 2022, 0.6, "CA Central Valley orchards and vineyards"),
    Site("napa", 38.43, -122.38, 2020, 2022, 0.6, "CA Napa vineyards, riparian woods"),
]

BY_NAME = {s.name: s for s in SITES}
