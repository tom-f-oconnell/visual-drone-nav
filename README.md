# naipnav: NAIP flight planner and image-based navigation for ArduPilot

Two separate parts:

* **`ground/`** is a web flight planner (FastAPI + Leaflet). It downloads and
  caches NAIP imagery, lets you click waypoints on a map, and exports a
  *mission bundle*: the ArduPilot mission plus a NAIP mosaic of the route corridor.
* **`onboard/`** runs on an NVIDIA Jetson next to the flight controller. It
  orthorectifies frames from a downward camera using the autopilot's attitude
  and height, matches them to the bundle's NAIP mosaic, filters the fixes, and
  sends the result to ArduPilot as a MAVLink `GPS_INPUT`, so the EKF flies the
  mission on vision instead of satellites.

See the system diagram artifact for hardware wiring and data flow.

## Ground planner

```bash
cd ground
pip install -r requirements.txt
uvicorn planner.app:app --port 8347      # open http://localhost:8347
```

1. Click the map to add waypoints (drag to move, set per-waypoint altitude if needed).
2. **Cache NAIP for route** downloads every 0.01 degree cell that the corridor touches.
   Reads are windowed range requests against the Planetary Computer COGs, so only
   the needed pixels are fetched. Cells are kept in `data/naip/` and reused across missions.
3. **Export mission bundle** gives a zip with `mission.waypoints` (QGC WPL 110),
   `mission.json` and `map.tif` (UTM, RGB, 0.6 m).

Settings (environment variables): `NAIPNAV_DATA` (data dir), `NAIPNAV_GSD`
(cache resolution, default 0.6 m), `NAIPNAV_SOURCE` (`pc` for Planetary Computer,
or a directory of NAIP GeoTIFFs you downloaded, e.g. from USGS EarthExplorer),
`NAIPNAV_YEARS` (limit to NAIP years, e.g. `2022,2023`; cached separately from
the default newest-first cache), `NAIPNAV_FETCH_THREADS` (cells downloaded in
parallel, default 6). Caching takes roughly 5 to 10 s per cell from Planetary
Computer (the imagery is stored in West Europe).

The planner warns when the camera footprint is too small for reliable matching.
Rule of thumb: the footprint should be at least ~250 map pixels (~150 m at
0.6 m), which means a wide lens (about 90° HFOV) at 100 to 120 m AGL.

## Onboard (Jetson)

JetPack 6 on an Orin Nano / Orin NX. Use JetPack's OpenCV (it has GStreamer
for CSI cameras), then:

```bash
cd onboard
sudo apt install python3-rasterio python3-pyproj   # or pip install rasterio pyproj
pip install pymavlink
pip install -e . --no-deps
```

Copy the unzipped bundle to the Jetson, then:

```bash
visnav upload --bundle ~/missions/mission1 --mavlink /dev/ttyTHS1        # mission to FC
visnav run    --bundle ~/missions/mission1 --mavlink /dev/ttyTHS1 --camera csi --calib camera.json
visnav sim    --bundle ~/missions/mission1        # bench test: synthetic flight, prints error stats
```

Place the vehicle at the mission home (the first waypoint) before power on:
the navigator starts there (`--start lat,lon` and `--start-sigma` otherwise), so
ArduPilot has a position to arm with and no GNSS is needed at any point. On the
ground it holds that position; from about 1 m up, frame-to-frame visual
odometry from the same camera tracks the vehicle through takeoff, landing and
gaps between map fixes; from about 60 m up (with a 90° lens) frames are also
matched to the NAIP map. `--matcher` selects how: `corr` (the default), `orb`,
`akaze`, `sift`, or several joined by `+` to try in that order. Set `--home-alt`
to the launch site's elevation (m AMSL) so the EKF origin has the right height.

`run` writes `visnav_<time>.csv` in the bundle with every fix, the FC position
and the real GPS (if fitted as GPS2), so flights can be scored afterwards.

Calibrate the camera with OpenCV's chessboard calibration and write the
intrinsics in the format of `camera.example.json`. Without `--calib` the
navigator assumes an ideal lens with the mission's HFOV. Mount the camera
square to the flight controller and measure any tilt: at 120 m each degree of
attitude or camera mount error moves map fixes by about 2 m. Frames and
attitude must be time-matched within a few ms (`--latency`).

### ArduPilot setup

Load `docs/ardupilot_visnav.parm` (checked against ArduCopter 4.7.2). In short:
Jetson on TELEM2 at 921600, `GPS1_TYPE=14` (MAVLink GPS), EKF3 horizontal
position only from that GPS (velocity from the IMU), height from baro, yaw from
compass, a long-range downward lidar, and a real GPS on GPS2 for comparison
only. [design.md](design.md) explains each choice.

## Flight simulator and 3D viewer

```bash
visnav sim3d --bundle ~/missions/mission1              # open http://127.0.0.1:27351/
visnav sim3d --bundle ~/missions/mission1 --realtime 1 # pace at real time
visnav sim3d --bundle ~/missions/mission1 --port 0 --export flight.html --exit
```

`sim3d` flies the mission in a closed loop with GPS denied for the whole flight:

* A multirotor model (tilt-to-accelerate, attitude lag, drag, wind with gusts)
  flies takeoff, every waypoint and the RTL or land at the end.
* Camera frames are rendered from the bundle's NAIP map at the true pose and
  passed, with attitude, heading and height errors, through the real visnav
  navigator. Those errors drift slowly, as an autopilot's do, rather than
  jumping every frame. Below NAIP's resolution a procedural ground texture,
  fixed to the ground, stands in for what a real camera sees low down.
* Its output feeds a stand-in for ArduPilot's EKF (per-axis position, velocity
  and accelerometer bias, driven by a noisy, biased IMU). **The controller only
  sees that estimate**, so navigation error moves the real vehicle, as in flight.

The viewer (three.js, served by the stdlib, no extra Python dependencies) shows
the NAIP map as the ground, the vehicle with its attitude, the true path, the
autopilot's estimate, every vision fix, the camera footprint and frustum, the
mission plan, the live camera frame and an error trace with the dead-reckoning
stretches shaded. Orbit, chase and top views; drag to rotate, right-drag or
shift-drag to pan, wheel or pinch to zoom, double-click to follow again.
`--export` writes one self-contained HTML file that replays the flight anywhere.

By default camera frames are rendered from the same imagery the navigator
matches against, which is optimistic. For a more honest test, export the same
mission a second time from a different NAIP year and render the camera from it
with `--world` (`sim` takes it too):

```bash
NAIPNAV_YEARS=2023 NAIPNAV_GSD=0.3 uvicorn planner.app:app --port 8348   # export the mission again
visnav sim3d --bundle ~/missions/mission1 --world ~/missions/mission1_2023/map.tif
```

To watch ArduPilot SITL (or a real vehicle) instead of the built-in model:

```bash
sim_vehicle.py -v ArduCopter --custom-location=<lat>,<lon>,<alt>,0 --out=udp:127.0.0.1:14551
visnav view --bundle ~/missions/mission1 --mavlink udpin:0.0.0.0:14551
```

With SITL, the white path is the simulator's truth (SIMSTATE) and the blue path
is ArduPilot's EKF. On a real vehicle the reference is the GNSS on GPS2, if fitted.

The built-in model is a simplified stand-in, not ArduPilot: no motor or
battery model, no ArduPilot control code, and a simple per-axis Kalman filter
instead of EKF3. Use it to tune matching and see failure modes quickly; use SITL
to check the autopilot integration.

## Development

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r ground/requirements.txt pymavlink pytest
pip install -e ground -e onboard --no-deps
```

[design.md](design.md) covers how the navigation works, why ArduPilot is set
up as it is, how far the simulators can be trusted, measured results and known
limits.

Fly a bundle in ArduCopter SITL, GNSS-denied from power on, with the prebuilt
SITL binary from `firmware.ardupilot.org/Copter/stable/SITL_x86_64_linux_gnu/`
and the matching `Tools/autotest/default_params/copter.parm`:

```bash
python tools/sitl_flight.py --bundle ~/missions/mission1 --arducopter ./arducopter \
    --defaults copter.parm --out sitl_out          # prints errors against SITL truth
```

Add `--visnav-args "--world OTHER_YEAR/map.tif"` to render the camera from another
NAIP year.

Compare map matchers across NAIP years on 12 sites (`pip install torch kornia`
for the LightGlue matchers):

```bash
python bench/fetch_sites.py DATA
python bench/match_bench.py DATA OUT --matchers orb corr lg:features=aliked
python bench/report.py OUT
```

Tests:

```bash
pytest tests
```

* `test_end_to_end.py` runs the planner API (cache job, tiles, export), unzips
  the bundle, and flies a synthetic mission through the onboard navigator.
* `test_end_to_end.py` also flies a short closed-loop mission in the simulator
  and exports the 3D replay.
* `test_mavlink.py` uploads a mission and sends GPS_INPUT to a fake flight
  controller over UDP, and checks the 3D viewer can follow it.
* `test_matcher.py` covers the correlation matcher, matcher chains and the
  re-lock rule.
* `test_naip.py` checks cached cells land exactly on the UTM grid, including
  when read from a COG overview.
* `test_odometry.py` checks odometry against known moves from 2 to 120 m, the
  navigator's position on the ground before any fix, the fix-bias model, and
  that uncertainty keeps up with the distance flown without fixes.

The tests use a procedurally generated stand-in for NAIP, so they run offline.
To run the same test on real imagery, download NAIP GeoTIFFs and set
`NAIPNAV_TEST_NAIP=<dir> NAIPNAV_TEST_LAT=<lat> NAIPNAV_TEST_LON=<lon>`.
