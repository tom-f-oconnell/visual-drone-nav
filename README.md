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
uvicorn planner.app:app --port 8000      # open http://localhost:8000
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
`NAIPNAV_YEARS` (limit to NAIP years, e.g. `2022,2023`).

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

`run` writes `visnav_<time>.csv` in the bundle with every fix, the FC position
and the real GPS (if fitted as GPS2), so flights can be scored afterwards.

Calibrate the camera with OpenCV's chessboard calibration and write the
intrinsics in the format of `camera.example.json`. Without `--calib` the
navigator assumes an ideal lens with the mission's HFOV.

### ArduPilot setup

Load `docs/ardupilot_visnav.parm` after checking each name against your firmware
version. In short: Jetson on TELEM2 at 921600, `GPS1_TYPE=14` (MAVLink GPS), EKF3
position from GPS, height from baro, yaw from compass, a downward rangefinder,
and a real GPS on GPS2 for comparison only.

### Testing in ArduPilot SITL (not yet run; written for it)

```bash
sim_vehicle.py -v ArduCopter --custom-location=<home lat>,<home lon>,<alt>,0 --out=udp:127.0.0.1:14550
# in MAVProxy: param set GPS1_TYPE 14 ... (as in the .parm file, GPS2 = SITL GPS)
visnav run --bundle <bundle> --mavlink udpin:127.0.0.1:14550 --camera sitl
```

`--camera sitl` renders camera frames from the map at SITL's true pose, so the
whole loop (MAVLink in, GPS_INPUT out, EKF fusion) runs without hardware.

## Flight simulator and 3D viewer

```bash
visnav sim3d --bundle ~/missions/mission1              # open http://127.0.0.1:8080/
visnav sim3d --bundle ~/missions/mission1 --realtime 1 # pace at real time
visnav sim3d --bundle ~/missions/mission1 --port 0 --export flight.html --exit
```

`sim3d` flies the mission in a closed loop with GPS denied for the whole flight:

* A multirotor model (tilt-to-accelerate, attitude lag, drag, wind with gusts)
  flies takeoff, every waypoint and the RTL or land at the end.
* Camera frames are rendered from the bundle's NAIP map at the true pose and
  passed, with attitude, heading and height errors, through the real visnav
  navigator.
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

To watch ArduPilot SITL (or a real vehicle) instead of the built-in model:

```bash
sim_vehicle.py -v ArduCopter --custom-location=<lat>,<lon>,<alt>,0 --out=udp:127.0.0.1:14551
visnav view --bundle ~/missions/mission1 --mavlink udpin:0.0.0.0:14551
```

With SITL, the white path is the simulator's truth (SIMSTATE) and the blue path
is ArduPilot's EKF. On a real vehicle the reference is the GNSS on GPS2, if fitted.
`visnav view` was tested against a fake MAVLink autopilot only, not SITL.

The built-in model is a simplified stand-in, not ArduPilot: no motor or
battery model, no ArduPilot control code, and a simple per-axis Kalman filter
instead of EKF3. Use it to tune matching and see failure modes quickly; use SITL
to check the autopilot integration.

## How the matching works

1. Ray-cast the four image corners through the camera model and the vehicle
   attitude onto a flat ground plane at the measured height. That gives a
   homography that warps the frame to north-up at the map's 0.6 m/px, with grid
   convergence and the UTM scale factor folded in.
2. Crop the NAIP mosaic around the filter's predicted position (radius from its
   uncertainty, 40 to 400 m).
3. ORB features (AKAZE and SIFT also available), ratio test, RANSAC similarity
   fit. Reject fits with scale outside 0.8 to 1.25 or rotation over 30°.
4. The pixel under the camera maps to lat/lon. A constant-velocity Kalman filter
   gates outliers (chi-square 99.9%) and re-locks after repeated strong
   disagreements.

## Tests

```bash
pip install -r ground/requirements.txt pymavlink pytest
pytest tests
```

* `test_end_to_end.py` runs the planner API (cache job, tiles, export), unzips
  the bundle, and flies a synthetic mission through the onboard navigator.
* `test_end_to_end.py` also flies a short closed-loop mission in the simulator
  and exports the 3D replay.
* `test_mavlink.py` uploads a mission and sends GPS_INPUT to a fake flight
  controller over UDP, and checks the 3D viewer can follow it.

The development sandbox could not reach NAIP servers, so these tests use a
procedurally generated stand-in for NAIP. To run the same test on real
imagery, download NAIP GeoTIFFs and set
`NAIPNAV_TEST_NAIP=<dir> NAIPNAV_TEST_LAT=<lat> NAIPNAV_TEST_LON=<lon>`.

## Known limits and next steps

* Flat-terrain assumption. In hilly areas, use the rangefinder or add a DEM
  (Copernicus 30 m) to get height above the ground under the camera.
* NAIP is flown every 2 to 3 years in summer. Snow, harvest and new construction
  reduce matches. A learned matcher (SuperPoint + LightGlue under TensorRT on the
  Jetson) is the upgrade path; `FeatureBackend` is the seam for it.
* Over water, forest canopy or uniform fields there may be no fixes. Plan routes
  over textured ground, and keep the corridor wide enough for drift.
* GPS_INPUT reuses ArduPilot's GPS path, which is the simplest integration.
  For tighter fusion, send `ODOMETRY` / `VISION_POSITION_ESTIMATE` with
  `EK3_SRC1_POSXY=6` instead.
