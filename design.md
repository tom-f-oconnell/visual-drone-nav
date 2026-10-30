# Design

How visnav navigates, how it plugs into ArduPilot and why, how far the
simulators can be trusted, and what has been measured. Usage is in the README.

## Navigation

Per camera frame (`onboard/visnav/navigator.py`):

1. **Orthorectification.** Ray-cast the image corners through the camera model
   and the autopilot's attitude onto a flat ground plane at the measured height,
   giving a homography that warps the frame north-up at the map's 0.6 m/px
   (grid convergence and UTM scale folded in).
2. **Odometry** (`odometry.py`, from ~1 m up). Each frame is warped to a
   north-up ground patch whose scale follows height, and phase-correlated with
   the previous frame's patch. The shift is a velocity measurement. Peaks with
   a response under 0.1 are rejected: every gross error measured had one, good
   matches score 0.3 and up.
3. **Map matching** (above `--min-alt`, 25 m). Crop the NAIP mosaic around the
   filter's prediction (radius 3 sigma, 40 to 400 m); ORB, ratio test, RANSAC
   similarity; reject scale outside 0.8 to 1.25 or rotation over 30°.
4. **Filter** (`filter.py`). Constant-velocity Kalman filter on map-grid
   metres with a chi-square gate, plus a **map-fix bias state**: consecutive
   fixes share most of their error (attitude or camera mount error times
   height, NAIP registration), so averaging them must not shrink the reported
   accuracy below that. Odometry measures velocity, which the bias does not
   affect. Process noise is 4 m/s² in the air and 0.1 m/s² on the ground.
   The rotation of each map fit is the remaining compass error, low-passed
   into a heading correction used by both odometry and orthorectification.

The filter starts at the mission home with a 5 m sigma (the vehicle is placed
there), holds still on the ground, and runs on odometry until map fixes
arrive. That is what makes a GNSS-free takeoff and landing possible: before,
nothing was sent until a map fix at ~65 m, and ArduPilot cannot arm without a
position.

At 120 m, attitude error dominates. Open-loop sim median error is 0.3 m with
perfect attitude, 1.7 m with 0.5° and 3.3 m with 1° (slowly varying) error.
Camera boresight calibration matters more than anything in the matcher.

## ArduPilot integration

visnav is GPS1 (`GPS1_TYPE 14`, MAVLink `GPS_INPUT`); a real GNSS on GPS2 is
for truth only. Every setting below was found or confirmed in ArduCopter 4.7.2
SITL with `tools/sitl_flight.py`:

| Setting | Why |
|---|---|
| `EK3_SRC1_VELXY 0` (position only) | Odometry at 120 m is too noisy (~2 m/s per frame) for the filter to follow accelerations; its velocity lagged truth by 3 to 5 m/s accelerating to 10 m/s, the EKF rejected it and the variance failsafe landed the vehicle. EKF3 derives velocity from its IMU far better. |
| `EK3_GSF_USE_MASK 0` | The GSF yaw estimator needs real GNSS velocity; fed visnav's it did emergency yaw resets to wrong headings. Yaw stays on the compass. |
| `GPS1_DELAY_MS 10` | visnav predicts its estimate to send time. With 150 the EKF ran 2.8 m ahead of truth; 0 means the 200 ms driver default. |
| GPS_INPUT at 10 Hz on a fixed schedule | ArduPilot marks a GPS unhealthy if messages average over 215 ms apart; sleeping 200 ms after the work always exceeded it. |
| HDOP sent as hacc / 5 | EKF3 uses `horiz_accuracy`; HDOP only gates arming (`GPS_HDOP_GOOD` 1.4), so this means "arm if vision is better than ~7 m". |
| Speed accuracy small on the ground | EKF3 flags a GPS glitch, and refuses to arm, while reported speed accuracy is over 1.5 m/s. |
| Rangefinder: long range (TF03, type 27), tilt corrected | It has to reach cruise height to help; the raw distance is along the tilted body axis. |

Arming needs only visnav's fix: a GNSS on GPS2 that never gets a fix (jammed
at launch) does not block it.

Result, fully GNSS-denied mission in SITL (arm on vision, takeoff, three
waypoints, RTL, land), frames rendered from the map's own year:

| vs SITL truth | median | p95 | max |
|---|---|---|---|
| ArduPilot EKF, above 25 m | 1.2 m | 3.7 m | 7.5 m |
| ArduPilot EKF, below 25 m (odometry only) | 0.2 m | 0.2 m | 0.2 m |
| visnav GPS1, above 25 m | 2.1 m | 5.3 m | 9.4 m |

It landed 0.3 m from home and stayed in AUTO throughout.

`GPS_INPUT` reuses ArduPilot's GPS path, the simplest integration. For tighter
fusion, `ODOMETRY` / `VISION_POSITION_ESTIMATE` with `EK3_SRC1_POSXY 6` is the
alternative.

## Simulation fidelity

* **`visnav sim3d`**: multirotor model with wind, the real navigator, a
  per-axis position-only Kalman filter standing in for EKF3 (as configured
  above), and a controller that only sees that estimate. Attitude, heading and
  height errors are slowly varying (Gauss-Markov), as an AHRS's are. Below
  NAIP's resolution, a procedural ground texture fixed to the ground stands in
  for grass and gravel, so odometry has something to track at low height.
  Not modelled: terrain relief, real optics, motion blur, motors, ArduPilot's
  controllers.
* **`--world`** renders the camera from other imagery than the map, e.g.
  another NAIP year. Without it, sims match the map against itself and are
  optimistic.
* **SITL** (`--camera sitl`) renders frames with the attitude visnav is told,
  not SIMSTATE's true attitude: the two come on different MAVLink streams and
  their timing jitter (up to a degree while yawing) alone swamped odometry.
  SITL tests the integration; attitude errors are `sim3d`'s job. On hardware
  the same sensitivity applies: at 30°/s, 10 ms of camera/attitude timing
  error is 0.3°, 0.6 m at 120 m.
* The planner's corridor includes the return-to-launch leg (it used not to, and
  sims flew the leg home over no imagery).

## Measured on real NAIP

Planner: caching works against Planetary Computer; a 4-cell route takes ~40 s
(286 s before reading each scene with one windowed read from the matching COG
overview and fetching cells in parallel).

Farm and suburb route east of Boulder (2.4 km, 120 m AGL), `sim3d`, two seeds:

| | median | p95 | max | landing error |
|---|---|---|---|---|
| same year (2021 map and camera) | 2.5 m | 7.2 to 7.6 m | 9 to 14 m | 1 to 2 m |
| 2021 map, camera from 2023 | 3.7 to 5.4 m | 17 to 21 m | 23 to 31 m | 2 to 7 m |
| same, before odometry and fixes | 23 m | 171 m | 198 m | 228 m |

## Matcher benchmark (`bench/`)

Twelve US sites (foothills, farm/suburb, Iowa row crops, Kansas pivots, Texas
suburbs, Arizona desert, Oregon pine, Georgia plantations, Vermont town and
forest, South Carolina marsh, California orchards and vineyards), map from one
NAIP year and camera from the next, 80 frames per site and search radius, prior
at truth plus noise. Correct within 10 m; thresholds set for at most 1% wrong
fixes.

| matcher | correct at ≤1% wrong, 40 / 150 m search | same, only ≥20 m counts as wrong | ms/frame (one CPU thread) |
|---|---|---|---|
| ORB (current) | 35 / 26% | 40 / 26% | 116 / 140 |
| gradient-orientation correlation | 45 / 48% | 74 / 70% | 19 / 24 |
| DISK + LightGlue | 39 / 56% | 72 / 56% | 3400 / 7400 |
| ALIKED + LightGlue | 60 / 42% | 66 / 50% | 2000 / 5000 |

* Correlation (`bench/matchers.py: CorrMatcher`) is the best general choice:
  FFT correlation of contrast-invariant gradient orientations at 2.4 m/px with
  ±4° template rotations, refined at full resolution, scored by the
  peak-to-sidelobe ratio of each quarter of the frame. ORB then correlation
  reaches 57 / 57%.
* ALIKED + LightGlue is strongest in towns, forest edges and marsh and worth
  revisiting under TensorRT on the Jetson. SuperPoint's weights are
  non-commercial, so it was not tried.
* Nothing works on single frames over row crops or centre pivots (≤10%):
  that needs multi-frame evidence or routes planned along roads and farmsteads.
* Most confident "wrong" fixes are 10 to 15 m near misses caused by NAIP years
  disagreeing with each other (8 m locally in Vermont, ~1 m in most places).

## Known limits and next steps

* Integrate the correlation matcher (alone or after ORB). It returns no
  inlier count, so the re-lock rule must move to its score.
* Farmland needs multi-frame matching or route planning around it.
* Flat-terrain assumption: hilly areas need the rangefinder or a DEM
  (Copernicus 30 m) for height above the ground under the camera.
* NAIP is flown every 2 to 3 years in summer; snow, harvest and construction
  reduce matches further. Water and fresh snow give odometry nothing to track,
  and long stretches without map fixes still drift.
* Camera/attitude time sync and boresight calibration need doing on hardware.
