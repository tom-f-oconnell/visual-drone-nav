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
3. **Map matching** (`matcher.py`, once the footprint reaches 8000 m², about
   60 m up with a 90° lens). Correlate the frame's contrast-invariant gradient
   orientations with the NAIP mosaic around the filter's prediction (radius 3
   sigma, 40 to 400 m): by FFT at 2.4 m/px with nine template rotations (±16°),
   then refined at full resolution. The score is how far the peak stands out
   from the rest of the search disc, as seen by each quarter of the frame on its
   own. A fix needs 2.6 with a 40 m search, plus 0.7 per e-fold of radius (3.5 at
   150 m, 4.2 at 400 m), since more candidates give chance peaks more
   opportunity. A peak on the rim of the disc is rejected: the surface is only
   rising toward something further out. `--matcher` also offers ORB, AKAZE or
   SIFT with a RANSAC similarity, alone or chained (`corr+orb`).
4. **Filter** (`filter.py`). Constant-velocity Kalman filter on map-grid
   metres with a chi-square gate, plus a **map-fix bias state**: consecutive
   fixes share most of their error (attitude or camera mount error times
   height, NAIP registration), so averaging them must not shrink the reported
   accuracy below that. Odometry measures velocity, which the bias does not
   affect. Process noise is 4 m/s² in the air and 0.1 m/s² on the ground.
   Odometry's heading and scale errors persist from frame to frame, so position
   uncertainty also grows by 5% of the distance flown since the last map fix.
   Without that the search radius stayed at its 40 m minimum while the real
   error passed 80 m, and the filter could never find the map again. Three
   rejected fixes in a row and a strong one (score 4, or 40 inliers) re-lock it.
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
waypoints, RTL, land) over the 2.4 km farm and suburb route east of Boulder, as
median / p95 / max against SITL truth. "Same year" renders the camera from the
map's own imagery; the other from NAIP two years later, with `visnav run --world`:

| | same year | 2021 map, 2023 camera |
|---|---|---|
| ArduPilot EKF, above 25 m | 0.8 / 1.6 / 2.2 m | 2.8 / 7.2 / 7.7 m |
| visnav GPS1, above 25 m | 1.3 / 2.5 / 3.1 m | 3.2 / 7.4 / 8.1 m |
| ArduPilot EKF, below 25 m (odometry only) | 0.3 m | 3.1 m |
| landing, from home | 0.4 m | 3.0 m |

Both stayed in AUTO with no GPS glitch or failsafe. The 2023 flight went up to
62 s between map fixes. On a bundle whose imagery stopped short of the return
leg, odometry and matching both lost the ground, visnav stopped reporting a fix,
and the EKF failsafe landed the vehicle where it was, as intended.

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
* **`--appearance`** (Camera appearance in the planner). Frames show the imagery
  as it is by default. Offsets to gamma, channel gains, blur and noise hold for
  the whole flight, as a stand-in for lighting, season and sensor differences from
  the NAIP flight. Brightness, colour and gamma can also wander (Gauss-Markov, from
  the offsets, 120 s by default), as clouds pass and auto exposure and white balance
  follow them. Only sensor noise is new each frame, and it does not move a frame's
  average. The renderer used to draw a new colour cast and gamma for every frame,
  which a real camera does not do, see below. `random`'s ranges
  (`onboard/visnav/appearance.py`) are judged from typical camera behaviour, not
  measured from flight footage. Not modelled: exposure following what is in the
  frame, cloud shadows within a frame.
* **SITL** (`--camera sitl`) renders frames with the attitude visnav is told,
  not SIMSTATE's true attitude: the two come on different MAVLink streams and
  their timing jitter (up to a degree while yawing) alone swamped odometry.
  SITL tests the integration; attitude errors are `sim3d`'s job. On hardware
  the same sensitivity applies: at 30°/s, 10 ms of camera/attitude timing
  error is 0.3°, 0.6 m at 120 m.
* The planner's corridor includes the return-to-launch leg (it used not to, and
  sims flew the leg home over no imagery).

Simulations started from the planner website (`ground/planner/simruns.py`) are
`visnav sim3d` subprocesses, at most three at a time. A separate process keeps a
heavy flight from stalling the planner, and keeps the onboard package an optional
dependency of it. The simulator appends to files in the run's directory and the
website's routes read them, so a live run and a replay take the same path and
runs survive a restart of the planner. The viewer is `sim3d`'s own page, served
per run under `/sim/<id>/`, which works because it only uses relative URLs. The
temporary bundle is deleted after each run, as the NAIP cache can rebuild it.

## Measured on real NAIP

Planner: caching works against Planetary Computer; a 4-cell route takes ~40 s
(286 s before reading each scene with one windowed read from the matching COG
overview and fetching cells in parallel).

Closed-loop `sim3d` flights over the benchmark's twelve sites (below), map from
one NAIP year and camera from the next, 90° lens, 120 m AGL. Errors are the
autopilot's estimate against truth, averaged over the sites; three seeds per
site unless noted:

| | median | p95 | max | landing error | worst landing | fixes |
|---|---|---|---|---|---|---|
| correlation (default) | 4.5 m | 14 m | 18 m | 5 m | 15 m | 61% |
| ORB, as the default used to be | 18 m | 33 m | 36 m | 22 m | 48 m | 17% |
| correlation, heading 15° off (one seed) | 38 m | 63 m | 72 m | 46 m | 257 m | 56% |
| correlation, height 5% off (one seed) | 7.6 m | 18 m | 22 m | 9 m | 34 m | 51% |

* On six sites (forest, towns, plantation, row crops, pivots) ORB gets almost no
  fixes and its error climbs to 30 to 50 m. Correlation's median stays under 8 m
  everywhere and its p95 under 20 m on all but the row crops and pivots (Iowa
  26 m, Kansas 41 m), where it loses the map for a while and then, thanks to the
  growing search radius, usually finds it again.
* Nine template rotations rather than three cost twice the time (37 ms against
  16 ms per frame) and make little difference in normal flight, but with a 15°
  heading error the median goes from 160 m to 38 m. Larger heading errors are
  not covered.
* Adding ORB as a fallback (`corr+orb`) gains 1% more fixes for about 5 times
  the time and barely helps with a 5% height error, so it is not the default.
* A wrong fix is most dangerous low down: nothing later corrects it. Of 960
  frames per height (three rotations, 40 m search), a score of 2.6 gives no
  errors over 20 m at 90 m and up, but 0.1% at 60 m and 0.4% at 40 m, so small
  footprints are not matched. Odometry is accurate there anyway.

The results in this file, except the table below, were measured when every frame
got its own random colour cast and gamma. That moved red/green by a typical 12%
between frames, against 0.2% from the ground passing below. Taking it out made no
clear difference to navigation on the one route checked, 5 km of suburbs near
Pasadena at 120 m with a NAIP 2022 map, four seeds per cell, sim3d, mean over seeds.
Median / p95 / landing error; 93% of frames got a fix in every cell:

| camera imagery | per-frame cast (old) | no change (default) | `random` |
|---|---|---|---|
| NAIP 2020 | 3.9 / 8.2 / 3.1 m | 3.7 / 7.6 / 2.7 m | 3.8 / 7.7 / 2.4 m |
| NAIP 2012 | 4.6 / 10.4 / 3.0 m | 5.3 / 10.6 / 3.1 m | 5.4 / 10.6 / 3.0 m |

Map matching against visual odometry alone (`visnav compare`), on the same 5.2 km
route with NAIP 2020 as the camera, three seeds, closed loop, mean over seeds. Odometry
alone starts from the first waypoint's coordinates and never looks at the map's pixels.
Median error / error at the end, then the end error as a share of distance flown:

| lasting error | with map matching | odometry only |
|---|---|---|
| none | 3.9 / 2.1 m, 0.04% | 45 / 79 m, 1.5% |
| compass 2° | 3.9 / 2.0 m, 0.04% | 104 / 175 m, 3.3% |
| height 2% | 3.3 / 1.3 m, 0.03% | 51 / 80 m, 1.6% |

* Map matching's error does not grow with distance; odometry's does. A 2° compass
  error over 5.2 km should push it off by `sin 2°`, 181 m. It ended 175 m off on
  average (129 to 253 m by seed), while map matching was unaffected.
* Odometry itself reads distance well: known moves on real NAIP at cruise (120 m,
  2.5 m per frame) read 0.997 of the truth. With no lasting error the drift is
  what the simulated heading wander predicts, 75 m on this route (3° standard
  deviation, 20 s correlation time).
* The price of the map is compute and storage: about 105 ms against 22 ms per frame
  with 18 flights running at once, and the imagery on board.
* This flatters odometry: frames come from a flat image, so there is no parallax,
  moving shadows or lens error to confuse it. The autopilot is also told to keep
  trusting it (accuracy capped at 5 m, `ODOMETRY_ONLY_SIGMA`), since its reported
  uncertainty only grows and an autopilot told so would stop following it. `visnav
  run` refuses `--matcher none`, so a real flight still stops reporting a fix instead.
* Frames with the camera past the corridor's imagery see nothing, unlike a real
  camera, so a wide enough corridor matters when comparing. The comparison counts
  them.

## Matcher benchmark (`bench/`)

Twelve US sites (foothills, farm/suburb, Iowa row crops, Kansas pivots, Texas
suburbs, Arizona desert, Oregon pine, Georgia plantations, Vermont town and
forest, South Carolina marsh, California orchards and vineyards), map from one
NAIP year and camera from the next, 80 frames per site and search radius, prior
at truth plus noise. Correct within 10 m; thresholds set for at most 1% wrong
fixes.

| matcher | correct at ≤1% wrong, 40 / 150 m search | same, only ≥20 m counts as wrong | ms/frame (one CPU thread) |
|---|---|---|---|
| ORB | 32 / 26% | 41 / 26% | 120 to 190 |
| gradient-orientation correlation | 38 / 32% | 74 / 70% | 30 to 40 |
| DISK + LightGlue (earlier run) | 39 / 56% | 72 / 56% | 3400 / 7400 |
| ALIKED + LightGlue (earlier run) | 60 / 42% | 66 / 50% | 2000 / 5000 |

* The filter tolerates near misses far better than gross errors, so the second
  column predicts flights better than the first: correlation's lead over ORB is
  much larger in flight than the first column suggests.
* With three template rotations instead of nine, correlation scores 46 / 47% and
  77 / 71%: the extra rotations give chance peaks more chances. They are worth it
  for the heading errors they tolerate, see the flights above.
* ALIKED + LightGlue is strongest in towns, forest edges and marsh and worth
  revisiting under TensorRT on the Jetson. SuperPoint's weights are
  non-commercial, so it was not tried.
* Nothing works on single frames over row crops or centre pivots (≤10%):
  that needs multi-frame evidence or routes planned along roads and farmsteads.
* Most confident "wrong" fixes are 10 to 15 m near misses caused by NAIP years
  disagreeing with each other (8 m locally in Vermont, ~1 m in most places).

## Known limits and next steps

* Farmland needs multi-frame matching or route planning around it.
* Correlation does not fit scale, so height errors over a few percent cost fixes.
* The heading estimate from correlation under-reads (about 0.4 of the true error
  between template rotations), so the correction converges slowly.
* Flat-terrain assumption: hilly areas need the rangefinder or a DEM
  (Copernicus 30 m) for height above the ground under the camera.
* NAIP is flown every 2 to 3 years in summer; snow, harvest and construction
  reduce matches further. Water and fresh snow give odometry nothing to track,
  and long stretches without map fixes still drift.
* Camera/attitude time sync and boresight calibration need doing on hardware.
