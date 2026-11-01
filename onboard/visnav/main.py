"""Command line entry points.

  visnav run    --bundle DIR --mavlink /dev/ttyTHS1 --camera csi     fly with vision as GPS
  visnav sim    --bundle DIR                                         synthetic flight, error stats
  visnav upload --bundle DIR --mavlink /dev/ttyTHS1                  upload the mission to the FC
  visnav sim3d  --bundle DIR                                         closed-loop flight sim + 3D viewer
  visnav compare --bundle DIR                                        the same flight with and without map matching
  visnav view   --bundle DIR --mavlink udpin:0.0.0.0:14551           3D viewer following SITL or a vehicle
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pyproj import Geod

from .appearance import Appearance, CameraLook
from .geo import MapRaster
from .matcher import ChainMatcher, CorrMatcher, FeatureBackend, MapMatcher
from .navigator import Navigator
from .ortho import CameraModel

log = logging.getLogger("visnav")
GEOD = Geod(ellps="WGS84")


def load_bundle(path: str):
    d = Path(path)
    meta = json.loads((d / "mission.json").read_text())
    return d, meta, MapRaster.load(d / meta["map"]["file"])


def load_world(args, map_: MapRaster) -> MapRaster:
    """Imagery to render simulated camera frames from: --world, else the bundle map."""
    return MapRaster.load(args.world) if args.world else map_


def appearance_from_args(args) -> Appearance:
    try:
        return Appearance.parse(args.appearance, getattr(args, "seed", None))
    except ValueError as exc:
        raise SystemExit(f"--appearance: {exc}") from exc


def camera_from_args(args, meta) -> CameraModel:
    if args.calib:
        return CameraModel.from_json(args.calib)
    c = meta["mission"]["camera"]
    return CameraModel.from_hfov(c["width_px"], c["height_px"], c["hfov_deg"])


def build_matcher(args, map_):
    """args.matcher: "corr", a feature type, several joined by "+" to try in that order, or "none"."""
    if args.matcher == "none":
        return None

    def one(name):
        if name == "corr":
            return CorrMatcher(map_, min_score=args.min_score)
        return MapMatcher(map_, FeatureBackend(name, args.nfeatures), min_inliers=args.min_inliers)

    parts = [one(name) for name in args.matcher.split("+")]
    return parts[0] if len(parts) == 1 else ChainMatcher(*parts)


def build_navigator(args, meta, map_, cam, start=None) -> Navigator:
    matcher = build_matcher(args, map_)
    lat, lon = start or (meta["home"]["lat"], meta["home"]["lon"])
    return Navigator(map_, cam, matcher, lat, lon, start_sigma_m=args.start_sigma, min_map_alt=args.min_alt)


# run ------------------------------------------------------------------------
def cmd_run(args):
    from .camera import FrameGrabber
    from .mavlink_io import Autopilot

    if args.matcher == "none":
        raise SystemExit("--matcher none is for the simulators (sim, sim3d, compare), not for flying: nothing corrects its drift")
    bundle, meta, map_ = load_bundle(args.bundle)
    cam = camera_from_args(args, meta)
    ap = Autopilot(args.mavlink, args.baud)
    ap.start()
    start = tuple(float(v) for v in args.start.split(",")) if args.start else None
    if start is None and ap.pos and ap.pos.lat != 0:
        start = (ap.pos.lat, ap.pos.lon)  # e.g. real GPS still healthy at takeoff
    nav = build_navigator(args, meta, map_, cam, start)
    if args.camera == "sitl":
        from .camera import SitlCamera

        look = CameraLook(appearance_from_args(args))
        print("camera appearance:", look.appearance.changes() or "unchanged")
        grabber = SitlCamera(ap, load_world(args, map_), cam, look=look)
        args.latency = 0.0
    else:
        grabber = FrameGrabber(args.camera, cam.width, cam.height)
    stop = threading.Event()

    def sender():
        # Fixed schedule: ArduPilot marks a GPS unhealthy if messages average over 215 ms apart.
        period, next_t = 1 / args.gps_rate, time.monotonic()
        while not stop.is_set():
            next_t = max(next_t + period, time.monotonic())
            time.sleep(max(next_t - time.monotonic(), 0.0))
            st = nav.state(time.monotonic())
            if st is None:
                continue
            healthy = st.last_update_age < args.max_fix_age and st.sigma_m < args.lost_sigma
            # Until the EKF has an origin (set from this very message) the FC has no altitude to echo.
            alt = ap.pos.alt_msl if ap.pos and ap.pos.lat != 0 else args.home_alt
            ap.send_gps_input(st.lat, st.lon, alt, st.vn, st.ve,
                              hacc=st.sigma_m, sacc=st.vel_sigma, fix_type=3 if healthy else 0)

    threading.Thread(target=sender, daemon=True).start()
    log_path = bundle / f"visnav_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["t", "lat", "lon", "score", "strong", "inliers", "rmse_px", "scale", "heading_err_deg", "sigma_m",
                      "h_agl", "fc_lat", "fc_lon", "gps2_lat", "gps2_lon"])
        last_t, last_status = 0.0, 0.0
        try:
            while True:
                frame, t = grabber.latest(last_t)
                if frame is None:
                    continue
                last_t = t
                t_cap = t - args.latency
                att, h = ap.attitude_at(t_cap), ap.height_agl()
                if att is None or h is None:
                    continue
                fix = nav.process(frame, t_cap, att.roll, att.pitch, att.yaw, h)
                if fix:
                    g2 = ap.gps_raw
                    out.writerow([f"{t_cap:.3f}", f"{fix.lat:.7f}", f"{fix.lon:.7f}", f"{fix.score:.2f}", int(fix.strong), fix.inliers,
                                  f"{fix.rmse_px:.2f}", f"{fix.scale:.3f}", f"{fix.heading_err_deg:.1f}",
                                  f"{fix.sigma_m:.1f}", f"{h:.1f}", ap.pos.lat if ap.pos else "",
                                  ap.pos.lon if ap.pos else "", g2.lat / 1e7 if g2 else "",
                                  g2.lon / 1e7 if g2 else ""])
                    fh.flush()  # the log must survive power being cut at landing
                if t - last_status > 5:
                    last_status = t
                    st = nav.state(time.monotonic())
                    mode = ("GND" if st.on_ground else "MAP" if st.last_fix_age < 2
                            else "ODO" if st.last_update_age < 2 else "LOST")
                    ap.statustext(f"VNAV {mode} sig {st.sigma_m:.0f}m")
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            grabber.close()
            ap.stop()


# sim ------------------------------------------------------------------------
def cmd_sim(args):
    from .sim import SensorErrors, fly_route, render

    _, meta, map_ = load_bundle(args.bundle)
    world = load_world(args, map_)
    cam = camera_from_args(args, meta)
    rng = np.random.default_rng(args.seed)
    look = CameraLook(appearance_from_args(args), [args.seed, 1])
    route = [(w["lat"], w["lon"], w["alt"] or meta["mission"]["default_alt"]) for w in meta["mission"]["waypoints"]]
    # Start the filter from a deliberately wrong position to exercise acquisition.
    # Odometry alone has no way to find its way back from a wrong start.
    start_error = 0.0 if args.matcher == "none" else args.start_error
    lon0, lat0, _ = GEOD.fwd(route[0][1], route[0][0], rng.uniform(0, 360), start_error)
    nav = build_navigator(args, meta, map_, cam, (lat0, lon0))
    sensors = SensorErrors(rng, args.att_noise, args.yaw_noise, h_frac=args.alt_noise,
                           yaw_bias_deg=args.compass_bias, h_bias_frac=args.height_bias / 100)
    errs, fixes, n, comp = [], 0, 0, []
    for pose in fly_route(map_, route, meta["mission"]["speed"], args.rate, rng):
        if args.frames and n >= args.frames:
            break
        n += 1
        img = look(render(world, cam, pose.lat, pose.lon, pose.h, pose.roll, pose.pitch, pose.yaw), pose.t)
        roll, pitch, yaw, h = sensors(pose.t, pose.roll, pose.pitch, pose.yaw, pose.h)
        t0 = time.perf_counter()
        fix = nav.process(img, pose.t, roll, pitch, yaw, h)
        comp.append(time.perf_counter() - t0)
        fixes += fix is not None
        st = nav.state(pose.t)
        if st:
            errs.append(GEOD.inv(pose.lon, pose.lat, st.lon, st.lat)[2])
        if args.verbose and fix:
            log.info("t=%.1f score=%.1f err=%.1fm", pose.t, fix.score, errs[-1])
    e = np.array(errs)
    report = {
        "frames": n, "fix_rate": round(fixes / max(n, 1), 3),
        "err_median_m": round(float(np.median(e)), 2) if len(e) else None,
        "err_p95_m": round(float(np.percentile(e, 95)), 2) if len(e) else None,
        "err_max_m": round(float(e.max()), 2) if len(e) else None,
        "ms_per_frame": round(1000 * float(np.mean(comp)), 1),
        "appearance": look.appearance.changes(),
    }
    print(json.dumps(report, indent=2))
    return report


# upload ---------------------------------------------------------------------
def cmd_upload(args):
    from .mavlink_io import Autopilot, read_wpl

    items = read_wpl((Path(args.bundle) / "mission.waypoints").read_text())
    ap = Autopilot(args.mavlink, args.baud)
    ap.start()
    ap.upload_mission(items)
    print(f"uploaded {len(items)} items")
    ap.stop()


# sim3d / view -----------------------------------------------------------------
def write_atomic(path: Path, text: str):
    """So that something following a run never reads a half written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def build_flight(args):
    """The closed-loop flight sim3d and compare fly."""
    from .flightsim import FlightSim, SimConfig

    _, meta, map_ = load_bundle(args.bundle)
    cam = camera_from_args(args, meta)
    nav = build_navigator(args, meta, map_, cam)
    cfg = SimConfig(seed=args.seed, camera_hz=args.camera_hz, wind=(args.wind_e, args.wind_n),
                    gust_sigma=args.gust, min_vision_alt=args.min_alt, appearance=appearance_from_args(args),
                    compass_bias_deg=args.compass_bias, height_bias_pct=args.height_bias)
    return FlightSim(map_, cam, nav, meta, cfg, world=load_world(args, map_)), map_


def cmd_sim3d(args):
    from .flightsim import map_texture
    from .viewer import FlightFeed, export_html, serve

    sim, map_ = build_flight(args)
    feed = FlightFeed(sim.scene(), map_texture(map_))
    if args.port:
        serve(feed, args.port, args.host)
        print(f"3D viewer: http://{args.host}:{args.port}/")
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir:
        (run_dir / "map.jpg").write_bytes(feed.map_jpg)
        records_out = open(run_dir / "records.jsonl", "w")
        write_atomic(run_dir / "scene.json", json.dumps(feed.scene))
    wall0 = time.monotonic()

    def on_record(rec):
        feed.push(rec)
        if run_dir:
            records_out.write(json.dumps(rec) + "\n")
            records_out.flush()
        if args.realtime:
            ahead = rec["t"] / args.realtime - (time.monotonic() - wall0)
            if ahead > 0:
                time.sleep(ahead)

    records, summary = sim.run(on_record=on_record)
    feed.finish(summary)
    if run_dir:
        records_out.close()
        write_atomic(run_dir / "summary.json", json.dumps(summary))
    print(json.dumps(summary, indent=2))
    if args.export:
        export_html(args.export, feed.scene, records, feed.map_jpg, summary)
        print(f"wrote {args.export}")
    if args.record:
        Path(args.record).write_text(json.dumps({"scene": feed.scene, "records": records, "summary": summary}))
    if args.port and not args.exit:
        print("flight finished; viewer still running, Ctrl-C to quit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return summary


def _compare_flight(job) -> dict:
    args, matcher, seed = job
    return build_flight(argparse.Namespace(**{**vars(args), "matcher": matcher, "seed": seed}))[0].run()[1]


def cmd_compare(args):
    """Fly each seed with map matching and on odometry alone, and print how they differ."""
    seeds = range(args.seed, args.seed + args.seeds)
    jobs = [(args, matcher, seed) for seed in seeds for matcher in (args.matcher, "none")]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        flights = list(pool.map(_compare_flight, jobs))
    with_map, alone = flights[::2], flights[1::2]
    rows = [("error, median (m)", "ekf_err_median_m"), ("error, 95% (m)", "ekf_err_p95_m"), ("error, largest (m)", "ekf_err_max_m"),
            ("error at the end (m)", "ekf_err_final_m"), ("  as % of distance", "final_err_pct_of_distance"),
            ("off target at the end (m)", "landing_error_m"), ("map fixes (fraction of frames)", "fix_rate"),
            ("navigator ms per frame", "nav_ms_per_frame"), ("frames off the imagery", "frames_off_imagery")]
    print(f"mean of {len(seeds)} seed(s), {with_map[0]['distance_m'] / 1000:.1f} km, compass bias {args.compass_bias} deg, "
          f"height bias {args.height_bias}%")
    print(f"{'':32s}{'with map matching':>20s}{'odometry only':>16s}")
    for label, key in rows:
        print(f"{label:32s}{np.mean([f[key] for f in with_map]):20.2f}{np.mean([f[key] for f in alone]):16.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"with_map": with_map, "odometry_only": alone}, indent=2))
    return with_map, alone


def cmd_view(args):
    from .flightsim import FlightSim, map_texture
    from .mavlink_io import Autopilot
    from .viewer import FlightFeed, mavlink_feed, serve

    _, meta, map_ = load_bundle(args.bundle)
    scene_src = FlightSim(map_, camera_from_args(args, meta), None, meta)
    feed = FlightFeed(scene_src.scene(), map_texture(map_))
    ap = Autopilot(args.mavlink, args.baud)
    ap.start()
    serve(feed, args.port, args.host)
    print(f"3D viewer: http://{args.host}:{args.port}/  (following {args.mavlink})")
    try:
        mavlink_feed(ap, map_, feed, scene_src.origin)
    except KeyboardInterrupt:
        feed.finish()
        ap.stop()


def main(argv=None):
    p = argparse.ArgumentParser(prog="visnav")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--bundle", required=True, help="unzipped mission bundle directory")
        sp.add_argument("--calib", help="camera calibration JSON (default: from mission camera HFOV)")
        sp.add_argument("--matcher", default="corr",
                        help='"corr", "orb", "akaze", "sift", or several joined by "+" to try in that order; '
                             'in the simulators, "none" for odometry alone, never looking at the map\'s imagery')
        sp.add_argument("--min-score", type=float, default=2.6, help="correlation score required of a fix")
        sp.add_argument("--nfeatures", type=int, default=3000)
        sp.add_argument("--min-inliers", type=int, default=20, help="RANSAC inliers required of a feature fix")
        sp.add_argument("--start-sigma", type=float, default=5.0,
                        help="m, uncertainty of the start position (vehicle placed at the mission home)")
        sp.add_argument("--min-alt", type=float, default=25.0,
                        help="m AGL below which frames are not matched to the map, only used for odometry")
        sp.add_argument("-v", "--verbose", action="store_true")

    def simulated_camera(sp):
        sp.add_argument("--world", help="GeoTIFF to render camera frames from (default: the bundle map), "
                                        "e.g. the map.tif of the same mission exported from another NAIP year")
        sp.add_argument("--appearance", default="",
                        help='how simulated frames differ from that imagery: "name=value,..." with gamma, red, green, '
                             "blue, blur, noise, brightness_drift, color_drift, gamma_drift, drift_tau_s, or "
                             '"random" for realistic values (from --seed where there is one). Default: no change')

    def lasting_errors(sp):
        sp.add_argument("--compass-bias", type=float, default=0.0,
                        help="deg of heading error lasting all flight, which only map fixes can correct")
        sp.add_argument("--height-bias", type=float, default=0.0,
                        help="%% height error lasting all flight, which scales odometry")

    r = sub.add_parser("run", help="fly: camera + MAVLink")
    common(r)
    simulated_camera(r)
    r.add_argument("--mavlink", default="/dev/ttyTHS1")
    r.add_argument("--baud", type=int, default=921600)
    r.add_argument("--camera", default="csi", help='"csi", "csi:N", V4L2 index, GStreamer pipeline, file, or "sitl"')
    r.add_argument("--start", help="lat,lon to start from (default: FC position, else mission home)")
    r.add_argument("--gps-rate", type=float, default=10.0, help="Hz of GPS_INPUT messages")
    r.add_argument("--latency", type=float, default=0.05, help="s from exposure to frame delivery")
    r.add_argument("--max-fix-age", type=float, default=4.0,
                   help="s without a map fix, odometry or ground update before reporting no fix")
    r.add_argument("--home-alt", type=float, default=0.0,
                   help="m AMSL sent before the FC has an altitude of its own (sets the EKF origin height)")
    r.add_argument("--lost-sigma", type=float, default=60.0)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sim", help="synthetic flight over the bundle map")
    common(s)
    simulated_camera(s)
    s.add_argument("--rate", type=float, default=2.0, help="frames per second of flight")
    s.add_argument("--frames", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--start-error", type=float, default=60.0, help="m offset of initial guess")
    s.set_defaults(start_sigma=150.0)
    s.add_argument("--att-noise", type=float, default=1.0, help="deg roll/pitch error (slowly varying)")
    s.add_argument("--yaw-noise", type=float, default=3.0, help="deg heading error (slowly varying)")
    s.add_argument("--alt-noise", type=float, default=0.01, help="fractional height error")
    lasting_errors(s)
    s.set_defaults(func=cmd_sim)

    u = sub.add_parser("upload", help="upload mission.waypoints to the FC")
    u.add_argument("--bundle", required=True)
    u.add_argument("--mavlink", default="/dev/ttyTHS1")
    u.add_argument("--baud", type=int, default=921600)
    u.set_defaults(func=cmd_upload)

    f = sub.add_parser("sim3d", help="closed-loop flight simulation with a 3D viewer")
    common(f)
    simulated_camera(f)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--camera-hz", type=float, default=4.0)
    f.add_argument("--wind-e", type=float, default=3.0, help="m/s wind toward east")
    f.add_argument("--wind-n", type=float, default=-1.5, help="m/s wind toward north")
    f.add_argument("--gust", type=float, default=1.5, help="m/s gust strength")
    f.add_argument("--port", type=int, default=27351, help="viewer port, 0 for none")
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--realtime", type=float, default=0.0,
                   help="pace the sim at this multiple of real time (0 = as fast as possible)")
    f.add_argument("--export", help="write a standalone replay HTML here")
    f.add_argument("--record", help="write records JSON here")
    f.add_argument("--exit", action="store_true", help="exit when the flight ends instead of serving on")
    f.add_argument("--run-dir", help="existing directory to write scene.json, map.jpg, records.jsonl (as the flight "
                                     "goes) and summary.json to; the planner website follows runs this way")
    lasting_errors(f)
    f.set_defaults(func=cmd_sim3d)

    c = sub.add_parser("compare", help="fly the same mission with and without map matching")
    common(c)
    simulated_camera(c)
    lasting_errors(c)
    c.add_argument("--seed", type=int, default=0, help="first seed")
    c.add_argument("--seeds", type=int, default=3, help="how many seeds to fly each way")
    c.add_argument("--jobs", type=int, default=6, help="flights at once")
    c.add_argument("--camera-hz", type=float, default=4.0)
    c.add_argument("--wind-e", type=float, default=3.0, help="m/s wind toward east")
    c.add_argument("--wind-n", type=float, default=-1.5, help="m/s wind toward north")
    c.add_argument("--gust", type=float, default=1.5, help="m/s gust strength")
    c.add_argument("--json", help="write every flight's summary here")
    c.set_defaults(func=cmd_compare)

    w = sub.add_parser("view", help="3D viewer following ArduPilot SITL or a vehicle over MAVLink")
    w.add_argument("--bundle", required=True)
    w.add_argument("--calib")
    w.add_argument("--mavlink", default="udpin:0.0.0.0:14551")
    w.add_argument("--baud", type=int, default=921600)
    w.add_argument("--port", type=int, default=27351)
    w.add_argument("--host", default="127.0.0.1")
    w.set_defaults(func=cmd_view)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    main()
