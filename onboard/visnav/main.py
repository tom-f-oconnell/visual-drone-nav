"""Command line entry points.

  visnav run    --bundle DIR --mavlink /dev/ttyTHS1 --camera csi     fly with vision as GPS
  visnav sim    --bundle DIR                                         synthetic flight, error stats
  visnav upload --bundle DIR --mavlink /dev/ttyTHS1                  upload the mission to the FC
  visnav sim3d  --bundle DIR                                         closed-loop flight sim + 3D viewer
  visnav view   --bundle DIR --mavlink udpin:0.0.0.0:14551           3D viewer following SITL or a vehicle
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np
from pyproj import Geod

from .geo import MapRaster
from .matcher import FeatureBackend, MapMatcher
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


def camera_from_args(args, meta) -> CameraModel:
    if args.calib:
        return CameraModel.from_json(args.calib)
    c = meta["mission"]["camera"]
    return CameraModel.from_hfov(c["width_px"], c["height_px"], c["hfov_deg"])


def build_navigator(args, meta, map_, cam, start=None) -> Navigator:
    matcher = MapMatcher(map_, FeatureBackend(args.features, args.nfeatures), min_inliers=args.min_inliers)
    lat, lon = start or (meta["home"]["lat"], meta["home"]["lon"])
    return Navigator(map_, cam, matcher, lat, lon, start_sigma_m=args.start_sigma)


# run ------------------------------------------------------------------------
def cmd_run(args):
    from .camera import FrameGrabber
    from .mavlink_io import Autopilot

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

        grabber = SitlCamera(ap, map_, cam)
        args.latency = 0.0
    else:
        grabber = FrameGrabber(args.camera, cam.width, cam.height)
    lock = threading.Lock()
    stop = threading.Event()

    def sender():
        period = 1 / args.gps_rate
        while not stop.is_set():
            time.sleep(period)
            with lock:
                st = nav.state(time.monotonic())
            if st is None or ap.pos is None:
                continue
            healthy = st.last_fix_age < args.max_fix_age and st.sigma_m < args.lost_sigma
            ap.send_gps_input(st.lat, st.lon, ap.pos.alt_msl, st.vn, st.ve,
                              hacc=st.sigma_m, sacc=st.vel_sigma, fix_type=3 if healthy else 0)

    threading.Thread(target=sender, daemon=True).start()
    log_path = bundle / f"visnav_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["t", "lat", "lon", "inliers", "rmse_px", "scale", "heading_err_deg", "sigma_m",
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
                if att is None or h is None or h < args.min_alt:
                    continue
                with lock:
                    fix = nav.process(frame, t_cap, att.roll, att.pitch, att.yaw, h)
                if fix:
                    g2 = ap.gps_raw
                    out.writerow([f"{t_cap:.3f}", f"{fix.lat:.7f}", f"{fix.lon:.7f}", fix.inliers,
                                  f"{fix.rmse_px:.2f}", f"{fix.scale:.3f}", f"{fix.heading_err_deg:.1f}",
                                  f"{fix.sigma_m:.1f}", f"{h:.1f}", ap.pos.lat if ap.pos else "",
                                  ap.pos.lon if ap.pos else "", g2.lat / 1e7 if g2 else "",
                                  g2.lon / 1e7 if g2 else ""])
                if t - last_status > 5:
                    last_status = t
                    st = nav.state(time.monotonic())
                    ap.statustext(f"VNAV {'ok' if st and st.last_fix_age < 2 else 'NOFIX'} "
                                  f"sig {st.sigma_m:.0f}m" if st else "VNAV waiting for first fix")
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            grabber.close()
            ap.stop()


# sim ------------------------------------------------------------------------
def cmd_sim(args):
    from .sim import fly_route, render

    _, meta, map_ = load_bundle(args.bundle)
    world = load_world(args, map_)
    cam = camera_from_args(args, meta)
    rng = np.random.default_rng(args.seed)
    route = [(w["lat"], w["lon"], w["alt"] or meta["mission"]["default_alt"]) for w in meta["mission"]["waypoints"]]
    # Start the filter from a deliberately wrong position to exercise acquisition.
    lon0, lat0, _ = GEOD.fwd(route[0][1], route[0][0], rng.uniform(0, 360), args.start_error)
    nav = build_navigator(args, meta, map_, cam, (lat0, lon0))
    errs, fixes, n, comp = [], 0, 0, []
    for pose in fly_route(map_, route, meta["mission"]["speed"], args.rate, rng):
        if args.frames and n >= args.frames:
            break
        n += 1
        img = render(world, cam, pose.lat, pose.lon, pose.h, pose.roll, pose.pitch, pose.yaw, rng)
        # Sensor errors in what the navigator is told.
        roll = pose.roll + math.radians(rng.normal(0, args.att_noise))
        pitch = pose.pitch + math.radians(rng.normal(0, args.att_noise))
        yaw = pose.yaw + math.radians(rng.normal(0, args.yaw_noise))
        h = pose.h * (1 + rng.normal(0, args.alt_noise))
        t0 = time.perf_counter()
        fix = nav.process(img, pose.t, roll, pitch, yaw, h)
        comp.append(time.perf_counter() - t0)
        fixes += fix is not None
        st = nav.state(pose.t)
        if st:
            errs.append(GEOD.inv(pose.lon, pose.lat, st.lon, st.lat)[2])
        if args.verbose and fix:
            log.info("t=%.1f inliers=%d err=%.1fm", pose.t, fix.inliers, errs[-1])
    e = np.array(errs)
    report = {
        "frames": n, "fix_rate": round(fixes / max(n, 1), 3),
        "err_median_m": round(float(np.median(e)), 2) if len(e) else None,
        "err_p95_m": round(float(np.percentile(e, 95)), 2) if len(e) else None,
        "err_max_m": round(float(e.max()), 2) if len(e) else None,
        "ms_per_frame": round(1000 * float(np.mean(comp)), 1),
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
def cmd_sim3d(args):
    from .flightsim import FlightSim, SimConfig, map_texture
    from .viewer import FlightFeed, export_html, serve

    _, meta, map_ = load_bundle(args.bundle)
    cam = camera_from_args(args, meta)
    nav = build_navigator(args, meta, map_, cam)
    cfg = SimConfig(seed=args.seed, camera_hz=args.camera_hz, wind=(args.wind_e, args.wind_n),
                    gust_sigma=args.gust)
    sim = FlightSim(map_, cam, nav, meta, cfg, world=load_world(args, map_))
    feed = FlightFeed(sim.scene(), map_texture(map_))
    if args.port:
        serve(feed, args.port, args.host)
        print(f"3D viewer: http://{args.host}:{args.port}/")
    wall0 = time.monotonic()

    def on_record(rec):
        feed.push(rec)
        if args.realtime:
            ahead = rec["t"] / args.realtime - (time.monotonic() - wall0)
            if ahead > 0:
                time.sleep(ahead)

    records, summary = sim.run(on_record=on_record)
    feed.finish(summary)
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
        sp.add_argument("--features", default="orb", choices=["orb", "akaze", "sift"])
        sp.add_argument("--nfeatures", type=int, default=3000)
        sp.add_argument("--min-inliers", type=int, default=20)
        sp.add_argument("--start-sigma", type=float, default=150.0, help="m, uncertainty of start position")
        sp.add_argument("-v", "--verbose", action="store_true")

    def world_arg(sp):
        sp.add_argument("--world", help="GeoTIFF to render camera frames from (default: the bundle map), "
                                        "e.g. the map.tif of the same mission exported from another NAIP year")

    r = sub.add_parser("run", help="fly: camera + MAVLink")
    common(r)
    r.add_argument("--mavlink", default="/dev/ttyTHS1")
    r.add_argument("--baud", type=int, default=921600)
    r.add_argument("--camera", default="csi", help='"csi", "csi:N", V4L2 index, GStreamer pipeline, file, or "sitl"')
    r.add_argument("--start", help="lat,lon to start from (default: FC position, else mission home)")
    r.add_argument("--gps-rate", type=float, default=5.0)
    r.add_argument("--latency", type=float, default=0.05, help="s from exposure to frame delivery")
    r.add_argument("--min-alt", type=float, default=25.0, help="m AGL below which matching is skipped")
    r.add_argument("--max-fix-age", type=float, default=4.0)
    r.add_argument("--lost-sigma", type=float, default=60.0)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sim", help="synthetic flight over the bundle map")
    common(s)
    world_arg(s)
    s.add_argument("--rate", type=float, default=2.0, help="frames per second of flight")
    s.add_argument("--frames", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--start-error", type=float, default=60.0, help="m offset of initial guess")
    s.add_argument("--att-noise", type=float, default=1.0, help="deg roll/pitch error")
    s.add_argument("--yaw-noise", type=float, default=3.0, help="deg heading error")
    s.add_argument("--alt-noise", type=float, default=0.03, help="fractional height error")
    s.set_defaults(func=cmd_sim)

    u = sub.add_parser("upload", help="upload mission.waypoints to the FC")
    u.add_argument("--bundle", required=True)
    u.add_argument("--mavlink", default="/dev/ttyTHS1")
    u.add_argument("--baud", type=int, default=921600)
    u.set_defaults(func=cmd_upload)

    f = sub.add_parser("sim3d", help="closed-loop flight simulation with a 3D viewer")
    common(f)
    world_arg(f)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--camera-hz", type=float, default=4.0)
    f.add_argument("--wind-e", type=float, default=3.0, help="m/s wind toward east")
    f.add_argument("--wind-n", type=float, default=-1.5, help="m/s wind toward north")
    f.add_argument("--gust", type=float, default=1.5, help="m/s gust strength")
    f.add_argument("--port", type=int, default=8080, help="viewer port, 0 for none")
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--realtime", type=float, default=0.0,
                   help="pace the sim at this multiple of real time (0 = as fast as possible)")
    f.add_argument("--export", help="write a standalone replay HTML here")
    f.add_argument("--record", help="write records JSON here")
    f.add_argument("--exit", action="store_true", help="exit when the flight ends instead of serving on")
    f.set_defaults(func=cmd_sim3d)

    w = sub.add_parser("view", help="3D viewer following ArduPilot SITL or a vehicle over MAVLink")
    w.add_argument("--bundle", required=True)
    w.add_argument("--calib")
    w.add_argument("--mavlink", default="udpin:0.0.0.0:14551")
    w.add_argument("--baud", type=int, default=921600)
    w.add_argument("--port", type=int, default=8080)
    w.add_argument("--host", default="127.0.0.1")
    w.set_defaults(func=cmd_view)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    main()
