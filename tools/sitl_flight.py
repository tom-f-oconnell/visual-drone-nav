"""Fly a mission bundle in ArduCopter SITL with visnav as the only position source.

Starts SITL at the mission home, loads docs/ardupilot_visnav.parm (SITL's
simulated rangefinder replaces the serial lidar, and SITL's simulated GNSS
feeds GPS2 as truth only), uploads the mission with `visnav upload`, runs
`visnav run --camera sitl`, arms on the navigator's start position, flies the
mission in AUTO and logs truth (SIMSTATE) against ArduPilot's EKF and against
visnav's GPS1 until the vehicle disarms. Prints a summary.

Get the SITL binary from
https://firmware.ardupilot.org/Copter/stable/SITL_x86_64_linux_gnu/arducopter and
copter.parm from Tools/autotest/default_params/ in the ArduPilot source at the
same version (see that directory's git-version.txt).

Usage:
    python tools/sitl_flight.py --bundle BUNDLE --arducopter PATH --defaults copter.parm --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from pymavlink import mavutil

ROOT = Path(__file__).resolve().parents[1]
VISNAV = Path(sys.executable).with_name("visnav")
mav = mavutil.mavlink

SITL_PARAMS = {  # instead of the hardware lines in the .parm file
    "RNGFND1_TYPE": 100, "RNGFND1_MAX": 180,
    "SIM_GPS1_ENABLE": 1,  # SITL's GNSS on SERIAL3, which ArduPilot detects as GPS2: truth only
}


def parm_file(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        f = line.split("#")[0].split()
        if len(f) >= 2 and not f[0].startswith(("SERIAL", "RNGFND")):
            out[f[0]] = float(f[1])
    return out


class Gcs:
    """Minimal ground station on SITL's SERIAL0."""

    def __init__(self, log):
        self.m = mavutil.mavlink_connection("tcp:127.0.0.1:5760", source_system=255)
        self.m.wait_heartbeat(timeout=60)
        self.log, self.state, self.acks = log, {}, []
        self.m.mav.request_data_stream_send(self.m.target_system, self.m.target_component,
                                            mav.MAV_DATA_STREAM_ALL, 10, 1)
        for msg_id, hz in ((mav.MAVLINK_MSG_ID_SIMSTATE, 10), (mav.MAVLINK_MSG_ID_GPS2_RAW, 5),
                           (mav.MAVLINK_MSG_ID_GPS_RAW_INT, 5), (mav.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)):
            self.command(mav.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz, wait=False)

    def pump(self, seconds: float = 0.05):
        end = time.time() + seconds
        while (msg := self.m.recv_match(blocking=True, timeout=max(end - time.time(), 0.001))) is not None:
            kind = msg.get_type()
            if kind == "STATUSTEXT":
                self.log("FC:", msg.text)
            elif kind == "COMMAND_ACK":
                self.acks.append((msg.command, msg.result))
            elif kind != "HEARTBEAT" or msg.get_srcComponent() == mav.MAV_COMP_ID_AUTOPILOT1:
                self.state[kind] = msg  # (visnav's own heartbeats are routed here too)
            if time.time() > end:
                return

    def wait(self, cond, seconds: float) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            self.pump(0.2)
            if cond():
                return True
        return False

    def set_param(self, name: str, value: float) -> bool:
        """Returns False if the firmware has no such parameter."""
        for _ in range(3):  # echoes are occasionally lost, e.g. right after a GPS type change
            self.state.pop("PARAM_VALUE", None)
            self.m.mav.param_set_send(self.m.target_system, self.m.target_component, name.encode(), float(value),
                                      mav.MAV_PARAM_TYPE_REAL32)
            if self.wait(lambda: "PARAM_VALUE" in self.state and self.state["PARAM_VALUE"].param_id == name, 2):
                return True
        return False

    def command(self, cmd: int, *params: float, wait: bool = True):
        self.acks = []
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, cmd, 0,
                                     *(list(params) + [0] * (7 - len(params))))
        if wait and self.wait(lambda: any(c == cmd for c, _ in self.acks), 3):
            return next(r for c, r in self.acks if c == cmd)
        return None

    def set_mode(self, name: str):
        mode = self.m.mode_mapping()[name]
        self.m.set_mode(mode)
        self.wait(lambda: "HEARTBEAT" in self.state and self.state["HEARTBEAT"].custom_mode == mode, 5)

    @property
    def armed(self) -> bool:
        hb = self.state.get("HEARTBEAT")
        return bool(hb and hb.base_mode & mav.MAV_MODE_FLAG_SAFETY_ARMED)


def dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dn = math.radians(lat2 - lat1) * 6371000
    de = math.radians(lon2 - lon1) * 6371000 * math.cos(math.radians(lat1))
    return math.hypot(dn, de)


def summarise(rows: list[dict], min_alt: float) -> dict:
    """Horizontal errors against SITL truth above and below map-matching height."""
    out = {}
    for band, sel in (("above_min_alt", lambda r: r["rel_alt"] > min_alt),
                      ("below_min_alt", lambda r: 0.5 < r["rel_alt"] <= min_alt)):
        rs = [r for r in rows if sel(r)]
        ekf = np.array([dist_m(r["true_lat"], r["true_lon"], r["ekf_lat"], r["ekf_lon"]) for r in rs])
        vis = np.array([dist_m(r["true_lat"], r["true_lon"], r["vis_lat"], r["vis_lon"]) for r in rs
                        if r["vis_fix"] >= 3])
        for name, e in (("ekf", ekf), ("visnav", vis)):
            if len(e):
                out[f"{name}_{band}"] = {"median_m": round(float(np.median(e)), 2),
                                         "p95_m": round(float(np.percentile(e, 95)), 2),
                                         "max_m": round(float(e.max()), 2)}
    home, last = rows[0], rows[-1]
    out["landing_offset_m"] = round(dist_m(home["true_lat"], home["true_lon"], last["true_lat"], last["true_lon"]), 2)
    out["final_mode"], out["duration_s"] = last["mode"], last["t"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bundle", required=True)
    p.add_argument("--arducopter", required=True, help="ArduCopter SITL binary")
    p.add_argument("--defaults", required=True, help="ArduPilot's copter.parm for that binary")
    p.add_argument("--parm", default=str(ROOT / "docs" / "ardupilot_visnav.parm"))
    p.add_argument("--out", required=True)
    p.add_argument("--home-alt", type=float, default=1550.0, help="m AMSL of the SITL home")
    p.add_argument("--timeout", type=float, default=1200.0)
    p.add_argument("--visnav-args", default="", help="extra arguments for visnav run")
    a = p.parse_args(argv)

    out = Path(a.out)
    sitl_dir = out / "sitl"
    sitl_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.bundle) / "mission.json").read_text())
    home = f"{meta['home']['lat']},{meta['home']['lon']},{a.home_alt},0"
    params = {**parm_file(Path(a.parm)), **SITL_PARAMS}
    t0 = time.time()

    def log(*args):
        print(f"[{time.time() - t0:6.1f}]", *args, flush=True)

    def start_sitl(*extra):
        return subprocess.Popen([str(Path(a.arducopter).resolve()), "--model", "quad", "--home", home,
                                 "--defaults", str(Path(a.defaults).resolve()), "-I0", *extra], cwd=sitl_dir,
                                stdout=open(out / "arducopter.log", "a"), stderr=subprocess.STDOUT)

    procs = []
    try:
        # First boot: store the parameters, then reboot so GPS and rangefinder types take effect.
        procs.append(start_sitl("--wipe"))
        gcs = Gcs(log)
        for name, value in params.items():
            if not gcs.set_param(name, value):
                log(f"warning: SITL has no parameter {name}")
        gcs.pump(2)
        procs.pop().terminate()
        time.sleep(2)
        procs.append(start_sitl())
        gcs = Gcs(log)
        if not gcs.wait(lambda: "GPS2_RAW" in gcs.state and gcs.state["GPS2_RAW"].fix_type >= 3, 90):
            sys.exit("SITL GNSS (truth on GPS2) never got a fix")
        up = subprocess.run([VISNAV, "upload", "--bundle", a.bundle, "--mavlink", "tcp:127.0.0.1:5763"],
                            capture_output=True, text=True, timeout=60)
        log("upload:", up.stdout.strip() or up.stderr.strip()[-300:])
        procs.append(subprocess.Popen(
            [VISNAV, "run", "--bundle", a.bundle, "--mavlink", "tcp:127.0.0.1:5762", "--camera", "sitl", "-v",
             "--home-alt", str(a.home_alt), *a.visnav_args.split()],
            stdout=open(out / "visnav.log", "w"), stderr=subprocess.STDOUT))

        # Arm on visnav's start position: it needs a few seconds to start, then the EKF to align.
        gcs.set_mode("GUIDED")
        end = time.time() + 180
        while not gcs.armed and time.time() < end:
            gcs.command(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            gcs.pump(3)
        if not gcs.armed:
            sys.exit("could not arm on vision")
        log("armed on vision")
        gcs.set_mode("AUTO")
        gcs.command(mav.MAV_CMD_MISSION_START, 0, 0)

        rows, next_row = [], 0.0
        while time.time() - t0 < a.timeout and gcs.armed:
            gcs.pump(0.05)
            s = gcs.state
            gp, ss, v1, hb = (s.get(k) for k in ("GLOBAL_POSITION_INT", "SIMSTATE", "GPS_RAW_INT", "HEARTBEAT"))
            if time.time() >= next_row and gp and ss and v1:
                next_row = time.time() + 0.2
                rows.append({"t": round(time.time() - t0, 2), "mode": mavutil.mode_string_v10(hb),
                             "true_lat": ss.lat / 1e7, "true_lon": ss.lng / 1e7, "rel_alt": gp.relative_alt / 1e3,
                             "ekf_lat": gp.lat / 1e7, "ekf_lon": gp.lon / 1e7, "vis_lat": v1.lat / 1e7,
                             "vis_lon": v1.lon / 1e7, "vis_fix": v1.fix_type})
        log("disarmed" if not gcs.armed else "timed out")
        with open(out / "track.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary = summarise(rows, min_alt=25.0)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        logs = sorted((sitl_dir / "logs").glob("*.BIN"))
        if logs:
            shutil.copy(logs[-1], out / "flight.BIN")
        return summary
    finally:
        for proc in procs[::-1]:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
