"""MAVLink link to the ArduPilot flight controller (pymavlink).

The Jetson talks to the FC over a UART (e.g. Jetson /dev/ttyTHS1 to the FC's
TELEM2) at 921600 baud. It identifies as an onboard computer, reads attitude,
altitude and rangefinder data, and feeds position back as GPS_INPUT so that
ArduPilot treats vision as a GPS (GPS1_TYPE = 14, "MAVLink").
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

from pymavlink import mavutil

log = logging.getLogger(__name__)
mavlink = mavutil.mavlink

GPS_EPOCH_UNIX = 315964800  # 1980-01-06T00:00:00Z
GPS_LEAP_SECONDS = 18
SECONDS_PER_WEEK = 604800


def gps_time(unix_t: float) -> tuple[int, int]:
    gps_t = unix_t - GPS_EPOCH_UNIX + GPS_LEAP_SECONDS
    week = int(gps_t // SECONDS_PER_WEEK)
    return week, int((gps_t - week * SECONDS_PER_WEEK) * 1000)


@dataclass
class Attitude:
    t: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class GlobalPos:
    t: float
    lat: float
    lon: float
    alt_msl: float
    rel_alt: float
    vn: float
    ve: float
    vd: float


class Autopilot:
    def __init__(self, url: str, baud: int = 921600, source_system: int = 1):
        self.conn = mavutil.mavlink_connection(
            url, baud=baud, source_system=source_system,
            source_component=mavlink.MAV_COMP_ID_ONBOARD_COMPUTER, autoreconnect=True)
        self._att: deque[Attitude] = deque(maxlen=400)
        self._att_t: deque[float] = deque(maxlen=400)
        self.pos: GlobalPos | None = None
        self.rangefinder: tuple[float, float] | None = None  # (t, metres)
        self.home: tuple[float, float, float] | None = None
        self.gps_raw = None  # real GPS (if fitted as GPS2) for logging truth
        self.simstate = None  # SITL only: true vehicle state, used by SitlCamera
        self._mission_msgs: deque = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # Lifecycle -------------------------------------------------------------
    def start(self, wait_heartbeat: float = 30.0):
        log.info("waiting for heartbeat")
        if self.conn.wait_heartbeat(timeout=wait_heartbeat) is None:
            raise TimeoutError("no heartbeat from flight controller")
        log.info("connected to system %d", self.conn.target_system)
        for msg_id, hz in ((mavlink.MAVLINK_MSG_ID_ATTITUDE, 50),
                           (mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10),
                           (mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR, 20),
                           (mavlink.MAVLINK_MSG_ID_HOME_POSITION, 0.5),
                           (mavlink.MAVLINK_MSG_ID_GPS2_RAW, 5),
                           (mavlink.MAVLINK_MSG_ID_SIMSTATE, 20)):
            self.set_message_interval(msg_id, hz)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def stop(self):
        self._stop.set()

    def set_message_interval(self, msg_id: int, hz: float):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, msg_id, 1e6 / hz, 0, 0, 0, 0, 0)

    def _heartbeat(self):
        while not self._stop.is_set():
            self.conn.mav.heartbeat_send(mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                         mavlink.MAV_AUTOPILOT_INVALID, 0, 0, mavlink.MAV_STATE_ACTIVE)
            time.sleep(1)

    def _reader(self):
        while not self._stop.is_set():
            msg = self.conn.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            now = time.monotonic()
            kind = msg.get_type()
            if kind == "ATTITUDE":
                with self._lock:
                    self._att.append(Attitude(now, msg.roll, msg.pitch, msg.yaw))
                    self._att_t.append(now)
            elif kind == "GLOBAL_POSITION_INT":
                self.pos = GlobalPos(now, msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1e3, msg.relative_alt / 1e3,
                                     msg.vx / 100, msg.vy / 100, msg.vz / 100)
            elif kind == "DISTANCE_SENSOR" and msg.orientation == mavlink.MAV_SENSOR_ROTATION_PITCH_270:
                if msg.min_distance <= msg.current_distance <= msg.max_distance:
                    self.rangefinder = (now, msg.current_distance / 100)
            elif kind == "HOME_POSITION":
                self.home = (msg.latitude / 1e7, msg.longitude / 1e7, msg.altitude / 1e3)
            elif kind == "GPS2_RAW":
                self.gps_raw = msg
            elif kind == "SIMSTATE":
                self.simstate = (now, msg)
            elif kind in ("MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"):
                self._mission_msgs.append(msg)
            elif kind == "STATUSTEXT":
                log.info("FC: %s", msg.text)

    # Telemetry access ------------------------------------------------------
    def attitude_at(self, t: float) -> Attitude | None:
        """Attitude sample closest to monotonic time t (frame capture time)."""
        with self._lock:
            if not self._att:
                return None
            i = bisect.bisect_left(self._att_t, t)
            cands = [self._att[j] for j in (i - 1, i) if 0 <= j < len(self._att)]
        best = min(cands, key=lambda a: abs(a.t - t))
        return best if abs(best.t - t) < 0.1 else None

    def height_agl(self, max_age: float = 0.3) -> float | None:
        """Rangefinder if fresh, else altitude above home (assumes flat terrain)."""
        now = time.monotonic()
        if self.rangefinder and now - self.rangefinder[0] < max_age and self.rangefinder[1] > 1:
            return self.rangefinder[1]
        if self.pos and now - self.pos.t < 1.0:
            return self.pos.rel_alt
        return None

    # Outputs ---------------------------------------------------------------
    def send_gps_input(self, lat: float, lon: float, alt_msl: float, vn: float, ve: float,
                       hacc: float, sacc: float, fix_type: int = 3, satellites: int = 12):
        week, week_ms = gps_time(time.time())
        flags = (mavlink.GPS_INPUT_IGNORE_FLAG_VDOP | mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT)
        self.conn.mav.gps_input_send(
            int(time.time() * 1e6), 0, flags, week_ms, week, fix_type,
            int(lat * 1e7), int(lon * 1e7), alt_msl,
            max(hacc / 2.5, 0.5), 9.99,       # hdop, vdop
            vn, ve, 0.0, sacc, hacc, max(hacc, 3.0), satellites)

    def statustext(self, text: str, severity: int = mavlink.MAV_SEVERITY_INFO):
        self.conn.mav.statustext_send(severity, text.encode()[:50])

    def upload_mission(self, items: list[dict], timeout: float = 5.0):
        """Upload mission items (dicts from read_wpl) with the MISSION_ITEM_INT protocol."""
        sysid, compid = self.conn.target_system, self.conn.target_component
        self._mission_msgs.clear()
        self.conn.mav.mission_count_send(sysid, compid, len(items), mavlink.MAV_MISSION_TYPE_MISSION)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._mission_msgs:
                time.sleep(0.01)
                continue
            msg = self._mission_msgs.popleft()
            if msg.get_type() == "MISSION_ACK":
                if msg.type != mavlink.MAV_MISSION_ACCEPTED:
                    raise RuntimeError(f"mission rejected: {msg.type}")
                return
            it = items[msg.seq]
            self.conn.mav.mission_item_int_send(
                sysid, compid, msg.seq, it["frame"], it["command"], it["current"], it["autocontinue"],
                it["p1"], it["p2"], it["p3"], it["p4"], int(it["lat"] * 1e7), int(it["lon"] * 1e7),
                it["alt"], mavlink.MAV_MISSION_TYPE_MISSION)
            deadline = time.monotonic() + timeout
        raise TimeoutError("mission upload timed out")


def read_wpl(text: str) -> list[dict]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines[0].startswith("QGC WPL 110"):
        raise ValueError("not a QGC WPL 110 file")
    items = []
    for ln in lines[1:]:
        f = ln.split("\t")
        items.append(dict(seq=int(f[0]), current=int(f[1]), frame=int(f[2]), command=int(f[3]),
                          p1=float(f[4]), p2=float(f[5]), p3=float(f[6]), p4=float(f[7]),
                          lat=float(f[8]), lon=float(f[9]), alt=float(f[10]), autocontinue=int(f[11])))
    return items
