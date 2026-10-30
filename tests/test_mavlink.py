"""MAVLink side against a fake flight controller over UDP (no SITL needed)."""

import threading
import time

from pymavlink import mavutil

from planner.mission import Mission
from visnav.mavlink_io import Autopilot, gps_time, read_wpl

mav = mavutil.mavlink
PORT = 14987


class FakeFC(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.conn = mavutil.mavlink_connection(f"udpout:127.0.0.1:{PORT}", source_system=1, source_component=1)
        self.items, self.gps_inputs, self.count = {}, [], None
        self.stop = False

    def run(self):
        last = 0
        while not self.stop:
            if time.time() - last > 0.05:
                last = time.time()
                self.conn.mav.heartbeat_send(mav.MAV_TYPE_QUADROTOR, mav.MAV_AUTOPILOT_ARDUPILOTMEGA, 0, 0, 0)
                self.conn.mav.attitude_send(0, 0.01, -0.02, 1.0, 0, 0, 0)
                self.conn.mav.global_position_int_send(0, 400000000, -1053000000, 1700000, 120000, 0, 0, 0, 0)
            msg = self.conn.recv_match(blocking=True, timeout=0.02)
            if msg is None:
                continue
            kind = msg.get_type()
            if kind == "MISSION_COUNT":
                self.count = msg.count
                self.conn.mav.mission_request_int_send(255, 191, 0, 0)
            elif kind == "MISSION_ITEM_INT":
                self.items[msg.seq] = msg
                if msg.seq + 1 < self.count:
                    self.conn.mav.mission_request_int_send(255, 191, msg.seq + 1, 0)
                else:
                    self.conn.mav.mission_ack_send(255, 191, mav.MAV_MISSION_ACCEPTED, 0)
            elif kind == "GPS_INPUT":
                self.gps_inputs.append(msg)


def test_gps_time():
    week, ms = gps_time(1792800000.0)  # 2026-10-24T00:00:00Z
    assert week == 2441 and ms == 518418000  # Saturday 00:00:18 GPS time


def test_upload_and_gps_input():
    m = Mission.model_validate({"waypoints": [{"lat": 40.0, "lon": -105.3}, {"lat": 40.01, "lon": -105.29}]})
    items = read_wpl(m.to_wpl())

    fc = FakeFC()
    fc.start()
    ap = Autopilot(f"udpin:127.0.0.1:{PORT}")
    try:
        ap.start(wait_heartbeat=5)
        ap.upload_mission(items)
        assert sorted(fc.items) == list(range(len(items)))
        wp = fc.items[3]
        assert wp.command == 16 and wp.x == 400000000 and abs(wp.z - 120) < 1e-6

        time.sleep(0.3)
        assert ap.attitude_at(time.monotonic()) is not None
        assert ap.height_agl() == 120.0

        ap.send_gps_input(40.0001, -105.3001, 1700.0, 3.0, -1.0, hacc=2.5, sacc=0.5)
        time.sleep(0.3)
        g = fc.gps_inputs[-1]
        assert g.lat == 400001000 and g.lon == -1053001000 and g.fix_type == 3
        assert abs(g.horiz_accuracy - 2.5) < 1e-6 and abs(g.vn - 3.0) < 1e-6
    finally:
        fc.stop = True
        ap.stop()


def test_viewer_follows_mavlink(tmp_path):
    """`visnav view` path: positions from the autopilot become viewer records."""
    from visnav.geo import MapRaster
    from visnav.sim import synthetic_map
    from visnav.viewer import FlightFeed, mavlink_feed

    synthetic_map(str(tmp_path / "m.tif"), 40.0, -105.3, size_m=300, seed=1)
    map_ = MapRaster.load(tmp_path / "m.tif")
    fc = FakeFC()
    fc.start()
    ap = Autopilot(f"udpin:127.0.0.1:{PORT}")
    feed = FlightFeed({}, b"")
    stop = threading.Event()
    try:
        ap.start(wait_heartbeat=5)
        threading.Timer(1.0, stop.set).start()
        mavlink_feed(ap, map_, feed, map_.ll_to_grid(40.0, -105.3), rate_hz=10, stop=stop)
        assert len(feed.records) >= 5
        rec = feed.records[-1]
        assert abs(rec["p"][0]) < 1 and abs(rec["p"][1]) < 1 and rec["p"][2] == 120.0
    finally:
        fc.stop = True
        ap.stop()
