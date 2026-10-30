"""3D flight viewer: a small HTTP server (stdlib only) plus a standalone HTML export.

Sources of flight data:
  * FlightSim running in a background thread (`visnav sim3d`)
  * a live MAVLink stream from ArduPilot SITL or a real vehicle (`visnav view`)

Routes: /  (viewer), /scene.json, /map.jpg, /three.min.js, /events (server-sent
events, one JSON record per message; replays history to late joiners), /run.json.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


class FlightFeed:
    """Thread-safe record log that SSE clients follow."""

    def __init__(self, scene: dict, map_jpg: bytes):
        self.scene, self.map_jpg = scene, map_jpg
        self.records: list[dict] = []
        self.done = False
        self.summary: dict | None = None
        self._cond = threading.Condition()

    def push(self, rec: dict):
        with self._cond:
            self.records.append(rec)
            self._cond.notify_all()

    def finish(self, summary: dict | None = None):
        with self._cond:
            self.done, self.summary = True, summary
            self._cond.notify_all()

    def follow(self):
        i = 0
        while True:
            with self._cond:
                self._cond.wait_for(lambda: i < len(self.records) or self.done, timeout=15)
                batch = self.records[i:]
                done = self.done
            i += len(batch)
            yield batch, done
            if done and not batch:
                return


def viewer_html(three_src: str) -> str:
    return (STATIC / "viewer.html").read_text().replace("__THREE_SRC__", three_src)


def export_html(path: str | Path, scene: dict, records: list[dict], map_jpg: bytes, summary: dict | None = None):
    """Write a single self-contained HTML file that replays the flight."""
    data = {"scene": scene, "records": records, "summary": summary,
            "map": "data:image/jpeg;base64," + base64.b64encode(map_jpg).decode()}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = viewer_html(THREE_CDN).replace(
        f'<script src="{THREE_CDN}"></script>',
        f'<script>window.NAVSIM = {payload};</script>\n<script src="{THREE_CDN}"></script>', 1)
    Path(path).write_text(html)
    return path


def serve(feed: FlightFeed, port: int = 27351, host: str = "127.0.0.1"):
    page = viewer_html("three.min.js").encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = self.path.split("?")[0]
            if route in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if route == "/scene.json":
                return self._send(json.dumps(feed.scene).encode(), "application/json")
            if route == "/map.jpg":
                return self._send(feed.map_jpg, "image/jpeg")
            if route == "/three.min.js":
                return self._send((STATIC / "three.min.js").read_bytes(), "application/javascript")
            if route == "/run.json":
                return self._send(json.dumps({"records": feed.records, "summary": feed.summary}).encode(),
                                  "application/json")
            if route == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    for batch, done in feed.follow():
                        for rec in batch:
                            self.wfile.write(f"data: {json.dumps(rec)}\n\n".encode())
                        if done and not batch:
                            self.wfile.write(b"event: done\ndata: {}\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_error(404)

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# Live MAVLink source ----------------------------------------------------------
def mavlink_feed(ap, map_, feed: FlightFeed, origin, rate_hz: float = 5.0, stop: threading.Event | None = None):
    """Follow a vehicle: SIMSTATE (SITL truth, if present) vs GLOBAL_POSITION_INT (EKF estimate)."""
    import math
    import time

    stop = stop or threading.Event()
    t0 = time.monotonic()
    while not stop.is_set():
        time.sleep(1 / rate_hz)
        pos = ap.pos
        if pos is None or (pos.lat == 0 and pos.lon == 0):  # EKF has no origin yet
            continue
        att = ap.attitude_at(time.monotonic()) or (ap._att[-1] if ap._att else None)
        ex, ey = map_.ll_to_grid(pos.lat, pos.lon)
        if ap.simstate:
            s = ap.simstate[1]
            tx, ty = map_.ll_to_grid(s.lat / 1e7, s.lng / 1e7)
            roll, pitch, yaw = s.roll, s.pitch, s.yaw
        else:  # real vehicle: no truth available, show the estimate as the path
            tx, ty = ex, ey
            roll, pitch, yaw = (att.roll, att.pitch, att.yaw) if att else (0.0, 0.0, 0.0)
        g2 = ap.gps_raw
        if not ap.simstate and g2 is not None and g2.fix_type >= 3:
            tx, ty = map_.ll_to_grid(g2.lat / 1e7, g2.lon / 1e7)  # GNSS on GPS2 as reference
        feed.push({
            "t": round(time.monotonic() - t0, 2),
            "p": [round(tx - origin[0], 2), round(ty - origin[1], 2), round(max(pos.rel_alt, 0.0), 2)],
            "att": [round(roll, 4), round(pitch, 4), round(yaw % (2 * math.pi), 4)],
            "ekf": [round(ex - origin[0], 2), round(ey - origin[1], 2)],
            "sig": None, "vis": True, "fix": None, "leg": 0,
        })
