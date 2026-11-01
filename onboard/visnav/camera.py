"""Frame capture on the Jetson, keeping only the newest frame with its capture time."""

from __future__ import annotations

import threading
import time

import cv2


def csi_pipeline(width=1280, height=800, fps=30, sensor_id=0) -> str:
    """GStreamer pipeline for a CSI camera via nvarguscamerasrc (Jetson ISP)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1"
    )


class FrameGrabber:
    """Spec: "csi", "csi:<sensor_id>", a V4L2 index like "0", a GStreamer pipeline, or a video file."""

    def __init__(self, spec: str, width: int = 1280, height: int = 800, fps: int = 30):
        if spec.startswith("csi"):
            sid = int(spec.split(":")[1]) if ":" in spec else 0
            self.cap = cv2.VideoCapture(csi_pipeline(width, height, fps, sid), cv2.CAP_GSTREAMER)
        elif spec.isdigit():
            self.cap = cv2.VideoCapture(int(spec), cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif "!" in spec:
            self.cap = cv2.VideoCapture(spec, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(spec)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {spec!r}")
        self._frame, self._t = None, 0.0
        self._cond = threading.Condition()
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._t = frame, t
                self._cond.notify_all()

    def latest(self, newer_than: float = 0.0, timeout: float = 1.0):
        """Block until a frame newer than `newer_than` arrives; returns (frame, t) or (None, 0)."""
        with self._cond:
            self._cond.wait_for(lambda: self._t > newer_than, timeout)
            return (self._frame, self._t) if self._t > newer_than else (None, 0.0)

    def close(self):
        self._stop = True
        self.cap.release()


class SitlCamera:
    """Renders frames from the map at SITL's true position (SIMSTATE), for software-in-the-loop runs.

    Lets the whole run path (MAVLink in, GPS_INPUT out, EKF fusion) be exercised
    in ArduPilot SITL with no camera. Frames are rendered with the attitude and
    height visnav will be told for them, not SIMSTATE's true attitude: the two
    arrive on different MAVLink streams, and their timing jitter alone (up to a
    degree while yawing) would swamp odometry. `visnav sim3d` models attitude
    errors instead.
    """

    def __init__(self, autopilot, map_, cam, rate_hz: float = 5.0, look=None):
        from .appearance import CameraLook
        from .sim import warm_up

        warm_up()
        self.ap, self.map, self.cam, self.period = autopilot, map_, cam, 1 / rate_hz
        self.look = look or CameraLook()
        self._last = 0.0

    def latest(self, newer_than: float = 0.0, timeout: float = 1.0):
        from .sim import render

        wait = self._last + self.period - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        st, h = self.ap.simstate, self.ap.height_agl()
        att = self.ap.attitude_at(st[0]) if st else None
        if att is None or h is None:
            time.sleep(0.1)
            return None, 0.0
        t, s = st
        frame = self.look(render(self.map, self.cam, s.lat / 1e7, s.lng / 1e7, max(h, 0.3), att.roll, att.pitch, att.yaw), t)
        self._last = time.monotonic()
        return frame, t

    def close(self):
        pass
