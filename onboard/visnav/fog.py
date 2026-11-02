"""Fog in simulated camera frames.

Fog is a layer of air from the ground up to a top height, which each ray from the
camera has to pass through. Along a ray the light from the ground fades, and light
scattered by the fog takes its place (Koschmieder's law):

    seen = ground * T + fog colour * (1 - T),    T = exp(-beta * density * path)

* ``beta`` comes from the meteorological visibility: the distance at which contrast
  falls to 2%, so beta = -ln(0.02) / visibility = 3.912 / visibility.
* ``path`` is the length of the ray inside the layer: min(height, top) / cos(angle
  from straight down). A camera in the fog looks through all of its height, and one
  above it through the layer's depth, so fog works at any altitude, and the edges
  of a wide lens's frame are foggier than the middle.
* ``density`` varies over the ground by fractal noise, in patches that drift with
  the wind, so that a frame can have a thick bank on one side and a hole on the other.

It is applied to each frame, before the camera's own appearance (see appearance.py),
as the sensor sees the fog. The noise is sampled where each ray meets the ground, on a
coarse grid of rays, since fog varies smoothly. Density does not vary with height.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

from .ortho import CameraModel, R_ned_body

VISIBILITY_CONTRAST = 3.912  # -ln(0.02)
NOISE_SIZE = 256
NOISE_PATCHES = 8  # patches across the noise tile, which repeats

LIMITS = {"visibility": (10.0, 100000.0), "top": (1.0, 20000.0), "patchiness": (0.0, 1.0), "patch": (5.0, 20000.0), "brightness": (0.0, 1.0)}


@dataclass(frozen=True)
class Fog:
    """A layer of fog, for a whole flight.

    Attributes:
        visibility: meteorological visibility in the fog, m. Under 1000 is fog by
            definition, 1000 to 5000 mist, more than that haze.
        top: height of the top of the layer above the ground, m.
        patchiness: 0 for even fog, 1 for thick banks and clear holes.
        patch: typical size of the patches, m.
        brightness: of the fog's light, from 0 to 1 (full scale).
    """

    visibility: float = 400.0
    top: float = 200.0
    patchiness: float = 0.4
    patch: float = 300.0
    brightness: float = 0.8

    def __post_init__(self):
        for name, (low, high) in LIMITS.items():
            if not low <= getattr(self, name) <= high:
                raise ValueError(f"{name} must be from {low} to {high}, not {getattr(self, name)}")

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "Fog":
        """Raises ValueError on an unknown name or a value out of range. Names left out keep their defaults."""
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown {', '.join(sorted(unknown))}; expected {', '.join(LIMITS)}")
        return cls(**{k: float(v) for k, v in values.items()})

    @classmethod
    def parse(cls, spec: str) -> "Fog | None":
        """Read a command line value: "" for no fog, "on" for the defaults, or "name=value,name=value"."""
        if not spec:
            return None
        if spec == "on":
            return cls()
        values = {}
        for item in spec.split(","):
            name, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f'expected "name=value", not {item!r}')
            values[name.strip()] = float(value)
        return cls.from_dict(values)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@functools.cache
def _patch_noise(seed: int) -> np.ndarray:
    """Periodic fractal noise of unit standard deviation, with blobs NOISE_SIZE / NOISE_PATCHES pixels across."""
    f = np.random.default_rng(seed).normal(size=(NOISE_SIZE, NOISE_SIZE))
    fy, fx = np.meshgrid(np.fft.fftfreq(NOISE_SIZE, 1 / NOISE_SIZE), np.fft.fftfreq(NOISE_SIZE, 1 / NOISE_SIZE), indexing="ij")
    r = np.hypot(fx, fy)
    r[0, 0] = 1e-9
    # Flat up to the patch size, then falling away, for wisps on the edges of the banks. No constant part.
    gain = np.minimum((r / NOISE_PATCHES) ** -1.5, 1.0)
    gain[0, 0] = 0.0
    n = np.real(np.fft.ifft2(np.fft.fft2(f) * gain))
    return ((n - n.mean()) / n.std()).astype(np.float32)


class FogLayer:
    """A `Fog` as a camera sees it.

    Args:
        fog: the fog.
        cam: the camera.
        seed: of the patches.
        wind: m/s east and north the patches drift at.
        step: pixels between the rays traced.
    """

    def __init__(self, fog: Fog, cam: CameraModel, seed: int = 0, wind: tuple[float, float] = (0.0, 0.0), step: int = 16):
        self.fog, self.cam, self.wind = fog, cam, wind
        self.beta = VISIBILITY_CONTRAST / fog.visibility
        self.noise = _patch_noise(seed)
        us = np.append(np.arange(0, cam.width, step), cam.width)
        vs = np.append(np.arange(0, cam.height, step), cam.height)
        u, v = np.meshgrid(us, vs)
        self.grid_shape = u.shape
        self.rays = np.linalg.inv(cam.K) @ np.vstack([u.ravel(), v.ravel(), np.ones(u.size)])

    def transmission(self, t: float, east: float, north: float, h: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Share of the ground's light reaching each pixel, from 0 to 1 (float32, camera height x width)."""
        fog = self.fog
        rays = R_ned_body(roll, pitch, yaw) @ self.cam.R_body_cam @ self.rays
        rays = rays / np.linalg.norm(rays, axis=0)
        down = np.maximum(rays[2], 0.1)  # the horizon does not come into a downward camera's view
        h = max(h, 0.0)
        density = 1.0
        if fog.patchiness > 0:
            scale = NOISE_SIZE / (fog.patch * NOISE_PATCHES)
            col = (east + rays[1] / down * h - self.wind[0] * t) * scale
            row = (north + rays[0] / down * h - self.wind[1] * t) * scale
            n = cv2.remap(self.noise, (col % NOISE_SIZE).astype(np.float32).reshape(self.grid_shape),
                          (row % NOISE_SIZE).astype(np.float32).reshape(self.grid_shape), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP).ravel()
            density = 1 - fog.patchiness + fog.patchiness * np.maximum(1 + 1.2 * n, 0.0)
        path = min(h, fog.top) / down
        grid = np.exp(-self.beta * density * path).reshape(self.grid_shape).astype(np.float32)
        return cv2.resize(grid, (self.cam.width, self.cam.height), interpolation=cv2.INTER_LINEAR)

    def __call__(self, img: np.ndarray, t: float, east: float, north: float, h: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """The BGR frame `img` as seen through the fog, for a camera at (east, north, h) m at time t."""
        T = self.transmission(t, east, north, h, roll, pitch, yaw)[..., None]
        seen = img.astype(np.float32) * T + 255 * self.fog.brightness * (1 - T)
        return np.clip(seen + 0.5, 0, 255).astype(np.uint8)
