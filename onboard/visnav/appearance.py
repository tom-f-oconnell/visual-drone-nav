"""How a simulated camera's picture differs from the imagery it is rendered from.

By default it does not: frames show the imagery as it is. An `Appearance` holds
offsets that last the whole flight, standing in for lighting, season and sensor
differences from the NAIP flight, and how far they wander during it. A real
camera's colour does not jump between frames. It wanders over tens of seconds to
minutes, as clouds pass and auto exposure and white balance follow them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

# Beyond these a frame is useless, or blurring it takes too long.
LIMITS = {
    "gamma": (0.1, 5.0), "red": (0.1, 5.0), "green": (0.1, 5.0), "blue": (0.1, 5.0),
    "blur": (0.0, 20.0), "noise": (0.0, 0.5),
    "brightness_drift": (0.0, 1.0), "color_drift": (0.0, 1.0), "gamma_drift": (0.0, 1.0),
    "drift_tau_s": (1.0, 86400.0),
}

# What `Appearance.random` draws from. Judged from typical camera behaviour, not measured from flight footage.
REALISTIC = {
    "gamma": (0.75, 1.3),            # tone curve against NAIP's
    "red": (0.85, 1.15),             # exposure and white balance
    "green": (0.85, 1.15),
    "blue": (0.85, 1.15),
    "blur": (0.5, 1.5),              # lens softness and vibration, px
    "noise": (0.005, 0.02),          # daylight sensor noise
    "brightness_drift": (0.0, 0.08),  # clouds, as far as auto exposure leaves them
    "color_drift": (0.0, 0.03),      # colour of the light, as far as white balance leaves it
    "gamma_drift": (0.0, 0.05),      # haze and flat light
    "drift_tau_s": (30.0, 300.0),    # passing clouds
}


@dataclass(frozen=True)
class Appearance:
    """Differences from the imagery, held for a whole flight. The defaults change nothing.

    Attributes:
        gamma: exponent of the tone curve on values from 0 to 1; above 1 darkens midtones.
        red: gain on the red channel, for exposure and white balance.
        green: gain on the green channel.
        blue: gain on the blue channel.
        blur: sigma of the softness of the picture, in pixels.
        noise: standard deviation of sensor noise, as a fraction of full scale. Like a
            real sensor's it is new each frame, but it does not move a frame's average.
        brightness_drift: standard deviation of the slow wander of all channels together,
            as a fraction.
        color_drift: standard deviation of the slow wander of each channel separately,
            as a fraction.
        gamma_drift: standard deviation of the slow wander of gamma, as a fraction.
        drift_tau_s: correlation time of the wander, in seconds.
    """

    gamma: float = 1.0
    red: float = 1.0
    green: float = 1.0
    blue: float = 1.0
    blur: float = 0.0
    noise: float = 0.0
    brightness_drift: float = 0.0
    color_drift: float = 0.0
    gamma_drift: float = 0.0
    drift_tau_s: float = 120.0

    def __post_init__(self):
        for name, (low, high) in LIMITS.items():
            if not low <= getattr(self, name) <= high:
                raise ValueError(f"{name} must be from {low} to {high}, not {getattr(self, name)}")

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "Appearance":
        """Raises ValueError on an unknown name or a value out of range."""
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown {', '.join(sorted(unknown))}; expected {', '.join(LIMITS)}")
        return cls(**{k: float(v) for k, v in values.items()})

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Appearance":
        """One realistic set of offsets and drifts."""
        return cls(**{k: float(f"{rng.uniform(*r):.3g}") for k, r in REALISTIC.items()})

    @classmethod
    def parse(cls, spec: str, seed=None) -> "Appearance":
        """Read a command line value.

        Args:
            spec: "" for no change, "random" for `random`, or "name=value,name=value".
            seed: for "random".
        """
        if not spec:
            return cls()
        if spec == "random":
            return cls.random(np.random.default_rng(seed))
        values = {}
        for item in spec.split(","):
            name, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f'expected "name=value", not {item!r}')
            values[name.strip()] = float(value)
        return cls.from_dict(values)

    def changes(self) -> dict[str, float]:
        """The values that differ from the defaults."""
        default = Appearance()
        return {k: v for k, v in asdict(self).items() if v != getattr(default, k)}

    @property
    def unchanged(self) -> bool:
        return not set(self.changes()) - {"drift_tau_s"}


class CameraLook:
    """An `Appearance` over one flight: applies its offsets and drift to each frame.

    The wander starts at zero, so the first frame has exactly the offsets asked for.

    Args:
        appearance: what to apply; the default leaves frames untouched.
        seed: of the drift and the noise, separate from everything else that is
            random in a flight, so a different look leaves the wind and IMU the same.
    """

    def __init__(self, appearance: Appearance | None = None, seed=None):
        self.appearance = appearance or Appearance()
        self.rng = np.random.default_rng(seed)
        self._t = None
        self._wander = np.zeros(5)  # in standard deviations: brightness, gamma, blue, green, red

    def _advance(self, t: float):
        if self._t is not None and t > self._t:
            k = math.exp(-(t - self._t) / self.appearance.drift_tau_s)
            self._wander = k * self._wander + math.sqrt(1 - k * k) * self.rng.normal(size=5)
        self._t = t if self._t is None else max(t, self._t)

    def __call__(self, img: np.ndarray, t: float = 0.0) -> np.ndarray:
        """The BGR frame `img` as this camera sees it at time t (seconds, never going back)."""
        a = self.appearance
        if a.unchanged:
            return img
        self._advance(t)
        brightness, gamma, *color = self._wander
        gains = np.array([a.blue, a.green, a.red]) * np.exp(a.brightness_drift * brightness
                                                            + a.color_drift * np.array(color))
        f = np.power(img.astype(np.float32) / 255, a.gamma * math.exp(a.gamma_drift * gamma)) * gains.astype(np.float32)
        if a.blur:
            f = cv2.GaussianBlur(f, (0, 0), a.blur)
        if a.noise:
            f += self.rng.normal(0, a.noise, f.shape).astype(np.float32)
        return np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8)
