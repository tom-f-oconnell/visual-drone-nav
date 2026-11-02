"""Real clouds on the imagery a simulated camera sees.

Maps are made clear of clouds, but a real flight happens under whatever the sky
is doing. A Sentinel-2 pass, kept as it was with its scene classification, says
where the clouds and their shadows were on that day, so they can be laid over
any imagery of the same ground, such as sharper NAIP. They do not move during a
flight, and the ground below a cloud is simply hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# Sentinel-2 scene classes: how much each hides the ground, and how much a cloud's shadow dims it.
CLOUD_OPACITY = {8: 0.85, 9: 1.0, 10: 0.35}  # medium probability, high probability, thin cirrus
SHADOW_CLASS = 3
SHADOW_GAIN = 0.55


def overlay(world: Path, pass_tif: Path, clouds: bool = True, shadows: bool = True) -> dict:
    """Lay one pass's clouds and shadows over a GeoTIFF of the same corridor, in place.

    The opacity goes in a fourth band, so that the simulator does not add its
    fine ground texture to a cloud.

    Args:
        world: RGB GeoTIFF to change.
        pass_tif: RGB and classification (fourth band), from build_map(..., scl=True).
        clouds: hide the ground below clouds.
        shadows: dim the ground below shadows. Not for the pass's own imagery, which has them already.

    Returns:
        cloud_pct and shadow_pct of the corridor.
    """
    with rasterio.open(pass_tif) as src:
        pass_rgb, scl = src.read([1, 2, 3]), src.read(4)
        pass_transform, pass_crs = src.transform, src.crs
    with rasterio.open(world) as src:
        rgb, profile = src.read([1, 2, 3]), src.profile
        transform, crs = src.transform, src.crs
    height, width = rgb.shape[1:]

    def onto_world(data: np.ndarray, resampling=Resampling.bilinear) -> np.ndarray:
        out = np.zeros((*data.shape[:-2], height, width), np.float32)
        reproject(data.astype(np.float32), out, src_transform=pass_transform, src_crs=pass_crs, dst_transform=transform,
                  dst_crs=crs, resampling=resampling)
        return out

    opacity = np.zeros(scl.shape, np.float32)
    for cls, value in CLOUD_OPACITY.items():
        opacity[scl == cls] = value
    alpha, shade = onto_world(opacity), onto_world((scl == SHADOW_CLASS).astype(np.float32))
    valid = rgb.any(axis=0)
    stats = {"cloud_pct": round(100 * float((alpha[valid] > 0.5).mean()), 1) if valid.any() else 0.0,
             "shadow_pct": round(100 * float((shade[valid] > 0.5).mean()), 1) if valid.any() else 0.0}

    out = rgb.astype(np.float32)
    if shadows:
        out *= 1 - (1 - SHADOW_GAIN) * shade
    if clouds:
        out = out * (1 - alpha) + onto_world(pass_rgb) * alpha
    else:
        alpha[:] = 0
    # Nothing that had imagery may become nodata (all zero) by being dark.
    out = np.clip(out + 0.5, 1, 255).astype(np.uint8)
    out[:, ~valid] = 0
    profile.update(count=4)
    profile.pop("photometric", None)
    with rasterio.open(world, "w", **profile) as dst:
        dst.write(out, [1, 2, 3])
        dst.write(np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8), 4)
    return stats
