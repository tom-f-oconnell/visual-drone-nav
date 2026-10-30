"""Register an orthorectified camera frame against the NAIP map.

Classical features (ORB, AKAZE or SIFT) with a ratio test and a RANSAC
similarity fit. On a Jetson the ORB path runs on CPU at a few Hz for
~1000 px frames, which is enough for GPS-rate updates. The FeatureBackend
seam is where a learned matcher (SuperPoint + LightGlue under TensorRT)
would plug in for better robustness to season and lighting changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geo import MapRaster
from .ortho import Ortho


class FeatureBackend:
    def __init__(self, kind: str = "orb", nfeatures: int = 3000):
        self.kind = kind
        if kind == "orb":
            self.det = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=6, fastThreshold=10)
            self.norm, self.ratio = cv2.NORM_HAMMING, 0.85
        elif kind == "akaze":
            if not hasattr(cv2, "AKAZE_create"):
                raise ValueError("AKAZE is not in OpenCV 5's main modules; use OpenCV 4.x (as in JetPack)")
            self.det = cv2.AKAZE_create(threshold=0.0005)
            self.norm, self.ratio = cv2.NORM_HAMMING, 0.85
        elif kind == "sift":
            self.det = cv2.SIFT_create(nfeatures=nfeatures)
            self.norm, self.ratio = cv2.NORM_L2, 0.8
        else:
            raise ValueError(f"unknown feature backend {kind}")
        self.bf = cv2.BFMatcher(self.norm)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, img: np.ndarray, mask: np.ndarray | None, normalise: bool = True):
        if normalise:
            img = self.clahe.apply(img)
        m = None if mask is None else mask.astype(np.uint8) * 255
        kps, desc = self.det.detectAndCompute(img, m)
        pts = np.float32([k.pt for k in kps]) if kps else np.empty((0, 2), np.float32)
        return pts, desc

    def match(self, d1, d2) -> list[tuple[int, int]]:
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
            return []
        pairs = self.bf.knnMatch(d1, d2, k=2)
        return [(m.queryIdx, m.trainIdx) for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < self.ratio * n.distance]


@dataclass
class Fix:
    col: float              # map pixel of the point below the camera
    row: float
    lat: float
    lon: float
    inliers: int
    matches: int
    rmse_px: float
    scale: float            # should be ~1 if altitude and intrinsics are right
    heading_err_deg: float  # yaw correction implied by the match
    sigma_m: float          # 1-sigma horizontal accuracy estimate


class MapMatcher:
    def __init__(self, map_: MapRaster, backend: FeatureBackend | None = None,
                 min_inliers: int = 20, ransac_px: float = 4.0):
        self.map = map_
        self.backend = backend or FeatureBackend()
        self.min_inliers = min_inliers
        self.ransac_px = ransac_px

    def locate(self, ortho: Ortho, prior_col: float, prior_row: float, search_radius_px: int) -> Fix | None:
        h, w = ortho.image.shape
        radius = int(search_radius_px + max(h, w) / 2)
        crop, cmask, (x0, y0) = self.map.crop(prior_col, prior_row, radius)
        if crop is None or cmask.mean() < 0.2:
            return None

        p1, d1 = self.backend.detect(ortho.image, ortho.mask)
        p2, d2 = self.backend.detect(crop, cmask, normalise=False)  # map is CLAHE'd at load
        pairs = self.backend.match(d1, d2)
        if len(pairs) < self.min_inliers:
            return None
        src = p1[[a for a, _ in pairs]]
        dst = p2[[b for _, b in pairs]]
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=self.ransac_px,
                                             maxIters=4000, confidence=0.999)
        if M is None:
            return None
        inl = inl.ravel().astype(bool)
        n_in = int(inl.sum())
        scale = math.hypot(M[0, 0], M[1, 0])
        rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        if n_in < self.min_inliers or not (0.8 < scale < 1.25) or abs(rot) > 30:
            return None

        proj = (M @ np.c_[src[inl], np.ones(n_in)].T).T
        rmse = float(np.sqrt(((proj - dst[inl]) ** 2).sum(axis=1).mean()))
        nx, ny = ortho.nadir
        col, row = M @ np.array([nx, ny, 1.0])
        col, row = col + x0, row + y0
        lat, lon = self.map.px_to_ll(col, row)
        # Rough accuracy model: residual noise averaged over inliers plus a floor
        # for map registration error (NAIP spec is about 6 m CE95, often better).
        sigma = math.hypot(rmse * self.map.gsd / math.sqrt(max(n_in, 1)) * 3, 2.0)
        return Fix(col, row, lat, lon, n_in, len(pairs), rmse, scale, rot, sigma)
