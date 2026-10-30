"""Register an orthorectified camera frame against the NAIP map.

Two matchers with the same interface, plus a chain of them:

* ``CorrMatcher``: FFT correlation of gradient orientations over the whole
  search area, refined at full map resolution. Fast, and the most robust to a
  different NAIP year (see design.md for the benchmark).
* ``MapMatcher``: classical features (ORB, AKAZE or SIFT) with a ratio test and
  a RANSAC similarity fit. Slower, but measures scale and larger heading errors.
  The FeatureBackend seam is where a learned matcher (ALIKED + LightGlue under
  TensorRT) would plug in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    inliers: int            # 0 from correlation
    matches: int
    rmse_px: float
    scale: float            # should be ~1 if altitude and intrinsics are right
    heading_err_deg: float  # yaw correction implied by the match
    sigma_m: float          # 1-sigma horizontal accuracy estimate
    score: float = 0.0      # the matcher's own confidence, in its own units
    strong: bool = False    # confident enough to believe over the filter (re-lock)
    diag: dict = field(default_factory=dict)


def similarity_fix(map_: MapRaster, ortho: Ortho, src: np.ndarray, dst: np.ndarray, x0: int, y0: int,
                   min_inliers: int, ransac_px: float = 4.0) -> Fix | None:
    """RANSAC similarity from ortho points to points in a map crop at (x0, y0), scored by inliers."""
    if len(src) < min_inliers:
        return None
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_px,
                                         maxIters=4000, confidence=0.999)
    if M is None:
        return None
    inl = inl.ravel().astype(bool)
    n_in = int(inl.sum())
    scale = math.hypot(M[0, 0], M[1, 0])
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    if n_in < min_inliers or not (0.8 < scale < 1.25) or abs(rot) > 30:
        return None

    proj = (M @ np.c_[src[inl], np.ones(n_in)].T).T
    rmse = float(np.sqrt(((proj - dst[inl]) ** 2).sum(axis=1).mean()))
    col, row = M @ np.array([*ortho.nadir, 1.0])
    col, row = col + x0, row + y0
    lat, lon = map_.px_to_ll(col, row)
    # Rough accuracy model: residual noise averaged over inliers plus a floor
    # for map registration error (NAIP spec is about 6 m CE95, often better).
    sigma = math.hypot(rmse * map_.gsd / math.sqrt(max(n_in, 1)) * 3, 2.0)
    return Fix(col, row, lat, lon, n_in, len(src), rmse, scale, rot, sigma, score=float(n_in))


class MapMatcher:
    """Features, ratio test, RANSAC similarity.

    Args:
        map_: reference map.
        backend: feature type.
        min_inliers: fixes with fewer RANSAC inliers are not returned.
        strong_inliers: inliers at which a fix counts as strong.
        ransac_px: RANSAC reprojection threshold.
    """

    def __init__(self, map_: MapRaster, backend: FeatureBackend | None = None,
                 min_inliers: int = 20, strong_inliers: int = 40, ransac_px: float = 4.0):
        self.map = map_
        self.backend = backend or FeatureBackend()
        self.min_inliers, self.strong_inliers = min_inliers, strong_inliers
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
        fix = similarity_fix(self.map, ortho, p1[[a for a, _ in pairs]], p2[[b for _, b in pairs]], x0, y0,
                             self.min_inliers, self.ransac_px)
        if fix is not None:
            fix.strong = fix.inliers >= self.strong_inliers
        return fix


def orientation_field(img: np.ndarray, sigma: float, noise: float) -> np.ndarray:
    """Doubled-angle gradient orientation as complex numbers.

    Doubling the angle makes it invariant to contrast reversal (a road darker
    or lighter than its verge between years). Magnitudes are close to 1 where
    the gradient is well above ``noise`` (in grey levels per pixel) and fall to
    0 in flat areas, so uniform fields and water add no random votes.
    """
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8
    m2 = gx * gx + gy * gy
    z = ((gx * gx - gy * gy) + 2j * gx * gy) / (m2 + noise * noise)
    return z.astype(np.complex64)


def parabolic_peak(a: float, b: float, c: float) -> float:
    """Offset of the top of the parabola through three evenly spaced values with b the highest, in -0.5..0.5."""
    d = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / d, -0.5, 0.5)) if d < 0 else 0.0


def parabolic_2d(R: np.ndarray, r: int, c: int) -> tuple[float, float]:
    """Sub-pixel peak offset (dc, dr) from a 3-point parabola in each axis."""
    dc = parabolic_peak(R[r, c - 1], R[r, c], R[r, c + 1]) if 0 < c < R.shape[1] - 1 else 0.0
    dr = parabolic_peak(R[r - 1, c], R[r, c], R[r + 1, c]) if 0 < r < R.shape[0] - 1 else 0.0
    return dc, dr


def padded_slice(a: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """a[y0:y0+h, x0:x0+w], zero where that falls outside a."""
    out = np.zeros((h, w), a.dtype)
    ya, xa, yb, xb = max(y0, 0), max(x0, 0), min(y0 + h, a.shape[0]), min(x0 + w, a.shape[1])
    if yb > ya and xb > xa:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
    return out


class CorrMatcher:
    """Gradient-orientation correlation of the ortho against the map.

    The score of a fix is how far its peak stands out from the rest of the
    search area, see ``locate``. More candidate positions give chance peaks more
    opportunity, so the score required rises with the search radius. Small
    footprints are not matched: too little structure to tell a chance alignment
    from a true one, and low down odometry is accurate anyway.

    Args:
        map_: reference map.
        down: coarse stage downsampling (4 gives 2.4 m/px on a 0.6 m map).
        sigma: gradient smoothing at the coarse scale, in coarse pixels.
        noise: gradient noise floor in grey levels per pixel (see orientation_field).
        angles: evenly spaced template rotations to try, degrees, for heading error.
        excl_m: radius around the peak excluded from the sidelobe statistics.
        fine: refine at full map resolution around the coarse peak.
        parts: score by how strongly each quarter of the template supports the
            peak (mean of their PSRs) rather than by the whole template's PSR.
        sigma_m: 1-sigma accuracy reported with each fix.
        min_score: fixes scoring below this are not returned, at or under
            base_radius_m of search.
        base_radius_m: see min_score.
        score_per_radius: added to min_score per e-fold of search radius above base_radius_m.
        min_area_m2: smaller footprints are not matched (8000 is 60 m up with a 90 degree lens).
        strong_score: score at which a fix counts as strong.
    """

    def __init__(self, map_: MapRaster, down: int = 4, sigma: float = 1.0, noise: float = 2.0,
                 angles: tuple[float, ...] = tuple(float(a) for a in range(-16, 17, 4)), excl_m: float = 12.0,
                 fine: bool = True, parts: bool = True, sigma_m: float = 3.0, min_score: float = 2.6,
                 base_radius_m: float = 40.0, score_per_radius: float = 0.7, min_area_m2: float = 8000.0,
                 strong_score: float = 4.0):
        self.map, self.down, self.sigma, self.noise = map_, down, sigma, noise
        self.angles, self.excl_m, self.fine = angles, excl_m, fine
        self.parts, self.sigma_m = parts, sigma_m
        self.min_score, self.base_radius_m, self.score_per_radius = min_score, base_radius_m, score_per_radius
        self.min_area_m2, self.strong_score = min_area_m2, strong_score
        g = cv2.resize(map_.gray, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
        v = cv2.resize(map_.valid.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
        self.valid_c = cv2.erode(v, np.ones((3, 3), np.uint8)) > 0
        self.field_c = orientation_field(g, sigma, noise) * self.valid_c
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def required_score(self, search_radius_px: int) -> float:
        radius_m = search_radius_px * self.map.gsd
        return self.min_score + self.score_per_radius * math.log(max(radius_m / self.base_radius_m, 1.0))

    def _template(self, img: np.ndarray, mask: np.ndarray, down: int, sigma: float, angle: float,
                  centre: tuple[float, float]):
        """Orientation field of the ortho rotated by angle (degrees) about centre, then downsampled."""
        if angle:
            h, w = img.shape
            R = cv2.getRotationMatrix2D(centre, angle, 1.0)
            img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask.astype(np.uint8), R, (w, h), flags=cv2.INTER_NEAREST) > 0
        if down > 1:
            img = cv2.resize(img, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA) > 0
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        t = orientation_field(img, sigma, self.noise) * mask
        return t, mask

    def locate(self, ortho: Ortho, prior_col: float, prior_row: float, search_radius_px: int) -> Fix | None:
        """Coarse FFT correlation over the search area, then fine refinement.

        The peak-to-sidelobe ratio (PSR) of a correlation surface is the
        peak's height above the mean of the other shifts in the search disc,
        in standard deviations, leaving out shifts within excl_m of the peak.
        The score is the mean over the template's four quarters of each
        quarter's PSR at the whole template's peak: a true match is supported
        across the footprint, a chance alignment mostly by one part of it.
        With parts=False the score is the whole template's PSR.
        """
        d = self.down
        img = self.clahe.apply(ortho.image)
        nx, ny = ortho.nadir
        s = int(math.ceil(search_radius_px / d))
        temps = [(ang, *self._template(img, ortho.mask, d, self.sigma, ang, (nx, ny))) for ang in self.angles]
        th, tw = temps[0][1].shape
        # Template top-left, in coarse map pixels, when the camera is at the prior.
        x0 = int(round((prior_col - nx) / d)) - s
        y0 = int(round((prior_row - ny) / d)) - s
        win_h, win_w = th + 2 * s, tw + 2 * s
        if padded_slice(self.valid_c, y0, x0, win_h, win_w).mean() < 0.3:
            return None
        win = padded_slice(self.field_c, y0, x0, win_h, win_w)
        shape = (cv2.getOptimalDFTSize(win_h), cv2.getOptimalDFTSize(win_w))
        Fw = np.fft.fft2(win, s=shape)

        def xcorr(t):
            """Re(sum conj(t) . win) / sum |t| for every shift of t in the window."""
            R = np.real(np.fft.ifft2(Fw * np.conj(np.fft.fft2(t, s=shape))))[:2 * s + 1, :2 * s + 1]
            return (R / max(float(np.abs(t).sum()), 1.0)).astype(np.float32)

        yy, xx = np.mgrid[:2 * s + 1, :2 * s + 1]
        d2 = (yy - s) ** 2 + (xx - s) ** 2
        in_disc = d2 <= s * s + 1
        best, peaks = None, []
        for i, (ang, t, tmask) in enumerate(temps):
            if tmask.mean() < 0.2:
                peaks.append(-np.inf)
                continue
            R = xcorr(t)
            r, c = np.unravel_index(int(np.argmax(np.where(in_disc, R, -np.inf))), R.shape)
            peaks.append(float(R[r, c]))
            if best is None or R[r, c] > best[0]:
                best = (float(R[r, c]), R, r, c, i, t, tmask)
        if best is None:
            return None
        peak, R, r, c, i, t, tmask = best
        if float(tmask.sum()) * (d * self.map.gsd) ** 2 < self.min_area_m2:
            return None
        # On the rim the surface is only rising towards something further out, not at a peak.
        if d2[r, c] > (s - 1) ** 2:
            return None
        ang = self.angles[i]
        if 0 < i < len(peaks) - 1 and np.isfinite(peaks[i - 1]) and np.isfinite(peaks[i + 1]):
            ang_fit = ang + parabolic_peak(*peaks[i - 1:i + 2]) * (self.angles[i + 1] - ang)
        else:
            ang_fit = ang

        # Score: PSR over the search disc, and the strongest rival peak.
        ex = max(1, int(round(self.excl_m / (self.map.gsd * d))))
        far = in_disc & ((yy - r) ** 2 + (xx - c) ** 2 > ex * ex)
        if far.sum() < 10:
            return None

        def psr_at(R):
            side = R[far]
            mu, sd = float(side.mean()), float(side.std()) + 1e-6
            return (float(R[r, c]) - mu) / sd, (float(side.max()) - mu) / sd

        psr, rival = psr_at(R)
        diag = {"peak": peak, "psr": psr, "rival": rival, "angle": ang}
        if self.parts:
            # Support for the peak from each quarter of the template on its own.
            my, mx = np.argwhere(tmask).mean(axis=0)
            ty, tx = np.mgrid[:th, :tw]
            sup = []
            for qy in (ty < my, ty >= my):
                for qx in (tx < mx, tx >= mx):
                    sup.append(round(psr_at(xcorr(t * (qy & qx)))[0], 2))
            diag["parts"] = sorted(sup)
            score = sum(sup) / 4
        else:
            score = psr
        if score < self.required_score(search_radius_px):
            return None
        dc, dr = parabolic_2d(R, r, c)
        col = (x0 + c + dc) * d + nx
        row = (y0 + r + dr) * d + ny
        if self.fine:
            col, row = self._refine(img, ortho, ang, col, row, diag)
        lat, lon = self.map.px_to_ll(col, row)
        # Negated to match MapMatcher, which reports the rotation taking the ortho onto the map.
        return Fix(col, row, lat, lon, 0, 0, 0.0, 1.0, -ang_fit, self.sigma_m, score=score,
                   strong=score >= self.strong_score, diag=diag)

    def _refine(self, img: np.ndarray, ortho: Ortho, ang: float, col: float, row: float, diag: dict):
        """Full-resolution correlation within +-(down + 2) map pixels of the coarse estimate."""
        nx, ny = ortho.nadir
        t, _ = self._template(img, ortho.mask, 1, self.sigma * self.down / 2, ang, (nx, ny))
        th, tw = t.shape
        m = self.down + 2
        x0, y0 = int(round(col - nx)) - m, int(round(row - ny)) - m
        gray = padded_slice(self.map.gray, y0, x0, th + 2 * m, tw + 2 * m)
        valid = padded_slice(self.map.valid, y0, x0, th + 2 * m, tw + 2 * m)
        f = orientation_field(gray, self.sigma * self.down / 2, self.noise) * valid
        # Re(sum conj(t) * f) as a two-channel correlation.
        a = np.dstack([f.real, f.imag]).astype(np.float32)
        b = np.dstack([t.real, t.imag]).astype(np.float32)
        R = cv2.matchTemplate(a, b, cv2.TM_CCORR) / max(float(np.abs(t).sum()), 1.0)
        r, c = np.unravel_index(int(np.argmax(R)), R.shape)
        dc, dr = parabolic_2d(R, r, c)
        diag["fine_peak"] = float(R[r, c])
        diag["fine_shift"] = float(math.hypot(c - m, r - m))
        return x0 + c + dc + nx, y0 + r + dr + ny


class ChainMatcher:
    """Try matchers in order: the first strong fix wins, else the first fix any of them found."""

    def __init__(self, *matchers):
        self.matchers = matchers

    def locate(self, ortho: Ortho, prior_col: float, prior_row: float, search_radius_px: int) -> Fix | None:
        first = None
        for m in self.matchers:
            fix = m.locate(ortho, prior_col, prior_row, search_radius_px)
            if fix is not None and fix.strong:
                return fix
            first = first or fix
        return first
