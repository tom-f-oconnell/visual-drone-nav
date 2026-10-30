"""Map matchers compared by the cross-year benchmark.

Each has the interface of ``visnav.matcher.MapMatcher``:
``locate(ortho, prior_col, prior_row, search_radius_px) -> Fix | None``, and
returns a ``ScoredFix``, which adds the matcher's own confidence (``score``) and
diagnostics. A fix should only be trusted when ``score`` reaches the threshold
the benchmark found for that matcher; ``min_score`` only bounds the work done
and is set low here so the benchmark can sweep the threshold.

* ``OrbMatcher``: the navigator's current matcher (ORB, ratio test, RANSAC
  similarity). Score: RANSAC inliers.
* ``CorrMatcher``: area correlation of gradient orientations, coarse (FFT over
  the whole search area, a few template rotations for heading error) then fine
  (full map resolution around the coarse peak). Score: how far the peak stands
  out from the rest of the search area, as seen by each quarter of the frame on
  its own, see ``CorrMatcher.locate``.
* ``LightGlueMatcher``: DISK or ALIKED features matched with LightGlue
  (kornia, PyTorch), then the same RANSAC similarity fit as ORB. Map features
  are extracted once for the whole map, as they would be before a flight.
  Score: RANSAC inliers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from visnav.geo import MapRaster
from visnav.matcher import FeatureBackend, Fix, MapMatcher
from visnav.ortho import Ortho


@dataclass
class ScoredFix(Fix):
    score: float = 0.0
    diag: dict = field(default_factory=dict)


def similarity_fix(map_: MapRaster, ortho: Ortho, src: np.ndarray, dst: np.ndarray, x0: int, y0: int,
                   min_inliers: int, ransac_px: float = 4.0, diag: dict | None = None) -> ScoredFix | None:
    """RANSAC similarity from ortho to map-crop points, as in MapMatcher, scored by inliers."""
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
    sigma = math.hypot(rmse * map_.gsd / math.sqrt(max(n_in, 1)) * 3, 2.0)
    return ScoredFix(col, row, lat, lon, n_in, len(src), rmse, scale, rot, sigma, float(n_in), diag or {})


class OrbMatcher:
    """The navigator's MapMatcher with ORB, reporting inliers as the score."""

    def __init__(self, map_: MapRaster, min_score: int = 8, nfeatures: int = 3000):
        self.map = map_
        self.inner = MapMatcher(map_, FeatureBackend("orb", nfeatures), min_inliers=min_score)

    def locate(self, ortho: Ortho, prior_col: float, prior_row: float, search_radius_px: int) -> ScoredFix | None:
        fix = self.inner.locate(ortho, prior_col, prior_row, search_radius_px)
        if fix is None:
            return None
        return ScoredFix(**vars(fix), score=float(fix.inliers))


# Area correlation -------------------------------------------------------------
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


def _parabolic(R: np.ndarray, r: int, c: int) -> tuple[float, float]:
    """Sub-pixel peak offset (dc, dr) from a 3-point parabola in each axis."""
    def one(a, b, cc):
        d = a - 2 * b + cc
        return float(np.clip(0.5 * (a - cc) / d, -0.5, 0.5)) if d < 0 else 0.0
    dc = one(R[r, c - 1], R[r, c], R[r, c + 1]) if 0 < c < R.shape[1] - 1 else 0.0
    dr = one(R[r - 1, c], R[r, c], R[r + 1, c]) if 0 < r < R.shape[0] - 1 else 0.0
    return dc, dr


class CorrMatcher:
    """Gradient-orientation correlation of the ortho against the map.

    Args:
        map_: reference map.
        down: coarse stage downsampling (4 gives 2.4 m/px on a 0.6 m map).
        sigma: gradient smoothing at the coarse scale, in coarse pixels.
        noise: gradient noise floor in grey levels per pixel (see orientation_field).
        angles: template rotations to try, degrees, for heading error.
        excl_m: radius around the peak excluded from the sidelobe statistics.
        fine: refine at full map resolution around the coarse peak.
        parts: score by how strongly each quarter of the template supports the
            peak (mean of their PSRs) rather than by the whole template's PSR.
        sigma_m: 1-sigma accuracy reported with each fix.
        min_score: fixes scoring below this are not returned.
    """

    def __init__(self, map_: MapRaster, down: int = 4, sigma: float = 1.0, noise: float = 2.0,
                 angles: tuple[float, ...] = (-4.0, 0.0, 4.0), excl_m: float = 12.0, fine: bool = True,
                 parts: bool = True, sigma_m: float = 3.0, min_score: float = 0.0):
        self.map, self.down, self.sigma, self.noise = map_, down, sigma, noise
        self.angles, self.excl_m, self.fine, self.min_score = angles, excl_m, fine, min_score
        self.parts, self.sigma_m = parts, sigma_m
        g = cv2.resize(map_.gray, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
        v = cv2.resize(map_.valid.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
        self.valid_c = cv2.erode(v, np.ones((3, 3), np.uint8)) > 0
        self.field_c = orientation_field(g, sigma, noise) * self.valid_c
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

    def locate(self, ortho: Ortho, prior_col: float, prior_row: float, search_radius_px: int) -> ScoredFix | None:
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
        Hc, Wc = self.field_c.shape
        if x0 < 0 or y0 < 0 or x0 + tw + 2 * s > Wc or y0 + th + 2 * s > Hc:
            return None
        win = self.field_c[y0:y0 + th + 2 * s, x0:x0 + tw + 2 * s]
        if self.valid_c[y0:y0 + th + 2 * s, x0:x0 + tw + 2 * s].mean() < 0.3:
            return None
        shape = (cv2.getOptimalDFTSize(win.shape[0]), cv2.getOptimalDFTSize(win.shape[1]))
        Fw = np.fft.fft2(win, s=shape)

        def xcorr(t):
            """Re(sum conj(t) . win) / sum |t| for every shift of t in the window."""
            R = np.real(np.fft.ifft2(Fw * np.conj(np.fft.fft2(t, s=shape))))[:2 * s + 1, :2 * s + 1]
            return (R / max(float(np.abs(t).sum()), 1.0)).astype(np.float32)

        best = None
        for ang, t, tmask in temps:
            if tmask.mean() < 0.2:
                continue
            R = xcorr(t)
            r, c = np.unravel_index(int(np.argmax(R)), R.shape)
            if best is None or R[r, c] > best[0]:
                best = (float(R[r, c]), R, r, c, ang, t, tmask)
        if best is None:
            return None
        peak, R, r, c, ang, t, tmask = best

        # Score: PSR over the search disc, and the strongest rival peak.
        yy, xx = np.mgrid[:R.shape[0], :R.shape[1]]
        in_disc = (yy - s) ** 2 + (xx - s) ** 2 <= s * s + 1
        ex = max(1, int(round(self.excl_m / (self.map.gsd * d))))
        far = in_disc & ((yy - r) ** 2 + (xx - c) ** 2 > ex * ex)
        if far.sum() < 10:
            return None

        def psr_at(R):
            side = R[far]
            mu, sd = float(side.mean()), float(side.std()) + 1e-6
            return (float(R[r, c]) - mu) / sd, (float(side.max()) - mu) / sd

        psr, rival = psr_at(R)
        diag = {"peak": peak, "psr": psr, "rival": rival, "angle": ang,
                "edge": int(r in (0, R.shape[0] - 1) or c in (0, R.shape[1] - 1))}
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
        dc, dr = _parabolic(R, r, c)
        col = (x0 + c + dc) * d + nx
        row = (y0 + r + dr) * d + ny
        if self.fine:
            ref = self._refine(img, ortho, ang, col, row, diag)
            if ref is None:
                return None
            col, row = ref
        if score < self.min_score:
            return None
        lat, lon = self.map.px_to_ll(col, row)
        # Negated to match MapMatcher, which reports the rotation taking the ortho onto the map.
        return ScoredFix(col, row, lat, lon, 0, 0, 0.0, 1.0, -ang, self.sigma_m, score, diag)

    def _refine(self, img: np.ndarray, ortho: Ortho, ang: float, col: float, row: float, diag: dict):
        """Full-resolution correlation within +-(down + 2) map pixels of the coarse estimate."""
        nx, ny = ortho.nadir
        t, _ = self._template(img, ortho.mask, 1, self.sigma * self.down / 2, ang, (nx, ny))
        th, tw = t.shape
        m = self.down + 2
        x0, y0 = int(round(col - nx)) - m, int(round(row - ny)) - m
        H, W = self.map.gray.shape
        if x0 < 0 or y0 < 0 or x0 + tw + 2 * m > W or y0 + th + 2 * m > H:
            return None
        sl = (slice(y0, y0 + th + 2 * m), slice(x0, x0 + tw + 2 * m))
        f = orientation_field(self.map.gray[sl], self.sigma * self.down / 2, self.noise) * self.map.valid[sl]
        # Re(sum conj(t) * f) as a two-channel correlation.
        a = np.dstack([f.real, f.imag]).astype(np.float32)
        b = np.dstack([t.real, t.imag]).astype(np.float32)
        R = cv2.matchTemplate(a, b, cv2.TM_CCORR) / max(float(np.abs(t).sum()), 1.0)
        r, c = np.unravel_index(int(np.argmax(R)), R.shape)
        dc, dr = _parabolic(R, r, c)
        diag["fine_peak"] = float(R[r, c])
        diag["fine_shift"] = float(math.hypot(c - m, r - m))
        return x0 + c + dc + nx, y0 + r + dr + ny


# Learned features + LightGlue -------------------------------------------------
class LightGlueMatcher:
    """DISK or ALIKED keypoints matched with LightGlue, then a RANSAC similarity fit.

    Args:
        map_: reference map.
        features: "disk" or "aliked".
        ortho_kpts: keypoints kept on the ortho (strongest first).
        map_density: map keypoints per million (feature image) pixels.
        scale: resample map and ortho by this before extracting features
            (0.5 gives 1.2 m/px on a 0.6 m map).
        min_score: minimum RANSAC inliers.
        device: torch device.
        tile: side of the tiles the map is cut into for feature extraction, pixels.
    """

    def __init__(self, map_: MapRaster, features: str = "disk", ortho_kpts: int = 1024,
                 map_density: float = 6000, scale: float = 1.0, min_score: int = 8, device: str = "cpu",
                 tile: int = 1024):
        import kornia.feature as KF
        import torch

        self.torch, self.map, self.features = torch, map_, features
        self.ortho_kpts, self.min_score, self.device = ortho_kpts, min_score, torch.device(device)
        self.scale = scale
        if features == "disk":
            self.extractor = KF.DISK.from_pretrained("depth", device=self.device).eval()
        elif features == "aliked":
            self.extractor = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=-1,
                                                       detection_threshold=0.01, device=self.device).eval()
        else:
            raise ValueError(f"unknown features {features}")
        self.glue = KF.LightGlue(features).eval().to(self.device)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kpts, self.desc = self._map_features(map_density, tile)

    def _extract(self, gray: np.ndarray, k: int, mask: np.ndarray | None = None):
        """Strongest k keypoints (x, y) and descriptors, optionally only where mask is set."""
        torch = self.torch
        x = torch.from_numpy(gray).float().div(255)[None, None].expand(1, 3, -1, -1).to(self.device)
        with torch.inference_mode():
            if self.features == "disk":
                f = self.extractor(x, n=None, score_threshold=0.0, pad_if_not_divisible=True)[0]
                kp, desc, score = f.keypoints, f.descriptors, f.detection_scores
            else:
                f = self.extractor(x)[0]
                kp, desc, score = f.keypoints, f.descriptors, f.keypoint_scores
        kp, desc, score = kp.cpu().numpy(), desc.cpu().numpy(), score.cpu().numpy()
        if mask is not None:
            h, w = mask.shape
            keep = mask[np.clip(kp[:, 1].astype(int), 0, h - 1), np.clip(kp[:, 0].astype(int), 0, w - 1)]
            kp, desc, score = kp[keep], desc[keep], score[keep]
        if len(score) > k:
            keep = np.argsort(-score)[:k]
            kp, desc = kp[keep], desc[keep]
        return kp.astype(np.float32), desc.astype(np.float32)

    def _map_features(self, density: float, tile: int):
        """Keypoints over the whole map, from overlapping tiles keeping each tile's core."""
        gray, valid = self._resample(self.map.gray, self.map.valid)
        valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        H, W = gray.shape
        pad, core = 32, tile - 64
        kps, descs = [], []
        for y in range(0, H, core):
            for x in range(0, W, core):
                ya, xa = max(y - pad, 0), max(x - pad, 0)
                if not valid[y:y + core, x:x + core].any():
                    continue
                sub = gray[ya:y + core + pad, xa:x + core + pad]
                own = np.zeros(sub.shape, bool)
                own[y - ya:y - ya + core, x - xa:x - xa + core] = True
                own &= valid[ya:ya + sub.shape[0], xa:xa + sub.shape[1]]
                kp, desc = self._extract(sub, int(density * own.sum() / 1e6), own)
                kps.append(kp + [xa, ya])
                descs.append(desc)
        return np.concatenate(kps) / self.scale, np.concatenate(descs)

    def _resample(self, img: np.ndarray, mask: np.ndarray):
        if self.scale == 1:
            return img, mask
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        return img, mask

    def locate(self, ortho: Ortho, prior_col: float, prior_row: float, search_radius_px: int) -> ScoredFix | None:
        torch = self.torch
        h, w = ortho.image.shape
        radius = int(search_radius_px + max(h, w) / 2)
        x0, y0 = int(prior_col) - radius, int(prior_row) - radius
        rel = self.kpts - [x0, y0]
        sel = (rel >= 0).all(axis=1) & (rel < 2 * radius).all(axis=1)
        if sel.sum() < self.min_score:
            return None
        kp2, d2 = rel[sel], self.desc[sel]
        img, mask = self._resample(self.clahe.apply(ortho.image), ortho.mask)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        kp1, d1 = self._extract(img, self.ortho_kpts, mask)
        kp1 = kp1 / self.scale
        if len(kp1) < self.min_score:
            return None
        data = {
            "image0": {"keypoints": torch.from_numpy(kp1)[None], "descriptors": torch.from_numpy(d1)[None],
                       "image_size": torch.tensor([[w, h]])},
            "image1": {"keypoints": torch.from_numpy(kp2)[None], "descriptors": torch.from_numpy(d2)[None],
                       "image_size": torch.tensor([[2 * radius, 2 * radius]])},
        }
        with torch.inference_mode():
            out = self.glue({k: {kk: vv.to(self.device) for kk, vv in v.items()} for k, v in data.items()})
        m = out["matches"][0].cpu().numpy()
        conf = out["scores"][0].cpu().numpy()
        diag = {"n_ortho": len(kp1), "n_map": int(sel.sum()), "matches": len(m),
                "conf_sum": float(conf.sum())}
        return similarity_fix(self.map, ortho, kp1[m[:, 0]], kp2[m[:, 1]], x0, y0, self.min_score, diag=diag)


def make(spec: str, map_: MapRaster):
    """Build a matcher from a spec like "orb", "corr:down=4,angles=-3/0/3" or "lg:features=aliked"."""
    name, _, args = spec.partition(":")
    kw = {}
    for item in filter(None, args.split(",")):
        k, v = item.split("=")
        if "/" in v:
            kw[k] = tuple(float(x) for x in v.split("/"))
        else:
            try:
                kw[k] = int(v)
            except ValueError:
                try:
                    kw[k] = float(v)
                except ValueError:
                    kw[k] = {"true": True, "false": False}.get(v.lower(), v)
    cls = {"orb": OrbMatcher, "corr": CorrMatcher, "lg": LightGlueMatcher}[name]
    return cls(map_, **kw)
