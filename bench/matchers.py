"""Map matchers compared by the cross-year benchmark.

Each has the interface of ``visnav.matcher.MapMatcher``:
``locate(ortho, prior_col, prior_row, search_radius_px) -> Fix | None``. Here
the thresholds are set low, so the benchmark can sweep them over ``Fix.score``.

* ``OrbMatcher``: ``visnav.matcher.MapMatcher`` with ORB. Score: RANSAC inliers.
* ``CorrMatcher``: ``visnav.matcher.CorrMatcher``. Score: how far the correlation
  peak stands out from the rest of the search area, see its ``locate``.
* ``LightGlueMatcher``: DISK or ALIKED features matched with LightGlue
  (kornia, PyTorch), then the same RANSAC similarity fit as ORB. Map features
  are extracted once for the whole map, as they would be before a flight.
  Score: RANSAC inliers.
"""

from __future__ import annotations

import cv2
import numpy as np

from visnav.geo import MapRaster
from visnav.matcher import CorrMatcher, FeatureBackend, Fix, MapMatcher, similarity_fix
from visnav.ortho import Ortho


class OrbMatcher:
    """The navigator's MapMatcher with ORB."""

    def __init__(self, map_: MapRaster, min_score: int = 8, nfeatures: int = 3000):
        self.map = map_
        self.inner = MapMatcher(map_, FeatureBackend("orb", nfeatures), min_inliers=min_score)

    def locate(self, ortho: Ortho, prior_col: float, prior_row: float, search_radius_px: int) -> Fix | None:
        return self.inner.locate(ortho, prior_col, prior_row, search_radius_px)


def corr_matcher(map_: MapRaster, **kw) -> CorrMatcher:
    """The navigator's CorrMatcher, returning every fix whatever its score."""
    return CorrMatcher(map_, **{"min_score": 0.0, "score_per_radius": 0.0, "min_area_m2": 0.0, **kw})


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

    def locate(self, ortho: Ortho, prior_col: float, prior_row: float, search_radius_px: int) -> Fix | None:
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
        fix = similarity_fix(self.map, ortho, kp1[m[:, 0]], kp2[m[:, 1]], x0, y0, self.min_score)
        if fix is not None:
            fix.diag = diag
        return fix


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
    cls = {"orb": OrbMatcher, "corr": corr_matcher, "lg": LightGlueMatcher}[name]
    return cls(map_, **kw)
