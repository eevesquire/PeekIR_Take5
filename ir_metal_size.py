"""
IR metal size -> threat level, by comparing against a no-metal reference.

The IR camera and the mannequin don't move, so:
  1. reference : average ~1 s of raw IR frames of the mannequin WITHOUT metal
  2. measure   : average ~1 s WITH metal, subtract the reference
  3. the pixels that changed (beyond sensor noise) are the metal; clean them up,
     find connected blobs, measure their area
  4. convert pixel area to cm^2 with a one-time calibration (a plate of known size)
  5. map the largest blob's area to a threat level

Needs: numpy, opencv-python (and httpx for the --pi command line options).
Runs on the dashboard laptop; frames come from the Pi via GET /api/ir/burst.

Command line:
  python3 ir_metal_size.py reference --pi http://<pi>:9000          # no metal on the mannequin
  python3 ir_metal_size.py calibrate --pi http://<pi>:9000 --width-cm 10 --height-cm 15
  python3 ir_metal_size.py measure   --pi http://<pi>:9000 --save result.png
  python3 ir_metal_size.py levels --low 25 --high 150               # thresholds in cm^2
  python3 ir_metal_size.py status

  # offline, on clips recorded with the Pi recorder:
  python3 ir_metal_size.py measure --ref-npz ref_clip.npz --npz metal_clip.npz

State (reference frames, calibration, levels) is kept in --state-dir
(default ./ir_metal_state), shared between the command line and the dashboard.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

LEVELS = ("NONE", "LOW", "MEDIUM", "HIGH")
LEVEL_COLORS_BGR = {"NONE": (160, 160, 160), "LOW": (80, 200, 80), "MEDIUM": (0, 200, 255), "HIGH": (60, 60, 255)}


@dataclass
class Settings:
    # threat levels by area of the largest metal blob, in cm^2
    low_max_cm2: float = 25.0       # area <  low_max            -> LOW
    high_min_cm2: float = 150.0     # low_max <= area < high_min -> MEDIUM ; >= high_min -> HIGH
    # detection
    min_delta_c: float = 0.3        # minimum temperature change counted as metal (deg C, y16 data)
    local_background: bool = True   # compare each spot with its surroundings, not just the whole frame
    bg_block_px: int = 24           # size of the "surroundings" (larger than any metal piece's short side)
    smooth_px: float = 1.0          # neighbour averaging (Gaussian sigma, px) for faint solid pieces; 0 = off
    half_contrast_edges: bool = True  # piece edge = half its own contrast, so faint and strong
                                      # identical pieces measure the same size
    # Ignore anything much hotter than the warm body (heating pads, ceiling lights): metal on the
    # mannequin never reads hotter than the mannequin, and a pad that switches on/off changes
    # temperature and would otherwise look like metal. Degrees C above the body's typical
    # temperature; 0 = off.
    ignore_hot_c: float = 5.0
    ignore_hot_margin_px: int = 3
    # Tolerate the mannequin/camera shifting by up to this many pixels between reference and
    # measurement (e.g. bumped while attaching metal). On a sharp edge of the reference such a
    # shift changes pixels by about (edge steepness x shift), so the threshold is raised there by
    # that much. Flat areas (where metal goes) are unaffected. 0 = off.
    motion_px: float = 0.5
    min_delta_8bit: float = 8.0     # same, for 8-bit frames (when IR_RAW_Y16=0 on the Pi)
    noise_k: float = 6.0            # also require change > noise_k * estimated sensor noise
    min_area_px: int = 6            # blobs smaller than this are ignored (noise / too small to see)
    min_area_cm2: float = 0.0       # when calibrated, also ignore pieces smaller than this many cm^2
                                    # (stray specks); 0 = off. Uncalibrated: only min_area_px applies
    morph_px: int = 1               # clean-up radius (open, then close) when outline_fill="none"
    # Metal often shows only as a thin bright/dark RING (its edges) with a face that blends in.
    # "holes": join small gaps in the ring (close_px), then fill its inside so the whole piece
    # is measured. "hull": fill each piece's convex outline (also covers rings with bigger gaps,
    # but over-fills L/odd shapes). "none": old behaviour (thin rings get removed as noise).
    outline_fill: str = "auto"      # "auto" | "hull" | "holes" | "none"
    close_px: int = 2               # gap-joining radius for outline_fill
    max_fill_px: int = 800          # never fill an enclosed area larger than this (a ring that big
                                    # usually means the mannequin moved, not metal)
    body_mask: bool = True          # only look where the mannequin is (warm region of the reference)
    # Shiny metal on a warmed mannequin reflects the cooler room, so it reads COLDER than the
    # reference. Warm handprints / fresh tape read WARMER. "cold" ignores those; use "both"
    # if the mannequin isn't warmed or metal shows up warm (e.g. reflecting a person or lamp).
    polarity: str = "cold"          # "cold" | "warm" | "both"
    capture_seconds: float = 1.0
    # calibration
    cm_per_px: float | None = None  # set by calibrate(); None = uncalibrated
    calibrated_at: str | None = None
    calibration_note: str = ""


@dataclass
class Blob:
    area_px: int
    area_cm2: float | None
    bbox: tuple[int, int, int, int]       # x, y, w, h (in IR pixels)
    centroid: tuple[float, float]
    mean_delta: float                     # mean change vs reference (deg C for y16, raw units otherwise)
    size_cm: tuple[float, float] | None   # bbox width/height in cm
    shape: dict | None = None             # shape_classify.py: gun / knife / suspicious object + measurements


@dataclass
class Result:
    level: str
    largest_area_cm2: float | None
    largest_area_px: int
    total_area_cm2: float | None
    total_area_px: int
    blobs: list[Blob] = field(default_factory=list)
    calibrated: bool = False
    threshold_used: float = 0.0
    noise: float = 0.0
    n_frames: int = 0
    reference_age_s: float | None = None
    notes: list[str] = field(default_factory=list)
    size_level: str | None = None         # level from size alone (before weapons raised it)
    weapons: list[str] = field(default_factory=list)   # e.g. ["gun"]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- core math

def _gray(frames: np.ndarray, ir_format: str) -> np.ndarray:
    """(N,H,W[,3]) -> float32 (N,H,W) in 'temperature units' (deg C for y16, raw otherwise)."""
    x = np.asarray(frames)
    if x.ndim == 2 or (x.ndim == 3 and ir_format == "bgr8" and x.shape[-1] == 3):
        x = x[None]
    x = x.astype(np.float32)
    if x.ndim == 4:
        x = x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299
    if ir_format == "y16":
        x = x / 100.0  # centikelvin -> kelvin; differences are then in deg C
    return x


def average(frames: np.ndarray, ir_format: str) -> np.ndarray:
    """Average a burst into one low-noise frame (median is robust to an odd glitchy frame)."""
    g = _gray(frames, ir_format)
    return np.median(g, axis=0) if len(g) >= 3 else g.mean(axis=0)


def _body_mask(ref: np.ndarray) -> np.ndarray | None:
    """Warm region of the reference (Otsu split), slightly grown. None if it looks unreliable."""
    import cv2
    lo, hi = np.percentile(ref, (1, 99))
    if hi - lo < 1e-6:
        return None
    u8 = np.clip((ref - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    frac = (m > 0).mean()
    if frac < 0.05 or frac > 0.9:
        return None
    return m > 0


def _fill_holes(m8: np.ndarray, max_fill_px: int) -> tuple[np.ndarray, int]:
    """Fill enclosed holes up to max_fill_px. Returns (mask, size of the largest hole left unfilled)."""
    import cv2
    h, w = m8.shape
    inv = (m8 == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = m8.copy()
    biggest_skipped = 0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + ww == w or y + hh == h
        if touches_border:
            continue                      # outside region, not a hole
        if area <= max_fill_px:
            out[labels == i] = 1
        else:
            biggest_skipped = max(biggest_skipped, int(area))
    return out, biggest_skipped


def _reference_edges(ref: np.ndarray) -> np.ndarray | None:
    """Strong edges of the reference image (e.g. the mannequin's outline), slightly widened."""
    import cv2
    gx = cv2.Sobel(ref.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    thr = np.percentile(g, 95)
    if thr <= 0:
        return None
    return cv2.dilate((g >= thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def _looks_like_movement(ref: np.ndarray, metal: np.ndarray) -> float:
    """
    Fraction of detected pixels sitting on strong edges of the REFERENCE (e.g. the mannequin's
    outline). Metal isn't in the reference, so real metal rarely lines up with its edges; a
    shifted mannequin does. Returns that fraction if it looks like movement, else 0.
    """
    import cv2
    n = int(metal.sum())
    if n < 10:
        return 0.0
    gx = cv2.Sobel(ref.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    thr = np.percentile(g, 95)
    if thr <= 0:
        return 0.0
    edges = cv2.dilate((g >= thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    frac = float((metal & edges).sum()) / n
    return frac if frac > 0.6 else 0.0


def _angular_coverage(piece: np.ndarray, hull: np.ndarray, bins: int = 16) -> float:
    """Fraction of directions (seen from the hull's centre) in which the piece has pixels."""
    import cv2
    mo = cv2.moments(hull)
    if mo["m00"] <= 0:
        return 0.0
    cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    ys, xs = np.nonzero(piece)
    if xs.size == 0:
        return 0.0
    ang = np.arctan2(ys - cy, xs - cx)
    idx = ((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    return len(np.unique(idx)) / bins


def _clean_mask(metal: np.ndarray, s: Settings, edges: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Remove noise and (optionally) turn metal outlines into filled shapes."""
    import cv2
    m8 = metal.astype(np.uint8)
    if s.outline_fill == "none":
        if s.morph_px > 0:
            k = np.ones((2 * s.morph_px + 1, 2 * s.morph_px + 1), np.uint8)
            m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
            m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
        return m8 > 0, 0
    # Outline mode: join small gaps in rings first (opening would erase 1-px rings)...
    if s.close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s.close_px + 1, 2 * s.close_px + 1))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    # ...then fill what they enclose.
    m8, skipped = _fill_holes(m8, s.max_fill_px)
    if s.outline_fill in ("hull", "auto"):
        k3 = np.ones((3, 3), np.uint8)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if s.outline_fill == "auto":
                # Solid pieces keep their real shape (L-brackets, T's, odd shapes). Only thin
                # outlines (a metal piece seen just as a ring around its edge) get their convex
                # outline filled, which also bridges gaps in the ring.
                one = np.zeros_like(m8)
                cv2.drawContours(one, [c], -1, 1, thickness=-1)
                piece = one & m8
                thick = cv2.morphologyEx(piece, cv2.MORPH_OPEN, k3)
                if thick.sum() >= 0.5 * max(1, piece.sum()):
                    continue
                # A ring goes (almost) all the way around its centre; a thin L, strip or curve
                # doesn't. Only fill thin pieces that surround their centre.
                if _angular_coverage(piece, cv2.convexHull(c)) < 0.75:
                    continue
            hull = cv2.convexHull(c)
            if cv2.contourArea(hull) <= s.max_fill_px:
                cv2.fillPoly(m8, [hull], 1)
    elif s.outline_fill != "holes":
        raise ValueError(f"outline_fill must be auto, holes, hull or none (got {s.outline_fill!r})")
    # Finally drop anything only 1-2 px thin that lies along edges already in the reference
    # (lines left by the mannequin shifting slightly). Thin metal elsewhere (strips, thin
    # brackets) is kept. Without edge information, all thin parts are dropped.
    if s.morph_px > 0:
        k = np.ones((2 * s.morph_px + 1, 2 * s.morph_px + 1), np.uint8)
        opened = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
        if edges is None:
            m8 = opened
        else:
            n, labels = cv2.connectedComponents(m8, connectivity=8)
            if n > 1:
                tot = np.bincount(labels.ravel(), minlength=n)
                on_edge = np.bincount(labels[edges].ravel(), minlength=n)
                along_edges = on_edge >= 0.5 * np.maximum(tot, 1)
                along_edges[0] = False
                m8 = np.where(along_edges[labels], opened, m8).astype(np.uint8)
    return m8 > 0, skipped


def _robust_sigma(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _local_background(diff: np.ndarray, block: int) -> np.ndarray:
    """
    Smooth, slowly varying part of the change (e.g. the mannequin warming up unevenly from a
    heating pad), estimated as block medians that ignore outliers (the metal itself), then
    interpolated back to full size. Subtracting it leaves compact changes such as metal.
    """
    import cv2
    import warnings
    h, w = diff.shape
    block = max(4, int(block))
    gy, gx = -(-h // block), -(-w // block)

    def coarse_median(a: np.ndarray) -> np.ndarray:
        pad = np.full((gy * block, gx * block), np.nan, np.float32)
        pad[:h, :w] = a
        b = pad.reshape(gy, block, gx, block).transpose(0, 2, 1, 3).reshape(gy, gx, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(b, axis=2)

    def upsample(c: np.ndarray) -> np.ndarray:
        c = np.where(np.isfinite(c), c, np.nanmedian(c) if np.isfinite(c).any() else 0.0).astype(np.float32)
        return cv2.resize(c, (gx * block, gy * block), interpolation=cv2.INTER_LINEAR)[:h, :w]

    c1 = coarse_median(diff)
    bg1 = upsample(c1)
    r1 = diff - bg1
    sig = max(_robust_sigma(r1), 1e-6)
    masked = np.where(np.abs(r1) > 3 * sig, np.nan, diff)   # drop outliers (metal) and redo
    c2 = coarse_median(masked)
    c2 = np.where(np.isfinite(c2), c2, c1)
    return upsample(c2)


def _hot_zones(ref: np.ndarray, cur: np.ndarray, ir_format: str, s: Settings) -> np.ndarray | None:
    """Areas much hotter than the scene in the reference OR the current frame, plus a margin."""
    import cv2
    if s.ignore_hot_c <= 0 or ir_format != "y16":
        return None

    def too_hot(img: np.ndarray) -> np.ndarray:
        # Measure "hot" against the warm BODY (mannequin or person), not the room: the body is
        # the bulk of the pixels warmer than the room; pads and lights are small and much hotter.
        room = float(np.median(img))
        warm = img > room + 1.0
        body = float(np.median(img[warm])) if warm.sum() > 0.01 * img.size else room
        return img > body + s.ignore_hot_c

    hot = too_hot(ref) | too_hot(cur)
    if s.ignore_hot_margin_px > 0 and hot.any():
        m = 2 * s.ignore_hot_margin_px + 1
        hot = cv2.dilate(hot.astype(np.uint8), np.ones((m, m), np.uint8)) > 0
    return hot


def _change_map(ref: np.ndarray, cur: np.ndarray, ir_format: str, s: Settings) -> dict:
    """Change vs reference with drift removed, plus the search region, noise and threshold."""
    notes = []
    diff = (cur - ref).astype(np.float32)
    mask_region = _body_mask(ref) if s.body_mask else None
    if s.body_mask and mask_region is None:
        notes.append("mannequin outline not found in reference (is it warmed?); searching the whole frame")
    region = mask_region if mask_region is not None else np.ones_like(diff, dtype=bool)
    hot = _hot_zones(ref, cur, ir_format, s)
    if hot is not None and hot.any():
        region = region & ~hot
    # remove global drift (Lepton flat-field correction / ambient change)
    diff = diff - np.median(diff[region])
    # remove slow, broad changes (mannequin warming/cooling unevenly), keep compact ones (metal)
    if s.local_background:
        diff = diff - _local_background(diff, s.bg_block_px)
    noise = _robust_sigma(diff[region])
    floor = s.min_delta_c if ir_format == "y16" else s.min_delta_8bit
    thr = max(floor, s.noise_k * noise)
    # Smoothed copy: averaging each pixel with its neighbours cuts per-pixel speckle (sensor noise,
    # and the Lepton's pixel offsets shifting after its periodic self-calibration) about 3x, while a
    # metal piece spanning many pixels keeps its contrast. Faint-but-solid pieces are found here;
    # thin outlines are still found in the unsmoothed copy.
    if s.smooth_px > 0:
        import cv2
        diff_s = cv2.GaussianBlur(diff, (0, 0), float(s.smooth_px))
        noise_s = _robust_sigma(diff_s[region])
        thr_s = max(floor, s.noise_k * noise_s)
    else:
        diff_s, noise_s, thr_s = diff, noise, thr
    edge_allow, edge_allow_s = _edge_allowance(ref, s)
    return {"diff": diff, "region": region, "noise": noise, "thr": thr, "notes": notes,
            "diff_s": diff_s, "noise_s": noise_s, "thr_s": thr_s, "hot": hot,
            "edge_allow": edge_allow, "edge_allow_s": edge_allow_s}


def _edge_allowance(ref: np.ndarray, s: Settings):
    """Extra change allowed on edges of the reference: |gradient| x motion_px (0 on flat areas)."""
    import cv2
    if s.motion_px <= 0:
        return 0.0, 0.0
    r = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 1.0)     # slope of real edges, not noise
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3) / 8.0     # per-pixel slope
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.hypot(gx, gy)
    g = np.maximum(g - 3.0 * float(np.median(g)), 0.0)     # flat areas (incl. noise) -> 0
    g = cv2.dilate(g, np.ones((3, 3), np.uint8))           # edges can move a pixel
    allow = (s.motion_px * 2.0) * g    # x2: the step spans ~2 px, so a shift moves it by twice the slope
    allow_s = cv2.GaussianBlur(allow, (0, 0), float(s.smooth_px)) if s.smooth_px > 0 else allow
    return allow, allow_s


def _grow_to_half_contrast(metal: np.ndarray, diff_s: np.ndarray, polarity: str, region: np.ndarray,
                           min_px: int, raw_hit: np.ndarray | None = None, allow=0.0,
                           reach: int = 4) -> np.ndarray:
    """
    Make a piece's edge depend on its OWN contrast, not on the fixed threshold. A faint piece
    otherwise only gets its strongest middle detected (and measures too small), while an
    identical strong piece is detected whole. Each detected piece is extended to connected
    pixels (within `reach` px) that changed at least half as much as the piece's typical change.
    """
    import cv2
    n, labels = cv2.connectedComponents(metal.astype(np.uint8), connectivity=8)
    out = metal.copy()
    k = np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)
    for i in range(1, n):
        comp = labels == i
        if comp.sum() < max(2, min_px // 2):
            continue
        # only solid pieces: thin structures (metal outlines/rings, edges of a shifted mannequin)
        # are left as they are; the outline filling handles rings.
        k3 = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, k3)   # fill speckle holes first
        thick = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k3)
        # ...unless it was found mainly through the smoothed copy (a faint solid piece whose
        # detection is patchy because it sits near the threshold)
        smooth_only = 1.0 if raw_hit is None else float((comp & ~raw_hit).sum()) / comp.sum()
        if thick.sum() < 0.5 * closed.sum() and smooth_only < 0.5:
            continue
        v = diff_s[comp]
        sign = -1.0 if polarity == "cold" else 1.0 if polarity == "warm" else (-1.0 if np.median(v) < 0 else 1.0)
        peak = float(np.percentile(sign * v, 70))      # typical contrast of this piece
        if peak <= 0:
            continue
        near = cv2.dilate(comp.astype(np.uint8), k) > 0
        cand = ((sign * diff_s - allow) >= 0.5 * peak) & near & region   # don't grow into shifted edges
        seeds = comp & cand
        if not seeds.any():
            continue                                    # keep the piece as detected
        _, lab2 = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        ids = np.unique(lab2[seeds])
        ids = ids[ids > 0]
        # REPLACE the piece with its half-contrast region: faint pieces grow to their real edge,
        # strong pieces lose the blurred halo the smoothing adds around them.
        out &= ~comp
        out |= np.isin(lab2, ids)
    return out


def _changed(diff: np.ndarray, thr: float, polarity: str) -> np.ndarray:
    if polarity == "cold":
        return diff < -thr
    if polarity == "warm":
        return diff > thr
    if polarity == "both":
        return np.abs(diff) > thr
    raise ValueError(f"polarity must be cold, warm or both (got {polarity!r})")


def detect(ref: np.ndarray, cur: np.ndarray, ir_format: str, s: Settings) -> tuple[Result, np.ndarray]:
    """ref, cur: averaged frames from average(). Returns (Result, metal mask bool HxW)."""
    import cv2
    cm = _change_map(ref, cur, ir_format, s)
    diff, region, noise, thr, notes = cm["diff"], cm["region"], cm["noise"], cm["thr"], list(cm["notes"])
    diff_s, thr_s = cm["diff_s"], cm["thr_s"]

    # per-pixel (thin outlines) OR smoothed (faint solid pieces). A pixel found via the smoothed
    # copy must also have changed at least half as much itself, so the blur doesn't make strong
    # pieces grow by the halo it spreads around them.
    ea, ea_s = cm["edge_allow"], cm["edge_allow_s"]

    def found(pol: str) -> np.ndarray:
        via_smooth = _changed(diff_s, thr_s + ea_s, pol) & _changed(diff, 0.5 * thr_s + ea, pol)
        return _changed(diff, thr + ea, pol) | via_smooth
    changed = found(s.polarity)
    if s.polarity != "both":
        ignored = int((found("warm" if s.polarity == "cold" else "cold") & region).sum())
    else:
        ignored = 0
    if ignored >= s.min_area_px * 3:
        notes.append(f"ignored {ignored} px that got {'warmer' if s.polarity == 'cold' else 'colder'} "
                     f"(handprints/tape?); set polarity=both if metal looks {'warm' if s.polarity == 'cold' else 'cold'} here")
    metal = changed & region
    if s.half_contrast_edges:
        metal = _grow_to_half_contrast(metal, diff_s, s.polarity, region, s.min_area_px,
                                       raw_hit=_changed(diff, thr + ea, s.polarity), allow=ea_s)
    metal, big_unfilled = _clean_mask(metal, s, edges=_reference_edges(ref))
    if big_unfilled:
        notes.append(f"an outline enclosing {big_unfilled} px was NOT filled (limit {s.max_fill_px} px); "
                     "if that's the mannequin's edge, it moved since the reference: retake the reference")
    moved = _looks_like_movement(ref, metal)
    if moved:
        notes.append(f"{moved:.0%} of the detected area lies on edges already in the reference: the "
                     "mannequin or camera probably MOVED since the reference. Retake the reference.")

    n, labels, stats, cents = cv2.connectedComponentsWithStats(metal.astype(np.uint8), connectivity=8)
    _shapes = _shape_module()
    shape_s = _shapes.load_settings() if _shapes is not None else None
    blobs: list[Blob] = []
    keep = np.zeros_like(metal)
    cpp = s.cm_per_px
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < s.min_area_px:
            continue
        if cpp and s.min_area_cm2 > 0 and area * cpp * cpp < s.min_area_cm2:
            continue                                       # too small to matter (noise speck)
        x, y, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                 cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        sel = labels == i
        keep |= sel
        shape = None
        if _shapes is not None:
            try:
                shape = _shapes.classify_mask(sel, cpp or 1.0, cpp or 1.0, in_cm=bool(cpp), s=shape_s)
            except Exception as e:                         # a shape problem must never stop the measurement
                notes.append(f"shape check failed: {e}")
        blobs.append(Blob(
            area_px=area,
            area_cm2=round(area * cpp * cpp, 1) if cpp else None,
            bbox=(x, y, w, h),
            centroid=(round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)),
            mean_delta=round(float(diff[sel].mean()), 2),
            size_cm=(round(w * cpp, 1), round(h * cpp, 1)) if cpp else None,
            shape=shape,
        ))
    blobs.sort(key=lambda b: b.area_px, reverse=True)

    largest_px = blobs[0].area_px if blobs else 0
    total_px = sum(b.area_px for b in blobs)
    if cpp:
        largest_cm2 = round(largest_px * cpp * cpp, 1)
        total_cm2 = round(total_px * cpp * cpp, 1)
        area_for_level = largest_cm2
    else:
        largest_cm2 = total_cm2 = None
        area_for_level = float(largest_px)  # uncalibrated: treat 1 px as 1 cm^2
        notes.append("not calibrated: sizes are in pixels and levels assume 1 px = 1 cm^2")

    if not blobs:
        level = "NONE"
    elif area_for_level < s.low_max_cm2:
        level = "LOW"
    elif area_for_level < s.high_min_cm2:
        level = "MEDIUM"
    else:
        level = "HIGH"

    size_level, weapons = level, []
    if _shapes is not None and blobs:
        shapes = [b.shape for b in blobs]
        weapons = _shapes.weapons_of(shapes)
        level = _shapes.raise_level(level, shapes, shape_s)
        if weapons and level != size_level:
            notes.append(f"{' + '.join(weapons)} shape found: threat raised from {size_level} to {level}")
        if not cpp and shape_s.enabled:
            notes.append("knife shapes need calibration (cm); only the gun (L) shape is checked")

    res = Result(level=level, largest_area_cm2=largest_cm2, largest_area_px=largest_px,
                 total_area_cm2=total_cm2, total_area_px=total_px, blobs=blobs,
                 calibrated=cpp is not None, threshold_used=round(thr, 3), noise=round(noise, 4),
                 notes=notes, size_level=size_level, weapons=weapons)
    return res, keep


def _shape_module():
    """shape_classify.py next to this file (None if missing: then threat is from size only)."""
    try:
        import shape_classify
        return shape_classify
    except ImportError:
        return None


def annotate(cur: np.ndarray, metal: np.ndarray, res: Result, scale: int = 4) -> np.ndarray:
    """Colorized IR frame with metal outlined and a threat banner. Returns BGR uint8."""
    import cv2
    lo, hi = np.percentile(cur, (1, 99))
    u8 = np.clip((cur - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    big = cv2.resize(metal.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
    for b in res.blobs:
        x, y, w, h = (v * scale for v in b.bbox)
        txt = f"{b.area_cm2:.0f} cm2" if b.area_cm2 is not None else f"{b.area_px} px"
        if b.shape:
            txt = f"{b.shape['display'].upper() if b.shape.get('weapon') else 'object'} " + txt
        cv2.putText(img, txt, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    color = LEVEL_COLORS_BGR[res.level]
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    if res.largest_area_cm2 is not None:
        size = f"largest {res.largest_area_cm2:.0f} cm2"
    else:
        size = f"largest {res.largest_area_px} px (uncalibrated)"
    weap = ("   " + " + ".join(w.upper() for w in res.weapons)) if getattr(res, "weapons", None) else ""
    cv2.putText(img, f"THREAT: {res.level}{weap}   {size}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- stateful helper

def diagnose_diff(ref: np.ndarray, cur: np.ndarray, ir_format: str, s: Settings,
                  scale: int = 4) -> tuple[dict, np.ndarray]:
    """
    Show HOW MUCH each pixel changed vs the reference, to tune detection when metal is missed.
    Returns (stats, image). Image: blue = colder than reference, red = warmer, white = unchanged;
    black outlines = what the current settings would detect.
    """
    import cv2
    cm = _change_map(ref, cur, ir_format, s)
    region = cm["region"]
    noise_raw, thr_raw = cm["noise"], cm["thr"]
    # show/judge the smoothed change: that's what finds solid pieces
    diff, noise, thr = cm["diff_s"], cm["noise_s"], cm["thr_s"]
    floor = s.min_delta_c if ir_format == "y16" else s.min_delta_8bit
    unit = "C" if ir_format == "y16" else "raw units"
    vals = diff[region]
    # Is the mannequin-outline mask hiding the strongest change?
    masked_out = None
    if s.body_mask and not region.all():
        allv = np.sort(diff.ravel())
        k_all = min(30, allv.size)
        cold_all = float(np.median(allv[:k_all]))
        warm_all = float(np.median(allv[-k_all:]))
        outside = ~region
        if outside.any():
            ov = np.sort(diff[outside])
            k_o = min(30, ov.size)
            if (s.polarity != "warm" and np.median(ov[:k_o]) <= cold_all * 0.8 and cold_all < -thr * 0.5) or \
               (s.polarity != "cold" and np.median(ov[-k_o:]) >= warm_all * 0.8 and warm_all > thr * 0.5):
                masked_out = round(cold_all if s.polarity != "warm" else warm_all, 3)
    steps = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0] if ir_format == "y16" else [2, 4, 8, 16, 32]
    colder = {str(t): int((vals < -t).sum()) for t in steps}
    warmer = {str(t): int((vals > t).sum()) for t in steps}
    p = np.percentile(vals, [0.2, 1, 50, 99, 99.8])
    # strongest compact change of each sign (median of the 30 most-changed pixels) ~ metal contrast
    k = min(30, vals.size)
    coldest = float(np.median(np.sort(vals)[:k]))
    warmest = float(np.median(np.sort(vals)[-k:]))
    sign = -1 if s.polarity == "cold" else (1 if s.polarity == "warm" else (-1 if -coldest >= warmest else 1))
    strongest = coldest if sign < 0 else warmest
    min_safe = max(4 * noise, 0.15 if ir_format == "y16" else 3)
    suggest = round(max(min_safe, 0.6 * abs(strongest)), 2)
    stats = {
        "unit": unit, "noise": round(noise, 3), "threshold_used": round(thr, 3),
        "min_delta_setting": floor, "noise_k": s.noise_k, "polarity": s.polarity,
        "percentiles": {"0.2%": round(float(p[0]), 3), "1%": round(float(p[1]), 3), "median": round(float(p[2]), 3),
                        "99%": round(float(p[3]), 3), "99.8%": round(float(p[4]), 3)},
        "strongest_cold_change": round(coldest, 3), "strongest_warm_change": round(warmest, 3),
        "pixels_colder_than": colder, "pixels_warmer_than": warmer,
        "suggested_min_delta": suggest,
        "suggestion_ok": abs(strongest) > min_safe / 0.6,
        "body_mask": s.body_mask,
        "body_mask_hides": masked_out,
        "local_background": s.local_background,
        "pixel_noise": round(noise_raw, 3), "pixel_threshold": round(thr_raw, 3),
        "ignored_hot_px": int(cm["hot"].sum()) if cm.get("hot") is not None else 0,
        "ignore_hot_c": s.ignore_hot_c,
        "smooth_px": s.smooth_px,
    }
    # picture
    span = max(abs(strongest) * 1.2, 3 * noise, 1e-6)
    t = np.clip(diff / span, -1, 1)
    img = np.empty(diff.shape + (3,), np.float32)       # BGR
    neg, pos = np.clip(-t, 0, 1), np.clip(t, 0, 1)
    img[..., 0] = 1 - pos          # blue channel stays high for cold, drops for warm
    img[..., 1] = 1 - neg - pos    # green drops either way
    img[..., 2] = 1 - neg          # red stays high for warm, drops for cold
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    _, metal = detect(ref, cur, ir_format, s)
    big = cv2.resize(metal.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 2)
    if cm.get("hot") is not None and cm["hot"].any():
        hb = cv2.resize(cm["hot"].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        img[hb] = (img[hb] * 0.25 + 90).astype(np.uint8)          # gray = ignored (too hot)
    if s.body_mask and not region.all():
        rb = cv2.resize(region.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        rc, _ = cv2.findContours(rb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, rc, -1, (0, 160, 0), 2)      # green = where it searches (body mask)
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (40, 40, 40), -1)
    u = "C" if ir_format == "y16" else ""
    cv2.putText(img, f"blue=colder red=warmer (+/-{span:.2f}{u})  threshold {thr:.2f}{u}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return stats, img


def explain_diagnosis(st: dict) -> list[str]:
    u = "°C" if st["unit"] == "C" else " " + st["unit"]
    lines = [
        f"Noise after neighbour averaging ~{st['noise']}{u} (per pixel {st.get('pixel_noise', '?')}{u}); "
        f"threshold {st['threshold_used']}{u} (max of min-delta {st['min_delta_setting']} and "
        f"{st['noise_k']}x noise); polarity {st['polarity']}.",
        f"Strongest local change: colder {st['strongest_cold_change']}{u}, warmer {st['strongest_warm_change']}{u}.",
        "Pixels colder than: " + ", ".join(f"{k}{u}: {v}" for k, v in st["pixels_colder_than"].items()),
    ]
    if st.get("ignored_hot_px"):
        lines.append(f"Ignoring {st['ignored_hot_px']} px hotter than the warm body by > {st['ignore_hot_c']}{u} "
                     "(heating pad, lights; gray in the image). Change with: levels --ignore-hot <degrees>")
    if st.get("body_mask_hides") is not None:
        lines.insert(0, f"The body mask is HIDING the strongest change ({st['body_mask_hides']}{u}, outside the "
                        "green outline). Run: levels --body-mask off")
    if st["suggestion_ok"]:
        lines.append(f"If the blue patch in the image is your metal, try: levels --min-delta-c {st['suggested_min_delta']}")
    else:
        lines.append("The change is close to the sensor noise: improve the contrast (warmer mannequin "
                     "around the metal, longer capture) rather than lowering the threshold.")
    return lines


class MetalSizer:
    """Keeps reference + settings on disk so the dashboard and command line share them."""

    def __init__(self, state_dir: str | Path = "ir_metal_state"):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.dir / "settings.json"
        self.ref_path = self.dir / "reference.npz"
        self.settings = Settings()
        if self.settings_path.exists():
            data = json.loads(self.settings_path.read_text())
            known = {k: v for k, v in data.items() if k in Settings.__dataclass_fields__}
            self.settings = Settings(**known)

    # ---- persistence
    def save_settings(self):
        self.settings_path.write_text(json.dumps(asdict(self.settings), indent=2))

    def update_settings(self, **kw):
        """Validate everything first, then apply; a rejected change leaves settings untouched."""
        defaults = Settings()
        new = asdict(self.settings)
        for k, v in kw.items():
            if v is None or k not in Settings.__dataclass_fields__:
                continue
            d = getattr(defaults, k)
            if isinstance(d, bool):
                if not isinstance(v, bool):
                    raise ValueError(f"{k} must be true or false")
            elif isinstance(d, (int, float)) and d is not None:
                try:
                    v = type(d)(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{k} must be a number")
                if v < 0:
                    raise ValueError(f"{k} must not be negative")
            new[k] = v
        if new["high_min_cm2"] <= new["low_max_cm2"]:
            raise ValueError("high threshold must be larger than low threshold")
        if new["polarity"] not in ("cold", "warm", "both"):
            raise ValueError("polarity must be cold, warm or both")
        if new["outline_fill"] not in ("auto", "holes", "hull", "none"):
            raise ValueError("outline_fill must be auto, holes, hull or none")
        self.settings = Settings(**new)
        self.save_settings()

    def load_reference(self):
        if not self.ref_path.exists():
            return None
        d = np.load(self.ref_path, allow_pickle=False)
        return d["mean"], str(d["ir_format"]), float(d["taken_at"]), int(d["n_frames"])

    # ---- actions
    def set_reference(self, frames: np.ndarray, ir_format: str) -> dict:
        ref = average(frames, ir_format)
        # raw frames are kept too, so scans can carry a copy and be re-measured offline later
        np.savez_compressed(self.ref_path, mean=ref, frames=np.asarray(frames), ir_format=np.array(ir_format),
                            taken_at=np.array(time.time()), n_frames=np.array(len(frames)))
        body = _body_mask(ref)
        return {"ok": True, "n_frames": int(len(frames)), "ir_format": ir_format,
                "mannequin_found": body is not None,
                "mannequin_fraction": round(float(body.mean()), 3) if body is not None else None}

    def measure(self, frames: np.ndarray, ir_format: str, settings: Settings | None = None):
        loaded = self.load_reference()
        if loaded is None:
            raise RuntimeError("no reference yet: capture one with NO metal on the mannequin first")
        ref, ref_fmt, taken_at, _ = loaded
        if ref_fmt != ir_format:
            raise RuntimeError(f"reference is {ref_fmt} but frames are {ir_format}; retake the reference")
        cur = average(frames, ir_format)
        if cur.shape != ref.shape:
            raise RuntimeError(f"frame size {cur.shape} != reference {ref.shape}; retake the reference")
        res, metal = detect(ref, cur, ir_format, settings or self.settings)
        res.n_frames = int(len(frames))
        res.reference_age_s = round(time.time() - taken_at, 1)
        if res.reference_age_s > 600:
            res.notes.append("reference is over 10 minutes old; retake it if results look noisy")
        return res, annotate(cur, metal, res)

    def diagnose(self, frames: np.ndarray, ir_format: str):
        loaded = self.load_reference()
        if loaded is None:
            raise RuntimeError("no reference yet: capture one with NO metal on the mannequin first")
        ref, ref_fmt, _, _ = loaded
        if ref_fmt != ir_format:
            raise RuntimeError(f"reference is {ref_fmt} but frames are {ir_format}; retake the reference")
        cur = average(frames, ir_format)
        stats, img = diagnose_diff(ref, cur, ir_format, self.settings)
        stats["ir_format"] = ir_format
        stats["n_frames"] = int(len(frames))
        stats["advice"] = explain_diagnosis(stats)
        if ir_format != "y16":
            stats["advice"].insert(0, f"IR is in {ir_format} mode, not raw y16: the camera's automatic "
                                      "contrast squeezes temperature differences. Check /api/health on the Pi.")
        return stats, img

    def calibrate(self, frames: np.ndarray, ir_format: str, known_area_cm2: float, note: str = "") -> dict:
        """Put a metal plate of known area on the mannequin, then call this."""
        s = Settings(**{**asdict(self.settings), "cm_per_px": None})
        res, _ = self.measure(frames, ir_format, settings=s)
        if not res.blobs:
            raise RuntimeError("no metal found for calibration; check the plate is in view and the reference was taken without it")
        px = res.blobs[0].area_px
        cpp = float(np.sqrt(known_area_cm2 / px))
        self.settings.cm_per_px = round(cpp, 4)
        self.settings.calibrated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.settings.calibration_note = note or f"{known_area_cm2:g} cm2 plate = {px} px"
        self.save_settings()
        return {"ok": True, "cm_per_px": self.settings.cm_per_px, "plate_area_px": px,
                "known_area_cm2": known_area_cm2, "other_blobs": len(res.blobs) - 1}

    def status(self) -> dict:
        loaded = self.load_reference()
        return {
            "has_reference": loaded is not None,
            "reference_age_s": round(time.time() - loaded[2], 1) if loaded else None,
            "reference_format": loaded[1] if loaded else None,
            "settings": asdict(self.settings),
        }


    def copy_reference_to(self, out_dir: str | Path) -> str | None:
        """Save the current reference's raw frames as <out_dir>/ir_reference.npz (for offline re-measuring)."""
        if not self.ref_path.exists():
            return None
        d = np.load(self.ref_path, allow_pickle=False)
        if "frames" not in d:
            return None  # reference taken by an older version; re-take it to get raw frames
        out = Path(out_dir) / "ir_reference.npz"
        np.savez_compressed(out, frames=d["frames"], ir_format=d["ir_format"], taken_at=d["taken_at"])
        return out.name


def measure_offline(ref_frames, ref_fmt, frames, fmt, settings: Settings):
    """Measure saved frames against saved reference frames with the given settings."""
    if ref_fmt != fmt:
        raise RuntimeError(f"reference is {ref_fmt} but frames are {fmt}")
    ref, cur = average(ref_frames, ref_fmt), average(frames, fmt)
    res, metal = detect(ref, cur, fmt, settings)
    res.n_frames = int(len(frames))
    return res, annotate(cur, metal, res)


def summarize_result(res: Result) -> dict:
    return {
        "level": res.level,
        "largest_area_cm2": res.largest_area_cm2,
        "largest_area_px": res.largest_area_px,
        "total_area_cm2": res.total_area_cm2,
        "pieces": len(res.blobs),
        "largest_size_cm": res.blobs[0].size_cm if res.blobs else None,
        "calibrated": res.calibrated,
        "reference_age_s": res.reference_age_s,
        "notes": res.notes,
        "size_level": res.size_level,
        "weapons": res.weapons,
        "objects": [(b.shape or {}).get("display", "suspicious object") for b in res.blobs],
    }


def capture_and_measure(sizer: "MetalSizer", base_url: str, out_dir: str | Path, tag: str,
                        seconds: float = 2.0, lock=None, frames_fmt=None) -> dict:
    """
    Grab a raw IR burst from the Pi, save it as <out_dir>/ir_<tag>.npz, and if a reference exists,
    measure it (ir_<tag>.json + ir_<tag>.png). Never raises: problems are reported in the result,
    so a scan is never lost because of the IR side.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tag": tag, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        frames, fmt = frames_fmt if frames_fmt is not None else fetch_burst(base_url, seconds)
    except Exception as e:
        summary["error"] = f"IR capture failed: {e}"
        return summary
    np.savez_compressed(out_dir / f"ir_{tag}.npz", frames=frames, ir_format=np.array(fmt))
    summary.update(file=f"ir_{tag}.npz", n_frames=int(len(frames)), ir_format=fmt)
    try:
        if lock is not None:
            with lock:
                res, img = sizer.measure(frames, fmt)
        else:
            res, img = sizer.measure(frames, fmt)
    except Exception as e:
        summary["measure_error"] = str(e)
        return summary
    (out_dir / f"ir_{tag}.json").write_text(json.dumps(res.to_dict(), indent=2))
    (out_dir / f"ir_{tag}.png").write_bytes(encode_png(img))
    summary.update(summarize_result(res), image=f"ir_{tag}.png")
    return summary


def encode_png(img_bgr: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- command line

def fetch_burst(base: str, seconds: float):
    import httpx
    r = httpx.get(f"{base.rstrip('/')}/api/ir/burst", params={"seconds": seconds}, timeout=seconds + 15)
    r.raise_for_status()
    d = np.load(io.BytesIO(r.content))
    return d["frames"], r.headers.get("X-IR-Format", "y16")


def load_frames_file(path) -> tuple[np.ndarray, str]:
    """Frames + format from an .npz saved by this module, the Pi recorder, or a scan folder."""
    d = np.load(path, allow_pickle=False)
    if "ir_format" in d:
        return d["frames"], str(d["ir_format"])
    meta = Path(path).with_suffix(".json")
    if meta.exists():
        try:
            return d["frames"], json.loads(meta.read_text()).get("ir_format", "y16")
        except Exception:
            pass
    return d["frames"], "y16"


def report(root: Path, remeasure: bool, settings: Settings, csv_path: Path | None = None) -> list[dict]:
    """Table of IR results for every scan folder under root that has ir_start.npz / ir_end.npz."""
    rows = []
    scan_dirs = sorted({p.parent for p in root.rglob("ir_*.npz") if p.name != "ir_reference.npz"})
    for d in scan_dirs:
        meta = {}
        for name in ("scan_meta.json", "ir_summary.json"):
            if (d / name).exists():
                try:
                    meta.update(json.loads((d / name).read_text()))
                except Exception:
                    pass
        row = {"scan": d.name, "folder": str(d.relative_to(root)) if d != root else ".",
               "view": meta.get("view", ""), "label": meta.get("label", ""), "notes": meta.get("notes", "")}
        ref = d / "ir_reference.npz"
        for tag in ("start", "end"):
            f = d / f"ir_{tag}.npz"
            if not f.exists():
                continue
            if remeasure:
                if not ref.exists():
                    row[f"{tag}_level"] = "no-ref"
                    continue
                rf, rfmt = load_frames_file(ref)
                fr, fmt = load_frames_file(f)
                try:
                    res, img = measure_offline(rf, rfmt, fr, fmt, settings)
                except RuntimeError as e:
                    row[f"{tag}_level"] = f"error: {e}"
                    continue
                (d / f"ir_{tag}_remeasured.png").write_bytes(encode_png(img))
                s = summarize_result(res)
            else:
                j = d / f"ir_{tag}.json"
                if not j.exists():
                    row[f"{tag}_level"] = "not measured"
                    continue
                r = json.loads(j.read_text())
                s = {"level": r["level"], "largest_area_cm2": r["largest_area_cm2"],
                     "largest_area_px": r["largest_area_px"], "pieces": len(r["blobs"])}
            row[f"{tag}_level"] = s["level"]
            row[f"{tag}_area_cm2"] = s["largest_area_cm2"]
            row[f"{tag}_area_px"] = s["largest_area_px"]
            row[f"{tag}_pieces"] = s["pieces"]
        if row.get("start_level") and row.get("end_level") and row["start_level"] != row["end_level"]:
            row["changed"] = "yes"
        rows.append(row)
    if csv_path and rows:
        import csv
        cols = ["scan", "folder", "view", "label", "notes", "start_level", "start_area_cm2", "start_area_px",
                "start_pieces", "end_level", "end_area_cm2", "end_area_px", "end_pieces", "changed"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return rows


def _frames_from_args(args, sizer: MetalSizer, which: str = "npz"):
    path = getattr(args, which, None)
    if path:
        return load_frames_file(path)
    if not args.pi:
        raise SystemExit("give --pi URL or --npz FILE")
    return fetch_burst(args.pi, args.seconds or sizer.settings.capture_seconds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default="ir_metal_state")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("reference", "calibrate", "measure", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("--pi", help="Pi camera server base URL, e.g. http://10.0.0.5:9000")
        p.add_argument("--npz", help="use a recorded clip instead of the Pi")
        p.add_argument("--seconds", type=float, default=None)
        if name == "calibrate":
            p.add_argument("--area-cm2", type=float)
            p.add_argument("--width-cm", type=float)
            p.add_argument("--height-cm", type=float)
        if name == "measure":
            p.add_argument("--ref-npz", help="offline: set the reference from this clip first")
            p.add_argument("--save", help="write the annotated image to this PNG")
            p.add_argument("--json", action="store_true")
        if name == "diagnose":
            p.add_argument("--save", default="ir_diagnose.png", help="difference image (default ir_diagnose.png)")
    p = sub.add_parser("levels")
    p.add_argument("--seconds", type=float, help="capture length for reference/measure (default 1; longer = less noise)")
    p.add_argument("--noise-k", type=float, help="threshold must exceed this many times the sensor noise (default 6)")
    p.add_argument("--min-delta-8bit", type=float, help="like --min-delta-c, for 8-bit IR frames (default 8)")
    p.add_argument("--local-background", choices=["on", "off"],
                   help="compare each spot with its surroundings (default on; handles uneven warming)")
    p.add_argument("--bg-block", type=int, help="size in px of the 'surroundings' for --local-background (default 24)")
    p.add_argument("--smooth", type=float, help="neighbour averaging in px for faint solid pieces (default 1; 0 = off)")
    p.add_argument("--motion-px", type=float,
                   help="tolerate the mannequin shifting this many px between reference and measurement (default 0.5; 0 = off)")
    p.add_argument("--ignore-hot", type=float,
                   help="ignore areas this many deg C hotter than the warm body, e.g. heating pads/lights (default 5; 0 = off)")
    p.add_argument("--half-contrast", choices=["on", "off"],
                   help="measure each piece's edge at half its own contrast, so faint and strong identical pieces match (default on)")
    p.add_argument("--low", type=float, help="below this many cm^2 = LOW")
    p.add_argument("--high", type=float, help="at or above this many cm^2 = HIGH")
    p.add_argument("--min-area-px", type=int)
    p.add_argument("--min-area-cm2", type=float,
                   help="ignore pieces smaller than this many cm^2 (needs calibration; 0 = off)")
    p.add_argument("--min-delta-c", type=float)
    p.add_argument("--polarity", choices=["cold", "warm", "both"],
                   help="which temperature change counts as metal (default cold)")
    p.add_argument("--fill", choices=["auto", "holes", "hull", "none"],
                   help="auto (default): fill thin metal outlines (rings) but keep solid pieces' real shape; "
                        "hull: fill every piece's convex outline; holes: only fill enclosed holes")
    p.add_argument("--close-px", type=int, help="gap size (px) joined when filling outlines (default 2)")
    p.add_argument("--body-mask", choices=["on", "off"],
                   help="only search the warm mannequin outline (turn off if the mannequin isn't heated)")
    sub.add_parser("status")
    p = sub.add_parser("report", help="table of IR results for every scan folder under a directory")
    p.add_argument("root", type=Path, help="e.g. outputs/")
    p.add_argument("--remeasure", action="store_true",
                   help="re-run the measurement with the CURRENT settings (after changing levels/polarity)")
    p.add_argument("--csv", type=Path, default=None, help="also write the table as CSV")
    args = ap.parse_args()

    sizer = MetalSizer(args.state_dir)
    if args.cmd == "report":
        rows = report(args.root, args.remeasure, sizer.settings, args.csv)
        if not rows:
            print(f"No scans with IR captures under {args.root}")
            return
        unit = "cm2" if sizer.settings.cm_per_px else "px"
        print(f"{'scan':28s} {'view':6s} {'label':14s} {'start':>8s} {'area':>7s} {'end':>8s} {'area':>7s}  notes")
        for r in rows:
            a0 = r.get(f"start_area_{unit}") if unit == "cm2" else r.get("start_area_px")
            a1 = r.get(f"end_area_{unit}") if unit == "cm2" else r.get("end_area_px")
            fmt_a = lambda a: "" if a is None else f"{a:.0f}"
            flag = "  <- changed during scan" if r.get("changed") else ""
            print(f"{r['scan'][:28]:28s} {str(r['view'])[:6]:6s} {str(r['label'])[:14]:14s} "
                  f"{str(r.get('start_level', '-')):>8s} {fmt_a(a0):>7s} {str(r.get('end_level', '-')):>8s} "
                  f"{fmt_a(a1):>7s}  {str(r['notes'])[:40]}{flag}")
        print(f"\n{len(rows)} scan(s); areas in {unit}" + (" (re-measured with current settings)" if args.remeasure else ""))
        if args.csv:
            print(f"CSV: {args.csv}")
        return
    if args.cmd == "status":
        print(json.dumps(sizer.status(), indent=2))
    elif args.cmd == "diagnose":
        frames, fmt = _frames_from_args(args, sizer)
        stats, img = sizer.diagnose(frames, fmt)
        Path(args.save).write_bytes(encode_png(img))
        for line in stats["advice"]:
            print(line)
        print(f"Image: {args.save}  (blue = colder than reference, red = warmer, black outline = detected)")
    elif args.cmd == "levels":
        sizer.update_settings(low_max_cm2=args.low, high_min_cm2=args.high,
                              min_area_px=args.min_area_px, min_area_cm2=args.min_area_cm2,
                              min_delta_c=args.min_delta_c,
                              polarity=args.polarity, outline_fill=args.fill, close_px=args.close_px,
                              capture_seconds=args.seconds, noise_k=args.noise_k,
                              min_delta_8bit=args.min_delta_8bit, bg_block_px=args.bg_block, smooth_px=args.smooth,
                              half_contrast_edges=None if args.half_contrast is None else args.half_contrast == "on",
                              ignore_hot_c=args.ignore_hot, motion_px=args.motion_px,
                              local_background=None if args.local_background is None else args.local_background == "on",
                              body_mask=None if args.body_mask is None else args.body_mask == "on")
        print(json.dumps(sizer.status()["settings"], indent=2))
    elif args.cmd == "reference":
        frames, fmt = _frames_from_args(args, sizer)
        print(json.dumps(sizer.set_reference(frames, fmt), indent=2))
    elif args.cmd == "calibrate":
        area = args.area_cm2 or (args.width_cm * args.height_cm if args.width_cm and args.height_cm else None)
        if not area:
            raise SystemExit("give --area-cm2, or --width-cm and --height-cm")
        frames, fmt = _frames_from_args(args, sizer)
        print(json.dumps(sizer.calibrate(frames, fmt, area), indent=2))
    elif args.cmd == "measure":
        if args.ref_npz:
            rf, rfmt = _frames_from_args(args, sizer, "ref_npz")
            sizer.set_reference(rf, rfmt)
        frames, fmt = _frames_from_args(args, sizer)
        res, img = sizer.measure(frames, fmt)
        if args.save:
            Path(args.save).write_bytes(encode_png(img))
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"THREAT LEVEL: {res.level}")
            if res.calibrated:
                print(f"  largest piece {res.largest_area_cm2} cm2, total {res.total_area_cm2} cm2, {len(res.blobs)} piece(s)")
            else:
                print(f"  largest piece {res.largest_area_px} px, total {res.total_area_px} px, {len(res.blobs)} piece(s)")
            for b in res.blobs:
                size = f"{b.size_cm[0]}x{b.size_cm[1]} cm box" if b.size_cm else f"{b.bbox[2]}x{b.bbox[3]} px box"
                print(f"   - {b.area_cm2 if b.area_cm2 is not None else b.area_px} {'cm2' if b.area_cm2 is not None else 'px'}"
                      f"  ({size}, change {b.mean_delta:+})")
            for n in res.notes:
                print(f"  note: {n}")
            if args.save:
                print(f"  image: {args.save}")


if __name__ == "__main__":
    main()