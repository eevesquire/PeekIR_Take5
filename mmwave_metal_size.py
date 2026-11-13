"""
mmWave metal size -> threat level, from sar_reconstruct.py output.

No calibration needed: the reconstruction grid is already in millimetres, so a piece's area in
cm^2 comes straight from how many image points it covers. Threat levels use the same cutoffs as
the IR side (read from ir_metal_state/settings.json), so the two can be compared directly.

Input: <scan folder>/sar_volume.npz, written by `sar_reconstruct.py ... --save-npz`
(sar_batch.py now adds --save-npz automatically). For older scans without it, the `reconstruct`
command re-runs sar_reconstruct.py with the settings stored in the scan's scan_meta.json.

How a scan is measured (all on the X-Y max-brightness image, mip_xy):
  1. optional depth gate: ignore returns whose brightest depth is outside a range (table, wall...)
  2. light smoothing (2 mm) to merge the radar's interference dots inside a piece
  3. background = median brightness of the scan (most of the scan is not metal)
  4. "cores": areas brighter than min_contrast x background (metal reflects far more than a
     plastic mannequin). Cores much weaker than the strongest one (side lobes/ringing around a
     strong piece) are dropped.
  5. each piece's edge = half of its OWN brightness (so the blurry fringe isn't counted and
     strong/weak pieces are measured the same way)
  6. area (cm^2) of each piece; threat level from the largest piece

Commands (run in the laptop folder, next to sar_reconstruct.py and ir_metal_size.py):
  python mmwave_metal_size.py measure outputs/front/metal/front_metal_000
  python mmwave_metal_size.py measure-all outputs/
  python mmwave_metal_size.py reconstruct outputs/test/check/test_check_017   # old scan without npz
  python mmwave_metal_size.py reconstruct-all outputs/                         # all old scans
  python mmwave_metal_size.py reconstruct-all outputs/ --viz                   # + viewer HTML where missing
  python mmwave_metal_size.py bake-outlines outputs/   # copies of the viewers with outlines drawn in
  python mmwave_metal_size.py levels --min-contrast 4 --depth-range 285 385
  python mmwave_metal_size.py report outputs/ --csv threat_report.csv          # IR + mmWave table

Files written into each scan folder: mmwave_result.json, mmwave_result.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
LEVELS = ("NONE", "LOW", "MEDIUM", "HIGH")
LEVEL_COLORS_BGR = {"NONE": (160, 160, 160), "LOW": (80, 200, 80), "MEDIUM": (0, 200, 255), "HIGH": (60, 60, 255)}


@dataclass
class MmSettings:
    min_contrast: float = 4.0       # a piece's core must be this many times brighter than the background
    rel_peak: float = 0.5           # ...and at least this fraction of the strongest piece (drops side
                                    # lobes and the mannequin's own, much weaker, reflection)
    edge_frac: float = 0.5          # piece edge at this fraction of its own brightness (half-max)...
    global_edge_frac: float = 0.35  # ...but never below this fraction of the scan's strongest return, so
                                    # the outline doesn't spread into the weaker surrounding glow
    join_mm: float = 4.0            # join parts of a piece separated by gaps up to ~2x this (radar stripes)
    smooth_mm: float = 2.0          # smoothing before measuring (merges interference dots)
    min_area_cm2: float = 2.0       # ignore pieces smaller than this
    depth_min_mm: float | None = None   # optional depth gate (mm): ignore returns outside this range
    depth_max_mm: float | None = None
    low_max_cm2: float | None = None    # None = use the IR cutoffs (ir_metal_state/settings.json)
    high_min_cm2: float | None = None


@dataclass
class MmPiece:
    area_cm2: float
    bbox_mm: tuple[float, float, float, float]   # x, y, width, height (mm, scan coordinates)
    centroid_mm: tuple[float, float]
    peak_contrast: float                         # piece brightness / background
    depth_mm: float | None                       # median depth of the brightest returns
    touches_edge: bool                           # piece is cut off by the scan border
    shape: dict | None = None                    # shape_classify.py: gun / knife / suspicious object


@dataclass
class MmResult:
    level: str
    largest_area_cm2: float
    total_area_cm2: float
    pieces: list[MmPiece] = field(default_factory=list)
    background: float = 0.0
    max_contrast: float = 0.0
    pixel_mm: tuple[float, float] = (0.0, 0.0)
    scan_mm: tuple[float, float] = (0.0, 0.0)
    cutoffs_cm2: tuple[float, float] = (25.0, 150.0)
    notes: list[str] = field(default_factory=list)
    size_level: str | None = None                # level from size alone (before weapons raised it)
    weapons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- settings

def _state_dir(state_dir: str | Path | None = None) -> Path:
    d = Path(state_dir or os.getenv("PEEKIR_METAL_STATE", str(REPO_ROOT / "ir_metal_state")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_settings(state_dir=None) -> MmSettings:
    p = _state_dir(state_dir) / "mmwave_settings.json"
    if p.exists():
        data = json.loads(p.read_text())
        return MmSettings(**{k: v for k, v in data.items() if k in MmSettings.__dataclass_fields__})
    return MmSettings()


def save_settings(s: MmSettings, state_dir=None) -> None:
    (_state_dir(state_dir) / "mmwave_settings.json").write_text(json.dumps(asdict(s), indent=2))


def cutoffs(s: MmSettings, state_dir=None) -> tuple[float, float]:
    """Threat cutoffs: mmWave's own if set, else the IR ones, else 25/150 cm^2."""
    low, high = 25.0, 150.0
    ir = _state_dir(state_dir) / "settings.json"
    if ir.exists():
        try:
            d = json.loads(ir.read_text())
            low, high = float(d.get("low_max_cm2", low)), float(d.get("high_min_cm2", high))
        except Exception:
            pass
    if s.low_max_cm2 is not None:
        low = s.low_max_cm2
    if s.high_min_cm2 is not None:
        high = s.high_min_cm2
    return low, high


# --------------------------------------------------------------------------- measurement

def measure_arrays(mip: np.ndarray, depth: np.ndarray | None, x_mm: np.ndarray, y_mm: np.ndarray,
                   s: MmSettings, cut: tuple[float, float]) -> tuple[MmResult, np.ndarray]:
    """mip/depth: (Y, X) arrays as saved by sar_reconstruct --save-npz. Returns (result, piece mask)."""
    import cv2
    notes: list[str] = []
    img = np.asarray(mip, np.float32).copy()
    ny, nx = img.shape
    dx = float(np.median(np.abs(np.diff(x_mm)))) if len(x_mm) > 1 else 1.0
    dy = float(np.median(np.abs(np.diff(y_mm)))) if len(y_mm) > 1 else 1.0
    px_cm2 = dx * dy / 100.0

    if depth is not None and (s.depth_min_mm is not None or s.depth_max_mm is not None):
        lo = -np.inf if s.depth_min_mm is None else s.depth_min_mm
        hi = np.inf if s.depth_max_mm is None else s.depth_max_mm
        out = (depth < lo) | (depth > hi)
        img[out] = 0.0
        if out.mean() > 0.5:
            notes.append(f"depth gate {lo}-{hi} mm removed {out.mean():.0%} of the scan")

    sm = img
    if s.smooth_mm > 0:
        sm = cv2.GaussianBlur(img, (0, 0), sigmaX=max(s.smooth_mm / dx, 0.3), sigmaY=max(s.smooth_mm / dy, 0.3))
    bg = float(np.median(sm))
    if bg <= 0:
        bg = float(np.mean(sm)) or 1.0
    contrast = sm / bg
    max_contrast = float(contrast.max())

    cores = (contrast >= s.min_contrast).astype(np.uint8)
    n, lab = cv2.connectedComponents(cores, connectivity=8)
    core_peaks = {}
    for i in range(1, n):
        v = sm[lab == i]
        core_peaks[i] = float(np.percentile(v, 95))
    mask = np.zeros_like(cores, dtype=bool)
    if core_peaks:
        strongest = max(core_peaks.values())
        for i, pk in core_peaks.items():
            if pk < s.rel_peak * strongest:
                continue                                       # side lobe of a stronger piece
            core = lab == i
            edge = max(s.edge_frac * pk, s.global_edge_frac * strongest)
            region = (sm >= edge).astype(np.uint8)
            _, lab2 = cv2.connectedComponents(region, connectivity=8)
            ids = np.unique(lab2[core])
            ids = ids[ids > 0]
            mask |= np.isin(lab2, ids)
    else:
        notes.append(f"no return stronger than {s.min_contrast:g}x background "
                     f"(strongest was {max_contrast:.1f}x)")

    if s.join_mm > 0 and mask.any():
        rx, ry = max(1, int(round(s.join_mm / dx))), max(1, int(round(s.join_mm / dy)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    n2, lab3, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    pieces: list[MmPiece] = []
    keep = np.zeros_like(mask)
    try:
        import shape_classify as _shapes
        shape_s = _shapes.load_settings()
    except ImportError:            # shape_classify.py missing: threat from size only
        _shapes = shape_s = None
    x0, y0 = float(np.min(x_mm)), float(np.min(y_mm))
    for i in range(1, n2):
        area = float(stats[i, cv2.CC_STAT_AREA]) * px_cm2
        if area < s.min_area_cm2:
            continue
        sel = lab3 == i
        keep |= sel
        bx, by, bw, bh = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                     cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        touches = bx == 0 or by == 0 or bx + bw >= nx or by + bh >= ny
        d = None
        if depth is not None:
            v = sm[sel]
            strong = sel & (sm >= np.percentile(v, 50))
            d = round(float(np.median(depth[strong])), 1)
        shape = None
        if _shapes is not None:
            try:      # cells in cm, scan coordinates
                shape = _shapes.classify_mask(sel, dx / 10.0, dy / 10.0, True, origin=(x0 / 10.0, y0 / 10.0),
                                              s=shape_s)
            except Exception as e:
                notes.append(f"shape check failed: {e}")
        pieces.append(MmPiece(
            area_cm2=round(area, 1),
            bbox_mm=(round(x0 + bx * dx, 1), round(y0 + by * dy, 1), round(bw * dx, 1), round(bh * dy, 1)),
            centroid_mm=(round(x0 + float(cents[i][0]) * dx, 1), round(y0 + float(cents[i][1]) * dy, 1)),
            peak_contrast=round(float(np.percentile(contrast[sel], 95)), 1),
            depth_mm=d,
            touches_edge=bool(touches),
            shape=shape,
        ))
    pieces.sort(key=lambda p: p.area_cm2, reverse=True)
    if any(p.touches_edge for p in pieces):
        notes.append("a piece touches the scan border: it may extend outside the scanned area "
                     "(its size is only the part inside)")

    largest = pieces[0].area_cm2 if pieces else 0.0
    low, high = cut
    level = "NONE" if not pieces else ("LOW" if largest < low else "MEDIUM" if largest < high else "HIGH")
    size_level, weapons = level, []
    if _shapes is not None and pieces:
        shapes = [p.shape for p in pieces]
        weapons = _shapes.weapons_of(shapes)
        level = _shapes.raise_level(level, shapes, shape_s)
        if weapons and level != size_level:
            notes.append(f"{' + '.join(weapons)} shape found: threat raised from {size_level} to {level}")
        if any(p.touches_edge and p.shape and p.shape.get("weapon") for p in pieces):
            notes.append("a weapon-shaped piece touches the scan border: its shape may be cut off")
    res = MmResult(level=level, largest_area_cm2=round(largest, 1),
                   total_area_cm2=round(sum(p.area_cm2 for p in pieces), 1), pieces=pieces,
                   background=round(bg, 1), max_contrast=round(max_contrast, 1),
                   pixel_mm=(round(dx, 3), round(dy, 3)),
                   scan_mm=(round(nx * dx, 1), round(ny * dy, 1)), cutoffs_cm2=(low, high), notes=notes,
                   size_level=size_level, weapons=weapons)
    return res, keep


def annotate(mip: np.ndarray, mask: np.ndarray, res: MmResult, px_per_mm: float = 2.0) -> np.ndarray:
    """Max-brightness image (log scale), pieces outlined, true mm aspect, y up. BGR uint8."""
    import cv2
    img = np.log10(np.maximum(np.asarray(mip, np.float32), 1e-12))
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.9)
    u8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    w = max(64, int(round(res.scan_mm[0] * px_per_mm)))
    h = max(64, int(round(res.scan_mm[1] * px_per_mm)))
    col = cv2.resize(col, (w, h), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    col, m = np.flipud(col).copy(), np.flipud(m).copy()          # y up, like the existing plots
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(col, contours, -1, (255, 255, 255), 2)
    x0 = res.pieces and min(p.bbox_mm[0] for p in res.pieces)
    for p in res.pieces:
        bx = int((p.bbox_mm[0] - 0) * px_per_mm)
        by = h - int((p.bbox_mm[1] + p.bbox_mm[3]) * px_per_mm)
        lbl = f"{p.area_cm2:.0f} cm2"
        if p.shape:
            lbl = f"{p.shape['display'].upper() if p.shape.get('weapon') else 'object'} " + lbl
        cv2.putText(col, lbl, (max(2, bx), max(36, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(col, (0, 0), (w, 26), (0, 0, 0), -1)
    weap = ("  " + " + ".join(w.upper() for w in res.weapons)) if res.weapons else ""
    cv2.putText(col, f"mmWave THREAT: {res.level}{weap}   largest {res.largest_area_cm2:.0f} cm2",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LEVEL_COLORS_BGR[res.level], 2, cv2.LINE_AA)
    return col


def measure_folder(scan_dir: str | Path, s: MmSettings | None = None, state_dir=None,
                   write: bool = True) -> MmResult:
    """Measure <scan_dir>/sar_volume.npz; write mmwave_result.json/.png into the folder."""
    import cv2
    scan_dir = Path(scan_dir)
    npz = scan_dir / "sar_volume.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found: reconstruct with --save-npz "
                                f"(python mmwave_metal_size.py reconstruct {scan_dir})")
    s = s or load_settings(state_dir)
    d = np.load(npz)
    depth = d["depth_xy"] if "depth_xy" in d else None
    res, mask = measure_arrays(d["mip_xy"], depth, d["x_mm"], d["y_mm"], s, cutoffs(s, state_dir))
    if write:
        out = res.to_dict()
        out["outlines_mm"] = outlines_mm(mask, d["x_mm"], d["y_mm"])
        (scan_dir / "mmwave_result.json").write_text(json.dumps(out, indent=2))
        cv2.imwrite(str(scan_dir / "mmwave_result.png"), annotate(d["mip_xy"], mask, res))
    return res


def outlines_mm(mask: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray) -> list[list[list[float]]]:
    """Piece outlines as polygons in scan mm coordinates (same axes as the Plotly slice viewer)."""
    import cv2
    m = np.asarray(mask, np.uint8)
    if not m.any():
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xs, ys = np.asarray(x_mm, float), np.asarray(y_mm, float)
    polys = []
    for c in contours:
        c = cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        polys.append([[round(float(xs[min(max(px, 0), len(xs) - 1)]), 1),
                       round(float(ys[min(max(py, 0), len(ys) - 1)]), 1)] for px, py in c])
    return polys


# --------------------------------------------------------------------------- per-scan threat summary (dashboard)

def find_scan_dir(root: str | Path, name: str) -> Path | None:
    """Scan folder called `name` under root (also tries without a trailing _N, as viewer HTMLs get one)."""
    root = Path(root)
    if not root.exists():
        return None
    cands = [name]
    m = re.match(r"^(.*)_\d+$", name)
    if m:
        cands.append(m.group(1))
    for n in cands:
        hits = [p for p in root.rglob(n) if p.is_dir() and "_failed" not in p.parts and not p.name.startswith(".")]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None


def list_scans(root: str | Path, limit: int = 200) -> list[dict]:
    """Scan folders that have any IR or mmWave result, newest first."""
    root = Path(root)
    if not root.exists():
        return []
    dirs = {p.parent for pat in ("ir_start.json", "ir_start.npz", "mmwave_result.json", "sar_volume.npz")
            for p in root.rglob(pat) if "_failed" not in p.parts and ".ir_pending" not in p.parts}
    out = [{"name": d.name, "folder": str(d.relative_to(root)), "mtime": d.stat().st_mtime} for d in dirs]
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out[:limit]


def scan_threat_summary(scan_dir: str | Path, state_dir=None) -> dict:
    """IR + mmWave results and the combined verdict for one scan folder (measures mmWave if needed)."""
    scan_dir = Path(scan_dir)
    ir = None
    if (scan_dir / "ir_start.json").exists():
        r = json.loads((scan_dir / "ir_start.json").read_text())
        ir = {"level": r["level"], "largest_area_cm2": r.get("largest_area_cm2"),
              "largest_area_px": r.get("largest_area_px"), "pieces": len(r.get("blobs", [])),
              "calibrated": r.get("calibrated"), "notes": r.get("notes", []),
              "image": (scan_dir / "ir_start.png").exists(),
              "weapons": r.get("weapons", []), "size_level": r.get("size_level"),
              "objects": _objects(r.get("blobs", []), "area_cm2")}
    elif (scan_dir / "ir_start.npz").exists():
        ir = {"level": None, "note": "IR captured but not measured (no reference was set)"}
    mm = None
    res_p, npz_p = scan_dir / "mmwave_result.json", scan_dir / "sar_volume.npz"
    if npz_p.exists() and _mm_result_stale(res_p, npz_p, state_dir):
        try:
            measure_folder(scan_dir, state_dir=state_dir)
        except Exception as e:
            mm = {"level": None, "note": f"mmWave measurement failed: {e}"}
    if mm is None and res_p.exists():
        r = json.loads(res_p.read_text())
        mm = {"level": r["level"], "largest_area_cm2": r.get("largest_area_cm2"), "pieces": len(r.get("pieces", [])),
              "notes": r.get("notes", []), "image": (scan_dir / "mmwave_result.png").exists(),
              "outlines": len(r.get("outlines_mm", [])),
              "weapons": r.get("weapons", []), "size_level": r.get("size_level"),
              "objects": _objects(r.get("pieces", []), "area_cm2")}
    elif mm is None:
        mm = {"level": None, "note": "no mmWave volume (sar_volume.npz) for this scan"}
    verdict, level = combine((ir or {}).get("level"), mm.get("level"))
    weapons = []
    for w in (ir or {}).get("weapons", []) + mm.get("weapons", []):
        if w not in [x["type"] for x in weapons]:
            weapons.append({"type": w, "sensors": [n for n, part in (("IR", ir), ("mmWave", mm))
                                                   if w in (part or {}).get("weapons", [])]})
    return {"name": scan_dir.name, "ir": ir, "mmwave": mm, "verdict": verdict, "level": level,
            "weapons": weapons, "headline": headline(verdict, weapons)}


def _objects(parts: list[dict], area_key: str) -> list[dict]:
    """[{display, weapon, shape, area_cm2}] for each piece/blob, largest first."""
    out = []
    for p in parts:
        sh = p.get("shape") or {}
        out.append({"display": sh.get("display", "suspicious object"), "weapon": sh.get("weapon"),
                    "shape": sh.get("shape_name"), "uncertain": bool(sh.get("uncertain")),
                    "area_cm2": p.get(area_key), "area_px": p.get("area_px")})
    return out


def headline(verdict: str, weapons: list[dict]) -> str:
    """e.g. 'GUN detected (IR + mmWave)' / 'possible KNIFE (mmWave only)' / ''."""
    parts = []
    for w in weapons:
        both = len(w["sensors"]) == 2
        parts.append((f"{w['type'].upper()} detected" if both else f"possible {w['type'].upper()}")
                     + f" ({' + '.join(w['sensors'])}{'' if both else ' only'})")
    return " · ".join(parts)


def _mm_result_stale(res_p: Path, npz_p: Path, state_dir=None) -> bool:
    """Re-measure when there is no result, the volume or settings are newer, or it predates shape labels."""
    if not res_p.exists():
        return True
    t = res_p.stat().st_mtime
    sd = _state_dir(state_dir)
    newer = [npz_p] + [sd / f for f in ("mmwave_settings.json", "shape_settings.json", "settings.json")]
    if any(f.exists() and f.stat().st_mtime > t for f in newer):
        return True
    try:
        return "size_level" not in json.loads(res_p.read_text())
    except Exception:
        return True


def plotly_overlay(scan_dir: str | Path) -> dict | None:
    """Plotly layout shapes + caption for the slice viewer: piece outlines and the threat level."""
    p = Path(scan_dir) / "mmwave_result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if "outlines_mm" not in r or "size_level" not in r:   # measured by an older version: redo
        try:
            measure_folder(scan_dir)
            r = json.loads(p.read_text())
        except Exception:
            return None
    color = {"NONE": "#bbbbbb", "LOW": "#3dcc7a", "MEDIUM": "#ffc14d", "HIGH": "#ff6b6b"}[r["level"]]
    shapes = [{"type": "path", "xref": "x", "yref": "y",
               "path": "M " + " L ".join(f"{x},{y}" for x, y in poly) + " Z",
               "line": {"color": "white", "width": 2}, "fillcolor": "rgba(0,0,0,0)"} for poly in r["outlines_mm"]]
    weap = r.get("weapons") or []
    caption = (f"<span style='color:{color}'><b>mmWave THREAT: {r['level']}"
               + (" — " + " + ".join(w.upper() for w in weap) if weap else "") + "</b></span>"
               + (f"  largest {r['largest_area_cm2']:.0f} cm²" if r.get("pieces") else "  no metal found"))
    annotations = []
    for piece in r.get("pieces", []):
        bx, by, bw, bh = piece["bbox_mm"]
        sh = piece.get("shape") or {}
        is_w = bool(sh.get("weapon"))
        name = sh["display"].upper() if is_w else ("suspicious object" if sh else "")
        annotations.append({"xref": "x", "yref": "y", "x": bx, "y": by + bh, "xanchor": "left", "yanchor": "bottom",
                            "showarrow": False,
                            "font": {"color": "#ff6b6b" if is_w else "white", "size": 13 if is_w else 11},
                            "text": (f"<b>{name}</b> · " if is_w else (f"{name} · " if name else ""))
                                    + f"{piece['area_cm2']:.0f} cm²"})
    return {"shapes": shapes, "annotations": annotations, "caption": caption, "level": r["level"]}


# --------------------------------------------------------------------------- reconstruction of old scans

def write_manual_meta(scan_dir: Path, a) -> Path:
    """For old scans without scan_meta.json: record the settings given on the command line."""
    need = {"width_mm": a.width_mm, "height_mm": a.height_mm, "num_frames": a.frames, "num_rows": a.rows,
            "speed_mms": a.speed, "frame_periodicity_ms": a.periodicity, "y_step_mm": a.y_step}
    flags = {"width_mm": "--width-mm", "height_mm": "--height-mm", "num_frames": "--frames", "num_rows": "--rows",
             "speed_mms": "--speed", "frame_periodicity_ms": "--periodicity", "y_step_mm": "--y-step"}
    missing = [flags[k] for k, v in need.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"no scan_meta.json, so the scan settings must be given: missing {' '.join(missing)} "
            "(also --snake if the gantry scanned back and forth, and --zstart/--zend if not 285/385)")
    meta = {"name": scan_dir.name, "view": a.view or "", "label": a.label or "", "notes": a.notes or "",
            "params": {**need, "samples": a.samples, "snake": bool(a.snake)},
            "reconstruction": {"zstart": a.zstart, "zend": a.zend, "zstep": a.zstep, "zindex": None,
                               "algo": a.algo, "window": "none"},
            "created_by": "mmwave_metal_size.py reconstruct (settings given by hand)"}
    out = scan_dir / "scan_meta.json"
    out.write_text(json.dumps(meta, indent=2))
    return out


def reconstruct_argv(scan_dir: Path, sar_reconstruct: Path, viz_dir: Path | None = None) -> list[str]:
    """Build the sar_reconstruct.py command from the scan's scan_meta.json (+ --save-npz)."""
    meta_p = scan_dir / "scan_meta.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"{meta_p} not found (needed for the scan settings)")
    meta = json.loads(meta_p.read_text())
    p, r = meta.get("params", {}), meta.get("reconstruction", {}) or {}
    argv = [sys.executable, "-X", "utf8", str(Path(sar_reconstruct).resolve()), "--folder", str(Path(scan_dir).resolve()),
            "--frames", str(p["num_frames"]), "--rows", str(p["num_rows"]),
            "--samples", str(p.get("samples", 512)), "--speed", str(p["speed_mms"]),
            "--periodicity", str(p["frame_periodicity_ms"]), "--y-step", str(p["y_step_mm"]),
            "--scan-width", str(p["width_mm"]), "--scan-height", str(p["height_mm"]),
            "--save-npz", "--xyonly"]
    if p.get("snake"):
        argv.append("--snake")
    for key, flag in (("zstart", "--zstart"), ("zend", "--zend"), ("zstep", "--zstep"),
                      ("zindex", "--zindex"), ("algo", "--algo"), ("window", "--window")):
        if r.get(key) not in (None, ""):
            argv += [flag, str(r[key])]
    if viz_dir is not None:                       # also write the Plotly slice viewer <scan>.html
        argv += ["--plotly", "--viz-html-root", str(Path(viz_dir).resolve())]
    return argv


def reconstruct_folder(scan_dir: str | Path, sar_reconstruct: Path | None = None, viz_dir: Path | None = None) -> None:
    scan_dir = Path(scan_dir)
    sar_reconstruct = Path(sar_reconstruct or (REPO_ROOT / "sar_reconstruct.py"))
    if not sar_reconstruct.exists():
        raise FileNotFoundError(f"{sar_reconstruct} not found; pass --sar-reconstruct PATH")
    argv = reconstruct_argv(scan_dir, sar_reconstruct, viz_dir)
    env = dict(os.environ, MPLBACKEND="Agg")       # never open a window
    print("  " + " ".join(argv[3:]))
    with open(scan_dir / "reconstruct_npz.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(argv, cwd=str(scan_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0 or not (scan_dir / "sar_volume.npz").exists():
        raise RuntimeError(f"reconstruction failed (exit {rc}); see {scan_dir / 'reconstruct_npz.log'}")
    if viz_dir is not None and find_viewer(viz_dir, scan_dir.name) is None:
        raise RuntimeError(f"no viewer HTML was written (is plotly installed?); see {scan_dir / 'reconstruct_npz.log'}")


def find_viewer(viz_dir: str | Path, name: str) -> Path | None:
    """visualizations/<name>.html (or <name>_N.html, as sar_reconstruct adds _N when the name is taken)."""
    viz_dir = Path(viz_dir)
    p = viz_dir / f"{name}.html"
    if p.is_file():
        return p
    hits = [q for q in viz_dir.glob(f"{name}_*.html") if re.fullmatch(re.escape(name) + r"_\d+", q.stem)]
    return max(hits, key=lambda q: q.stat().st_mtime) if hits else None


_OVERLAY_JS = """
<script id="peekir-threat-overlay">
(function () {
  var ov = __OVERLAY__;
  var tries = 0;
  function apply() {
    var gd = document.querySelector('.js-plotly-plot');
    if (!gd || !window.Plotly || !gd.layout) {
      if (tries++ < 200) setTimeout(apply, 50);
      return;
    }
    var base = (gd.layout.title && (gd.layout.title.text || gd.layout.title)) || 'SAR Reconstruction';
    if (typeof base !== 'string') base = 'SAR Reconstruction';
    window.Plotly.relayout(gd, {
      shapes: (gd.layout.shapes || []).concat(ov.shapes || []),
      annotations: (gd.layout.annotations || []).concat(ov.annotations || []),
      'title.text': base + '<br>' + ov.caption
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', apply);
  else apply();
})();
</script>
"""


def bake_outlines(html: str, overlay: dict) -> str:
    """Viewer HTML with the outlines + threat caption added (works when opened directly in a browser)."""
    script = _OVERLAY_JS.replace("__OVERLAY__", json.dumps(overlay).replace("</", "<\\/"))
    i = html.lower().rfind("</body>")
    return html[:i] + script + html[i:] if i >= 0 else html + script


# --------------------------------------------------------------------------- combined IR + mmWave report

ORDER = {lv: i for i, lv in enumerate(LEVELS)}


def combine(ir_level: str | None, mm_level: str | None) -> tuple[str, str]:
    """(verdict, level). Both see metal -> CONFIRMED at the higher level; one -> POSSIBLE."""
    ir_hit = ir_level not in (None, "NONE")
    mm_hit = mm_level not in (None, "NONE")
    if ir_hit and mm_hit:
        return "CONFIRMED", max(ir_level, mm_level, key=lambda l: ORDER[l])
    if ir_hit:
        return ("POSSIBLE (IR only)" if mm_level is not None else "IR only (no mmWave result)"), ir_level
    if mm_hit:
        return ("POSSIBLE (mmWave only)" if ir_level is not None else "mmWave only (no IR result)"), mm_level
    if ir_level is None and mm_level is None:
        return "no results", "-"
    return "CLEAR", "NONE"


def report(root: Path, csv_path: Path | None = None) -> list[dict]:
    root = Path(root)
    dirs = sorted({p.parent for pat in ("scan_meta.json", "ir_start.npz", "sar_volume.npz", "mmwave_result.json")
                   for p in root.rglob(pat) if "_failed" not in p.parts})
    rows = []
    for d in dirs:
        meta = {}
        for name in ("scan_meta.json", "ir_summary.json"):
            if (d / name).exists():
                try:
                    meta.update(json.loads((d / name).read_text()))
                except Exception:
                    pass
        ir = None
        if (d / "ir_start.json").exists():
            ir = json.loads((d / "ir_start.json").read_text())
        mm = None
        if (d / "mmwave_result.json").exists():
            mm = json.loads((d / "mmwave_result.json").read_text())
        verdict, level = combine(ir["level"] if ir else None, mm["level"] if mm else None)
        rows.append({
            "scan": d.name, "folder": str(d.relative_to(root)) if d != root else ".",
            "view": meta.get("view", ""), "label": meta.get("label", ""), "notes": meta.get("notes", ""),
            "ir_level": ir["level"] if ir else "", "ir_area_cm2": ir.get("largest_area_cm2") if ir else "",
            "mm_level": mm["level"] if mm else "", "mm_area_cm2": mm.get("largest_area_cm2") if mm else "",
            "verdict": verdict, "level": level,
        })
    if csv_path and rows:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


# --------------------------------------------------------------------------- command line

def _print_result(name: str, r: MmResult) -> None:
    print(f"{name}: mmWave THREAT {r.level}  largest {r.largest_area_cm2} cm2, {len(r.pieces)} piece(s), "
          f"strongest return {r.max_contrast}x background")
    for p in r.pieces:
        depth = f", depth ~{p.depth_mm:.0f} mm" if p.depth_mm is not None else ""
        edge = "  [touches scan edge]" if p.touches_edge else ""
        kind = f"{p.shape['display'].upper() if p.shape.get('weapon') else p.shape['shape_name']}: " if p.shape else ""
        print(f"   - {kind}{p.area_cm2} cm2  ({p.bbox_mm[2]:.0f} x {p.bbox_mm[3]:.0f} mm box at "
              f"x={p.bbox_mm[0]:.0f}, y={p.bbox_mm[1]:.0f}; {p.peak_contrast}x background{depth}){edge}")
    for n in r.notes:
        print(f"   note: {n}")


def _folders_with(root: Path, name: str) -> list[Path]:
    return sorted({p.parent for p in Path(root).rglob(name) if "_failed" not in p.parts})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default=None, help="settings folder (default ./ir_metal_state, shared with IR)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("measure", help="measure one or more scan folders")
    p.add_argument("folders", nargs="+", type=Path)
    p = sub.add_parser("measure-all", help="measure every scan folder under ROOT that has sar_volume.npz")
    p.add_argument("root", type=Path)
    for name in ("reconstruct", "reconstruct-all"):
        p = sub.add_parser(name, help="re-run sar_reconstruct.py with --save-npz from scan_meta.json, then measure")
        p.add_argument("target", type=Path)
        p.add_argument("--sar-reconstruct", type=Path, default=None)
        p.add_argument("--force", action="store_true", help="also redo folders that already have sar_volume.npz")
        p.add_argument("--viz", action="store_true",
                       help="also make the Plotly slice viewer (visualizations/<scan>.html) for scans that don't have one")
        p.add_argument("--viz-dir", type=Path, default=REPO_ROOT / "visualizations",
                       help="where viewer HTMLs go (default: visualizations/ next to this file)")
        g = p.add_argument_group("settings for old scans WITHOUT scan_meta.json (written into one)")
        g.add_argument("--width-mm", type=float); g.add_argument("--height-mm", type=float)
        g.add_argument("--frames", type=int); g.add_argument("--rows", type=int)
        g.add_argument("--samples", type=int, default=512)
        g.add_argument("--speed", type=float, help="mm/s"); g.add_argument("--periodicity", type=float, help="ms")
        g.add_argument("--y-step", type=float, help="mm"); g.add_argument("--snake", action="store_true")
        g.add_argument("--zstart", default="285"); g.add_argument("--zend", default="385")
        g.add_argument("--zstep", default=None); g.add_argument("--algo", default="mf")
        g.add_argument("--view", default=None); g.add_argument("--label", default=None); g.add_argument("--notes", default=None)
    p = sub.add_parser("levels", help="show/change mmWave settings")
    p.add_argument("--min-contrast", type=float)
    p.add_argument("--rel-peak", type=float)
    p.add_argument("--edge-frac", type=float)
    p.add_argument("--global-edge-frac", type=float, help="piece edge never below this fraction of the strongest return (default 0.35)")
    p.add_argument("--smooth-mm", type=float)
    p.add_argument("--join-mm", type=float, help="join parts of a piece separated by small gaps (default 4)")
    p.add_argument("--min-area-cm2", type=float)
    p.add_argument("--depth-range", type=float, nargs=2, metavar=("MIN_MM", "MAX_MM"))
    p.add_argument("--no-depth-range", action="store_true")
    p.add_argument("--low", type=float, help="own LOW cutoff (default: same as IR)")
    p.add_argument("--high", type=float, help="own HIGH cutoff (default: same as IR)")
    p.add_argument("--use-ir-cutoffs", action="store_true", help="go back to using the IR cutoffs")
    p = sub.add_parser("bake-outlines", help="save copies of the viewer HTMLs with the metal outlines drawn in")
    p.add_argument("root", type=Path, help="folder with the scan folders (e.g. outputs/)")
    p.add_argument("--viz-dir", type=Path, default=REPO_ROOT / "visualizations")
    p.add_argument("--out", type=Path, default=None, help="default: <viz-dir>/outlined/")
    p = sub.add_parser("report", help="combined IR + mmWave table for all scans under ROOT")
    p.add_argument("root", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    s = load_settings(args.state_dir)
    if args.cmd == "measure":
        for f in args.folders:
            try:
                _print_result(f.name, measure_folder(f, s, args.state_dir))
            except Exception as e:
                print(f"{f.name}: {e}")
    elif args.cmd == "measure-all":
        dirs = _folders_with(args.root, "sar_volume.npz")
        if not dirs:
            print(f"No sar_volume.npz under {args.root}. Run: python mmwave_metal_size.py reconstruct-all {args.root}")
        for d in dirs:
            try:
                _print_result(d.name, measure_folder(d, s, args.state_dir))
            except Exception as e:
                print(f"{d.name}: {e}")
    elif args.cmd in ("reconstruct", "reconstruct-all"):
        if args.cmd == "reconstruct":
            dirs = [args.target]
        else:   # scans made by sar_batch (scan_meta.json) and older ones (just raw row files)
            dirs = sorted(set(_folders_with(args.target, "scan_meta.json")) | set(_folders_with(args.target, "row_1*.bin")))
        for d in dirs:
            if not (d / "scan_meta.json").exists():
                try:
                    write_manual_meta(d, args)
                    print(f"{d.name}: no scan_meta.json; wrote one from the settings you gave")
                except FileNotFoundError as e:
                    print(f"{d.name}: {e}")
                    continue
            need_viz = args.viz and find_viewer(args.viz_dir, d.name) is None
            if (d / "sar_volume.npz").exists() and not args.force and not need_viz and args.cmd == "reconstruct-all":
                print(f"{d.name}: already has sar_volume.npz" + (" and a viewer" if args.viz else "")
                      + " (use --force to redo)")
            else:
                print(f"{d.name}: reconstructing" + (" + viewer" if need_viz or (args.viz and args.force) else "")
                      + " (this can take a few minutes)...")
                try:
                    reconstruct_folder(d, args.sar_reconstruct,
                                       args.viz_dir if (need_viz or (args.viz and args.force)) else None)
                except Exception as e:
                    print(f"{d.name}: {e}")
                    continue
            try:
                _print_result(d.name, measure_folder(d, s, args.state_dir))
            except Exception as e:
                print(f"{d.name}: {e}")
    elif args.cmd == "levels":
        upd = dict(min_contrast=args.min_contrast, rel_peak=args.rel_peak, edge_frac=args.edge_frac,
                   global_edge_frac=args.global_edge_frac,
                   smooth_mm=args.smooth_mm, join_mm=args.join_mm, min_area_cm2=args.min_area_cm2, low_max_cm2=args.low,
                   high_min_cm2=args.high)
        for k, v in upd.items():
            if v is not None:
                setattr(s, k, v)
        if args.depth_range:
            s.depth_min_mm, s.depth_max_mm = sorted(args.depth_range)
        if args.no_depth_range:
            s.depth_min_mm = s.depth_max_mm = None
        if args.use_ir_cutoffs:
            s.low_max_cm2 = s.high_min_cm2 = None
        lo, hi = cutoffs(s, args.state_dir)
        if hi <= lo:
            raise SystemExit("HIGH cutoff must be larger than LOW cutoff")
        save_settings(s, args.state_dir)
        print(json.dumps({**asdict(s), "cutoffs_in_use_cm2": [lo, hi]}, indent=2))
    elif args.cmd == "bake-outlines":
        out = Path(args.out or (Path(args.viz_dir) / "outlined"))
        htmls = sorted(Path(args.viz_dir).glob("*.html"))
        if not htmls:
            print(f"No viewer HTMLs in {args.viz_dir}. Make them with: reconstruct-all {args.root} --viz")
        out.mkdir(parents=True, exist_ok=True)
        for h in htmls:
            d = find_scan_dir(args.root, h.stem)
            if d is None:
                print(f"{h.name}: no scan folder called {h.stem} under {args.root}; skipped")
                continue
            if not (d / "sar_volume.npz").exists():
                print(f"{h.name}: {d.name} has no sar_volume.npz; run reconstruct-all {args.root} first; skipped")
                continue
            try:
                scan_threat_summary(d, state_dir=args.state_dir)       # measures if needed
                ov = plotly_overlay(d)
                (out / h.name).write_text(bake_outlines(h.read_text(encoding="utf-8"), ov), encoding="utf-8")
                print(f"{h.name}: {ov['level']} -> {out / h.name}")
            except Exception as e:
                print(f"{h.name}: {e}")
    elif args.cmd == "report":
        rows = report(args.root, args.csv)
        if not rows:
            print(f"No scans under {args.root}")
            return
        f = lambda v: "" if v in ("", None) else (f"{v:.0f}" if isinstance(v, (int, float)) else str(v))
        print(f"{'scan':24s} {'view':6s} {'label':12s} {'IR':>7s} {'cm2':>5s} {'mmWave':>7s} {'cm2':>5s}  verdict")
        for r in rows:
            print(f"{r['scan'][:24]:24s} {str(r['view'])[:6]:6s} {str(r['label'])[:12]:12s} "
                  f"{r['ir_level'] or '-':>7s} {f(r['ir_area_cm2']):>5s} {r['mm_level'] or '-':>7s} "
                  f"{f(r['mm_area_cm2']):>5s}  {r['verdict']} {'' if r['level'] in ('-', 'NONE') else r['level']}")
        print(f"\n{len(rows)} scan(s)" + (f"; CSV: {args.csv}" if args.csv else ""))


if __name__ == "__main__":
    main()