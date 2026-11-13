"""
Training-data export for the next team.

Copies each scan's raw sensor data into a clean, consistently organised training set, with
text-free images and machine-readable labels. The demo outputs (annotated images with text and
outlines) stay in the scan folders; nothing in training_data/ has text drawn on it.

Layout (see the README.md written into the folder):

  training_data/
    README.md
    index.csv                       one row per sample
    paired/                         IR and mmWave from the same scene (sar_batch.py scans)
      front_metal_training/  side_metal_training/  front_nometal_training/  side_nometal_training/
        <sample_id>/
          label.json
          ir/      reference.npz start.npz end.npz  (raw frames)
                   reference.png start.png end.png (clean, no text)  mask_start.png
          mmwave/  sar_volume.npz  mip.png  depth.png (clean, no text)  mask.png
    ir_only/      same class folders, ir/ only      (dashboard "Save IR training sample", capture-ir)
    mmwave_only/  same class folders, mmwave/ only  (sar_batch.py --no-ir)

Commands (run in the laptop folder):
  python training_export.py export outputs/front/metal/front_metal_000      # one scan
  python training_export.py export-all outputs/                             # every scan not yet exported
  python training_export.py capture-ir --pi http://<pi>:9000 --view front --metal yes --object plateA
  python training_export.py import-legacy-ir --src knife_dataset            # old colourised IR images
  python training_export.py summary                                         # counts per section/class

Used automatically by sar_batch.py (after each scan's mmWave measurement) and by the dashboard.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.getenv("PEEKIR_TRAINING_DIR", str(REPO_ROOT / "training_data")))
SECTIONS = ("paired", "ir_only", "mmwave_only")
INDEX_FIELDS = ["sample_id", "section", "class", "view", "metal", "object", "true_area_cm2", "notes",
                "ir_level", "ir_area_cm2", "mmwave_level", "mmwave_area_cm2", "scene_id",
                "created", "source", "path", "weapon_type", "ir_detected", "mmwave_detected"]
WEAPON_TYPES = ("gun", "knife", "other", "none")   # ground truth: what was REALLY on the mannequin
_NO_METAL_WORDS = ("nometal", "no_metal", "no-metal", "empty", "none", "clear", "baseline")


# --------------------------------------------------------------------------- labels

def parse_weapon_type(value, metal: bool | None = None) -> str:
    """Ground-truth weapon type: gun / knife / other / none, or '' when not given.
    No metal at all means 'none'."""
    v = str(value or "").strip().lower()
    if not v:
        return "none" if metal is False else ""
    if v not in WEAPON_TYPES:
        raise ValueError(f"weapon type must be one of {', '.join(WEAPON_TYPES)} (got {value!r})")
    return v


def _detected(parts: list[dict] | None) -> str:
    """'gun+suspicious object' style summary of what the detector called each piece (largest first)."""
    return "+".join((p.get("shape") or {}).get("display", "suspicious object") for p in (parts or []))


def infer_metal(label: str | None, explicit: bool | None = None) -> bool | None:
    """Metal yes/no from an explicit flag, else from the scan label. None = can't tell."""
    if explicit is not None:
        return explicit
    lab = (label or "").strip().lower()
    if not lab:
        return None
    if any(w in lab for w in _NO_METAL_WORDS):
        return False
    if "metal" in lab:
        return True
    return None


def parse_yes_no(v) -> bool | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("y", "yes", "true", "1", "metal"):
        return True
    if s in ("n", "no", "false", "0", "nometal", "no_metal", "none"):
        return False
    raise ValueError(f"metal must be yes or no (got {v!r})")


def class_name(view: str, metal: bool) -> str:
    v = re.sub(r"[^A-Za-z0-9_-]+", "_", (view or "unknown").strip().lower()) or "unknown"
    return f"{v}_{'metal' if metal else 'nometal'}_training"


# --------------------------------------------------------------------------- clean images

def _png(path: Path, img: np.ndarray) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def ir_clean_png(mean_c: np.ndarray) -> np.ndarray:
    """Averaged IR frame -> inferno colours, native 160x120, no text."""
    import cv2
    lo, hi = np.percentile(mean_c, (1, 99))
    u8 = np.clip((mean_c - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def mm_clean_mip(mip: np.ndarray) -> np.ndarray:
    """mmWave max-brightness image -> log-scaled jet, y up (like the plots), native grid, no text."""
    import cv2
    img = np.log10(np.maximum(np.asarray(mip, np.float32), 1e-12))
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.9)
    u8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.flipud(cv2.applyColorMap(u8, cv2.COLORMAP_JET)).copy()


def mm_clean_depth(depth: np.ndarray, mip: np.ndarray, z_mm: np.ndarray, faint_frac: float = 0.2) -> np.ndarray:
    """Depth of the brightest return -> viridis (yellow = nearer), faint areas grey, y up, no text."""
    import cv2
    zmin, zmax = float(np.min(z_mm)), float(np.max(z_mm))
    t = 1.0 - np.clip((np.asarray(depth, np.float32) - zmin) / max(zmax - zmin, 1e-6), 0, 1)
    col = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    col[np.asarray(mip) < faint_frac * float(np.max(mip))] = (217, 217, 217)
    return np.flipud(col).copy()


def mask_png(mask: np.ndarray, flip: bool = False) -> np.ndarray:
    m = (np.asarray(mask, bool).astype(np.uint8) * 255)
    return np.flipud(m).copy() if flip else m


# --------------------------------------------------------------------------- per-sensor writers

def _write_ir(dst: Path, ref_frames, ref_fmt, captures: dict, state_dir=None) -> dict:
    """captures: tag -> (frames, fmt). Writes raw npz + clean pngs + auto mask for 'start'."""
    import ir_metal_size as ims
    dst.mkdir(parents=True, exist_ok=True)
    info: dict = {"ir_format": ref_fmt}
    np.savez_compressed(dst / "reference.npz", frames=np.asarray(ref_frames), ir_format=np.array(ref_fmt))
    ref_mean = ims.average(ref_frames, ref_fmt)
    _png(dst / "reference.png", ir_clean_png(ref_mean))
    sizer = ims.MetalSizer(state_dir or os.getenv("PEEKIR_METAL_STATE", str(REPO_ROOT / "ir_metal_state")))
    for tag, (frames, fmt) in captures.items():
        np.savez_compressed(dst / f"{tag}.npz", frames=np.asarray(frames), ir_format=np.array(fmt))
        mean = ims.average(frames, fmt)
        _png(dst / f"{tag}.png", ir_clean_png(mean))
        if fmt == ref_fmt and mean.shape == ref_mean.shape:
            res, mask = ims.detect(ref_mean, mean, fmt, sizer.settings)
            _png(dst / f"mask_{tag}.png", mask_png(mask))
            info[tag] = ims.summarize_result(res)
            info[tag]["blobs_px"] = [{"area_px": b.area_px, "bbox_px": list(b.bbox), "shape": b.shape}
                                     for b in res.blobs]
    info["settings"] = {k: v for k, v in ims.asdict(sizer.settings).items()}
    info["units"] = ("raw y16 = centikelvin (deg C = value/100 - 273.15)" if ref_fmt == "y16"
                     else f"{ref_fmt}: 8-bit display values, not temperatures")
    return info


def _write_mmwave(dst: Path, npz_path: Path, state_dir=None) -> dict:
    import mmwave_metal_size as mms
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(npz_path, dst / "sar_volume.npz")
    d = np.load(npz_path)
    mip = d["mip_xy"]
    depth = d["depth_xy"] if "depth_xy" in d else None
    s = mms.load_settings(state_dir)
    res, mask = mms.measure_arrays(mip, depth, d["x_mm"], d["y_mm"], s, mms.cutoffs(s, state_dir))
    _png(dst / "mip.png", mm_clean_mip(mip))
    if depth is not None and "z_mm" in d:
        _png(dst / "depth.png", mm_clean_depth(depth, mip, d["z_mm"]))
    _png(dst / "mask.png", mask_png(mask, flip=True))
    info = res.to_dict()
    info["settings"] = mms.asdict(s)
    info["grid"] = {"x_mm_first_last": [float(d["x_mm"][0]), float(d["x_mm"][-1])],
                    "y_mm_first_last": [float(d["y_mm"][0]), float(d["y_mm"][-1])],
                    "z_mm_first_last": [float(d["z_mm"][0]), float(d["z_mm"][-1])] if "z_mm" in d else None}
    return info


# --------------------------------------------------------------------------- index / readme

def _index_add(root: Path, row: dict) -> None:
    p = root / "index.csv"
    if p.exists():                       # older index without the newer columns: rewrite it with them
        with open(p, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            old_fields, rows = rd.fieldnames or [], list(rd)
        if old_fields != INDEX_FIELDS:
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _already_exported(root: Path, source: str) -> Path | None:
    p = root / "index.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("source") == source and (root / r["path"]).exists():
                return root / r["path"]
    return None


def ensure_readme(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for sec in SECTIONS:
        (root / sec).mkdir(exist_ok=True)
    p = root / "README.md"
    if not p.exists():
        p.write_text(README + WEAPON_README, encoding="utf-8")
    elif "## Weapon type" not in p.read_text(encoding="utf-8"):
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(WEAPON_README)


WEAPON_README = """
## Weapon type (added Nov 2026)
`label.json` `weapon_type` is the GROUND TRUTH entered when scanning: what was really on the
mannequin. `gun` = the L-shaped panel (8 x 5 in, 1 in arms), `knife` = a long rectangle (two
3 x 1 in rectangles taped end to end), `other` = any other metal (squares, short rectangles...),
`none` = no metal, `''` = not given (older samples). `index.csv` has it in the `weapon_type` column.

The rule-based detector's own guesses are stored separately and may be wrong (it only looks at
the outline's shape): `ir.start.objects` / `ir.start.blobs_px[].shape` and `mmwave.pieces[].shape`
(gun / knife / suspicious object, plus the measurements behind it: length, width, fill...).
`index.csv` columns `ir_detected` / `mmwave_detected`. Train on `weapon_type`, not on the guesses.
"""


README = """# PeekIR training data

Collected with the PeekIR scanner (FLIR Lepton thermal camera on a Raspberry Pi + 77 GHz mmWave
SAR on a gantry) on a mannequin with bare metal pieces taped to it. Nothing in this folder has
text or outlines drawn on it.

## Sections
- `paired/`      IR and mmWave captured of the SAME scene (one sar_batch.py scan). Use these for
                 fusion models. Each sample has `ir/` and `mmwave/`.
- `ir_only/`     IR captures with no mmWave scan (`ir/` only).
- `mmwave_only/` mmWave scans with no IR (`mmwave/` only).
For a single-sensor model, use that sensor's folder from `paired/` AND its `*_only/` section.
`index.csv` lists every sample with its section, class and path.

## Classes
`<view>_<metal|nometal>_training`, e.g. `front_metal_training`, `side_nometal_training`.
"view" is the mannequin's orientation to the sensors; "metal" means at least one metal piece was
attached. `label.json` has the details: object name, true area (if entered), notes.

## Files
`label.json`  labels, scan parameters, and the automatic detector's results at export time.

`ir/`
- `reference.npz`  frames with NO metal on the mannequin, taken just before the capture
- `start.npz`      frames with the scene as labelled (just before the mmWave scan, for paired)
- `end.npz`        frames just after the mmWave scan (paired only; checks for drift/movement)
  Each npz: `frames` (N, 120, 160) and `ir_format`. y16 = raw Lepton values in centikelvin
  (deg C = value / 100 - 273.15). Other formats are 8-bit display values (not temperatures).
- `reference.png`, `start.png`, `end.png`  averaged frames, inferno colours, 160x120, no text.
- `mask_start.png`  AUTOMATIC metal mask (255 = metal) from the rule-based detector
  (start vs reference). Not hand-checked: treat as weak labels.

`mmwave/`
- `sar_volume.npz`  `volume` (Z, Y, X) linear magnitude, `mip_xy` (Y, X) max over Z,
  `depth_xy` (Y, X) depth (mm) of the brightest return, `x_mm`, `y_mm`, `z_mm` axes.
  Array row 0 = smallest y (bottom of the scan area).
- `mip.png`   log-scaled max-brightness image, jet colours, y UP (row 0 = top), no text.
- `depth.png` depth of the brightest return, yellow = nearer, grey = faint, y up, no text.
- `mask.png`  AUTOMATIC metal mask (255 = metal), same orientation as mip.png. Weak labels.

## How the data was made (important for using it well)
- Bare metal is a mirror in thermal IR. The mannequin was warmed with a heating pad so metal
  reads COLDER than the mannequin (it reflects the cooler room). Heating pads and lights appear
  as very hot areas.
- IR detection compares `start` with `reference` (no-metal) frames; no single-image model was used.
- mmWave metal is far brighter than the plastic mannequin; the detector thresholds relative to the
  scan's background and strongest return.
- Split train/validation by `scene_id` (one physical setup), never by sample, or near-identical
  scenes will leak between the sets.
"""


# --------------------------------------------------------------------------- export of a scan folder

def _read_meta(scan_dir: Path) -> dict:
    meta = {}
    for name in ("scan_meta.json", "ir_summary.json"):
        p = scan_dir / name
        if p.exists():
            try:
                meta.update(json.loads(p.read_text()))
            except Exception:
                pass
    return meta


def export_scan(scan_dir: str | Path, root: str | Path | None = None, *, view: str | None = None,
                metal: bool | None = None, obj: str | None = None, true_area_cm2: float | None = None,
                notes: str | None = None, include_raw: bool = False, force: bool = False,
                state_dir=None, log=print, weapon_type: str | None = None) -> Path | None:
    """Export one scan folder (from sar_batch.py or the dashboard). Returns the sample folder."""
    import ir_metal_size as ims
    scan_dir = Path(scan_dir)
    root = Path(root or DEFAULT_ROOT)
    ensure_readme(root)
    source = str(scan_dir.resolve())
    prev = _already_exported(root, source)
    if prev and not force:
        log(f"training: {scan_dir.name} already exported -> {prev.relative_to(root)}")
        return prev

    meta = _read_meta(scan_dir)
    tr = meta.get("training") or {}
    view = view or tr.get("view") or meta.get("view") or ""
    metal = metal if metal is not None else parse_yes_no(tr.get("metal")) if tr.get("metal") is not None \
        else infer_metal(meta.get("label"))
    obj = obj if obj is not None else tr.get("object", "")
    true_area_cm2 = true_area_cm2 if true_area_cm2 is not None else tr.get("true_area_cm2")
    notes = notes if notes is not None else meta.get("notes", "")
    if metal is None:
        log(f"training: {scan_dir.name} skipped: can't tell metal / no metal from label "
            f"{meta.get('label')!r} (use --metal yes|no)")
        return None
    if view.lower() not in ("front", "side"):
        log(f"training: note: view {view!r} is not front/side; using it as given")
    weapon_type = parse_weapon_type(weapon_type if weapon_type is not None else tr.get("weapon_type"), metal)

    has_ir = (scan_dir / "ir_start.npz").exists() and (scan_dir / "ir_reference.npz").exists()
    has_mm = (scan_dir / "sar_volume.npz").exists()
    if not has_ir and not has_mm:
        log(f"training: {scan_dir.name} skipped: no IR (ir_start/ir_reference.npz) and no "
            "sar_volume.npz (run: python mmwave_metal_size.py reconstruct <folder>)")
        return None
    section = "paired" if has_ir and has_mm else "ir_only" if has_ir else "mmwave_only"
    cls = class_name(view, metal)
    sample_id = scan_dir.name
    dst = root / section / cls / sample_id
    if dst.exists():
        if not force:
            sample_id = f"{scan_dir.name}_{dt.datetime.now():%Y%m%d%H%M%S}"
            dst = root / section / cls / sample_id
        else:
            shutil.rmtree(dst)
    dst.mkdir(parents=True)

    label = {
        "sample_id": sample_id, "section": section, "class": cls, "view": view, "metal": metal,
        "object": obj, "true_area_cm2": true_area_cm2, "notes": notes,
        "weapon_type": weapon_type,
        "weapon_type_note": "ground truth entered when scanning (gun/knife/other/none, '' = not given); "
                            "the detector's own guesses are in ir.*.objects / mmwave.pieces[].shape",
        "scene_id": meta.get("batch_id", "") + ":" + scan_dir.name if meta.get("batch_id") else scan_dir.name,
        "source_scan": source, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "scan_params": meta.get("params"), "reconstruction": meta.get("reconstruction"),
    }
    if has_ir:
        rf, rfmt = ims.load_frames_file(scan_dir / "ir_reference.npz")
        caps = {}
        for tag in ("start", "end"):
            if (scan_dir / f"ir_{tag}.npz").exists():
                caps[tag] = ims.load_frames_file(scan_dir / f"ir_{tag}.npz")
        label["ir"] = _write_ir(dst / "ir", rf, rfmt, caps, state_dir)
    if has_mm:
        label["mmwave"] = _write_mmwave(dst / "mmwave", scan_dir / "sar_volume.npz", state_dir)
        if include_raw:
            raw = dst / "mmwave" / "raw_rows"
            raw.mkdir()
            for f in sorted(scan_dir.glob("row_*.bin")):
                shutil.copy2(f, raw / f.name)
            label["mmwave"]["raw_rows"] = "raw_rows/ (DCA1000 row files; see scan_params)"
    (dst / "label.json").write_text(json.dumps(label, indent=2, default=str), encoding="utf-8")
    _index_add(root, {
        "sample_id": sample_id, "section": section, "class": cls, "view": view, "metal": metal,
        "object": obj, "true_area_cm2": true_area_cm2, "notes": notes,
        "ir_level": (label.get("ir", {}).get("start") or {}).get("level", ""),
        "ir_area_cm2": (label.get("ir", {}).get("start") or {}).get("largest_area_cm2", ""),
        "mmwave_level": label.get("mmwave", {}).get("level", ""),
        "mmwave_area_cm2": label.get("mmwave", {}).get("largest_area_cm2", ""),
        "scene_id": label["scene_id"], "created": label["created"], "source": source,
        "path": str(dst.relative_to(root)), "weapon_type": weapon_type,
        "ir_detected": _detected((label.get("ir", {}).get("start") or {}).get("blobs_px")),
        "mmwave_detected": _detected(label.get("mmwave", {}).get("pieces")),
    })
    log(f"training: saved {section}/{cls}/{sample_id}")
    return dst


# --------------------------------------------------------------------------- IR-only capture

def save_ir_sample(ref_frames, ref_fmt, frames, fmt, root=None, *, view: str, metal: bool,
                   obj: str = "", true_area_cm2: float | None = None, notes: str = "",
                   source: str = "ir capture", state_dir=None, log=print, weapon_type: str = "") -> Path:
    """Save one IR-only sample (reference + capture) into ir_only/<class>/."""
    root = Path(root or DEFAULT_ROOT)
    ensure_readme(root)
    cls = class_name(view, metal)
    weapon_type = parse_weapon_type(weapon_type, metal)
    sample_id = f"ir_{dt.datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]
    dst = root / "ir_only" / cls / sample_id
    info = _write_ir(dst / "ir", ref_frames, ref_fmt, {"start": (frames, fmt)}, state_dir)
    label = {"sample_id": sample_id, "section": "ir_only", "class": cls, "view": view, "metal": metal,
             "object": obj, "true_area_cm2": true_area_cm2, "notes": notes, "weapon_type": weapon_type,
             "scene_id": sample_id, "source": source, "created": dt.datetime.now().isoformat(timespec="seconds"), "ir": info}
    (dst / "label.json").write_text(json.dumps(label, indent=2, default=str), encoding="utf-8")
    st = info.get("start") or {}
    _index_add(root, {"sample_id": sample_id, "section": "ir_only", "class": cls, "view": view, "metal": metal,
                      "object": obj, "true_area_cm2": true_area_cm2, "notes": notes,
                      "ir_level": st.get("level", ""), "ir_area_cm2": st.get("largest_area_cm2", ""),
                      "scene_id": sample_id, "created": label["created"], "source": source,
                      "path": str(dst.relative_to(root)), "weapon_type": weapon_type,
                      "ir_detected": _detected(st.get("blobs_px"))})
    log(f"training: saved ir_only/{cls}/{sample_id}")
    return dst


def summary(root=None) -> dict:
    root = Path(root or DEFAULT_ROOT)
    out = {}
    for sec in SECTIONS + ("legacy_ir",):
        d = root / sec
        if d.exists():
            out[sec] = {c.name: sum(1 for x in c.iterdir() if x.is_dir())
                        for c in sorted(d.iterdir()) if c.is_dir()}
    return out


# --------------------------------------------------------------------------- legacy IR images

LEGACY_README = """
## legacy_ir/ (older IR images from the previous team, Nov 2026)
Same camera (FLIR Lepton), but NOT comparable to `ir_only/` / `paired/`; keep them separate:
- colorised 8-bit PNGs with automatic contrast (no temperatures), no no-metal reference frame,
  so there are no masks or sizes;
- metal usually reads WARM (hand-held or reflecting lights), not cold against a heated mannequin;
- scenes were people/objects (knife / no knife), not metal taped to a mannequin; view unknown unless
  given at import.
Per sample: `original.png` (as recorded, no text), `values.npz` (`values` = 8-bit image values
recovered from the colours, `palette`), `label.json`. `scene_id` groups images taken in one sitting
(split by it). Images with text burned in, matplotlib figures and duplicates were rejected.
Folder names were mapped knife -> metal, no_knife -> nometal.
"""

LEGACY_DEFAULT_MAP = {"knife": True, "no_knife": False}


def import_legacy_ir(src: Path, root=None, *, view: str = "unknown", mapping: dict | None = None,
                     session_gap_min: float = 30.0, log=print) -> dict:
    """Old colourised IR PNGs (folders per class, timestamps in file names) -> training_data/legacy_ir/."""
    import cv2
    import hashlib
    from ir_preprocess import decode_legacy_png
    root = Path(root or DEFAULT_ROOT)
    ensure_readme(root)
    rd = root / "README.md"
    if "## legacy_ir/" not in rd.read_text(encoding="utf-8"):
        with open(rd, "a", encoding="utf-8") as fh:
            fh.write(LEGACY_README)
    mapping = {**LEGACY_DEFAULT_MAP, **(mapping or {})}
    stamp_re = re.compile(r"(\d{8})_(\d{6})_(\d{6})")
    items, rejected, seen = [], {}, {}
    for folder in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        metal = mapping.get(folder.name.lower(), infer_metal(folder.name))
        if metal is None:
            log(f"legacy: skipping folder {folder.name!r} (unknown class; use --map {folder.name}=metal|nometal)")
            continue
        for f in sorted(folder.glob("*.png")):
            m = stamp_re.search(f.name)
            values, info = decode_legacy_png(f)
            reason = info["reject"] or (None if m else "no_timestamp")
            if values is not None and reason is None:
                h = hashlib.sha1(values.tobytes()).hexdigest()
                if h in seen:
                    reason = "duplicate"
                seen.setdefault(h, f.name)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            ts = dt.datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S%f")
            items.append((ts, f, metal, values, info["palette"]))
    items.sort(key=lambda t: t[0])
    existing = set()
    idx = root / "index.csv"
    if idx.exists():
        with open(idx, newline="", encoding="utf-8") as fh:
            existing = {r["source"] for r in csv.DictReader(fh)}
    saved, skipped_existing, scene, last = 0, 0, None, None
    per_scene: dict = {}
    for ts, f, metal, values, palette in items:
        if last is None or (ts - last).total_seconds() > session_gap_min * 60:
            scene = f"legacy-{ts:%Y%m%d-%H%M}"
        last = ts
        per_scene.setdefault(scene, {True: 0, False: 0})[metal] += 1
        source = str(f.resolve())
        if source in existing:
            skipped_existing += 1
            continue
        cls = class_name(view, metal)
        sid = f.stem
        dst = root / "legacy_ir" / cls / sid
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / "original.png")
        np.savez_compressed(dst / "values.npz", values=values, palette=np.array(palette))
        label = {"sample_id": sid, "section": "legacy_ir", "class": cls, "view": view, "metal": metal,
                 "original_class": f.parent.name, "object": "knife" if metal else "", "scene_id": scene,
                 "captured": ts.isoformat(), "palette": palette, "source": source,
                 "notes": "legacy 8-bit colourised image; no reference frame; metal usually reads warm"}
        (dst / "label.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
        _index_add(root, {"sample_id": sid, "section": "legacy_ir", "class": cls, "view": view, "metal": metal,
                          "object": label["object"], "notes": "legacy", "scene_id": scene,
                          "created": dt.datetime.now().isoformat(timespec="seconds"), "source": source,
                          "path": str(dst.relative_to(root))})
        saved += 1
    log(f"legacy: saved {saved} image(s), already imported {skipped_existing}, rejected "
        f"{sum(rejected.values())} {rejected or ''}")
    for sc, c in per_scene.items():
        if not (c[True] and c[False]):
            log(f"legacy: note: {sc} has only {'metal' if c[True] else 'no-metal'} images "
                "(anything specific to that sitting can act as the label)")
    return {"saved": saved, "rejected": rejected, "already": skipped_existing}


# --------------------------------------------------------------------------- command line

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="training data folder (default ./training_data)")
    ap.add_argument("--state-dir", default=None, help="detector settings folder (default ./ir_metal_state)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "export-all"):
        p = sub.add_parser(name)
        p.add_argument("target", type=Path, nargs="+" if name == "export" else None)
        p.add_argument("--view", choices=["front", "side"], help="override the scan's view")
        p.add_argument("--metal", choices=["yes", "no"], help="override metal / no metal")
        p.add_argument("--object", default=None)
        p.add_argument("--true-area-cm2", type=float, default=None)
        p.add_argument("--weapon-type", choices=WEAPON_TYPES, default=None,
                       help="ground truth: what was really on the mannequin")
        p.add_argument("--include-raw", action="store_true", help="also copy raw mmWave row files (large)")
        p.add_argument("--force", action="store_true", help="re-export even if already exported")
    p = sub.add_parser("capture-ir", help="interactive IR-only capture: reference, then metal")
    p.add_argument("--pi", required=True)
    p.add_argument("--view", choices=["front", "side"], required=True)
    p.add_argument("--metal", choices=["yes", "no"], required=True)
    p.add_argument("--object", default="")
    p.add_argument("--true-area-cm2", type=float, default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--weapon-type", choices=WEAPON_TYPES, default="",
                   help="ground truth: gun (L panel) / knife (long rectangle) / other / none")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--count", type=int, default=1, help="number of samples (new reference each time)")
    p = sub.add_parser("import-legacy-ir", help="old colourised IR PNGs (knife/ no_knife/ folders) -> legacy_ir/")
    p.add_argument("--src", type=Path, required=True, help="folder containing class folders, e.g. knife_dataset")
    p.add_argument("--view", default="unknown", help="front / side / unknown (default unknown)")
    p.add_argument("--map", default="", help="extra folder mappings, e.g. 'gun=metal,phone=metal,empty=nometal'")
    sub.add_parser("summary")
    args = ap.parse_args()

    if args.cmd in ("export", "export-all"):
        targets = args.target if args.cmd == "export" else sorted(
            {p.parent for pat in ("scan_meta.json", "ir_summary.json") for p in args.target.rglob(pat)
             if "_failed" not in p.parts})
        for t in targets:
            try:
                export_scan(t, args.root, view=args.view,
                            metal=parse_yes_no(args.metal) if args.metal else None, obj=args.object,
                            true_area_cm2=args.true_area_cm2, include_raw=args.include_raw,
                            force=args.force, state_dir=args.state_dir, weapon_type=args.weapon_type)
            except Exception as e:
                print(f"training: {Path(t).name} failed: {e}")
    elif args.cmd == "capture-ir":
        import ir_metal_size as ims
        for i in range(args.count):
            input(f"\n[{i + 1}/{args.count}] Remove ALL metal from the mannequin, then press Enter: ")
            rf, rfmt = ims.fetch_burst(args.pi, args.seconds)
            input("Now set up the scene (attach metal if --metal yes), step out of view, press Enter: ")
            fr, fmt = ims.fetch_burst(args.pi, args.seconds)
            save_ir_sample(rf, rfmt, fr, fmt, args.root, view=args.view, metal=parse_yes_no(args.metal),
                           obj=args.object, true_area_cm2=args.true_area_cm2, notes=args.notes,
                           source="training_export capture-ir", state_dir=args.state_dir,
                           weapon_type=args.weapon_type)
    elif args.cmd == "import-legacy-ir":
        extra = {}
        for part in filter(None, (x.strip() for x in args.map.split(","))):
            k, _, v = part.partition("=")
            extra[k.strip().lower()] = parse_yes_no(v)
        import_legacy_ir(args.src, args.root, view=args.view, mapping=extra)
    elif args.cmd == "summary":
        s = summary(args.root)
        if not any(s.values()):
            print(f"No samples yet in {args.root}")
        for sec, classes in s.items():
            print(f"{sec}:")
            for c, n in classes.items():
                print(f"  {c:28s} {n}")


if __name__ == "__main__":
    main()