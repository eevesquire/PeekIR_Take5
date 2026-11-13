#!/usr/bin/env python3
"""
prepare_labels.py -- get every SAR scan ready for training.

Place next to sar_reconstruct.py and run:

    python prepare_labels.py                      # uses ./outputs
    python prepare_labels.py -o "D:\\scans\\outputs"

For every scan folder under the outputs directory it:

  1. Makes sure sar_volume.npz exists. Missing ones are rebuilt by running
     sar_reconstruct.py with the parameters stored in scan_meta.json
     (use --force-recon to rebuild all of them).
  2. Works out the class from the label:
        metal     -> needs a hand-checked mask
        negative  -> nometal / plastic / empty / subject-only scans
        unknown   -> label not recognised; fix it in labels.csv
  3. Writes masks:
        negative  -> mask.png, all black (final, nothing to draw)
        metal     -> mask_draft.png (automatic guess) + mask_ref.png (the scan
                     as a greyscale picture of the same size). Correct the
                     draft in any paint program and save it as mask.png.
  4. Writes / updates <outputs>/labels.csv (one row per scan). Columns you
     edit by hand -- label, object_id, material, size_mm, tilt_deg, cover,
     distance_mm, split, notes -- are kept when you re-run the script.
     Change a scan's label there (not by moving folders) and re-run.
  5. Writes review images to <outputs>/_review/:
        contact_<class>_NN.png  thumbnails of every scan on ONE shared
                                brightness scale, so a mislabelled scan
                                stands out (a "negative" with a bright blob).
        <scan>/review.png       heatmap with the mask outline drawn on it.

Mask PNG convention: same pixel size as mip_xy in sar_volume.npz, flipped so
the TOP row of the image is the TOP of the scan (as in the plots).
White (>127) = metal, black = not metal.

Re-running is safe: existing mask.png files are never overwritten unless you
pass --overwrite-masks; drafts and review images are regenerated.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent

# Labels are compared after lower-casing and removing '_', '-' and spaces,
# so "no_metal", "No-Metal" and "nometal" are all the same.
DEFAULT_NEGATIVE_CONTAINS = "nometal,nonmetal,plastic,empty,blank,nothing"
DEFAULT_NEGATIVE_EXACT = "subject,subjectonly,bare,none,clean,negative,background"
DEFAULT_METAL_CONTAINS = "metal"

USER_COLS = ["label", "object_id", "material", "size_mm", "tilt_deg", "cover",
             "distance_mm", "split", "notes"]
CSV_COLS = (["folder", "name", "view", "label", "class", "mask_status", "check"]
            + USER_COLS[1:]
            + ["y_step_mm", "x_step_mm", "rows", "cols", "peak_intensity",
               "draft_blobs", "draft_area_mm2", "draft_length_mm", "label_from"])

ROW_FILE_RE = re.compile(r"^row_\d+(_Raw_\d+)?\.bin$")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[\s_\-]+", "", (s or "").lower())


class LabelRules:
    def __init__(self, args):
        split = lambda s: [_norm(t) for t in s.split(",") if t.strip()]
        self.neg_contains = split(args.negative_labels)
        self.neg_exact = split(args.negative_exact)
        self.metal_contains = split(args.metal_labels)

    def classify(self, label):
        s = _norm(label)
        if not s:
            return "unknown"
        if s in self.neg_exact or any(t in s for t in self.neg_contains):
            return "negative"
        if any(t in s for t in self.metal_contains):
            return "metal"
        return "unknown"


def infer_view_label(rel, meta, rules):
    """(view, label, source) from scan_meta.json, else the folder path, else the name."""
    view = (meta or {}).get("view") or ""
    label = (meta or {}).get("label") or ""
    source = "scan_meta" if label else ""
    parents = list(rel.parts[:-1])
    if not label or not view:
        p_view = p_label = ""
        if len(parents) >= 2:
            p_view, p_label = parents[-2], parents[-1]
        elif len(parents) == 1:
            if rules.classify(parents[0]) != "unknown":
                p_label = parents[0]
            else:
                p_view = parents[0]
        view = view or p_view
        if not label and p_label:
            label, source = p_label, "folder"
    if not label and rules.classify(rel.name) != "unknown":
        # e.g. front_nometal_003 -> "nometal"
        label, source = rel.name, "name"
        m = re.match(r"^(.*?)_?\d+$", rel.name)
        if m and m.group(1):
            label = m.group(1)
            if view and label.lower().startswith(view.lower() + "_"):
                label = label[len(view) + 1:]
    return view, label, source or "none"


# ---------------------------------------------------------------------------
# Scan discovery and reconstruction
# ---------------------------------------------------------------------------
def find_scans(root):
    """Folders that contain row_*.bin files or sar_volume.npz (skips _failed, _review...)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith("_") and d != "visualizations")
        if "sar_volume.npz" in filenames or any(ROW_FILE_RE.match(f) for f in filenames):
            found.append(Path(dirpath))
            dirnames[:] = []
    return found


def load_meta(scan_dir):
    p = scan_dir / "scan_meta.json"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def npz_ok(path):
    if not path.is_file():
        return False
    try:
        with np.load(path) as d:
            return all(k in d.files for k in ("mip_xy", "x_mm", "y_mm"))
    except Exception:
        return False


def reconstruct(scan_dir, meta, args):
    """Run sar_reconstruct.py for one scan. Returns (ok, message)."""
    p = (meta or {}).get("params") or {}
    need = ("num_frames", "num_rows", "speed_mms", "frame_periodicity_ms",
            "y_step_mm", "width_mm", "height_mm")
    if not all(k in p for k in need):
        return False, "no sar_volume.npz and no scan parameters in scan_meta.json"
    r = (meta or {}).get("reconstruction") or {}
    cmd = [sys.executable, "-X", "utf8", str(args.reconstruct_script),
           "--folder", str(scan_dir),
           "--frames", str(p["num_frames"]),
           "--rows", str(p["num_rows"]),
           "--samples", str(p.get("samples") or 512),
           "--speed", str(p["speed_mms"]),
           "--periodicity", str(p["frame_periodicity_ms"]),
           "--y-step", str(p["y_step_mm"]),
           "--scan-width", str(p["width_mm"]),
           "--scan-height", str(p["height_mm"]),
           # row_N.bin first, then the DCA1000 names row_N_Raw_0.bin etc.
           "--filename-pattern", "row_{y}.bin",
           "--zstart", str(args.zstart or r.get("zstart") or 200),
           "--zend", str(args.zend or r.get("zend") or 400),
           "--algo", r.get("algo") or "mf",
           "--window", r.get("window") or "none",
           "--save-npz"]
    zstep = args.zstep or r.get("zstep")
    if zstep:
        cmd += ["--zstep", str(zstep)]
    if p.get("snake"):
        cmd.append("--snake")
    env = dict(os.environ, MPLBACKEND="Agg")   # never open a window
    log_path = scan_dir / "reprocess.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.run(cmd, cwd=str(scan_dir), env=env,
                            stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not npz_ok(scan_dir / "sar_volume.npz"):
        return False, f"reconstruction failed (exit {rc}); see {log_path.name}"
    return True, "reconstructed"


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def load_scan_arrays(npz_path):
    with np.load(npz_path) as d:          # never loads the big 'volume' array
        mip = d["mip_xy"].astype(np.float64)
        x = d["x_mm"].astype(np.float64)
        y = d["y_mm"].astype(np.float64)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1.0
    return mip, x, y, abs(dx), abs(dy)


def draft_mask(mip, dx, dy, frac, min_area_mm2):
    """Bright regions of the max-projection, cleaned up. Returns (mask, blobs)."""
    if mip.max() <= 0:
        return np.zeros(mip.shape, bool), []
    m = mip > frac * mip.max()
    m = ndimage.binary_closing(m, structure=np.ones((3, 3)))
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    blobs = []
    keep = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        area = len(ys) * dx * dy
        if area < min_area_mm2:
            continue
        keep[i] = True
        blobs.append({"area_mm2": area, "length_mm": blob_length(ys, xs, dx, dy)})
    blobs.sort(key=lambda b: -b["area_mm2"])
    return keep[lab], blobs


def blob_length(ys, xs, dx, dy):
    """Extent of a blob along its longest direction, in mm."""
    pts = np.column_stack([xs * dx, ys * dy])
    if len(pts) < 2:
        return max(dx, dy)
    c = pts - pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(c.T))
    proj = c @ vecs[:, -1]
    return float(proj.max() - proj.min() + max(dx, dy))


def save_mask_png(mask, path):
    from PIL import Image
    Image.fromarray(np.flipud(mask).astype(np.uint8) * 255, mode="L").save(path)


def read_mask_png(path, shape):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"))
    if a.shape != tuple(shape):
        raise ValueError(f"{path.name} is {a.shape[1]}x{a.shape[0]} px, "
                         f"expected {shape[1]}x{shape[0]}")
    return np.flipud(a > 127)


def save_ref_png(mip, path, lo, hi):
    from PIL import Image
    v = np.clip((np.log10(mip + 1e-12) - lo) / max(hi - lo, 1e-9), 0, 1)
    Image.fromarray((np.flipud(v) * 255).astype(np.uint8), mode="L").save(path)


# ---------------------------------------------------------------------------
# Review images
# ---------------------------------------------------------------------------
def save_review(path, mip, x, y, mask, title, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    img = np.log10(mip + 1e-12)
    mesh = ax.pcolormesh(x, y, img, cmap="jet", shading="nearest", vmin=lo, vmax=hi)
    if mask is not None and mask.any():
        ax.contour(x, y, mask.astype(float), levels=[0.5], colors="white", linewidths=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal (mm)")
    ax.set_ylabel("Vertical (mm)")
    ax.set_title(title, fontsize=9)
    fig.colorbar(mesh, ax=ax, label="log10 intensity (shared scale)")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheets(review_dir, items, lo, hi, per_page=30, ncols=6):
    """items: list of (title, mip, x, y, mask). One shared colour scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_class = {}
    for cls, *rest in items:
        by_class.setdefault(cls, []).append(rest)
    written = []
    for cls, lst in sorted(by_class.items()):
        for page in range(0, len(lst), per_page):
            chunk = lst[page:page + per_page]
            nrows = int(np.ceil(len(chunk) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6),
                                     squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, (title, mip, x, y, mask) in zip(axes.ravel(), chunk):
                ax.pcolormesh(x, y, np.log10(mip + 1e-12), cmap="jet",
                              shading="nearest", vmin=lo, vmax=hi)
                if mask is not None and mask.any():
                    ax.contour(x, y, mask.astype(float), levels=[0.5],
                               colors="white", linewidths=0.8)
                ax.set_aspect("equal")
                ax.set_title(title, fontsize=7)
            fig.suptitle(f"class = {cls}   (all thumbnails share one brightness scale)",
                         fontsize=11)
            fig.tight_layout()
            out = review_dir / f"contact_{cls}_{page // per_page + 1:02d}.png"
            fig.savefig(out, dpi=80)
            plt.close(fig)
            written.append(out)
    return written


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def read_csv(path):
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["folder"]: row for row in csv.DictReader(f) if row.get("folder")}


def write_csv(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="Prepare SAR scans + labels + masks for training.")
    ap.add_argument("-o", "--output-dir", default=str(REPO_ROOT / "outputs"),
                    help="Scan outputs folder (default: <this folder>/outputs)")
    ap.add_argument("--reconstruct-script", default=str(REPO_ROOT / "sar_reconstruct.py"))
    ap.add_argument("--force-recon", action="store_true",
                    help="Rebuild sar_volume.npz for every scan that has scan_meta.json")
    ap.add_argument("--skip-recon", action="store_true",
                    help="Never run sar_reconstruct.py; scans without an npz are just listed")
    ap.add_argument("--zstart", default=None, help="Override Z start when reconstructing")
    ap.add_argument("--zend", default=None, help="Override Z end when reconstructing")
    ap.add_argument("--zstep", default=None, help="Override Z step when reconstructing")
    ap.add_argument("--draft-frac", type=float, default=0.35,
                    help="Draft mask: pixels brighter than this fraction of the scan's "
                         "peak count as metal (default 0.35)")
    ap.add_argument("--min-area", type=float, default=20.0, metavar="MM2",
                    help="Draft mask: drop blobs smaller than this many mm^2 (default 20)")
    ap.add_argument("--overwrite-masks", action="store_true",
                    help="Replace existing mask.png files (normally they are kept)")
    ap.add_argument("--no-review", action="store_true", help="Skip review images")
    ap.add_argument("--metal-labels", default=DEFAULT_METAL_CONTAINS,
                    help=f"Comma list: a label containing any of these is metal "
                         f"(default '{DEFAULT_METAL_CONTAINS}')")
    ap.add_argument("--negative-labels", default=DEFAULT_NEGATIVE_CONTAINS,
                    help=f"Comma list: a label containing any of these is negative; checked "
                         f"before metal (default '{DEFAULT_NEGATIVE_CONTAINS}')")
    ap.add_argument("--negative-exact", default=DEFAULT_NEGATIVE_EXACT,
                    help=f"Comma list: a label exactly equal to one of these is negative "
                         f"(default '{DEFAULT_NEGATIVE_EXACT}')")
    return ap


def main():
    args = build_parser().parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Outputs folder not found: {root}")
    args.reconstruct_script = Path(args.reconstruct_script).resolve()
    rules = LabelRules(args)
    csv_path = root / "labels.csv"
    old = read_csv(csv_path)
    review_dir = root / "_review"

    scans = find_scans(root)
    print(f"Found {len(scans)} scan folders under {root}")

    # ---- pass 1: npz + labels ------------------------------------------------
    rows, loaded = [], []
    for scan_dir in scans:
        rel = scan_dir.relative_to(root)
        key = rel.as_posix()
        meta = load_meta(scan_dir)
        if meta and meta.get("status") == "failed":
            print(f"  skip {key}: marked failed in scan_meta.json")
            continue

        view, label, src = infer_view_label(rel, meta, rules)
        prev = old.get(key, {})
        row = {c: prev.get(c, "") for c in USER_COLS}
        if row["label"]:
            src = "labels.csv"          # your edit wins
        else:
            row["label"] = label
        if not row["notes"] and meta and meta.get("notes"):
            row["notes"] = meta["notes"]
        row.update(folder=key, name=rel.name, view=view, label_from=src,
                   **{"class": rules.classify(row["label"])})

        npz = scan_dir / "sar_volume.npz"
        have = npz_ok(npz)
        if (args.force_recon and meta) or not have:
            if args.skip_recon:
                if not have:
                    row["mask_status"] = "no_npz"
                    row["check"] = "no sar_volume.npz (re-run without --skip-recon)"
            else:
                print(f"  reconstructing {key} ...", flush=True)
                ok, msg = reconstruct(scan_dir, meta, args)
                have = ok or (have and npz_ok(npz))
                if not ok:
                    row["mask_status"] = "no_npz"
                    row["check"] = msg
                    print(f"    {msg}")
        if have:
            mip, x, y, dx, dy = load_scan_arrays(npz)
            row.update(rows=mip.shape[0], cols=mip.shape[1],
                       x_step_mm=f"{dx:.3g}", y_step_mm=f"{dy:.3g}",
                       peak_intensity=f"{mip.max():.4g}")
            loaded.append((row, scan_dir, mip, x, y, dx, dy))
        rows.append(row)

    # Shared log-brightness scale so scans can be compared with each other.
    if loaded:
        logs = np.concatenate([np.log10(m.ravel() + 1e-12) for _, _, m, *_ in loaded])
        lo, hi = np.percentile(logs, [5, 99.9])
    else:
        lo, hi = 0.0, 1.0

    # Brightness sanity check: negatives as bright as typical metal (and vice versa).
    peaks = {c: [float(r["peak_intensity"]) for r, *_ in loaded if r["class"] == c]
             for c in ("metal", "negative")}
    metal_med = np.median(peaks["metal"]) if peaks["metal"] else None
    neg_med = np.median(peaks["negative"]) if peaks["negative"] else None

    # ---- pass 2: masks + review ---------------------------------------------
    if not args.no_review:
        review_dir.mkdir(exist_ok=True)
        for old_sheet in review_dir.glob("contact_*.png"):
            old_sheet.unlink()
    contact = []
    for row, scan_dir, mip, x, y, dx, dy in loaded:
        cls = row["class"]
        mask_path = scan_dir / "mask.png"
        checks = []
        shown_mask = None
        peak = float(row["peak_intensity"])

        if cls == "negative":
            if neg_med is not None and metal_med is not None and peak >= metal_med:
                checks.append("as bright as a typical metal scan - is the label right?")
            existing = None
            if mask_path.is_file() and not args.overwrite_masks:
                try:
                    existing = read_mask_png(mask_path, mip.shape)
                except Exception as e:
                    checks.append(f"bad mask.png: {e}")
            if existing is not None and existing.any():
                row["mask_status"] = "conflict"
                checks.append("label is negative but mask.png marks metal; "
                              "fix the label or delete mask.png")
                shown_mask = existing
            else:
                save_mask_png(np.zeros(mip.shape, bool), mask_path)
                row["mask_status"] = "final_empty"
            for f in ("mask_draft.png", "mask_ref.png"):
                (scan_dir / f).unlink(missing_ok=True)

        elif cls == "metal":
            if neg_med is not None and metal_med is not None and peak <= neg_med:
                checks.append("no brighter than a typical negative - is metal really there?")
            draft, blobs = draft_mask(mip, dx, dy, args.draft_frac, args.min_area)
            save_mask_png(draft, scan_dir / "mask_draft.png")
            save_ref_png(mip, scan_dir / "mask_ref.png", lo, hi)
            row.update(draft_blobs=len(blobs),
                       draft_area_mm2=f"{sum(b['area_mm2'] for b in blobs):.0f}",
                       draft_length_mm=f"{max((b['length_mm'] for b in blobs), default=0):.0f}")
            shown_mask = draft
            if mask_path.is_file() and not args.overwrite_masks:
                try:
                    m = read_mask_png(mask_path, mip.shape)
                    shown_mask = m
                    if m.any():
                        row["mask_status"] = "reviewed"
                    else:
                        row["mask_status"] = "conflict"
                        checks.append("label is metal but mask.png is empty "
                                      "(old negative mask?); delete it and re-run")
                except Exception as e:
                    row["mask_status"] = "conflict"
                    checks.append(f"bad mask.png: {e}")
            else:
                if args.overwrite_masks:
                    mask_path.unlink(missing_ok=True)
                row["mask_status"] = "needs_review"
            if not blobs:
                checks.append("draft found nothing; try a lower --draft-frac")
        else:
            row["mask_status"] = "unknown_label"
            checks.append(f"label '{row['label']}' not recognised; set it in labels.csv")

        row["check"] = "; ".join(checks)

        if not args.no_review:
            d = review_dir / row["folder"].replace("/", "__")
            d.mkdir(exist_ok=True)
            title = (f"{row['folder']}\nlabel={row['label']}  class={cls}  "
                     f"mask={row['mask_status']}")
            if cls == "metal":
                title += (f"\ndraft: {row['draft_blobs']} blob(s), "
                          f"{row['draft_area_mm2']} mm^2, longest {row['draft_length_mm']} mm")
            save_review(d / "review.png", mip, x, y, shown_mask, title, lo, hi)
            flag = " (!)" if checks else ""
            contact.append((cls, f"{row['folder']}{flag}", mip, x, y, shown_mask))

    rows.sort(key=lambda r: r["folder"])
    write_csv(csv_path, rows)

    sheets = []
    if not args.no_review and contact:
        sheets = save_contact_sheets(review_dir, contact, lo, hi)

    # ---- summary ---------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"labels.csv: {csv_path}  ({len(rows)} scans)")
    by = Counter((r["view"] or "-", r["class"]) for r in rows)
    views = sorted({v for v, _ in by})
    classes = ["metal", "negative", "unknown"]
    print(f"\n{'view':<12}" + "".join(f"{c:>10}" for c in classes))
    for v in views:
        print(f"{v:<12}" + "".join(f"{by[(v, c)]:>10}" for c in classes))
    steps = Counter((r["y_step_mm"] or "?", r["class"]) for r in rows)
    print("\ny-step (mm) by class (watch for one class being mostly one step size):")
    for s in sorted({s for s, _ in steps}):
        print(f"  {s:>5} mm: " + ", ".join(f"{c} {steps[(s, c)]}" for c in classes
                                          if steps[(s, c)]))
    st = Counter(r["mask_status"] for r in rows)
    print("\nmask status: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))
    flagged = [r for r in rows if r["check"]]
    if flagged:
        print(f"\n{len(flagged)} scan(s) to look at (also in the 'check' column):")
        for r in flagged[:40]:
            print(f"  {r['folder']}: {r['check']}")
        if len(flagged) > 40:
            print(f"  ... {len(flagged) - 40} more")
    if sheets:
        print(f"\nReview images: {review_dir}  ({len(sheets)} contact sheet(s))")
    print("=" * 64)


if __name__ == "__main__":
    main()
