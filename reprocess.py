#!/usr/bin/env python3
"""
Rebuild the images and sar_volume.npz for scans you already collected, from
their raw row_N_Raw_0.bin files. No rescanning needed.

Use it to add new outputs (e.g. the depth map) to old scans, or to rebuild a
whole dataset at a different depth range.

Usage:
    python reprocess.py outputs                                  # every scan under outputs\\
    python reprocess.py outputs\\front\\metal                       # one class
    python reprocess.py outputs\\front\\metal\\front_metal_003       # one scan
    python reprocess.py outputs --zstart 290 --zend 390          # override the depth range
    python reprocess.py outputs --dry-run                        # list what would be done
    python reprocess.py outputs\\scan_18 -W 150 -H 150 --zstart 600 --zend 700
                                                                 # old scan without scan_meta.json

Settings come from each scan's scan_meta.json (written by sar_batch.py). The
depth range comes from, in order: --zstart/--zend on the command line, the
range recorded in scan_meta.json, or else the scan is skipped with a message.

Skips anything under a _failed folder and scans whose scan_meta.json says
status "failed". Writes reprocess.log in each scan folder and records the
settings used under "reprocessed" in scan_meta.json.
"""

import argparse
import datetime as _dt
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import sar_complete as scomp

REPO_ROOT = Path(__file__).resolve().parent


def is_scan_dir(p: Path) -> bool:
    return bool(list(p.glob("row_1_Raw*.bin")) or list(p.glob("row_1.bin")))


def find_scans(patterns):
    found = []
    for pat in patterns:
        matches = sorted(glob.glob(pat)) or [pat]
        for m in matches:
            p = Path(m)
            if not p.is_dir():
                print(f"Skipping {p}: not a folder")
                continue
            if is_scan_dir(p):
                found.append(p)
            else:
                for root, dirs, _files in os.walk(p):
                    dirs[:] = sorted(d for d in dirs if not d.startswith("_"))  # skip _failed etc.
                    rp = Path(root)
                    if is_scan_dir(rp):
                        found.append(rp)
                        dirs[:] = []
    out, seen = [], set()
    for p in found:
        key = str(p.resolve())
        if key not in seen and "_failed" not in p.resolve().parts:
            seen.add(key)
            out.append(p)
    return out


def build_args(scan_dir, cli):
    """Namespace for scomp.build_reconstruct_argv, or (None, reason) to skip."""
    a = scomp.build_parser().parse_args([])   # all sar_complete defaults
    meta_path = scan_dir / "scan_meta.json"
    meta = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "failed":
            return None, meta, "scan_meta.json says this scan failed"
        p = meta.get("params", {})
        a.width, a.height = p.get("width_mm", a.width), p.get("height_mm", a.height)
        a.y_step, a.speed = p.get("y_step_mm", a.y_step), p.get("speed_mms", a.speed)
        a.frame_periodicity = p.get("frame_periodicity_ms", a.frame_periodicity)
        a.num_frames = p.get("num_frames", a.num_frames)
        a.samples, a.snake = p.get("samples", a.samples), p.get("snake", a.snake)
        r = meta.get("reconstruction", {})
        for k in ("zstart", "zend", "zstep", "zindex", "algo", "window"):
            if r.get(k) is not None:
                setattr(a, k, r[k])
        has_recorded_z = r.get("zstart") is not None
    else:
        if cli.width is None or cli.height is None:
            return None, None, "no scan_meta.json; pass -W and -H (and --zstart/--zend)"
        a.width, a.height = cli.width, cli.height
        a.y_step, a.speed, a.frame_periodicity = cli.y_step, cli.speed, cli.period
        a.snake = not cli.no_snake
        has_recorded_z = False

    for k in ("zstart", "zend", "zstep", "zindex", "algo", "window"):
        v = getattr(cli, k)
        if v is not None:
            setattr(a, k, v)
    if not has_recorded_z and cli.zstart is None and cli.zindex is None:
        return None, meta, "depth range unknown; pass --zstart and --zend"

    a.viz = "plotly" if cli.plotly else "matplotlib"
    a.viz_html_root = str(REPO_ROOT / "visualizations") if cli.plotly else None
    a.save_npz = True
    return a, meta, None


def main():
    p = argparse.ArgumentParser(description="Rebuild images + sar_volume.npz from raw scan data.")
    p.add_argument("scans", nargs="+", help="Scan folders or parent folders (wildcards OK)")
    p.add_argument("--zstart"), p.add_argument("--zend"), p.add_argument("--zstep")
    p.add_argument("--zindex"), p.add_argument("--algo"), p.add_argument("--window")
    p.add_argument("--plotly", action="store_true", help="Also write the Plotly slice viewer HTML")
    p.add_argument("--dry-run", action="store_true", help="Only list what would be reprocessed")
    old = p.add_argument_group("for old scans without scan_meta.json")
    old.add_argument("-W", "--width", type=float)
    old.add_argument("-H", "--height", type=float)
    old.add_argument("--y-step", type=float, default=1)
    old.add_argument("--speed", type=float, default=25)
    old.add_argument("--period", type=float, default=40)
    old.add_argument("--no-snake", action="store_true")
    cli = p.parse_args()

    scans = find_scans(cli.scans)
    if not scans:
        print("No scan folders found.")
        return
    print(f"Found {len(scans)} scan folder(s).")

    ok, skipped, failed = 0, [], []
    for i, scan_dir in enumerate(scans, 1):
        a, meta, why = build_args(scan_dir, cli)
        tag = f"[{i}/{len(scans)}] {scan_dir}"
        if a is None:
            print(f"{tag}: skipped ({why})")
            skipped.append(str(scan_dir))
            continue
        z_desc = f"z {a.zindex}" if a.zindex else f"z {a.zstart}-{a.zend}"
        print(f"{tag}: {a.width}x{a.height} mm, {z_desc}", end="", flush=True)
        if cli.dry_run:
            print("  (dry run)")
            continue

        argv = scomp.build_reconstruct_argv(a, scan_dir.resolve())
        cmd = [sys.executable, "-X", "utf8", str(REPO_ROOT / "sar_reconstruct.py"), *argv[1:]]
        t0 = time.time()
        with open(scan_dir / "reprocess.log", "w", encoding="utf-8") as logf:
            rc = subprocess.run(cmd, cwd=str(scan_dir), stdout=logf, stderr=subprocess.STDOUT,
                                env=dict(os.environ, MPLBACKEND="Agg")).returncode
        if rc != 0:
            print(f"  FAILED (exit {rc}); see reprocess.log")
            failed.append(str(scan_dir))
            continue
        print(f"  done in {time.time() - t0:.0f}s")
        ok += 1
        if meta is not None:
            meta["reprocessed"] = {
                "when": _dt.datetime.now().isoformat(timespec="seconds"),
                "zstart": a.zstart, "zend": a.zend, "zstep": a.zstep, "zindex": a.zindex,
                "algo": a.algo, "window": a.window,
            }
            (scan_dir / "scan_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not cli.dry_run:
        print(f"\nDone: {ok} rebuilt, {len(skipped)} skipped, {len(failed)} failed.")
        for f in failed:
            print(f"  failed: {f}")


if __name__ == "__main__":
    main()