#!/usr/bin/env python3
"""
Find the depth (Z) at which a SAR scan is in focus.

Works on sar_batch scans (reads the scan settings from scan_meta.json) and on
older scans (pass -W/-H/--y-step etc. yourself).

Usage:
    python focus_search.py outputs/front/metal/front_metal_000
    python focus_search.py outputs/front/metal/front_metal_000 --zmin 150 --zmax 600
    python focus_search.py outputs/scan_18 -W 150 -H 150          # old scan, no scan_meta.json
    python focus_search.py outputs/front/metal/front_metal_00*    # several scans at once

Prints the depth where the reconstruction is brightest (the in-focus depth)
plus a suggested --zstart/--zend for sar_batch.py. Accurate to about +/-15 mm:
the radar's range bins are ~21 mm apart.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.fft import fft

import sar_coordinator as sc
import sar_reconstruct as sr

C, FS, K, TI = 299792458.0, 9121e3, 63.343e12, 4.5225e-10
N_FFT = 1024


def load_settings(scan_dir, args):
    """Scan settings from scan_meta.json, falling back to the CLI values."""
    s = {
        "width_mm": args.width, "height_mm": args.height, "y_step_mm": args.y_step,
        "speed_mms": args.speed, "frame_periodicity_ms": args.period,
        "samples": args.samples, "snake": not args.no_snake, "num_frames": None,
    }
    meta_path = scan_dir / "scan_meta.json"
    source = "command line"
    if meta_path.exists():
        params = json.loads(meta_path.read_text(encoding="utf-8")).get("params", {})
        s.update({k: v for k, v in params.items() if k in s and v is not None})
        source = "scan_meta.json"
    if s["num_frames"] is None:   # same rule as sar_coordinator.compute_num_frames
        s["num_frames"] = int((s["width_mm"] / s["speed_mms"]) * 1000 / s["frame_periodicity_ms"]) + 10
    return s, source


def focus_curve(scan_dir, s, z_values):
    W, H, ystep = s["width_mm"], s["height_mm"], s["y_step_mm"]
    dx = s["speed_mms"] * s["frame_periodicity_ms"] / 1000.0
    rows = sc.num_rows_from_height_step(H, ystep)
    n_move = int(round(W / dx))

    # option 5 = average of the 4 RX channels, same as sar_reconstruct.py
    raw = sr.stack(s["samples"], s["num_frames"], rows, 5, str(scan_dir),
                   lambda y: f"row_{y}_Raw_0.bin", snake=s["snake"], n_move=n_move)
    raw_fft = fft(raw, n=N_FFT, axis=0)

    out = []
    for z in z_values:
        k = int(round(K / FS * (2 * z * 1e-3 / C + TI) * N_FFT))
        if not 0 <= k < N_FFT:
            continue
        mf = sr.create_matched_filter(N_FFT, dx, N_FFT, ystep, z)
        img, _, _ = sr.reconstruct_sar_image(raw_fft[k], mf, dx, ystep, W, H)
        mag = np.abs(img)
        out.append((float(z), float(mag.max())))
    return out


def main():
    p = argparse.ArgumentParser(description="Find the in-focus depth of SAR scans.")
    p.add_argument("scans", nargs="+", help="Scan folder(s); wildcards OK")
    p.add_argument("--zmin", type=float, default=100)
    p.add_argument("--zmax", type=float, default=900)
    p.add_argument("--zstep", type=float, default=5)
    p.add_argument("--margin", type=float, default=50,
                   help="Half-width of the suggested --zstart/--zend window (mm, default 50)")
    p.add_argument("--quiet", action="store_true", help="Only print the summary per scan")
    old = p.add_argument_group("settings for scans without scan_meta.json")
    old.add_argument("-W", "--width", type=float, default=150)
    old.add_argument("-H", "--height", type=float, default=150)
    old.add_argument("--y-step", type=float, default=1)
    old.add_argument("--speed", type=float, default=25)
    old.add_argument("--period", type=float, default=40)
    old.add_argument("--samples", type=int, default=512)
    old.add_argument("--no-snake", action="store_true")
    args = p.parse_args()

    folders = []
    for pat in args.scans:
        folders += [Path(f) for f in (sorted(glob.glob(pat)) or [pat])]
    z_values = np.arange(args.zmin, args.zmax + args.zstep / 2, args.zstep)

    bests = []
    for scan_dir in folders:
        if not scan_dir.is_dir():
            print(f"Skipping {scan_dir}: not a folder")
            continue
        s, source = load_settings(scan_dir, args)
        print(f"\n=== {scan_dir} ===")
        print(f"settings from {source}: {s['width_mm']} x {s['height_mm']} mm, "
              f"y-step {s['y_step_mm']}, speed {s['speed_mms']}, period {s['frame_periodicity_ms']} ms, "
              f"{s['num_frames']} frames, snake={s['snake']}")
        res = focus_curve(scan_dir, s, z_values)
        if not res:
            print("No valid depths in that range.")
            continue
        top_peak = max(r[1] for r in res)
        if not args.quiet:
            for z, peak in res:
                bar = "#" * int(40 * peak / top_peak)
                print(f"  z = {z:5.0f} mm   peak = {peak:.3e}  {bar}")
        best = max(res, key=lambda r: r[1])[0]
        bests.append(best)
        print(f"In focus at: {best:.0f} mm  (brightest reconstruction)")
        # Another strong, separate reflector (stand, wall, table edge)?
        others = [z for i, (z, pk) in enumerate(res)
                  if pk > 0.5 * top_peak and abs(z - best) > 60
                  and pk >= max(r[1] for r in res[max(0, i - 2):i + 3])]
        if others:
            print(f"  Warning: also strong at {', '.join(f'{z:.0f}' for z in others)} mm. Another object "
                  "(stand, wall) may be reflecting; check the X-Y image at each depth.")

    if bests:
        lo = max(0, min(bests) - args.margin)
        hi = max(bests) + args.margin
        print(f"\nSuggested for sar_batch.py:  --zstart {lo:.0f} --zend {hi:.0f}")


if __name__ == "__main__":
    main()
