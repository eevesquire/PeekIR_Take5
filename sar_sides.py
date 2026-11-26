#!/usr/bin/env python3
"""
sar_sides.py - side passes with the arm, after the normal front scan.

For each side (default: xmax, then xmin):
    1. ARM RETRACT, then XEND <max|min>: drive X to that end's reed switch
    2. move Y to the side pass start height
    3. ROTATE <side>        (panel turns while the arm is still retracted, clear of the mannequin)
    4. ARM EXTEND           (timed move into the hard stop)
    5. one vertical pass: radar ARM + gantry MOVE up=<height> + radar CAPTURE, like one front-scan row
    6. ARM RETRACT, ROTATE front
Finally: XEND min and Y back down, so the gantry ends at home with the arm parked.

Each side is saved to <scan>/<scan>_side_<xmax|xmin>/row_1.bin (+ side_meta.json) and reconstructed
with sar_reconstruct.py as a one-row scan. Pictures (drawn by make_panorama):
    <scan>/<scan>_side_<side>/side_profile.png   height (up) vs depth from the panel (across)
    <scan>/panorama.png                          min-X side | front | max-X side, same height scale
The dashboard shows these in its "Side views" panel.

Used by sar_complete.py --side-scans; the Lua radar script is unchanged.
"""

import copy
import json
import threading
import time
from pathlib import Path

import sar_coordinator as sc


def add_side_args(parser):
    g = parser.add_argument_group("side scans (arm)")
    g.add_argument("--side-scans", action="store_true",
                   help="After the front scan, run a vertical side pass at each X end with the arm extended")
    g.add_argument("--side-order", default="xmax,xmin",
                   help="Which sides, in order (default 'xmax,xmin'; e.g. 'xmax' for one side only)")
    g.add_argument("--side-height", type=float, default=300.0,       # PLACEHOLDER: calibrate_arm step 11
                   help="Vertical travel of each side pass in mm (default 300)")
    g.add_argument("--side-y-start", type=float, default=0.0,        # PLACEHOLDER: calibrate_arm step 11
                   help="Y height (from Y home) where each side pass starts, mm (default 0)")
    g.add_argument("--side-speed", type=float, default=None,
                   help="Y speed during the side pass, mm/s (default: same as -s/--speed)")
    g.add_argument("--side-zstart", type=str, default="50",          # PLACEHOLDER: side standoff - ~100 mm
                   help="Reconstruction depth start for the side passes (default 50 mm)")
    g.add_argument("--side-zend", type=str, default="300",           # PLACEHOLDER: side standoff + ~150 mm
                   help="Reconstruction depth end for the side passes (default 300 mm)")
    g.add_argument("--side-settle", type=float, default=1.0, metavar="SEC",  # PLACEHOLDER: calibrate_arm step 4
                   help="Wait after extending the arm before capturing, for the panel to stop wobbling (default 1 s)")
    g.add_argument("--arm-timeout", type=float, default=60.0, metavar="SEC",
                   help="Max seconds for one ARM/ROTATE command on the Pi (default 60)")
    return parser


def _progress(**kw):
    print("__PEEKIR_PROGRESS__ " + json.dumps({"phase": "side", **kw}, separators=(",", ":")), flush=True)


def _gantry_cmd(gantry, cmd, timeout):
    """Send an ARM/ROTATE/XEND/MOVE command and wait for MOTOR_STARTED + MOVE_COMPLETE.
    Any ERROR reply (e-stop, arm problem, refused move) raises GantryStopped, which ends the scan."""
    gantry.send(cmd)
    sc._gantry_expect(gantry, "MOTOR_STARTED", 30)
    sc._gantry_expect(gantry, "MOVE_COMPLETE", timeout)


def side_speed(args):
    return args.side_speed or args.speed


def side_frames(args):
    """Frames for one vertical pass (same rule as compute_num_frames, with the side height/speed)."""
    return int((args.side_height / side_speed(args)) * 1000 / args.frame_periodicity) + 10


def front_end_y(args):
    """Y of the gantry (from home) when run_scan finishes."""
    rows = sc.num_rows_from_height_step(args.height, args.y_step)
    return args.y_start + (rows - 1) * args.y_step


def side_dir_for(scan_dir, side):
    scan_dir = Path(scan_dir)
    return scan_dir / f"{scan_dir.name}_side_{side}"


def run_side_passes(gantry, radar, args, scan_dir):
    """Run after sc.run_scan() on the same connections. Returns the list of side folders written."""
    sides = [s.strip() for s in args.side_order.split(",") if s.strip()]
    for s in sides:
        if s not in ("xmax", "xmin"):
            raise ValueError(f"--side-order: unknown side '{s}' (use xmax and/or xmin)")
    speed = side_speed(args)
    frames = side_frames(args)
    sep = "\\" if "\\" in str(scan_dir) else "/"
    t_arm = args.arm_timeout
    t_x = 636 / max(1.0, args.return_speed) * 1.5 + 30     # full X travel (~636 mm) with margin
    y_cur = front_end_y(args)
    done = []
    args.side_dirs = done               # filled in as each side finishes (kept even if a later side fails)

    print(f"\n{'=' * 60}\nSide passes: {', '.join(sides)}")
    print(f"  Height {args.side_height} mm from Y={args.side_y_start} mm at {speed} mm/s, "
          f"{frames} frames @ {args.frame_periodicity} ms")
    print(f"{'=' * 60}\n")

    radar.send(f"CONFIGURE {frames} {int(args.frame_periodicity)}")
    radar.expect("CONFIGURED")
    time.sleep(args.dca_record_gap)          # DCA1000 gap after the front scan's last row

    for i, side in enumerate(sides, 1):
        sdir = side_dir_for(scan_dir, side)
        sdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _progress(status="start", side=side, index=i, total=len(sides))
        print(f"--- Side {i}/{len(sides)}: {side} ---")

        _gantry_cmd(gantry, "ARM RETRACT", t_arm)
        _gantry_cmd(gantry, f"XEND {'max' if side == 'xmax' else 'min'} speed={args.return_speed}mms", t_x)
        dy = args.side_y_start - y_cur
        if abs(dy) > 0.01:
            _gantry_cmd(gantry, f"MOVE {'up' if dy > 0 else 'down'}={abs(dy):.2f}mm speed={args.return_speed}mms", 300)
        y_cur = args.side_y_start
        _gantry_cmd(gantry, f"ROTATE {side}", t_arm)
        _gantry_cmd(gantry, "ARM EXTEND", t_arm)
        if args.side_settle > 0:
            time.sleep(args.side_settle)

        # One vertical pass, exactly like one row of the front scan.
        radar.send(f"ARM {sdir}{sep}row_1.bin")
        resp = radar.recv(timeout=30)
        if resp != "ARMED":
            raise sc.ScanAborted(f"side {side}: radar ARM failed: {resp}")
        gantry.send(f"MOVE up={args.side_height:.2f}mm speed={speed}mms")
        sc._gantry_expect(gantry, "MOTOR_STARTED", 30)
        if args.stabilization > 0:
            time.sleep(args.stabilization / 1000.0)
        radar.send("CAPTURE")

        results, errors = {}, []

        def _wait(client, key):
            try:
                results[key] = client.recv(timeout=180)
            except Exception as e:
                errors.append(f"{client.name}: {e}")

        th = [threading.Thread(target=_wait, args=(gantry, "gantry")),
              threading.Thread(target=_wait, args=(radar, "radar"))]
        for t in th:
            t.start()
        for t in th:
            t.join(timeout=180)
        if errors or results.get("gantry") != "MOVE_COMPLETE":
            raise sc.GantryStopped(f"side {side}: {errors or results.get('gantry')}")
        if results.get("radar") != "CAPTURE_DONE":
            raise sc.ScanAborted(f"side {side}: radar sent {results.get('radar')!r}")
        y_cur += args.side_height
        time.sleep(args.dca_record_gap)

        _gantry_cmd(gantry, "ARM RETRACT", t_arm)
        _gantry_cmd(gantry, "ROTATE front", t_arm)

        meta = {"side": side, "height_mm": args.side_height, "y_start_mm": args.side_y_start,
                "speed_mms": speed, "frames": frames, "frame_periodicity_ms": args.frame_periodicity,
                "zstart": args.side_zstart, "zend": args.side_zend,
                "duration_s": round(time.time() - t0, 1),
                "note": "One vertical pass. In sar_heatmap_xz.png, Horizontal = height along the pass and "
                        "Depth Z = distance from the panel; side_profile.png shows the same data upright."}
        (sdir / "side_meta.json").write_text(json.dumps(meta, indent=2))
        done.append(sdir)
        _progress(status="done", side=side, index=i, total=len(sides), elapsed_s=meta["duration_s"])

    # Park: X home, Y home (arm already retracted, panel front).
    _gantry_cmd(gantry, f"XEND min speed={args.return_speed}mms", t_x)
    if y_cur > 0.01:
        _gantry_cmd(gantry, f"MOVE down={y_cur:.2f}mm speed={args.return_speed}mms", 300)
    print(f"Side passes complete: {', '.join(str(d) for d in done)}")
    return done


def side_args(args):
    """A copy of args that makes sar_complete.build_reconstruct_argv describe one side pass:
    1 row, width = side height, speed = side speed, side depth range, no snake."""
    a = copy.copy(args)
    a.width = args.side_height
    a.height = 0.0                     # -> 1 row
    a.y_step = 1.0
    a.speed = side_speed(args)
    a.num_frames = None                # computed from width/speed, same as side_frames()
    a.snake = False
    a.zstart, a.zend = args.side_zstart, args.side_zend
    a._scan_height_override = 2.0      # sar_reconstruct needs a crop window >= 1 row wide
    return a


def reconstruct_sides(args, side_dirs, run_reconstruction):
    """Reconstruct each side folder with sar_complete.run_reconstruction (in-process),
    then draw side_profile.png in each and panorama.png in the scan folder."""
    a = side_args(args)
    if a.viz == "matplotlib":
        a.viz = "plotly"               # never open a blocking window for the side passes
    for d in side_dirs:
        try:
            run_reconstruction(a, d)
        except BaseException as e:
            print(f"Side reconstruction failed for {d}: {e}")
    if side_dirs:
        try:
            out = make_panorama(Path(side_dirs[0]).parent, side_dirs, front_y_start=args.y_start,
                                front_x_start=args.x_start)
            print(f"Saved side views: {out}")
            _progress(status="views", folder=str(Path(side_dirs[0]).parent))
        except Exception as e:
            print(f"Could not draw the side views: {e}")


# ---------------------------------------------------------------------------
# Pictures: per-side profile and a "panorama" (min-X side | front | max-X side)
# ---------------------------------------------------------------------------
def find_side_dirs(scan_dir):
    """Side folders of a scan that have been reconstructed, as {side: path}."""
    scan_dir = Path(scan_dir)
    out = {}
    for side in ("xmin", "xmax"):
        d = side_dir_for(scan_dir, side)
        if (d / "sar_volume.npz").exists():
            out[side] = d
    return out


def _side_profile(sdir):
    """(height_mm, depth_mm, image[height, depth]) for one side pass, height in gantry Y from home."""
    import numpy as np
    d = np.load(Path(sdir) / "sar_volume.npz")
    vol = d["volume"]                                # (Z, Y, X); X = position along the vertical pass
    prof = vol.max(axis=1)                           # (Z, X)
    y0 = 0.0
    try:
        y0 = float(json.loads((Path(sdir) / "side_meta.json").read_text()).get("y_start_mm", 0.0))
    except Exception:
        pass
    return y0 + d["x_mm"], d["z_mm"], prof.T          # rows = height, cols = depth


def make_panorama(scan_dir, side_dirs=None, front_y_start=None, front_x_start=None):
    """Write side_profile.png into each side folder and panorama.png into scan_dir. Returns the path.
    The front scan's start position comes from the arguments, else scan_meta.json, else 0."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scan_dir = Path(scan_dir)
    if front_x_start is None or front_y_start is None:
        params = {}
        try:
            params = json.loads((scan_dir / "scan_meta.json").read_text()).get("params") or {}
        except Exception:
            pass
        if front_x_start is None:
            front_x_start = float(params.get("x_start_mm") or 0.0)
        if front_y_start is None:
            front_y_start = float(params.get("y_start_mm") or 0.0)
    sides = find_side_dirs(scan_dir) if side_dirs is None else \
        {Path(d).name.rsplit("_side_", 1)[-1]: Path(d) for d in side_dirs if (Path(d) / "sar_volume.npz").exists()}
    profiles = {s: _side_profile(d) for s, d in sides.items()}
    titles = {"xmin": "Min-X side", "xmax": "Max-X side"}

    # One picture per side: height up, depth from the panel across.
    for s, (hgt, dep, img) in profiles.items():
        fig, ax = plt.subplots(figsize=(5, 7))
        m = ax.pcolormesh(dep, hgt, img, cmap="jet", shading="auto")
        ax.set_xlabel("Depth from panel (mm)")
        ax.set_ylabel("Height (mm from Y home)")
        ax.set_title(f"{titles[s]} profile")
        if s == "xmax":
            ax.invert_xaxis()                         # both sides "look" toward the mannequin's centre
        fig.colorbar(m, ax=ax, label="Intensity")
        fig.tight_layout()
        fig.savefig(sides[s] / "side_profile.png", dpi=100)
        plt.close(fig)

    panels = []
    if "xmin" in profiles:
        panels.append("xmin")
    front = scan_dir / "sar_volume.npz"
    if front.exists():
        panels.append("front")
    if "xmax" in profiles:
        panels.append("xmax")
    if not panels:
        raise FileNotFoundError(f"nothing reconstructed in {scan_dir}")

    widths = []
    fd = np.load(front) if "front" in panels else None
    for pnl in panels:
        if pnl == "front":
            widths.append(float(np.ptp(fd["x_mm"])) or 1.0)
        else:
            widths.append(float(np.ptp(profiles[pnl][1])) or 1.0)
    fig, axes = plt.subplots(1, len(panels), figsize=(min(18, 4 + 0.03 * sum(widths)), 7), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    axes = np.atleast_1d(axes)
    for ax, pnl in zip(axes, panels):
        if pnl == "front":
            ax.pcolormesh(front_x_start + fd["x_mm"], front_y_start + fd["y_mm"], fd["mip_xy"], cmap="jet", shading="auto")
            ax.set_xlabel("Gantry X (mm)")
            ax.set_title("Front")
        else:
            hgt, dep, img = profiles[pnl]
            ax.pcolormesh(dep, hgt, img, cmap="jet", shading="auto")
            ax.set_xlabel("Depth from panel (mm)")
            ax.set_title(titles[pnl])
            if pnl == "xmax":
                ax.invert_xaxis()
    axes[0].set_ylabel("Height (mm from Y home)")
    fig.suptitle(f"{scan_dir.name}: side views and front (each panel scaled to its own brightest return)")
    fig.tight_layout()
    out = scan_dir / "panorama.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
