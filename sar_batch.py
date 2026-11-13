#!/usr/bin/env python3
"""
SAR Batch — run many SAR scans back-to-back, unattended, for collecting
training data.

Why this exists
---------------
``sar_complete.py`` runs ONE scan and then sends SHUTDOWN to both clients, so
the mmWave Studio Lua script (sar_scan_tcp.lua) has to be restarted by hand for
every scan. This script instead:

  * opens the coordinator TCP server ONCE and keeps the RADAR (Lua) connection
    alive for the whole batch -- start the Lua script once, then walk away;
  * between scans, shuts down only the gantry client, re-homes the gantry over
    SSH (``move.py origin``) and relaunches it (or, with --skip-pi-ssh, drives
    the gantry back to the start position over the existing connection);
  * calls ``sar_coordinator.run_scan`` unchanged for each scan;
  * verifies every row file was actually written and retries failed scans;
  * optionally grabs an RGB + IR snapshot from the Pi camera server per scan;
  * captures raw IR bursts at the start and end of every scan and measures the
    metal size / threat level against a no-metal IR reference (ir_metal_size.py),
    prompting you to take that reference (see --ir-reference);
  * writes ``scan_meta.json`` into each scan folder and appends one line per
    scan to ``<output>/batch_manifest.jsonl`` (label, params, status, files);
  * reconstructs each scan in a background process while the next one runs.

Usage (place next to sar_complete.py / sar_coordinator.py / sar_reconstruct.py):

    # 20 scans named metal_000 .. metal_019, all labelled "metal"
    python sar_batch.py --count 20 --prefix metal --label metal

    # Pause before each scan so you can move/swap the object and change label
    python sar_batch.py --count 50 --prefix mixed --label empty --pause

    # Organise by viewing angle: outputs/front/metal/front_metal_000 ...
    python sar_batch.py --count 20 --view front --label metal
    python sar_batch.py --count 20 --view side  --label nometal

    # One session, switching at the prompt ("side/metal", "nometal", ...)
    python sar_batch.py --count 40 --view front --label metal --pause

    # Any sar_complete / sar_coordinator flag still works
    python sar_batch.py --count 10 --prefix knife -W 150 -H 150 --y-step 1

    # IR: ask for a no-metal IR reference before every scan (default with --pause),
    # once at the start (default without --pause), or reuse the dashboard's (keep)
    python sar_batch.py --count 10 --view front --label metal --pause
    python sar_batch.py --count 10 --view front --label metal --ir-reference keep
    python sar_batch.py --count 10 --no-ir

Then, once, start sar_scan_tcp.lua in mmWave Studio when the script says it is
waiting for RADAR. Ctrl+C stops cleanly after shutting both clients down.

IR files written into every scan folder (needs ir_metal_size.py next to this file):
  ir_start.npz / ir_end.npz   raw thermal frames just before / just after the scan
  ir_start.json / .png        metal measurement + annotated image (when a reference exists)
  ir_end.json / .png
  ir_reference.npz            the no-metal reference used (so it can be re-measured later)
and an "ir" block in scan_meta.json / batch_manifest.jsonl.

mmWave: reconstruction also saves sar_volume.npz (--save-npz) and, when it finishes,
mmwave_metal_size.py measures the metal (mmwave_result.json / .png, "mmwave" block in
scan_meta.json). Turn off with --no-mmwave-measure. Afterwards:
  python mmwave_metal_size.py report outputs/          (IR + mmWave + combined verdict)

Training data: after each scan's reconstruction, the scan is also exported to
training_data/<paired|mmwave_only>/<view>_<metal|nometal>_training/<scan>/ (raw data, clean images
without text, auto masks, label.json; see training_data/README.md). Metal yes/no comes from
--metal or the label ('nometal' -> no). --object / --true-area-cm2 / --weapon-type are stored in the labels.
--no-training turns it off.
  python ir_metal_size.py report outputs/            (table of all scans)
  python ir_metal_size.py report outputs/ --remeasure  (after changing levels/polarity)
"""

import argparse
import copy
import datetime as _dt
import glob
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import sar_complete as scomp
import sar_coordinator as sc

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    parser = scomp.build_parser()
    parser.description = "Run many SAR scans back-to-back for training-data collection."

    b = parser.add_argument_group("batch")
    b.add_argument("-n", "--count", type=int, default=10,
                   help="Number of successful scans to collect (default 10)")
    b.add_argument("--prefix", type=str, default=None,
                   help="Scan names are <prefix>_000, <prefix>_001, ... (existing names are skipped). "
                        "Default: <view>_<label> (or 'batch' if neither is set)")
    b.add_argument("--label", type=str, default="",
                   help="Class label (e.g. 'metal', 'nometal'). Scans go in <output>/[<view>/]<label>/")
    b.add_argument("--view", type=str, default="",
                   help="Viewing angle / orientation (e.g. 'front', 'side'). "
                        "Scans go in <output>/<view>/[<label>/]")
    b.add_argument("--notes", type=str, default="",
                   help="Free-text notes stored with every scan in this batch")
    b.add_argument("--delay", type=float, default=0.0, metavar="SEC",
                   help="Seconds to wait between scans (default 0)")
    b.add_argument("--pause", action="store_true",
                   help="Prompt before every scan (lets you reposition the object and change "
                        "label or view: type 'nometal', 'side/', or 'side/metal')")
    b.add_argument("--retries", type=int, default=2,
                   help="Retries per scan when rows are missing or the scan errors (default 2)")
    b.add_argument("--rehome-every", type=int, default=1, metavar="N",
                   help="Re-home the gantry via SSH every N scans (default 1 = every scan). "
                        "Between re-homes the gantry is driven back to start over TCP. "
                        "Ignored with --skip-pi-ssh (never re-homes).")
    b.add_argument("--radar-wait", type=float, default=600.0, metavar="SEC",
                   help="How long to wait for the Lua RADAR client to (re)connect (default 600)")
    b.add_argument("--gantry-wait", type=float, default=120.0, metavar="SEC",
                   help="How long to wait for the GANTRY client to connect (default 120)")
    b.add_argument("--no-radar-ping", action="store_true",
                   help="Don't PING the radar before each scan (use if the Lua script lacks PING)")
    b.add_argument("--configure-once", action="store_true",
                   help="Send CONFIGURE to the radar only on the first scan and answer later "
                        "identical CONFIGUREs locally. Use if the Lua script errors on re-CONFIGURE.")
    b.add_argument("--reconstruct", choices=["png", "plotly", "none"], default="png",
                   help="Per-scan reconstruction: 'png' = heatmap PNGs only (default), "
                        "'plotly' = PNGs + slice-viewer HTML, 'none' = raw data only")
    b.add_argument("--recon-parallel", type=int, default=1, metavar="N",
                   help="Max background reconstructions running at once (default 1; 0 = run inline)")
    b.add_argument("--no-snapshots", action="store_true",
                   help="Don't save RGB/IR snapshots from the Pi camera server")
    b.add_argument("--pi-camera-url", type=str, default=None,
                   help="Pi camera server base URL (default http://<pi-host>:9000)")

    ir = parser.add_argument_group("IR metal size (thermal camera)")
    ir.add_argument("--no-ir", action="store_true",
                    help="Don't capture raw IR bursts or measure metal size")
    ir.add_argument("--ir-reference", choices=["prompt", "once", "keep"], default=None,
                    help="When to take the no-metal IR reference: 'prompt' = before every scan "
                         "(default with --pause), 'once' = at the start of the batch (default "
                         "otherwise), 'keep' = reuse the current one (e.g. set in the dashboard)")
    ir.add_argument("--ir-seconds", type=float, default=2.0, metavar="SEC",
                    help="Length of each IR burst (default 2)")
    tr = parser.add_argument_group("training data (for the next team)")
    tr.add_argument("--metal", choices=["yes", "no"], default=None,
                    help="Whether metal is on the mannequin. If omitted it is inferred from the label "
                         "('nometal', 'no_metal', 'empty' -> no; anything with 'metal' -> yes)")
    tr.add_argument("--object", type=str, default="", help="Metal piece name, e.g. plateA (stored in labels)")
    tr.add_argument("--weapon-type", choices=["gun", "knife", "other", "none"], default=None,
                    help="Ground truth for training: gun (L panel), knife (long rectangle), other metal, none")
    tr.add_argument("--true-area-cm2", type=float, default=None, help="Real area of the metal piece (cm^2), if known")
    tr.add_argument("--training-dir", type=str,
                    default=os.getenv("PEEKIR_TRAINING_DIR", str(REPO_ROOT / "training_data")),
                    help="Training data folder (default ./training_data)")
    tr.add_argument("--no-training", action="store_true", help="Don't export scans to the training folder")
    tr.add_argument("--include-raw", action="store_true",
                    help="Also copy the raw mmWave row files into the training sample (large: ~300 MB/scan)")
    ir.add_argument("--no-mmwave-measure", action="store_true",
                    help="Don't save sar_volume.npz / measure mmWave metal size after reconstruction")
    ir.add_argument("--ir-state-dir", type=str,
                    default=os.getenv("PEEKIR_METAL_STATE", str(REPO_ROOT / "ir_metal_state")),
                    help="Reference/calibration folder, shared with the dashboard (default ./ir_metal_state)")
    return parser


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[batch {ts}] {msg}", flush=True)


def progress(**kw):
    """Same __PEEKIR_PROGRESS__ convention the dashboard already parses."""
    print("__PEEKIR_PROGRESS__ " + json.dumps({"phase": "batch", **kw}, separators=(",", ":")), flush=True)


def check_part(kind, value):
    """view/label become folder names, so keep them filesystem-safe."""
    if value:
        try:
            scomp._sanitize_scan_name(value)
        except ValueError:
            raise ValueError(f"{kind} '{value}' may contain only letters, digits, '_' and '-' "
                             "(no spaces or slashes); e.g. 'no_metal'")
    return value


def group_dir(root, view, label):
    d = root
    if view:
        d = d / view
    if label:
        d = d / label
    return d


def scan_prefix(args, view, label):
    if args.prefix:
        return args.prefix
    return "_".join(p for p in (view, label) if p) or "batch"


def next_free_name(prefix, start_idx, outputs_base):
    i = start_idx
    while True:
        name = f"{prefix}_{i:03d}"
        if scomp._scan_name_slot_available(name, outputs_base):
            return name, i
        i += 1


class PersistentRadar(sc.ClientConnection):
    """Radar connection that can skip redundant CONFIGUREs (see --configure-once)."""

    def __init__(self, base, configure_once=False):
        super().__init__(base.sock, base.addr, base.name)
        self._buf = base._buf
        self.configure_once = configure_once
        self.last_configure = None
        self._fake = []
        self.errors = []          # ERROR replies seen during the current scan

    def send(self, msg):
        if self.configure_once and msg.startswith("CONFIGURE ") and msg == self.last_configure:
            print(f"  [{self.name}] (skipping repeat {msg}; already configured)")
            self._fake.append("CONFIGURED")
            return
        if msg.startswith("CONFIGURE "):
            self.last_configure = msg
        super().send(msg)

    def recv(self, timeout=None):
        if self._fake:
            return self._fake.pop(0)
        line = super().recv(timeout)
        if line.startswith("ERROR"):
            self.errors.append(line)
        return line

    def drain(self, quiet_s=1.5):
        """Discard any late replies left over from an aborted scan."""
        self._fake.clear()
        end = time.time() + quiet_s
        while time.time() < end:
            try:
                self.sock.settimeout(0.3)
                data = self.sock.recv(4096)
                if not data:
                    raise ConnectionError("RADAR disconnected")
                self._buf += data.decode(errors="replace")
            except socket.timeout:
                pass
            finally:
                self.sock.settimeout(None)
        if self._buf.strip():
            print(f"  [RADAR] drained stale: {self._buf.strip()!r}")
        self._buf = ""


# ---------------------------------------------------------------------------
# Connection management (one server socket for the whole batch)
# ---------------------------------------------------------------------------
class Clients:
    def __init__(self, server, args):
        self.server = server
        self.args = args
        self.radar = None
        self.gantry = None
        self.gantry_ssh = None   # Popen of `ssh ... move.py tcp`
        self.coordinator_host = args.coordinator_host or scomp._detect_local_ip()

    # -- accepting ----------------------------------------------------------
    def wait_for(self, roles, timeout):
        """Accept connections until every role in *roles* is connected."""
        deadline = time.time() + timeout
        last_note = 0.0
        while True:
            missing = [r for r in roles if getattr(self, r.lower()) is None]
            if not missing:
                return
            now = time.time()
            if now > deadline:
                raise TimeoutError(f"timed out waiting for {', '.join(missing)}")
            if now - last_note > 30:
                hint = ""
                if "RADAR" in missing:
                    hint = "  -> start sar_scan_tcp.lua in mmWave Studio"
                log(f"Waiting for {', '.join(missing)} on port {self.args.port}{hint}")
                last_note = now
            self.server.settimeout(min(5.0, max(0.1, deadline - now)))
            try:
                conn, addr = self.server.accept()
            except socket.timeout:
                continue
            self._register(conn, addr)

    def _register(self, conn, addr):
        conn.settimeout(10)
        buf = ""
        try:
            while "\n" not in buf:
                data = conn.recv(1024)
                if not data:
                    conn.close()
                    return
                buf += data.decode(errors="replace")
        except (socket.timeout, OSError):
            conn.close()
            return
        conn.settimeout(None)
        ident, rest = buf.split("\n", 1)
        ident = ident.strip()
        if ident == "RADAR" and self.radar is None:
            base = sc.ClientConnection(conn, addr, "RADAR")
            base._buf = rest
            self.radar = PersistentRadar(base, self.args.configure_once)
            log(f"RADAR connected from {addr}")
        elif ident == "GANTRY" and self.gantry is None:
            self.gantry = sc.ClientConnection(conn, addr, "GANTRY")
            self.gantry._buf = rest
            log(f"GANTRY connected from {addr}")
        else:
            log(f"Rejected client '{ident}' from {addr}")
            conn.close()

    # -- radar --------------------------------------------------------------
    def radar_alive(self):
        if self.radar is None:
            return False
        if self.args.no_radar_ping:
            return True
        try:
            self.radar.send("PING")
            resp = self.radar.recv(timeout=10)
            if resp == "PONG":
                return True
            log(f"RADAR answered PING with {resp!r}; draining")
            self.radar.drain()
            return True
        except Exception as e:
            log(f"RADAR not responding ({e}); dropping connection")
            self.radar.close()
            self.radar = None
            return False

    def ensure_radar(self):
        if not self.radar_alive():
            self.wait_for(["RADAR"], self.args.radar_wait)

    # -- gantry -------------------------------------------------------------
    def shutdown_gantry(self):
        if self.gantry is not None:
            try:
                self.gantry.send("SHUTDOWN")
                self.gantry.recv(timeout=5)
            except Exception:
                pass
            self.gantry.close()
            self.gantry = None
        if self.gantry_ssh is not None:
            try:
                self.gantry_ssh.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log("Gantry SSH session still running; terminating")
                self.gantry_ssh.terminate()
                try:
                    self.gantry_ssh.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.gantry_ssh.kill()
            self.gantry_ssh = None

    def rehome_and_connect_gantry(self):
        """SSH: stop gantry client, `move.py origin`, relaunch tcp client, accept."""
        self.shutdown_gantry()
        if not self.args.skip_pi_origin:
            scomp.pi_home_gantry(self.args)
        self.gantry_ssh = scomp.pi_launch_tcp_client(self.args, self.coordinator_host, self.args.port)
        self.wait_for(["GANTRY"], self.args.gantry_wait)

    def move_gantry(self, parts, speed):
        self.gantry.send(f"MOVE {' '.join(parts)} speed={speed}mms")
        self.gantry.expect("MOTOR_STARTED", timeout=30)
        self.gantry.expect("MOVE_COMPLETE", timeout=300)

    def close_all(self):
        for c in (self.gantry, self.radar):
            if c is not None:
                try:
                    c.send("SHUTDOWN")
                except Exception:
                    pass
                c.close()
        self.gantry = self.radar = None
        if self.gantry_ssh is not None:
            try:
                self.gantry_ssh.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.gantry_ssh.terminate()
        try:
            self.server.close()
        except Exception:
            pass


def end_position(args):
    """(x, y) of the gantry relative to home after run_scan finishes."""
    rows = sc.num_rows_from_height_step(args.height, args.y_step)
    if args.snake:
        ends_right = rows % 2 == 1
    else:
        ends_right = False  # raster mode returns left after every row
    x = args.x_start + (args.width if ends_right else 0)
    y = args.y_start + (rows - 1) * args.y_step
    return x, y


def return_to_home(clients, args):
    x, y = end_position(args)
    parts = []
    if x > 0:
        parts.append(f"left={x:.2f}mm")
    if y > 0:
        parts.append(f"down={y:.2f}mm")
    if parts:
        log(f"Returning gantry to start: {' '.join(parts)}")
        clients.move_gantry(parts, args.return_speed)


# ---------------------------------------------------------------------------
# Per-scan pieces
# ---------------------------------------------------------------------------
def check_rows(scan_dir, num_rows):
    """Return (missing_rows, sizes) for row_<n>*.bin files in scan_dir."""
    missing, sizes = [], {}
    for r in range(1, num_rows + 1):
        files = [f for f in glob.glob(os.path.join(scan_dir, f"row_{r}_Raw*.bin"))
                 + glob.glob(os.path.join(scan_dir, f"row_{r}.bin"))]
        size = max((os.path.getsize(f) for f in files), default=0)
        if size == 0:
            missing.append(r)
        sizes[r] = size
    # A row much smaller than the typical row was cut short (e.g. DCA1000 timeout).
    good = sorted(v for v in sizes.values() if v > 0)
    if good:
        typical = good[len(good) // 2]
        missing += [r for r, v in sizes.items() if 0 < v < 0.9 * typical]
    return sorted(set(missing)), sizes


def grab_snapshots(base_url, scan_dir, warned):
    saved = {}
    for kind, path in (("rgb", "/api/camera/frame"), ("ir", "/api/ir/frame")):
        try:
            with urllib.request.urlopen(base_url + path, timeout=5) as resp:
                data = resp.read()
            out = os.path.join(scan_dir, f"snapshot_{kind}.jpg")
            with open(out, "wb") as f:
                f.write(data)
            saved[kind] = os.path.basename(out)
        except Exception as e:
            if kind not in warned:
                log(f"Snapshot '{kind}' unavailable ({e}); continuing without it")
                warned.add(kind)
    return saved


# ---------------------------------------------------------------------------
# IR metal size (thermal)
# ---------------------------------------------------------------------------
def ir_setup(args):
    """Return a MetalSizer, or None if IR is disabled/unavailable."""
    if args.no_ir:
        return None
    try:
        from ir_metal_size import MetalSizer
    except ImportError as e:
        log(f"IR disabled: can't import ir_metal_size ({e}). Put ir_metal_size.py next to "
            "sar_batch.py and pip install numpy opencv-python httpx.")
        return None
    sizer = MetalSizer(args.ir_state_dir)
    st = sizer.status()
    s = st["settings"]
    log(f"IR: state {args.ir_state_dir}; "
        + (f"calibrated {s['cm_per_px']} cm/px" if s["cm_per_px"] else "NOT calibrated (sizes in px)")
        + f"; LOW < {s['low_max_cm2']} <= MEDIUM < {s['high_min_cm2']} <= HIGH cm2; polarity {s['polarity']}")
    return sizer


def ir_take_reference(sizer, cam_url, args, allow_keep=True):
    """Interactive: ask for the mannequin without metal, capture the reference."""
    from ir_metal_size import fetch_burst
    has_ref = sizer.status()["has_reference"]
    while True:
        hint = " (s = keep current reference)" if allow_keep and has_ref else ""
        ans = input(f"\n>>> IR reference: remove ALL metal from the mannequin, then press Enter{hint}: ").strip().lower()
        if ans == "s" and allow_keep and has_ref:
            log("IR: keeping the current reference")
            break
        try:
            frames, fmt = fetch_burst(cam_url, args.ir_seconds)
            r = sizer.set_reference(frames, fmt)
            log(f"IR reference saved ({r['n_frames']} frames, {fmt})"
                + ("" if r["mannequin_found"] else "; no warm mannequin outline found (unheated? "
                   "consider: python ir_metal_size.py levels --body-mask off)"))
            break
        except Exception as e:
            ans = input(f"    IR reference failed: {e}\n    r = retry, s = skip IR reference: ").strip().lower()
            if ans == "s":
                break
    input(">>> Now set up the object for this scan (attach the metal, if any), then press Enter: ")


def ir_capture(sizer, cam_url, scan_dir, tag, args):
    from ir_metal_size import capture_and_measure
    r = capture_and_measure(sizer, cam_url, scan_dir, tag, args.ir_seconds)
    if "error" in r:
        log(f"IR {tag}: {r['error']}")
    elif "measure_error" in r:
        log(f"IR {tag}: saved {r['n_frames']} frames; not measured ({r['measure_error']})")
    else:
        size = (f"{r['largest_area_cm2']:.0f} cm2" if r["largest_area_cm2"] is not None
                else f"{r['largest_area_px']} px")
        log(f"IR {tag}: THREAT {r['level']} (largest {size}, {r['pieces']} piece(s))")
        for note in r.get("notes") or []:
            log(f"IR {tag} note: {note}")
        progress(status="ir", tag=tag, level=r["level"], area_cm2=r["largest_area_cm2"])
    return r


def start_reconstruction(args, scan_dir):
    a = copy.copy(args)
    a.viz = "plotly" if args.reconstruct == "plotly" else "matplotlib"
    if a.viz == "plotly" and not a.viz_html_root:
        a.viz_html_root = str(REPO_ROOT / "visualizations")
    argv = list(scomp.build_reconstruct_argv(a, scan_dir))
    if not a.no_mmwave_measure and "--save-npz" not in argv:
        argv.append("--save-npz")      # raw volume for mmwave_metal_size.py (and later training)
    cmd = [sys.executable, "-X", "utf8", str(REPO_ROOT / "sar_reconstruct.py"), *argv[1:]]
    env = dict(os.environ, MPLBACKEND="Agg")  # never pop up a window
    logf = open(os.path.join(scan_dir, "reconstruct.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(scan_dir), env=env, stdout=logf, stderr=subprocess.STDOUT)
    proc._logf = logf
    proc._scan = os.path.basename(scan_dir)
    proc._dir = str(scan_dir)
    proc._measure = not a.no_mmwave_measure
    proc._state = getattr(a, "ir_state_dir", None)
    proc._train = not a.no_training and not a.no_mmwave_measure
    proc._train_root = a.training_dir
    proc._include_raw = a.include_raw
    return proc


def training_export(scan_dir, root, include_raw=False, state_dir=None):
    try:
        import training_export as tx
    except ImportError as e:
        log(f"training export skipped: can't import training_export ({e})")
        return
    try:
        tx.export_scan(scan_dir, root, include_raw=include_raw, state_dir=state_dir,
                       log=lambda m: log(m))
    except Exception as e:
        log(f"training export failed for {os.path.basename(scan_dir)}: {e}")


def mmwave_measure(scan_dir, state_dir=None):
    """Metal size / threat level from the reconstruction; adds an "mmwave" block to scan_meta.json."""
    try:
        import mmwave_metal_size as mms
    except ImportError as e:
        log(f"mmWave measurement skipped: can't import mmwave_metal_size ({e})")
        return
    try:
        r = mms.measure_folder(scan_dir, state_dir=state_dir)
    except Exception as e:
        log(f"mmWave measurement failed for {os.path.basename(scan_dir)}: {e}")
        return
    log(f"mmWave {os.path.basename(scan_dir)}: THREAT {r.level} (largest {r.largest_area_cm2:.0f} cm2, "
        f"{len(r.pieces)} piece(s))")
    for n in r.notes:
        log(f"mmWave note: {n}")
    progress(status="mmwave", name=os.path.basename(scan_dir), level=r.level, area_cm2=r.largest_area_cm2)
    meta_p = os.path.join(scan_dir, "scan_meta.json")
    try:
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        meta["mmwave"] = {"level": r.level, "largest_area_cm2": r.largest_area_cm2,
                          "pieces": len(r.pieces), "result": "mmwave_result.json"}
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        pass


def reap(procs, block_until_below=None):
    """Collect finished reconstructions; optionally wait until fewer than N running."""
    failures = []
    while True:
        for p in list(procs):
            rc = p.poll()
            if rc is not None:
                p._logf.close()
                procs.remove(p)
                if rc == 0:
                    log(f"Reconstruction done: {p._scan}")
                    if getattr(p, "_measure", False):
                        mmwave_measure(p._dir, getattr(p, "_state", None))
                    if getattr(p, "_train", False):
                        training_export(p._dir, p._train_root, p._include_raw, getattr(p, "_state", None))
                else:
                    log(f"Reconstruction FAILED for {p._scan} (exit {rc}); see reconstruct.log")
                    failures.append(p._scan)
        if block_until_below is None or len(procs) < block_until_below:
            return failures
        time.sleep(1)


def append_manifest(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()

    if args.scan_name:
        print("Error: use --prefix with sar_batch.py (not --scan-name).")
        sys.exit(2)
    try:
        check_part("--view", args.view)
        check_part("--label", args.label)
        if args.prefix:
            scomp._sanitize_scan_name(f"{args.prefix}_000")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)
    if args.count < 1:
        print("Error: --count must be >= 1")
        sys.exit(2)

    outputs_base = Path(args.output_dir).expanduser().resolve()
    outputs_base.mkdir(parents=True, exist_ok=True)
    manifest = outputs_base / "batch_manifest.jsonl"
    cam_url = (args.pi_camera_url or f"http://{args.pi_host}:9000").rstrip("/")
    num_rows = sc.num_rows_from_height_step(args.height, args.y_step)
    num_frames = sc.compute_num_frames(args)
    batch_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    log(f"Batch {batch_id}: {args.count} scans, view '{args.view}', label '{args.label}'")
    log(f"Scan: {args.width} x {args.height} mm, {num_rows} rows, {num_frames} frames/row, "
        f"speed {args.speed} mm/s, snake={args.snake}")
    log(f"Output: {outputs_base}   manifest: {manifest.name}")
    if not args.no_training:
        if args.reconstruct == "none" or args.no_mmwave_measure:
            log("training: scans are NOT exported automatically without reconstruction + --save-npz. Later run: "
                "python mmwave_metal_size.py reconstruct-all outputs/ && python training_export.py export-all outputs/")
        else:
            log(f"training: exporting each scan to {args.training_dir} after its reconstruction")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(4)
    clients = Clients(server, args)

    recon_procs, recon_failures = [], []
    snap_warned = set()
    sizer = ir_setup(args)
    ir_mode = args.ir_reference or ("prompt" if args.pause else "once")
    if sizer is not None:
        log(f"IR reference mode: {ir_mode}")
        if ir_mode == "keep" and not sizer.status()["has_reference"]:
            log("IR: no reference yet; raw IR is still saved per scan, but nothing is measured. "
                "Set one in the dashboard, or use --ir-reference once/prompt.")
    label = args.label
    view = args.view
    done, attempts_total, idx = 0, 0, 0
    scans_since_home = None          # None => position unknown, must home
    results = []

    try:
        if sizer is not None and ir_mode == "once":
            ir_take_reference(sizer, cam_url, args)

        # Radar first: this is the one that needs a human (once).
        clients.wait_for(["RADAR"], args.radar_wait)
        if args.skip_pi_ssh:
            log("--skip-pi-ssh: start the gantry yourself: "
                f"python3 move.py tcp host={clients.coordinator_host} port={args.port}")
            clients.wait_for(["GANTRY"], args.gantry_wait)
            scans_since_home = 0     # assume user placed gantry at start

        while done < args.count:
            if args.pause:
                while True:
                    ans = input(f"\n>>> Scan {done + 1}/{args.count} (view='{view}', label='{label}'). "
                                "Enter = go, 'label' or 'view/label' to change, 'q' = quit: ").strip()
                    if ans.lower() == "q" or not ans:
                        break
                    new_view, new_label = (ans.split("/", 1) if "/" in ans else (view, ans))
                    try:
                        view = check_part("view", new_view.strip())
                        label = check_part("label", new_label.strip() or label)
                    except ValueError as e:
                        print(f"    {e}")
                        continue
                    log(f"Now view='{view}', label='{label}'")
                if ans.lower() == "q":
                    break
            elif done > 0 and args.delay > 0:
                log(f"Waiting {args.delay:.0f}s before next scan")
                time.sleep(args.delay)

            if sizer is not None and ir_mode == "prompt":
                ir_take_reference(sizer, cam_url, args)

            scan_base = group_dir(outputs_base, view, label)
            scan_base.mkdir(parents=True, exist_ok=True)
            name, idx = next_free_name(scan_prefix(args, view, label), 0, scan_base)

            training_info = None
            if not args.no_training:
                try:
                    import training_export as tx
                    m = tx.infer_metal(label, tx.parse_yes_no(args.metal) if args.metal else None)
                except ImportError:
                    m = None
                training_info = {"view": view, "metal": m, "object": args.object,
                                 "true_area_cm2": args.true_area_cm2,
                                 "weapon_type": args.weapon_type or ("none" if m is False else "")}
                if m is None:
                    log(f"training: can't tell metal / no metal from label '{label}'; this scan won't be "
                        "exported (use --metal yes|no, or a label like 'metal' / 'nometal')")

            ok = False
            for attempt in range(1, args.retries + 2):
                attempts_total += 1
                scan_dir = scan_base / name
                t0 = time.time()
                started = _dt.datetime.now().isoformat(timespec="seconds")
                progress(status="scan_start", index=done + 1, total=args.count, name=name, attempt=attempt)
                log(f"=== Scan {done + 1}/{args.count}: {name} (attempt {attempt}) ===")
                error = None
                snaps = {}
                ir_info = None
                missing, sizes = list(range(1, num_rows + 1)), {}
                try:
                    clients.ensure_radar()

                    # Put gantry at home.
                    need_home = (scans_since_home is None
                                 or (not args.skip_pi_ssh and scans_since_home >= args.rehome_every))
                    if need_home:
                        if args.skip_pi_ssh:
                            raise RuntimeError("gantry position unknown and --skip-pi-ssh set; "
                                               "home it manually and restart the batch")
                        clients.rehome_and_connect_gantry()
                        scans_since_home = 0
                    elif clients.gantry is None:
                        raise RuntimeError("gantry not connected")

                    scan_dir.mkdir(parents=True, exist_ok=False)
                    if not args.no_snapshots:
                        snaps = grab_snapshots(cam_url, str(scan_dir), snap_warned)
                    if sizer is not None:
                        ir_info = {"reference": sizer.copy_reference_to(scan_dir),
                                   "start": ir_capture(sizer, cam_url, scan_dir, "start", args)}

                    scan_args = copy.copy(args)
                    scan_args.output_dir = str(scan_dir)
                    scans_since_home = None          # unknown until run_scan finishes cleanly
                    clients.radar.errors = []
                    sc.run_scan(clients.gantry, clients.radar, scan_args)

                    time.sleep(1.0)                  # let the DCA1000 finish the last file
                    if ir_info is not None:
                        ir_info["end"] = ir_capture(sizer, cam_url, scan_dir, "end", args)
                        lv = (ir_info["start"].get("level"), ir_info["end"].get("level"))
                        ir_info["changed_during_scan"] = bool(lv[0] and lv[1] and lv[0] != lv[1])
                        if ir_info["changed_during_scan"]:
                            log(f"IR: threat level changed during the scan ({lv[0]} -> {lv[1]}); "
                                "did something move, or did the mannequin/metal warm up?")
                    missing, sizes = check_rows(str(scan_dir), num_rows)
                    if clients.radar.errors:
                        error = f"radar reported: {clients.radar.errors[0]}"
                    elif missing:
                        error = f"{len(missing)}/{num_rows} rows missing, empty or short (first: {missing[:5]})"
                    else:
                        # run_scan only returns normally-with-all-rows when every row completed,
                        # so the gantry is at the known end position.
                        # Drive back to start unless we're about to re-home over SSH anyway.
                        if args.skip_pi_ssh or args.rehome_every > 1:
                            return_to_home(clients, args)
                        scans_since_home = 1
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"

                duration = round(time.time() - t0, 1)
                record = {
                    "batch_id": batch_id,
                    "name": name,
                    "view": view,
                    "label": label,
                    "notes": args.notes,
                    "status": "ok" if error is None else "failed",
                    "error": error,
                    "attempt": attempt,
                    "started": started,
                    "duration_s": duration,
                    "folder": str(scan_dir),
                    "rows_expected": num_rows,
                    "rows_ok": num_rows - len(missing),
                    "row_bytes_min": min(sizes.values()) if sizes else 0,
                    "row_bytes_max": max(sizes.values()) if sizes else 0,
                    "snapshots": snaps,
                    "ir": ir_info,
                    "training": training_info,
                    "params": {
                        "width_mm": args.width, "height_mm": args.height,
                        "x_start_mm": args.x_start, "y_start_mm": args.y_start,
                        "y_step_mm": args.y_step, "speed_mms": args.speed,
                        "frame_periodicity_ms": args.frame_periodicity,
                        "num_frames": num_frames, "num_rows": num_rows,
                        "samples": args.samples, "snake": args.snake,
                    },
                    "reconstruction": {
                        "zstart": args.zstart, "zend": args.zend, "zstep": args.zstep,
                        "zindex": args.zindex, "algo": args.algo, "window": args.window,
                    },
                }

                if error is None:
                    with open(scan_dir / "scan_meta.json", "w", encoding="utf-8") as f:
                        json.dump(record, f, indent=2)
                    append_manifest(manifest, record)
                    log(f"Scan {name} OK in {duration}s")
                    progress(status="scan_ok", index=done + 1, total=args.count, name=name)
                    ok = True
                    break

                log(f"Scan {name} FAILED: {error}")
                append_manifest(manifest, record)
                progress(status="scan_failed", index=done + 1, total=args.count, name=name, error=error)
                if clients.radar is not None:
                    try:
                        clients.radar.drain()
                    except Exception:
                        clients.radar.close()
                        clients.radar = None
                if scan_dir.exists():
                    failed_dir = outputs_base / "_failed" / f"{name}_{_dt.datetime.now():%H%M%S}"
                    failed_dir.parent.mkdir(exist_ok=True)
                    try:
                        # os.rename, not shutil.move: if a file is still locked we want a clean
                        # failure, not a half-copied folder.
                        os.rename(scan_dir, failed_dir)
                        meta_dir = failed_dir
                    except OSError as e:
                        # e.g. a DCA1000 recorder process or Excel still has a row file open.
                        log(f"Couldn't move {name} to _failed ({e}). Leaving it in place, "
                            "marked failed in its scan_meta.json; retrying under a new name.")
                        meta_dir = scan_dir
                        name, _ = next_free_name(scan_prefix(args, view, label), 0, scan_base)
                    try:
                        with open(meta_dir / "scan_meta.json", "w", encoding="utf-8") as f:
                            json.dump(record, f, indent=2)
                    except OSError:
                        pass
                if args.skip_pi_ssh and scans_since_home is None:
                    raise RuntimeError("scan failed mid-way with --skip-pi-ssh; gantry position "
                                       "unknown. Home it and restart the batch.")

            results.append((name, ok))
            if not ok:
                log(f"Giving up on {name} after {args.retries + 1} attempts; moving on")
                idx += 1
                continue

            done += 1
            idx += 1
            if args.reconstruct != "none":
                if args.recon_parallel <= 0:
                    p = start_reconstruction(args, scan_dir)
                    p.wait()
                    recon_failures += reap([p])
                else:
                    recon_failures += reap(recon_procs, block_until_below=args.recon_parallel)
                    recon_procs.append(start_reconstruction(args, scan_dir))
            recon_failures += reap(recon_procs)

    except KeyboardInterrupt:
        log("Interrupted by user.")
    except Exception as e:
        log(f"Batch aborted: {type(e).__name__}: {e}")
    finally:
        clients.close_all()
        if recon_procs:
            log(f"Waiting for {len(recon_procs)} reconstruction(s) to finish...")
            while recon_procs:
                recon_failures += reap(recon_procs)
                time.sleep(1)

    ok_n = sum(1 for _, ok in results if ok)
    log("=" * 60)
    log(f"Batch finished: {ok_n} ok, {len(results) - ok_n} gave up, {attempts_total} attempts")
    if recon_failures:
        log(f"Reconstruction failures: {', '.join(recon_failures)}")
    log(f"Manifest: {manifest}")
    progress(status="complete", ok=ok_n, failed=len(results) - ok_n)


if __name__ == "__main__":
    main()