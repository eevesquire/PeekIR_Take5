#!/usr/bin/env python3
"""
PeekIR unified web dashboard (run on the laptop next to mmWave Studio).

  python peekir_dashboard.py
      Opens the dashboard in your default browser (disable: PEEKIR_OPEN_BROWSER=0).

  Or: uvicorn peekir_dashboard:app --host 127.0.0.1 --port 8080

By default this SSHs to the Pi and starts ``pi_camera_server.py`` so you only run
one command on the laptop. Disable with PI_SSH_AUTO_START=0 if you start the Pi
server yourself.

Environment:
  PI_BASE_URL       — Pi camera server base (default http://10.244.13.117:9000)
  PI_SSH_AUTO_START — 1 (default): SSH to Pi and run pi_camera_server.py; 0 to skip
  PEEKIR_POI_DB     — Path to POI embedding JSON (default facial_rec/demo_embedding.json)
  PEEKIR_FACE_COSINE_THRESHOLD — ArcFace match threshold (default 0.68, same as match_suspect.py)
  PEEKIR_FACE_WARMUP — 1 (default): background-load DeepFace models at startup so the first
    capture is not stuck on TensorFlow/ArcFace/RetinaFace load (POI vectors always come from JSON only).
  PI_SSH_HOST       — Pi hostname/IP for SSH (default: host from PI_BASE_URL)
  PI_SSH_USER       — SSH user (default peekir)
  PI_SSH_PROJECT_DIR — Project dir on Pi under $HOME (default PeekIR_Pi)
  PI_SSH_VENV       — venv folder on Pi (default venv)
  PEEKIR_DATA_ROOT  — Extra allowed root for loading Plotly HTML (optional)
  PEEKIR_DEBUG=1    — Extra logs (e.g. /api/config, traceback on snapshot httpx errors)
  PEEKIR_METAL_STATE — Folder for the IR metal-size reference/calibration (default ./ir_metal_state).
                      Needs ir_metal_size.py next to this file and numpy + opencv-python.
  PEEKIR_SCAN_IR    — 1 (default): capture + measure IR automatically before and after each scan,
                      saved in outputs/<scan_name>/ (ir_start.*, ir_end.*, ir_summary.json); 0 to disable
  PEEKIR_SCAN_IR_SECONDS — length of each automatic IR capture (default 2)
  PEEKIR_TRAINING_DIR — training data folder for "Save IR training sample" (default ./training_data;
                      needs training_export.py next to this file)

Debug:
  - Server logs SNAPSHOT/PROBE lines for Pi HTTP from this machine.
  - Live MJPEG <img> URLs are loaded by the browser; if those fail but PROBE works,
    check firewall / mixed content / Pi bound to 0.0.0.0:9000.
  - GET http://127.0.0.1:8080/api/pi/probe  — httpx GET {PI_BASE_URL}/api/health
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.responses import Response

REPO_ROOT = Path(__file__).resolve().parent
# Plotly/HTML viewers dropped here for the dashboard dropdown (see GET /api/viz/list).
VIZ_DIR = (REPO_ROOT / "visualizations").resolve()
OUTPUTS_DIR = (REPO_ROOT / "outputs").resolve()
# The page (HTML + CSS + JavaScript) lives in peekir_dashboard.html next to this file.
# It is read on every page load, so edits show up after a browser refresh (no restart).
DASHBOARD_HTML_PATH = REPO_ROOT / "peekir_dashboard.html"

_SCAN_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def _sanitize_scan_name(name: str) -> str:
    s = str(name).strip()
    if not s:
        raise ValueError("scan_name must not be empty")
    if len(s) > 120:
        raise ValueError("scan_name is too long (max 120 characters)")
    if not _SCAN_NAME_RE.fullmatch(s):
        raise ValueError(
            "scan_name may contain only letters, digits, underscores, and hyphens "
            "(must start with a letter or digit)"
        )
    return s


def _next_available_scan_name() -> str:
    n = 0
    while True:
        candidate = f"scan_{n}"
        if not (OUTPUTS_DIR / candidate).exists() and not (VIZ_DIR / f"{candidate}.html").exists():
            return candidate
        n += 1
        if n > 100_000:
            raise RuntimeError("exhausted scan_* names")


def _scan_name_slot_available(name: str) -> bool:
    return not (OUTPUTS_DIR / name).exists() and not (VIZ_DIR / f"{name}.html").exists()

PI_BASE_URL = os.getenv("PI_BASE_URL", "http://10.245.61.148:9000").rstrip("/")
PEEKIR_DATA_ROOT = os.getenv("PEEKIR_DATA_ROOT", "").strip()
# Set PEEKIR_DEBUG=1 for extra Pi connectivity logs (browser img URLs are client-side only).
PEEKIR_DEBUG = os.getenv("PEEKIR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

# ArcFace cosine distance threshold (same as facial_rec/match_suspect.py)
PEEKIR_FACE_COSINE_THRESHOLD = float(os.getenv("PEEKIR_FACE_COSINE_THRESHOLD", "0.68"))
PEEKIR_POI_DB = Path(
    os.getenv("PEEKIR_POI_DB", str(REPO_ROOT / "facial_rec" / "demo_embedding.json"))
).resolve()
# At runtime we only load vectors from this JSON (see build_poi_db.py); we never re-embed POI images per request.
PEEKIR_FACE_WARMUP = os.getenv("PEEKIR_FACE_WARMUP", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
PEEKIR_IR_GUN_WEIGHTS = Path(
    os.getenv(
        "PEEKIR_IR_GUN_WEIGHTS",
        str(REPO_ROOT / "weapon_detection" / "runs" / "detect" / "ir_gun_detector" / "weights" / "best.pt"),
    )
).resolve()
PEEKIR_IR_GUN_CONF = float(os.getenv("PEEKIR_IR_GUN_CONF", "0.25"))

# Display metadata when a POI face match succeeds (keys matched case-insensitively to embedding "name").
_POI_WANTED_RECORDS: list[dict[str, Any]] = [
    {
        "name": "Havish",
        "image_filename": "Havish.PNG",
        "full_name": "Havish Bandapalli",
        "age": 21,
        "dob": "05/22/2004",
        "crime": "Went 120 mph in a school zone and crashed into 3 kids",
        "wanted": True,
        "extra_info": "All 3 kids passed away on impact",
    },
    {
        "name": "Trisha",
        "image_filename": "Trisha.jpeg",
        "full_name": "Trisha Kommitla",
        "age": 21,
        "dob": "08/27/2004",
        "crime": "Got caught shoplifting at Sephora",
        "wanted": False,
        "extra_info": "She tried to steal items worth $345",
    },
    {
        "name": "Vidhya",
        "image_filename": "Vidhya.JPG",
        "full_name": "Vidhya Kommineni",
        "age": 21,
        "dob": "04/25/2004",
        "crime": "Hit and run",
        "wanted": True,
    },
    {
        "name": "Rayaan",
        "image_filename": "Rayaan.PNG",
        "full_name": "Rayaan Naveed",
        "age": 21,
        "dob": "06/06/2004",
        "crime": "Got caught selling illegal substances to middle schoolers",
        "wanted": False,
        "extra_info": "Spent a month in jail before getting released",
    },
    {
        "name": "Sean",
        "image_filename": "Sean.png",
        "full_name": "Sean Conway",
        "age": 23,
        "dob": "01/08/2003",
        "crime": "Got into a fight on campus grounds",
        "wanted": False,
        "extra_info": "He lost the fight",
    },
    {
        "name": "Rajat",
        "image_filename": "Rajat.jpeg",
        "full_name": "Rajat Sarkar",
        "age": 22,
        "dob": "11/27/2003",
        "crime": "Stabbed an elderly lady in the park and fled the crime scene",
        "wanted": True,
        "extra_info": "Last seen at the Aggie Park",
    },
]
_POI_RECORD_BY_NAME_LOWER: dict[str, dict[str, Any]] = {
    str(r["name"]).strip().lower(): r for r in _POI_WANTED_RECORDS
}


def _poi_detail_for_match(name: str | None) -> dict[str, Any] | None:
    if not name or not str(name).strip():
        return None
    return _POI_RECORD_BY_NAME_LOWER.get(str(name).strip().lower())


_poi_db_rows: list[dict[str, Any]] | None = None
_poi_db_mtime: float | None = None
_ir_gun_model: Any | None = None

PI_SSH_AUTO_START = os.getenv("PI_SSH_AUTO_START", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
PI_SSH_USER = os.getenv("PI_SSH_USER", "takefive")
PI_SSH_PROJECT_DIR = os.getenv("PI_SSH_PROJECT_DIR", "PeekIR_Final")
PI_SSH_VENV = os.getenv("PI_SSH_VENV", ".venv")


def _pi_ssh_host() -> str:
    h = os.getenv("PI_SSH_HOST", "").strip()
    if h:
        return h
    parsed = urlparse(PI_BASE_URL)
    if parsed.hostname:
        return parsed.hostname
    return "10.244.13.117"


def _remote_pi_cd_and_activate() -> str:
    return f"cd $HOME/{PI_SSH_PROJECT_DIR} && source {PI_SSH_VENV}/bin/activate"


def _build_pi_ssh_camera_cmd() -> list[str]:
    """SSH to Pi and run pi_camera_server (same idea as run_pi_camera_server.py)."""
    host = _pi_ssh_host()
    remote = f"{_remote_pi_cd_and_activate()} && exec python3 pi_camera_server.py"
    target = f"{PI_SSH_USER}@{host}"
    return [
        "ssh",
        "-T",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=15",
        target,
        f'bash -lc "{remote}"',
    ]


_pi_ssh_proc: subprocess.Popen | None = None


def _start_pi_ssh_camera_server() -> None:
    """Spawn background SSH; Pi runs pi_camera_server on port 9000."""
    global _pi_ssh_proc
    if not PI_SSH_AUTO_START:
        print(
            "[peekir_dashboard] PI_SSH_AUTO_START=0 — not starting Pi camera via SSH",
            flush=True,
        )
        return
    if _pi_ssh_proc is not None and _pi_ssh_proc.poll() is None:
        return
    cmd = _build_pi_ssh_camera_cmd()
    host = _pi_ssh_host()
    print(
        f"[peekir_dashboard] SSH → {PI_SSH_USER}@{host}  (remote: pi_camera_server.py)",
        flush=True,
    )
    try:
        _pi_ssh_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print(
            "[peekir_dashboard] 'ssh' not found on PATH — install OpenSSH client or set PI_SSH_AUTO_START=0",
            flush=True,
        )
    except OSError as e:
        print(f"[peekir_dashboard] Failed to spawn SSH for Pi camera: {e}", flush=True)


def _stop_pi_ssh_camera_server() -> None:
    global _pi_ssh_proc
    p = _pi_ssh_proc
    if p is None or p.poll() is not None:
        return
    print("[peekir_dashboard] Stopping Pi camera SSH session…", flush=True)
    try:
        p.terminate()
        p.wait(timeout=12)
    except subprocess.TimeoutExpired:
        try:
            p.kill()
            p.wait(timeout=5)
        except Exception:
            pass
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
    _pi_ssh_proc = None


def _wait_for_pi_camera_http(timeout_s: float = 60.0) -> None:
    """Background: poll /api/health until Pi responds (after SSH start)."""
    if not PI_SSH_AUTO_START:
        return
    url = f"{PI_BASE_URL}/api/health"
    deadline = time.time() + timeout_s
    print(f"[peekir_dashboard] Waiting for Pi camera {url} …", flush=True)
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code == 200:
                print("[peekir_dashboard] Pi camera /api/health OK", flush=True)
                return
            print(f"[peekir_dashboard] Pi camera health status={r.status_code}", flush=True)
        except httpx.RequestError as e:
            if PEEKIR_DEBUG:
                print(f"[peekir_dashboard] Pi camera wait: {e!r}", flush=True)
        time.sleep(1.0)
    print(
        "[peekir_dashboard] Pi camera not ready yet — check SSH/Pi or open /api/pi/probe",
        flush=True,
    )

_scan_lock = threading.Lock()
_scan_proc: subprocess.Popen | None = None
_scan_status: dict[str, Any] = {
    "state": "idle",
    "last_progress": None,
    "exit_code": None,
    "width_mm": None,
    "height_mm": None,
    "scan_name": None,
    "log_tail": [],
}
_sse_loop: asyncio.AbstractEventLoop | None = None
_sse_subscribers: list[asyncio.Queue] = []

_viz_docs: dict[str, Path] = {}


def _allowed_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    roots = [REPO_ROOT.resolve()]
    if PEEKIR_DATA_ROOT:
        try:
            roots.append(Path(PEEKIR_DATA_ROOT).resolve())
        except OSError:
            pass
    for r in roots:
        try:
            resolved.relative_to(r)
            return True
        except ValueError:
            continue
    return False


def _peekir_viz_pixel_size(path: Path) -> tuple[int | None, int | None]:
    """Read `<!-- PEEKIR_VIZ_SIZE:WxH -->` from sar_reconstruct HTML (first 512 bytes)."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
        text = head.decode("utf-8", errors="replace")
    except OSError:
        return None, None
    m = re.search(r"<!--\s*PEEKIR_VIZ_SIZE:(\d+)x(\d+)\s*-->", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


_VIZ_EMBED_STYLE = (
    '<style id="peekir-viz-embed">'
    "html,body{margin:0!important;padding:0!important;overflow:hidden!important;"
    "height:100%!important;width:100%!important;box-sizing:border-box}"
    ".plotly-graph-div,.js-plotly-plot{overflow:hidden!important}"
    "</style>"
)


def _inject_viz_embed_css(html: str) -> str:
    """Remove default body margins and iframe scrollbars for embedded Plotly HTML."""
    if "peekir-viz-embed" in html:
        return html
    low = html.lower()
    i = low.find("<head")
    if i < 0:
        return _VIZ_EMBED_STYLE + html
    j = html.find(">", i)
    if j < 0:
        return _VIZ_EMBED_STYLE + html
    return html[: j + 1] + _VIZ_EMBED_STYLE + html[j + 1 :]


def _safe_visualizations_file(name: str) -> Path | None:
    """Resolve ``name`` to a single HTML file under ``visualizations/`` (no path traversal)."""
    n = name.strip()
    if not n or "/" in n or "\\" in n or ".." in n:
        return None
    try:
        base = VIZ_DIR.resolve()
    except OSError:
        return None
    if not base.is_dir():
        return None
    try:
        p = (base / n).resolve()
        p.relative_to(base)
    except (OSError, ValueError):
        return None
    if p.suffix.lower() not in {".html", ".htm"} or not p.is_file():
        return None
    return p


def _broadcast_sse_line(line: str) -> None:
    obj: dict[str, Any] = {"line": line.rstrip("\n")}
    if "__PEEKIR_PROGRESS__" in line:
        try:
            idx = line.index("__PEEKIR_PROGRESS__")
            rest = line[idx + len("__PEEKIR_PROGRESS__") :].strip()
            p = json.loads(rest)
            obj["progress"] = p
            with _scan_lock:
                _scan_status["last_progress"] = p
        except (json.JSONDecodeError, ValueError):
            pass
    if _sse_loop is None:
        return
    for q in _sse_subscribers:
        try:
            _sse_loop.call_soon_threadsafe(q.put_nowait, obj)
        except Exception:
            pass


def _scan_reader_thread(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    with _scan_lock:
        scan_name_for_ir = _scan_status.get("scan_name")
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        with _scan_lock:
            tail = list(_scan_status.get("log_tail") or [])
            tail.append(line.rstrip("\n"))
            _scan_status["log_tail"] = tail[-200:]
        _broadcast_sse_line(line)
        if scan_name_for_ir and "__PEEKIR_PROGRESS__" in line and '"ready_for_ir"' in line:
            _ir_after_homing(scan_name_for_ir)
    proc.wait()
    with _scan_lock:
        global _scan_proc
        _scan_status["exit_code"] = proc.returncode
        _scan_status["state"] = "done" if proc.returncode == 0 else "error"
        if _scan_proc is proc:
            _scan_proc = None
    _broadcast_sse_line(
        "__PEEKIR_PROGRESS__ "
        + json.dumps({"phase": "process", "exit_code": proc.returncode})
        + "\n"
    )
    if PEEKIR_SCAN_IR and scan_name_for_ir:
        _finish_scan_ir(scan_name_for_ir, proc.returncode)
    if scan_name_for_ir:
        _after_dashboard_scan(scan_name_for_ir, proc.returncode)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _sse_loop
    _sse_loop = asyncio.get_event_loop()
    print(
        "[peekir_dashboard] Startup: PI_BASE_URL="
        f"{PI_BASE_URL!r}  (live camera <img> tags use this host in the browser; "
        "snapshots use httpx here)",
        flush=True,
    )
    if PEEKIR_DATA_ROOT:
        print(f"[peekir_dashboard] PEEKIR_DATA_ROOT={PEEKIR_DATA_ROOT!r}", flush=True)

    _start_pi_ssh_camera_server()

    async def _wait_pi_bg() -> None:
        await asyncio.to_thread(_wait_for_pi_camera_http)

    asyncio.create_task(_wait_pi_bg())

    if PEEKIR_FACE_WARMUP:
        threading.Thread(
            target=_deepface_models_warmup,
            daemon=True,
            name="peekir-deepface-warmup",
        ).start()
    yield
    _stop_pi_ssh_camera_server()


app = FastAPI(title="PeekIR Dashboard", lifespan=_lifespan)


@app.get("/api/config")
async def api_config():
    if PEEKIR_DEBUG:
        print(f"[peekir_dashboard] GET /api/config -> pi_base_url={PI_BASE_URL!r}", flush=True)
    return {
        "pi_base_url": PI_BASE_URL,
        "suggested_scan_name": _next_available_scan_name(),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        return DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(500, f"{DASHBOARD_HTML_PATH.name} not found: keep it in the same folder as "
                                 f"{Path(__file__).name}")


# --------------------------------------------------------------------------- IR snapshot after homing
# sar_complete.py --ir-sync-file: after homing the gantry it prints {"phase":"ready_for_ir"} and waits
# for the file; the dashboard takes the "before" IR snapshot, then creates the file. With
# --home-after-scan the gantry is homed again right after the capture, so the "after" snapshot
# (taken when sar_complete.py exits) is not blocked either.
#   PEEKIR_IR_AFTER_HOMING=0  -> old behaviour: "before" snapshot as soon as Start scan is pressed
#   PEEKIR_HOME_AFTER_SCAN=0  -> leave the gantry where the scan ended
def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


PEEKIR_IR_AFTER_HOMING = _env_on("PEEKIR_IR_AFTER_HOMING")
PEEKIR_HOME_AFTER_SCAN = _env_on("PEEKIR_HOME_AFTER_SCAN")


def _ir_after_homing(scan_name: str) -> None:
    """Called from the scan reader thread on 'ready_for_ir'. Never raises; always releases the scan."""
    sync = (_scan_requests.get(scan_name) or {}).get("ir_sync")
    try:
        ir_start = _scan_ir_capture(scan_name, "start", _IR_PENDING_DIR / scan_name)
        with _scan_lock:
            _scan_status["ir"] = {"start": ir_start}
    except Exception as e:
        print(f"[peekir_dashboard] IR after homing failed for {scan_name}: {e}", flush=True)
    finally:
        if sync is not None:
            try:
                Path(sync).touch()
            except OSError as e:
                print(f"[peekir_dashboard] could not release the scan ({sync}): {e}", flush=True)


# --------------------------------------------------------------------------- scan options (same as sar_batch.py)
# Optional fields on POST /api/scan/start, passed to sar_complete.py:
#   x_start_mm (-x), y_start_mm (-y), y_step_mm (--y-step), zstart_mm / zend_mm / zstep_mm (--zstart/--zend/--zstep)
# Labels for the scan_meta.json + training export (same meaning as sar_batch.py):
#   view, label, metal ("yes"/"no"/""), object, true_area_cm2, notes, training (true/false)
_scan_requests: dict[str, dict[str, Any]] = {}
_SAFE_PART = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")


def _opt_num(body: dict[str, Any], key: str, *, positive: bool = False) -> float | None:
    v = body.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{key} must be a number")
    if f != f or f in (float("inf"), float("-inf")):
        raise HTTPException(400, f"{key} must be a number")
    if positive and f <= 0:
        raise HTTPException(400, f"{key} must be greater than 0")
    return f


def _num_arg(f: float) -> str:
    return str(int(f)) if float(f).is_integer() else str(f)


def _parse_scan_options(body: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    x0 = _opt_num(body, "x_start_mm")
    y0 = _opt_num(body, "y_start_mm")
    ys = _opt_num(body, "y_step_mm", positive=True)
    z0 = _opt_num(body, "zstart_mm", positive=True)
    z1 = _opt_num(body, "zend_mm", positive=True)
    zs = _opt_num(body, "zstep_mm", positive=True)
    for key, val in (("X start", x0), ("Y start", y0)):
        if val is not None and val < 0:
            raise HTTPException(400, f"{key} must be 0 or more (the gantry starts from its home position)")
    if z0 is not None and z1 is not None and z1 <= z0:
        raise HTTPException(400, "Z end must be larger than Z start")
    area = _opt_num(body, "true_area_cm2", positive=True)
    parts = {}
    for key in ("view", "label"):
        v = str(body.get(key) or "").strip()
        if v and not _SAFE_PART.fullmatch(v):
            raise HTTPException(400, f"{key} may contain only letters, digits, '_' and '-' (e.g. no_metal)")
        parts[key] = v
    metal = str(body.get("metal") or "").strip().lower()
    if metal not in ("", "yes", "no"):
        raise HTTPException(400, "metal must be yes, no or empty (= from the label)")
    weapon_type = str(body.get("weapon_type") or "").strip().lower()
    if weapon_type not in ("", "gun", "knife", "other", "none"):
        raise HTTPException(400, "weapon type must be gun, knife, other, none or empty")
    args: list[str] = ["-o", str(OUTPUTS_DIR)]
    for flag, val in (("-x", x0), ("-y", y0), ("--y-step", ys), ("--zstart", z0), ("--zend", z1), ("--zstep", zs)):
        if val is not None:
            args += [flag, _num_arg(val)]
    opts = {
        "x_start_mm": x0, "y_start_mm": y0, "y_step_mm": ys, "zstart_mm": z0, "zend_mm": z1, "zstep_mm": zs,
        "view": parts["view"], "label": parts["label"], "metal": metal,
        "object": str(body.get("object") or "").strip()[:80], "true_area_cm2": area, "weapon_type": weapon_type,
        "notes": str(body.get("notes") or "").strip()[:500],
        "training": body.get("training", True) not in (False, "false", "0", 0, "no"),
    }
    return args, opts


def _scan_params_from_args(cmd_args: list[str]) -> tuple[dict, dict]:
    """Exact scan/reconstruction settings, parsed the same way sar_complete.py parses them."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import sar_complete as scomp
    import sar_coordinator as sc

    a = scomp.build_parser().parse_args(cmd_args)
    params = {
        "width_mm": a.width, "height_mm": a.height,
        "x_start_mm": getattr(a, "x_start", None), "y_start_mm": getattr(a, "y_start", None),
        "y_step_mm": a.y_step, "speed_mms": a.speed, "frame_periodicity_ms": a.frame_periodicity,
        "num_frames": sc.compute_num_frames(a), "num_rows": sc.num_rows_from_height_step(a.height, a.y_step),
        "samples": a.samples, "snake": a.snake,
    }
    recon = {"zstart": a.zstart, "zend": a.zend, "zstep": a.zstep, "zindex": a.zindex,
             "algo": a.algo, "window": a.window}
    return params, recon


def _write_dashboard_scan_meta(scan_name: str, returncode: int | None) -> dict[str, Any] | None:
    req = _scan_requests.get(scan_name) or {}
    opts = req.get("opts") or {}
    final = OUTPUTS_DIR / scan_name
    if not final.is_dir():
        return None
    try:
        params, recon = _scan_params_from_args(req.get("cmd_args") or [])
    except Exception as e:
        print(f"[peekir_dashboard] scan_meta: could not read sar_complete settings ({e}); saving what is known",
              flush=True)
        params = {"width_mm": req.get("width_mm"), "height_mm": req.get("height_mm"),
                  "x_start_mm": opts.get("x_start_mm"), "y_start_mm": opts.get("y_start_mm"),
                  "y_step_mm": opts.get("y_step_mm")}
        recon = {"zstart": opts.get("zstart_mm"), "zend": opts.get("zend_mm"), "zstep": opts.get("zstep_mm")}
    metal = None
    try:
        import training_export as tx

        metal = tx.infer_metal(opts.get("label"), tx.parse_yes_no(opts["metal"]) if opts.get("metal") else None)
    except Exception:
        metal = {"yes": True, "no": False}.get(opts.get("metal") or "")
    ir = None
    with contextlib.suppress(Exception):
        ir = json.loads((final / "ir_summary.json").read_text(encoding="utf-8"))
    record = {
        "name": scan_name, "view": opts.get("view", ""), "label": opts.get("label", ""),
        "notes": opts.get("notes", ""), "status": "ok" if returncode == 0 else "failed",
        "error": None if returncode == 0 else f"sar_complete.py exited with code {returncode}",
        "started": req.get("started"), "folder": str(final),
        "ir": ir,
        "training": {"view": opts.get("view", ""), "metal": metal, "object": opts.get("object", ""),
                     "true_area_cm2": opts.get("true_area_cm2"),
                     "weapon_type": opts.get("weapon_type") or ("none" if metal is False else "")},
        "params": params, "reconstruction": recon,
        "created_by": "peekir_dashboard.py (Start scan)",
    }
    (final / "scan_meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _scan_log(msg: str) -> None:
    print(f"[peekir_dashboard] {msg}", flush=True)
    _broadcast_sse_line(f"[dashboard] {msg}\n")


def _after_dashboard_scan(scan_name: str, returncode: int | None) -> None:
    """scan_meta.json, threat result to the GUI, then the training export (never raises)."""
    req = _scan_requests.get(scan_name) or {}
    opts = req.get("opts") or {}
    record = None
    try:
        record = _write_dashboard_scan_meta(scan_name, returncode)
    except Exception as e:
        _scan_log(f"scan_meta.json not written for {scan_name}: {e}")
    _broadcast_scan_threat(scan_name)
    if returncode != 0 or record is None or not opts.get("training", True):
        _scan_requests.pop(scan_name, None)
        return
    if record["training"]["metal"] is None:
        _scan_log(f"training: {scan_name} not exported; set Metal to yes/no or use a label like "
                  "'metal' / 'nometal'")
    else:
        try:
            import training_export as tx

            tx.export_scan(OUTPUTS_DIR / scan_name, PEEKIR_TRAINING_DIR,
                           state_dir=Path(__file__).resolve().parent / "ir_metal_state",
                           log=_scan_log)
        except Exception as e:
            _scan_log(f"training export failed for {scan_name}: {e}")
    _scan_requests.pop(scan_name, None)



@app.post("/api/scan/start")
async def scan_start(body: dict[str, Any]):
    global _scan_proc
    w = body.get("width_mm")
    h = body.get("height_mm")
    if w is None or h is None:
        raise HTTPException(400, "width_mm and height_mm required")
    try:
        wf = float(w)
        hf = float(h)
    except (TypeError, ValueError):
        raise HTTPException(400, "width_mm and height_mm must be numbers")
    if wf <= 0 or hf <= 0:
        raise HTTPException(400, "dimensions must be positive")

    raw_sn = body.get("scan_name")
    if raw_sn is None or not str(raw_sn).strip():
        raise HTTPException(400, "scan_name required")
    try:
        scan_name = _sanitize_scan_name(str(raw_sn))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not _scan_name_slot_available(scan_name):
        raise HTTPException(
            409,
            f"Name '{scan_name}' is already used under outputs/ or visualizations/. "
            "Choose a different scan name.",
        )
    extra_args, scan_opts = _parse_scan_options(body)

    with _scan_lock:
        if _scan_proc is not None and _scan_proc.poll() is None:
            raise HTTPException(409, "A scan is already running")
        _scan_status.update(
            {
                "state": "running",
                "last_progress": None,
                "exit_code": None,
                "width_mm": wf,
                "height_mm": hf,
                "scan_name": scan_name,
                "log_tail": [],
                "ir": None,
            }
        )

    # Automatic IR capture + metal measurement just before the scan starts (see _scan_ir_capture).
    # Normally it is taken later, once sar_complete.py has homed the gantry (see _ir_after_homing).
    if PEEKIR_SCAN_IR and not PEEKIR_IR_AFTER_HOMING:
        ir_start = await asyncio.to_thread(
            _scan_ir_capture, scan_name, "start", _IR_PENDING_DIR / scan_name
        )
        with _scan_lock:
            _scan_status["ir"] = {"start": ir_start}

    # Child stdout is a pipe (not a console); without UTF-8 mode, prints containing
    # Unicode in sar_coordinator can raise UnicodeEncodeError on Windows (cp1252),
    # aborting run_scan before CONFIGURE so the Lua client only sees SHUTDOWN.
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(REPO_ROOT / "sar_complete.py"),
        "-W",
        str(wf),
        "-H",
        str(hf),
        "--scan-name",
        scan_name,
        "--viz-html-root",
        str(VIZ_DIR),
    ]
    cmd += extra_args
    ir_sync = None
    if PEEKIR_SCAN_IR and PEEKIR_IR_AFTER_HOMING:
        ir_sync = OUTPUTS_DIR / ".ir_pending" / f"{scan_name}.go"
        ir_sync.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            ir_sync.unlink()
        cmd += ["--ir-sync-file", str(ir_sync)]
    if PEEKIR_SCAN_IR and PEEKIR_HOME_AFTER_SCAN:
        cmd += ["--home-after-scan"]
    _scan_requests[scan_name] = {"opts": scan_opts, "cmd_args": cmd[4:], "width_mm": wf, "height_mm": hf,
                                 "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "ir_sync": ir_sync}
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _scan_lock:
        _scan_proc = proc

    threading.Thread(target=_scan_reader_thread, args=(proc,), daemon=True).start()
    return {"ok": True, "pid": proc.pid}


@app.post("/api/scan/cancel")
async def scan_cancel():
    global _scan_proc
    with _scan_lock:
        p = _scan_proc
    if p is None or p.poll() is not None:
        return {"ok": False, "detail": "no running scan"}
    p.terminate()
    try:
        p.wait(timeout=8)
    except subprocess.TimeoutExpired:
        p.kill()
    with _scan_lock:
        _scan_status["state"] = "cancelled"
        _scan_proc = None
    return {"ok": True}


@app.get("/api/scan/status")
async def scan_status():
    with _scan_lock:
        return dict(_scan_status)


@app.get("/api/scan/stream")
async def scan_stream():
    q: asyncio.Queue = asyncio.Queue()
    _sse_subscribers.append(q)

    async def gen():
        try:
            yield f"data: {json.dumps({'hello': True})}\n\n"
            while True:
                item = await q.get()
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            if q in _sse_subscribers:
                _sse_subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/viz/list")
async def viz_list():
    """List ``*.html`` files in the repo ``visualizations/`` folder."""
    files: list[dict[str, Any]] = []
    try:
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if not VIZ_DIR.is_dir():
        return {"dir": str(VIZ_DIR), "files": []}
    try:
        for p in sorted(VIZ_DIR.glob("*.html")):
            if p.is_file():
                try:
                    st = p.stat()
                    files.append(
                        {"name": p.name, "size": st.st_size, "mtime": st.st_mtime}
                    )
                except OSError:
                    files.append({"name": p.name, "size": None, "mtime": None})
    except OSError:
        pass
    return {"dir": str(VIZ_DIR), "files": files}


@app.post("/api/viz/load")
async def viz_load(body: dict[str, Any]):
    body = body or {}
    explicit_path = body.get("path")
    if explicit_path is not None and isinstance(explicit_path, str) and explicit_path.strip():
        raw = explicit_path.strip()
        p = Path(raw).expanduser()
        if not _allowed_path(p):
            raise HTTPException(403, "path must be under repo or PEEKIR_DATA_ROOT")
        if not p.is_file():
            raise HTTPException(404, "file not found")
        if p.suffix.lower() not in {".html", ".htm"}:
            raise HTTPException(400, "only HTML files are supported")
        resolved = p.resolve()
    else:
        vname = body.get("file") or body.get("name") or body.get("visualization")
        if not vname or not isinstance(vname, str):
            raise HTTPException(
                400,
                "Provide {\"file\": \"name.html\"} for files in visualizations/, or {\"path\": \"...\"}",
            )
        p = _safe_visualizations_file(vname)
        if p is None:
            raise HTTPException(
                404,
                "Unknown or invalid file. Use the name of an HTML file inside visualizations/.",
            )
        resolved = p

    doc_id = uuid.uuid4().hex
    _viz_docs[doc_id] = resolved
    wpx, hpx = _peekir_viz_pixel_size(resolved)
    out: dict[str, Any] = {"ok": True, "doc_id": doc_id, "name": resolved.name}
    if wpx is not None and hpx is not None:
        out["width_px"] = wpx
        out["height_px"] = hpx
    return out


@app.get("/api/viz/document/{doc_id}")
async def viz_document(doc_id: str):
    p = _viz_docs.get(doc_id)
    if p is None or not p.is_file():
        raise HTTPException(404, "unknown or expired document id")
    raw = await asyncio.to_thread(p.read_text, encoding="utf-8")
    html = _inject_viz_embed_css(raw)
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


# --------------------------------------------------------------------------- threat assessment (IR + mmWave)
# Uses mmwave_metal_size.py (same folder as this file). Looks for scan folders in outputs/ plus any
# folders listed in PEEKIR_THREAT_ROOTS (separated by os.pathsep), e.g. where sar_batch.py writes scans.

def _threat_roots() -> list[Path]:
    roots = [OUTPUTS_DIR]
    for part in os.getenv("PEEKIR_THREAT_ROOTS", "").split(os.pathsep):
        if part.strip():
            roots.append(Path(part.strip()).expanduser().resolve())
    return roots


def _threat_state_dir() -> Path:
    return Path(__file__).resolve().parent / "ir_metal_state"


def _threat_find(name: str) -> Path | None:
    from mmwave_metal_size import find_scan_dir

    n = (name or "").strip()
    if not n or "/" in n or "\\" in n or ".." in n:
        return None
    for root in _threat_roots():
        d = find_scan_dir(root, n)
        if d is not None:
            return d
    return None


def _threat_summary_for(name: str) -> dict:
    from mmwave_metal_size import scan_threat_summary

    d = _threat_find(name)
    if d is None:
        return {"found": False, "name": name}
    s = scan_threat_summary(d, state_dir=_threat_state_dir())
    s["found"] = True
    s["folder"] = str(d)
    return s


def _broadcast_scan_threat(scan_name: str) -> None:
    """After a dashboard scan: push the combined IR + mmWave result to open dashboards."""
    try:
        s = _threat_summary_for(scan_name)
    except Exception as e:
        s = {"found": False, "name": scan_name, "error": str(e)}
    s["phase"] = "threat"
    s["scan_name"] = scan_name
    _broadcast_sse_line("__PEEKIR_PROGRESS__ " + json.dumps(s) + "\n")


@app.get("/api/threat/scans")
async def threat_scans():
    """Scan folders that have IR and/or mmWave data, newest first."""
    from mmwave_metal_size import list_scans

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in _threat_roots():
        for r in await asyncio.to_thread(list_scans, root):
            if r["name"] not in seen:
                seen.add(r["name"])
                out.append(r)
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return {"scans": out, "roots": [str(r) for r in _threat_roots()]}


@app.get("/api/threat/scan")
async def threat_scan(name: str):
    """IR result, mmWave result and the combined verdict for one scan (measures mmWave if needed)."""
    try:
        return await asyncio.to_thread(_threat_summary_for, name)
    except Exception as e:
        raise HTTPException(500, f"threat summary failed: {e}")


@app.get("/api/threat/image")
async def threat_image(name: str, kind: str = "ir"):
    d = await asyncio.to_thread(_threat_find, name)
    fn = {"ir": "ir_start.png", "ir_end": "ir_end.png", "mmwave": "mmwave_result.png"}.get(kind)
    if d is None or fn is None or not (d / fn).is_file():
        raise HTTPException(404, "no image")
    return Response(content=(d / fn).read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


def _viz_scan_name(doc_id: str) -> str | None:
    p = _viz_docs.get(doc_id)
    return None if p is None else p.stem


@app.get("/api/threat/for-viz/{doc_id}")
async def threat_for_viz(doc_id: str):
    """Threat summary for the scan a loaded visualization belongs to (matched by file name)."""
    name = _viz_scan_name(doc_id)
    if name is None:
        raise HTTPException(404, "unknown or expired document id")
    try:
        return await asyncio.to_thread(_threat_summary_for, name)
    except Exception as e:
        return {"found": False, "name": name, "error": str(e)}


_VIZ_OVERLAY_JS = """
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


@app.get("/api/viz/document-overlay/{doc_id}")
async def viz_document_overlay(doc_id: str):
    """Same viewer as /api/viz/document, with the detected metal outlined and the threat level in the title."""
    from mmwave_metal_size import plotly_overlay

    p = _viz_docs.get(doc_id)
    if p is None or not p.is_file():
        raise HTTPException(404, "unknown or expired document id")
    raw = await asyncio.to_thread(p.read_text, encoding="utf-8")
    html = _inject_viz_embed_css(raw)
    d = await asyncio.to_thread(_threat_find, p.stem)
    ov = None
    if d is not None:
        await asyncio.to_thread(_threat_summary_for, p.stem)      # measures mmWave if the volume is new
        ov = await asyncio.to_thread(plotly_overlay, d)
    if ov:
        script = _VIZ_OVERLAY_JS.replace("__OVERLAY__", json.dumps(ov).replace("</", "<\\/"))
        i = html.lower().rfind("</body>")
        html = html[:i] + script + html[i:] if i >= 0 else html + script
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8",
                        headers={"X-Peekir-Overlay": "1" if ov else "0"})



async def _http_get_pi_frame_jpeg(which: str) -> bytes:
    """Fetch a single JPEG frame from the Pi (rgb or ir)."""
    path = "/api/camera/frame" if which.lower() == "rgb" else "/api/ir/frame"
    url = f"{PI_BASE_URL}{path}"
    print(f"[peekir_dashboard] FRAME which={which!r} GET {url}", flush=True)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
    except httpx.RequestError as e:
        print(
            f"[peekir_dashboard] FRAME httpx.RequestError: {type(e).__name__}: {e!r}",
            flush=True,
        )
        if PEEKIR_DEBUG:
            import traceback

            traceback.print_exc()
        raise HTTPException(502, f"Pi request failed: {e}") from e
    print(f"[peekir_dashboard] FRAME response status={r.status_code} bytes={len(r.content)}", flush=True)
    if r.status_code != 200:
        snippet = (r.text or "")[:400]
        print(f"[peekir_dashboard] FRAME non-200 body snippet: {snippet!r}", flush=True)
        raise HTTPException(502, f"Pi returned {r.status_code}: {snippet[:200]}")
    return r.content


def _resolve_poi_image_path(stored: str) -> Path:
    p = Path(stored.replace("\\", "/"))
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _load_poi_db_cached() -> list[dict[str, Any]]:
    """Load POI rows from JSON once; reload if the file is replaced (mtime change)."""
    global _poi_db_rows, _poi_db_mtime
    if not PEEKIR_POI_DB.is_file():
        raise FileNotFoundError(str(PEEKIR_POI_DB))
    mtime = PEEKIR_POI_DB.stat().st_mtime
    if _poi_db_rows is not None and _poi_db_mtime == mtime:
        return _poi_db_rows
    with open(PEEKIR_POI_DB, encoding="utf-8") as f:
        _poi_db_rows = json.load(f)
    _poi_db_mtime = mtime
    return _poi_db_rows


def _deepface_models_warmup() -> None:
    """One represent() call loads ArcFace + RetinaFace + TF; POI matching still uses JSON vectors only."""
    try:
        from deepface import DeepFace
    except ImportError:
        print(
            "[peekir_dashboard] Face model warm-up skipped (deepface not installed)",
            flush=True,
        )
        return
    poi_dir = REPO_ROOT / "facial_rec" / "poi"
    sample: Path | None = None
    if poi_dir.is_dir():
        for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            found = sorted(poi_dir.glob(pat))
            if found:
                sample = found[0]
                break
    if sample is None:
        print(
            "[peekir_dashboard] Face model warm-up skipped (no facial_rec/poi/*.jpg|png)",
            flush=True,
        )
        return
    print(
        f"[peekir_dashboard] Face model warm-up (ArcFace/RetinaFace) using {sample.name} …",
        flush=True,
    )
    try:
        DeepFace.represent(
            img_path=str(sample),
            model_name="ArcFace",
            detector_backend="retinaface",
        )
        print("[peekir_dashboard] Face model warm-up done", flush=True)
    except Exception as e:
        print(f"[peekir_dashboard] Face warm-up failed (non-fatal): {e}", flush=True)


def _face_match_rgb_jpeg(jpeg_bytes: bytes) -> dict[str, Any]:
    """Embed the capture with ArcFace; match POI using precomputed vectors in demo_embedding.json only."""
    import numpy as np
    from PIL import Image, ImageDraw
    from deepface import DeepFace

    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return float(1 - np.dot(a, b))

    pil = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    np_img = np.array(pil)

    try:
        poi_db = _load_poi_db_cached()
    except FileNotFoundError:
        return {"ok": False, "error": f"POI database not found: {PEEKIR_POI_DB}"}

    try:
        sus_reps = DeepFace.represent(
            img_path=np_img,
            model_name="ArcFace",
            detector_backend="retinaface",
        )
    except Exception as e:
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return {
            "ok": True,
            "face_detected": False,
            "matched": False,
            "name": None,
            "distance": None,
            "threshold": PEEKIR_FACE_COSINE_THRESHOLD,
            "annotated_image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "poi_image_b64": None,
            "poi_detail": None,
            "no_face_detail": str(e),
        }

    sus_rep = sus_reps[0] if isinstance(sus_reps, list) else sus_reps
    sus_emb = np.array(sus_rep["embedding"], dtype="float32")

    fa = sus_rep.get("facial_area") or {}
    x, y, w, h = fa.get("x"), fa.get("y"), fa.get("w"), fa.get("h")
    draw = ImageDraw.Draw(pil)
    if None not in (x, y, w, h):
        x, y, w, h = int(x), int(y), int(w), int(h)
        t = max(2, int(round(min(pil.size) * 0.004)))
        draw.rectangle([x, y, x + w, y + h], outline="#ff3333", width=t)

    best_name: str | None = None
    best_path: str | None = None
    best_dist = 999.0
    for poi in poi_db:
        poi_emb = np.array(poi["embedding"], dtype="float32")
        dist = cosine_distance(sus_emb, poi_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = poi.get("name")
            best_path = poi.get("image_path")

    matched = best_dist <= PEEKIR_FACE_COSINE_THRESHOLD

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    poi_b64: str | None = None
    if matched and best_path:
        p = _resolve_poi_image_path(str(best_path))
        if p.is_file():
            poi_b64 = base64.b64encode(p.read_bytes()).decode("ascii")

    return {
        "ok": True,
        "face_detected": True,
        "matched": matched,
        "name": best_name,
        "distance": float(best_dist),
        "threshold": PEEKIR_FACE_COSINE_THRESHOLD,
        "annotated_image_b64": annotated_b64,
        "poi_image_b64": poi_b64,
        "poi_detail": _poi_detail_for_match(best_name) if matched else None,
    }


def _load_ir_gun_model_cached() -> Any:
    global _ir_gun_model
    if _ir_gun_model is not None:
        return _ir_gun_model
    if not PEEKIR_IR_GUN_WEIGHTS.is_file():
        raise FileNotFoundError(str(PEEKIR_IR_GUN_WEIGHTS))
    from ultralytics import YOLO

    _ir_gun_model = YOLO(str(PEEKIR_IR_GUN_WEIGHTS))
    return _ir_gun_model


def _detect_ir_gun_jpeg(jpeg_bytes: bytes) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw

    pil = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    np_img = np.array(pil)

    model = _load_ir_gun_model_cached()
    results = model.predict(source=np_img, conf=PEEKIR_IR_GUN_CONF, verbose=False)
    result = results[0]
    boxes = result.boxes

    draw = ImageDraw.Draw(pil)
    detections: list[dict[str, Any]] = []
    has_gun = boxes is not None and len(boxes) > 0
    if boxes is not None:
        t = max(2, int(round(min(pil.size) * 0.004)))
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            draw.rectangle([x1, y1, x2, y2], outline="#ffd400", width=t)
            draw.text((x1 + 2, max(4, y1 - 18)), f"gun {conf:.2f}", fill="#ffd400")
            detections.append(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf, "class_name": "gun"}
            )

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "ok": True,
        "gun_detected": has_gun,
        "count": len(detections),
        "threshold": PEEKIR_IR_GUN_CONF,
        "detections": detections,
        "annotated_image_b64": annotated_b64,
    }


# --------------------------------------------------------------------------- Capture IR: AI metal check
# PEEKIR_IR_DETECTOR = "ai" (default): the team's ThermalCNN metal / no-metal model (ir_ai_metal.py,
# model file ir_metal_model.pth / ir_knife_model.pth or PEEKIR_IR_AI_MODEL).
#                    = "yolo": the previous team's YOLO gun detector (/api/pi/ir-gun-detect).
PEEKIR_IR_DETECTOR = os.getenv("PEEKIR_IR_DETECTOR", "ai").strip().lower()


@app.get("/api/pi/ir-detect")
async def pi_ir_detect():
    """Capture IR button: one IR frame from the Pi -> the configured detector."""
    if PEEKIR_IR_DETECTOR == "yolo":
        out = await pi_ir_gun_detect()
        out["summary"] = ("Gun detected" if out.get("gun_detected") else "No gun detected") + " (YOLO, previous team)"
        return out
    jpeg_bytes = await _http_get_pi_frame_jpeg("ir")
    try:
        import ir_ai_metal

        return await asyncio.to_thread(ir_ai_metal.predict_jpeg, jpeg_bytes)
    except ImportError as e:
        raise HTTPException(501, f"AI check needs PyTorch: pip install torch pillow ({e})") from e
    except FileNotFoundError as e:
        raise HTTPException(500, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"AI metal check failed: {e}") from e


@app.get("/api/pi/probe")
async def pi_probe():
    """GET Pi /api/health via httpx — same path as snapshot connectivity checks."""
    url = f"{PI_BASE_URL}/api/health"
    print(f"[peekir_dashboard] PROBE GET {url}", flush=True)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
    except httpx.RequestError as e:
        print(f"[peekir_dashboard] PROBE FAILED: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(502, f"Pi unreachable: {e}") from e
    txt = (r.text or "")[:500]
    print(f"[peekir_dashboard] PROBE status={r.status_code} body[:500]={txt!r}", flush=True)
    try:
        data = r.json()
    except json.JSONDecodeError:
        data = {"raw": txt}
    return {"ok": r.status_code == 200, "status_code": r.status_code, "pi_url": url, "data": data}


@app.get("/api/pi/snapshot")
async def pi_snapshot(which: str = "rgb"):
    print(f"[peekir_dashboard] SNAPSHOT which={which!r}", flush=True)
    content = await _http_get_pi_frame_jpeg(which)
    return Response(content=content, media_type="image/jpeg")


@app.get("/api/pi/rgb-face-match")
async def pi_rgb_face_match():
    """Capture RGB frame from Pi, run POI match (ArcFace + cosine), return JSON with base64 PNGs."""
    try:
        jpeg_bytes = await _http_get_pi_frame_jpeg("rgb")
    except HTTPException:
        raise
    try:
        out = await asyncio.to_thread(_face_match_rgb_jpeg, jpeg_bytes)
    except ImportError as e:
        raise HTTPException(
            501,
            "deepface/Pillow not installed — pip install deepface pillow numpy",
        ) from e
    if not out.get("ok"):
        raise HTTPException(500, out.get("error", "face match failed"))
    return out


@app.get("/api/pi/ir-gun-detect")
async def pi_ir_gun_detect():
    """Capture IR frame from Pi, run local gun detector, return annotated image + status."""
    try:
        jpeg_bytes = await _http_get_pi_frame_jpeg("ir")
    except HTTPException:
        raise
    try:
        out = await asyncio.to_thread(_detect_ir_gun_jpeg, jpeg_bytes)
    except FileNotFoundError as e:
        raise HTTPException(500, f"IR detector weights not found: {e}") from e
    except ImportError as e:
        raise HTTPException(
            501,
            "IR detector dependencies missing — pip install ultralytics pillow numpy",
        ) from e
    except Exception as e:
        raise HTTPException(500, f"IR gun detection failed: {e}") from e
    if not out.get("ok"):
        raise HTTPException(500, out.get("error", "IR gun detection failed"))
    return out


# --------------------------------------------------------------------------- IR metal size -> threat level
# Uses ir_metal_size.py (same folder as this file). Frames come from the Pi's /api/ir/burst.
# Flow: "Set reference" with NO metal on the mannequin -> (once) "Calibrate" with a plate of
# known size -> "Measure metal". State is kept in PEEKIR_METAL_STATE (default ./ir_metal_state),
# shared with the ir_metal_size.py command line.
PEEKIR_METAL_STATE = Path(
    os.getenv("PEEKIR_METAL_STATE", str(REPO_ROOT / "ir_metal_state"))
).resolve()
_metal_sizer: Any | None = None
_metal_lock = threading.Lock()


def _get_metal_sizer() -> Any:
    global _metal_sizer
    if _metal_sizer is None:
        from ir_metal_size import MetalSizer

        _metal_sizer = MetalSizer(PEEKIR_METAL_STATE)
    return _metal_sizer


def _metal_call(fn):
    """Run a MetalSizer action under a lock, mapping errors to HTTP responses."""
    try:
        with _metal_lock:
            return fn()
    except ImportError as e:
        raise HTTPException(
            501, f"IR metal sizing needs numpy + opencv-python (pip install numpy opencv-python): {e}"
        ) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


async def _http_get_pi_ir_burst(seconds: float) -> tuple[Any, str]:
    """Fetch `seconds` of raw IR frames from the Pi as (frames ndarray, ir_format)."""
    import numpy as np

    url = f"{PI_BASE_URL}/api/ir/burst"
    print(f"[peekir_dashboard] IR BURST GET {url} seconds={seconds}", flush=True)
    try:
        async with httpx.AsyncClient(timeout=seconds + 20.0) as client:
            r = await client.get(url, params={"seconds": seconds})
    except httpx.RequestError as e:
        raise HTTPException(502, f"Pi request failed: {e}") from e
    if r.status_code == 404:
        raise HTTPException(
            502, "Pi camera server has no /api/ir/burst; copy the updated pi_camera_server.py to the Pi"
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Pi returned {r.status_code}: {(r.text or '')[:200]}")
    data = np.load(io.BytesIO(r.content))
    return data["frames"], r.headers.get("X-IR-Format", "y16")


def _capture_seconds(body: dict[str, Any] | None, sizer: Any) -> float:
    try:
        s = float((body or {}).get("seconds") or sizer.settings.capture_seconds)
    except (TypeError, ValueError):
        raise HTTPException(400, "seconds must be a number")
    return max(0.3, min(s, 30.0))


# ---- automatic IR capture around each dashboard scan ("Start scan")
# Before sar_complete.py starts: raw IR burst + metal measurement ("start"); after it exits
# successfully: another ("end"). Files land in outputs/<scan_name>/ (ir_start.*, ir_end.*,
# ir_reference.npz, ir_summary.json). Disable with PEEKIR_SCAN_IR=0.
PEEKIR_SCAN_IR = os.getenv("PEEKIR_SCAN_IR", "1").strip().lower() not in {"0", "false", "no", "off"}
try:
    PEEKIR_SCAN_IR_SECONDS = float(os.getenv("PEEKIR_SCAN_IR_SECONDS", "2"))
except ValueError:
    PEEKIR_SCAN_IR_SECONDS = 2.0
_IR_PENDING_DIR = OUTPUTS_DIR / ".ir_pending"


def _broadcast_scan_ir(scan_name: str, r: dict) -> None:
    payload = {
        "phase": "ir", "scan_name": scan_name, "tag": r.get("tag"), "level": r.get("level"),
        "largest_area_cm2": r.get("largest_area_cm2"), "largest_area_px": r.get("largest_area_px"),
        "pieces": r.get("pieces"), "calibrated": r.get("calibrated"),
        "error": r.get("error") or r.get("measure_error") or r.get("skipped"),
        "image": bool(r.get("image")),
        "notes": r.get("notes") or [],
        "weapons": r.get("weapons") or [], "objects": r.get("objects") or [],
    }
    _broadcast_sse_line("__PEEKIR_PROGRESS__ " + json.dumps(payload) + "\n")


def _scan_ir_capture(scan_name: str, tag: str, out_dir: Path) -> dict:
    """Never raises: an IR problem must not stop or fail the mmWave scan."""
    try:
        from ir_metal_size import capture_and_measure

        sizer = _get_metal_sizer()
        r = capture_and_measure(sizer, PI_BASE_URL, out_dir, tag, PEEKIR_SCAN_IR_SECONDS, lock=_metal_lock)
        if tag == "start":
            with _metal_lock:
                r["reference"] = sizer.copy_reference_to(out_dir)
    except Exception as e:
        r = {"tag": tag, "error": f"IR unavailable: {e}"}
    print(f"[peekir_dashboard] SCAN IR {tag} {scan_name}: "
          f"{r.get('level') or r.get('error') or r.get('measure_error')}", flush=True)
    _broadcast_scan_ir(scan_name, r)
    return r


def _finish_scan_ir(scan_name: str, returncode: int | None) -> None:
    """After the scan process exits: move the start capture into outputs/<scan>/, capture the end."""
    import shutil

    final = OUTPUTS_DIR / scan_name
    try:
        final.mkdir(parents=True, exist_ok=True)
        pending = _IR_PENDING_DIR / scan_name
        if pending.is_dir():
            for f in pending.iterdir():
                shutil.move(str(f), str(final / f.name))
            with contextlib.suppress(OSError):
                pending.rmdir()
        if returncode == 0:
            end = _scan_ir_capture(scan_name, "end", final)
        else:
            end = {"tag": "end", "skipped": f"scan exited with code {returncode}; no end capture"}
            _broadcast_scan_ir(scan_name, end)
        with _scan_lock:
            ir = dict(_scan_status.get("ir") or {})
            ir["end"] = end
            lv = ((ir.get("start") or {}).get("level"), end.get("level"))
            ir["changed_during_scan"] = bool(lv[0] and lv[1] and lv[0] != lv[1])
            _scan_status["ir"] = ir
            summary = {
                "scan_name": scan_name, "scan_exit_code": returncode,
                "width_mm": _scan_status.get("width_mm"), "height_mm": _scan_status.get("height_mm"),
                **ir,
            }
        (final / "ir_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[peekir_dashboard] SCAN IR finish failed for {scan_name}: {e}", flush=True)


@app.get("/api/ir/metal/scan-image")
async def ir_metal_scan_image(scan_name: str, tag: str = "start"):
    """Annotated IR image saved with a scan (ir_start.png / ir_end.png)."""
    try:
        name = _sanitize_scan_name(scan_name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if tag not in ("start", "end"):
        raise HTTPException(400, "tag must be start or end")
    for d in (OUTPUTS_DIR / name, _IR_PENDING_DIR / name):
        p = d / f"ir_{tag}.png"
        if p.is_file():
            return Response(content=p.read_bytes(), media_type="image/png",
                            headers={"Cache-Control": "no-store"})
    raise HTTPException(404, "no IR image for that scan yet")


@app.get("/api/ir/metal/status")
async def ir_metal_status():
    return _metal_call(lambda: _get_metal_sizer().status())


@app.post("/api/ir/metal/reference")
async def ir_metal_reference(body: dict[str, Any] | None = None):
    """Capture the no-metal reference."""
    sizer = _metal_call(_get_metal_sizer)
    frames, fmt = await _http_get_pi_ir_burst(_capture_seconds(body, sizer))
    return await asyncio.to_thread(_metal_call, lambda: sizer.set_reference(frames, fmt))


@app.post("/api/ir/metal/calibrate")
async def ir_metal_calibrate(body: dict[str, Any]):
    """Body: {"width_cm": 10, "height_cm": 15} or {"area_cm2": 150}. Plate must be on the mannequin."""
    sizer = _metal_call(_get_metal_sizer)
    try:
        if body.get("area_cm2") is not None:
            area = float(body["area_cm2"])
        else:
            area = float(body["width_cm"]) * float(body["height_cm"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "give area_cm2, or width_cm and height_cm")
    if area <= 0:
        raise HTTPException(400, "plate area must be positive")
    frames, fmt = await _http_get_pi_ir_burst(_capture_seconds(body, sizer))
    return await asyncio.to_thread(_metal_call, lambda: sizer.calibrate(frames, fmt, area))


@app.post("/api/ir/metal/measure")
async def ir_metal_measure(body: dict[str, Any] | None = None):
    """Measure metal against the reference; returns threat level, sizes and an annotated PNG."""
    sizer = _metal_call(_get_metal_sizer)
    frames, fmt = await _http_get_pi_ir_burst(_capture_seconds(body, sizer))

    def _run():
        from ir_metal_size import encode_png

        res, img = sizer.measure(frames, fmt)
        out = res.to_dict()
        out["annotated_image_b64"] = base64.b64encode(encode_png(img)).decode("ascii")
        out["settings"] = sizer.status()["settings"]
        return out

    return await asyncio.to_thread(_metal_call, _run)


@app.post("/api/ir/metal/diagnose")
async def ir_metal_diagnose(body: dict[str, Any] | None = None):
    """How much each pixel changed vs the reference (to tune detection when metal is missed)."""
    sizer = _metal_call(_get_metal_sizer)
    frames, fmt = await _http_get_pi_ir_burst(_capture_seconds(body, sizer))

    def _run():
        from ir_metal_size import encode_png

        stats, img = sizer.diagnose(frames, fmt)
        stats["image_b64"] = base64.b64encode(encode_png(img)).decode("ascii")
        return stats

    return await asyncio.to_thread(_metal_call, _run)


PEEKIR_TRAINING_DIR = Path(
    os.getenv("PEEKIR_TRAINING_DIR", str(REPO_ROOT / "training_data"))
).resolve()


@app.get("/api/training/summary")
async def training_summary():
    try:
        import training_export as tx
    except ImportError as e:
        raise HTTPException(501, f"training_export.py not found next to the dashboard: {e}")
    return {"root": str(PEEKIR_TRAINING_DIR), "counts": tx.summary(PEEKIR_TRAINING_DIR)}


@app.post("/api/training/ir-sample")
async def training_ir_sample(body: dict[str, Any]):
    """Save an IR-only training sample: the current no-metal reference + a fresh capture."""
    import numpy as np

    try:
        import training_export as tx
    except ImportError as e:
        raise HTTPException(501, f"training_export.py not found next to the dashboard: {e}")
    view = str(body.get("view", "")).strip().lower()
    if view not in ("front", "side"):
        raise HTTPException(400, "view must be front or side")
    try:
        metal = tx.parse_yes_no(body.get("metal"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if metal is None:
        raise HTTPException(400, "choose metal or no metal")
    area = body.get("true_area_cm2")
    try:
        area = float(area) if area not in (None, "") else None
    except (TypeError, ValueError):
        raise HTTPException(400, "true area must be a number (cm2)")
    try:
        weapon_type = tx.parse_weapon_type(body.get("weapon_type"), metal)
    except (ValueError, AttributeError) as e:
        raise HTTPException(400, str(e))
    sizer = _metal_call(_get_metal_sizer)
    ref_p = sizer.ref_path
    if not ref_p.exists():
        raise HTTPException(400, "no reference yet: click Set reference with NO metal on the mannequin first")
    ref = np.load(ref_p, allow_pickle=False)
    if "frames" not in ref:
        raise HTTPException(400, "reference was taken by an older version: click Set reference again")
    frames, fmt = await _http_get_pi_ir_burst(_capture_seconds(body, sizer))

    def _save():
        return tx.save_ir_sample(ref["frames"], str(ref["ir_format"]), frames, fmt, PEEKIR_TRAINING_DIR,
                                 view=view, metal=metal, obj=str(body.get("object", "")).strip(),
                                 true_area_cm2=area, notes=str(body.get("notes", "")).strip(),
                                 source="dashboard", state_dir=PEEKIR_METAL_STATE, log=lambda m: print(m, flush=True),
                                 weapon_type=weapon_type)

    dst = await asyncio.to_thread(_metal_call, _save)
    return {"ok": True, "saved": str(dst.relative_to(PEEKIR_TRAINING_DIR)),
            "counts": tx.summary(PEEKIR_TRAINING_DIR)}


@app.post("/api/ir/metal/settings")
async def ir_metal_settings(body: dict[str, Any]):
    """Change e.g. {"low_max_cm2": 25, "high_min_cm2": 150, "polarity": "cold", "min_area_px": 6}."""
    sizer = _metal_call(_get_metal_sizer)
    allowed = {"low_max_cm2", "high_min_cm2", "polarity", "min_area_px", "min_area_cm2", "min_delta_c",
               "noise_k", "body_mask", "capture_seconds", "outline_fill", "close_px", "max_fill_px",
               "min_delta_8bit", "local_background", "bg_block_px", "smooth_px",
               "half_contrast_edges", "ignore_hot_c", "ignore_hot_margin_px",
               "motion_px"}
    unknown = set(body) - allowed
    if unknown:
        raise HTTPException(400, f"unknown settings {sorted(unknown)}; allowed: {sorted(allowed)}")
    if "polarity" in body and body["polarity"] not in ("cold", "warm", "both"):
        raise HTTPException(400, "polarity must be cold, warm or both")
    _metal_call(lambda: sizer.update_settings(**body))
    return _metal_call(sizer.status)



def _open_dashboard_browser(url: str, delay_s: float) -> None:
    def _run() -> None:
        time.sleep(delay_s)
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("PEEKIR_HOST", "127.0.0.1")
    port = int(os.getenv("PEEKIR_PORT", "8080"))
    browse_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if browse_host == "::":
        browse_host = "127.0.0.1"
    dash_url = f"http://{browse_host}:{port}/"

    if os.getenv("PEEKIR_OPEN_BROWSER", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        try:
            delay = float(os.getenv("PEEKIR_BROWSER_DELAY", "1.5"))
        except ValueError:
            delay = 1.5
        print(f"[peekir_dashboard] Browser: {dash_url} (after {delay}s)", flush=True)
        _open_dashboard_browser(dash_url, delay)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
    )