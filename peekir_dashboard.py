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

PI_BASE_URL = os.getenv("PI_BASE_URL", "http://10.244.13.117:9000").rstrip("/")
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
PI_SSH_USER = os.getenv("PI_SSH_USER", "peekir")
PI_SSH_PROJECT_DIR = os.getenv("PI_SSH_PROJECT_DIR", "PeekIR_Pi")
PI_SSH_VENV = os.getenv("PI_SSH_VENV", "venv")


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

_metal_queue: queue.Queue = queue.Queue()
_metal_ws_clients: list[WebSocket] = []
_metal_state: dict[str, Any] = {"thread": None, "loop": None, "stop": None}
_metal_fanout_task: asyncio.Task | None = None


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
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        with _scan_lock:
            tail = list(_scan_status.get("log_tail") or [])
            tail.append(line.rstrip("\n"))
            _scan_status["log_tail"] = tail[-200:]
        _broadcast_sse_line(line)
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


def _metal_queue_get() -> Any | None:
    try:
        return _metal_queue.get(timeout=0.4)
    except queue.Empty:
        return None


def _stop_metal_worker() -> None:
    loop = _metal_state.get("loop")
    stop = _metal_state.get("stop")
    if loop is not None and stop is not None and not stop.is_set():

        def _set() -> None:
            stop.set()

        loop.call_soon_threadsafe(_set)
    th = _metal_state.get("thread")
    if th is not None and th.is_alive():
        th.join(timeout=3.0)


def _metal_worker(device_name: str | None, address: str | None) -> None:
    import metal_detector as md

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop = asyncio.Event()
    _metal_state["loop"] = loop
    _metal_state["stop"] = stop

    def on_notify(d: dict[str, Any]) -> None:
        _metal_queue.put({"type": "notify", "data": d})

    def on_status(kind: str, payload: dict[str, Any]) -> None:
        _metal_queue.put({"type": "status", "kind": kind, "payload": payload})

    try:
        loop.run_until_complete(
            md.monitor(
                device_name,
                address,
                3.0,
                stop_event=stop,
                on_notify=on_notify,
                on_status=on_status,
                verbose=False,
            )
        )
    finally:
        loop.close()
        if _metal_state.get("loop") is loop:
            _metal_state["loop"] = None
            _metal_state["stop"] = None
        if _metal_state.get("thread") is threading.current_thread():
            _metal_state["thread"] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _sse_loop, _metal_fanout_task
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

    async def _metal_fanout():
        while True:
            msg = await asyncio.to_thread(_metal_queue_get)
            if msg is None:
                await asyncio.sleep(0)
                continue
            dead: list[WebSocket] = []
            for ws in list(_metal_ws_clients):
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in _metal_ws_clients:
                    _metal_ws_clients.remove(ws)

    _metal_fanout_task = asyncio.create_task(_metal_fanout())
    if PEEKIR_FACE_WARMUP:
        threading.Thread(
            target=_deepface_models_warmup,
            daemon=True,
            name="peekir-deepface-warmup",
        ).start()
    yield
    _stop_pi_ssh_camera_server()
    if _metal_fanout_task:
        _metal_fanout_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await _metal_fanout_task


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
    return _DASHBOARD_HTML


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
            }
        )

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


@app.post("/api/metal/pair")
async def metal_pair(body: dict[str, Any] | None = None):
    try:
        import metal_detector as md
    except ImportError as e:
        raise HTTPException(501, "BLE dependencies missing (pip install bleak)") from e

    body = body or {}
    name = body.get("name") or md.DEFAULT_DEVICE_NAME
    address = body.get("address") or None
    if isinstance(address, str) and not address.strip():
        address = None

    await asyncio.to_thread(_stop_metal_worker)

    t = threading.Thread(
        target=_metal_worker,
        args=(name, address),
        daemon=True,
    )
    _metal_state["thread"] = t
    t.start()
    return {"ok": True}


@app.post("/api/metal/stop")
async def metal_stop():
    await asyncio.to_thread(_stop_metal_worker)
    return {"ok": True}


@app.websocket("/ws/metal")
async def metal_ws(websocket: WebSocket):
    await websocket.accept()
    _metal_ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _metal_ws_clients:
            _metal_ws_clients.remove(websocket)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PeekIR Dashboard</title>
  <style>
    :root {
      --bg: #0e1014;
      --panel: #161a20;
      --border: #2a3140;
      --text: #e8eaef;
      --muted: #8b93a3;
      --accent: #4f8cff;
      --danger: #e85d5d;
      --ok: #3dcc7a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, Segoe UI, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    h2 {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin: 0 0 0.75rem 0;
    }
    section {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
      padding: 1rem;
      margin: 1rem;
    }
    .top-panels {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 1rem;
      margin: 1rem;
      align-items: start;
    }
    .top-panels section {
      margin: 0;
      min-width: 0;
    }
    @media (max-width: 1100px) {
      .top-panels {
        grid-template-columns: 1fr;
      }
    }
    label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 0.25rem; }
    input[type="text"], input[type="number"] {
      width: 100%;
      max-width: 280px;
      padding: 0.45rem 0.5rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: #0b0d11;
      color: var(--text);
    }
    button {
      cursor: pointer;
      padding: 0.5rem 1rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: #1e2430;
      color: var(--text);
      font-weight: 600;
    }
    button.primary { background: var(--accent); border-color: #3a6fd0; color: #fff; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-end; }
    .progress-wrap { margin-top: 0.75rem; }
    .bar {
      height: 8px;
      background: #0b0d11;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 0.35rem;
    }
    .bar > div {
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 0.2s;
    }
    #scanLog {
      font-family: ui-monospace, monospace;
      font-size: 0.75rem;
      color: var(--muted);
      max-height: 120px;
      overflow: auto;
      white-space: pre-wrap;
      margin-top: 0.5rem;
    }
    #vizSection #vizFrameHost { display: none; }
    #vizSection.viz-has-viewer #vizFrameHost {
      display: block;
      margin-top: 0.75rem;
    }
    .viz-frame-wrap {
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      background: #000;
    }
    /* Legacy HTML without PEEKIR_VIZ_SIZE */
    .viz-frame-wrap:not(.viz-sized) .viz-frame-scale-inner {
      display: block;
      width: 100%;
    }
    .viz-frame-wrap:not(.viz-sized) .viz-frame-scale-inner iframe {
      width: 100%;
      height: 70vh;
      border: none;
      display: block;
    }
    .viz-cam-row .viz-frame-wrap:not(.viz-sized) .viz-frame-scale-inner iframe {
      height: min(55vh, 42vw);
    }
    /* Sized Plotly: fill mmWave column; --viz-scale set in JS = min(1, columnWidth / export width) */
    .viz-cam-col--viz {
      min-width: 0;
      width: 100%;
    }
    .viz-frame-wrap.viz-sized {
      position: relative;
      min-height: 0;
      width: 100%;
      box-sizing: border-box;
      aspect-ratio: var(--viz-ar-w, 1) / var(--viz-ar-h, 1);
      overflow: hidden;
    }
    .viz-frame-wrap.viz-sized .viz-frame-scale-inner {
      position: absolute;
      left: 0;
      top: 0;
      width: calc(100% / var(--viz-scale, 1));
      height: calc(100% / var(--viz-scale, 1));
      transform: scale(var(--viz-scale, 1));
      transform-origin: top left;
    }
    .viz-frame-wrap.viz-sized .viz-frame-scale-inner iframe {
      width: 100%;
      height: 100%;
      min-height: 0;
      border: none;
      display: block;
    }
    select#vizSelect {
      width: 100%;
      max-width: 420px;
      padding: 0.45rem 0.5rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: #0b0d11;
      color: var(--text);
    }
    /* mmWave / RGB / IR: 40% / 30% / 30% on wide screens */
    .viz-cam-row {
      display: grid;
      grid-template-columns: 4fr 3fr 3fr;
      gap: 1rem;
      align-items: start;
      margin-top: 0.75rem;
    }
    @media (max-width: 1100px) {
      .viz-cam-row {
        grid-template-columns: 1fr;
      }
    }
    .viz-panel {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.5rem;
      background: #0b0d11;
    }
    .viz-panel h3 {
      margin: 0 0 0.5rem 0;
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: none;
      letter-spacing: normal;
    }
    .viz-panel .row { width: 100%; }
    .viz-panel select#vizSelect { max-width: 100%; }
    .cam-box {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.5rem;
      background: #0b0d11;
    }
    .cam-box h3 { margin: 0 0 0.5rem 0; font-size: 0.8rem; color: var(--muted); }
    .cam-box .live {
      width: 100%;
      aspect-ratio: 4/3;
      object-fit: contain;
      background: #000;
      display: block;
      max-height: 360px;
    }
    .thumb { margin-top: 0.5rem; max-width: 100%; max-height: 200px; object-fit: contain; }
    .face-match-result { margin-top: 0.5rem; width: 100%; }
    .face-match-split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
      align-items: stretch;
    }
    .face-match-cap-wrap {
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #000;
      min-height: 100px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .face-match-cap-img {
      max-width: 100%;
      max-height: 220px;
      object-fit: contain;
      display: block;
    }
    .poi-match-stack {
      position: relative;
      min-height: 100px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #0b0d11;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .poi-match-img {
      max-width: 100%;
      max-height: 220px;
      object-fit: contain;
      display: block;
    }
    .poi-wanted-overlay {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      padding: 0.55rem 0.4rem;
      background: linear-gradient(transparent, rgba(120, 0, 0, 0.92));
      color: #fff;
      font-weight: 800;
      font-size: 0.8rem;
      text-align: center;
      letter-spacing: 0.02em;
      display: none;
      pointer-events: none;
    }
    .face-no-match-msg {
      padding: 1.25rem 0.5rem;
      text-align: center;
      color: var(--muted);
      font-size: 0.9rem;
      width: 100%;
    }
    .poi-detail-panel {
      margin-top: 0.65rem;
      padding: 0.65rem 0.75rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #07090d;
      font-size: 0.82rem;
      line-height: 1.45;
      width: 100%;
      box-sizing: border-box;
    }
    .poi-detail-panel h4 {
      margin: 0 0 0.45rem 0;
      font-size: 0.95rem;
      font-weight: 700;
      color: #e8ecf1;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .poi-detail-badge {
      display: inline-block;
      padding: 0.12rem 0.45rem;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .poi-detail-badge--wanted {
      background: rgba(232, 93, 93, 0.22);
      color: #ff8a8a;
      border: 1px solid rgba(232, 93, 93, 0.45);
    }
    .poi-detail-badge--clear {
      background: rgba(120, 140, 160, 0.18);
      color: var(--muted);
      border: 1px solid var(--border);
    }
    .poi-detail-grid {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.25rem 0.65rem;
      align-items: start;
    }
    .poi-detail-grid dt {
      margin: 0;
      color: var(--muted);
      font-weight: 600;
      font-size: 0.78rem;
    }
    .poi-detail-grid dd {
      margin: 0;
      color: #c9d0da;
    }
    .ir-detect-stack {
      background: #0b0d11;
      min-height: 120px;
      padding: 0.75rem;
    }
    .ir-detect-msg {
      width: 100%;
      text-align: center;
      font-size: 0.95rem;
      color: var(--muted);
      font-weight: 700;
    }
    .ir-detect-msg.detected { color: #ff6f6f; }
    .ir-detect-msg.clear { color: var(--muted); }
    .ir-detect-result-single .face-match-cap-wrap {
      min-height: 180px;
      padding: 0.4rem;
    }
    .ir-detect-result-single .face-match-cap-img {
      max-height: 300px;
      width: 100%;
      object-fit: contain;
    }
    #rgbFacePanel.face-alert-flash {
      animation: rgbFaceFlash 0.55s ease-in-out 3;
    }
    @keyframes rgbFaceFlash {
      0%, 100% { box-shadow: none; }
      50% { box-shadow: inset 0 0 0 3px var(--danger); background: rgba(232, 93, 93, 0.14); }
    }
    #metalPanel.metal-alert-active {
      animation: metalPanelAlert 0.85s ease-in-out infinite;
    }
    @keyframes metalPanelAlert {
      0%, 100% {
        box-shadow: inset 0 0 0 2px transparent;
        background: var(--panel);
      }
      50% {
        box-shadow: inset 0 0 0 3px var(--danger);
        background: rgba(232, 93, 93, 0.12);
      }
    }
    .metal-conn-tag {
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      font-weight: 700;
      font-size: 0.8rem;
    }
    .metal-conn-tag--disconnected { background: #1a1f28; color: var(--muted); }
    .metal-conn-tag--stopped { background: #1a1f28; color: var(--muted); }
    .metal-conn-tag--connecting { background: #3a3020; color: #e5b84a; }
    .metal-conn-tag--connected { background: #1a3024; color: var(--ok); }
    .metal-conn-tag--error { background: #3a2020; color: var(--danger); }
    .metal-high {
      display: inline-block;
      padding: 0.25rem 0.6rem;
      border-radius: 4px;
      font-weight: 700;
      background: #3a2020;
      color: var(--danger);
    }
    .metal-high.off { background: #1a2520; color: var(--muted); }
    .bat { font-size: 1.2rem; font-weight: 600; }
  </style>
</head>
<body>
  <div class="top-panels">
    <section>
      <h2>mmWave data collection</h2>
      <div class="row">
        <div>
          <label>Width (mm)</label>
          <input type="number" id="w" value="150" min="1" step="0.1">
        </div>
        <div>
          <label>Height (mm)</label>
          <input type="number" id="h" value="150" min="1" step="0.1">
        </div>
        <div>
          <label for="scanName">Scan name</label>
          <input type="text" id="scanName" name="scanName" autocomplete="off"
            spellcheck="false" placeholder="scan_0" maxlength="120"
            title="Letters, digits, underscore, hyphen; must start with a letter or digit">
        </div>
        <div>
          <button class="primary" id="btnScan">Start scan</button>
          <button id="btnCancel">Cancel</button>
        </div>
      </div>
      <div class="progress-wrap">
        <div id="scanStatus">Idle</div>
        <div class="bar"><div id="scanBar"></div></div>
        <div id="scanLog"></div>
      </div>
    </section>

    <section id="metalPanel">
      <h2>Metal detector</h2>
      <div class="row">
        <div>
          <button class="primary" id="btnPair">Pair</button>
          <button id="btnMetalStop">Stop</button>
        </div>
      </div>
      <p style="margin-top:0.75rem">
        Status: <span id="metalConn" class="metal-conn-tag metal-conn-tag--disconnected">Disconnected</span>
        &nbsp;|&nbsp; Battery: <span class="bat" id="metalBat">—</span>
        &nbsp;|&nbsp; <span class="metal-high off" id="metalAlert">METAL: LOW</span>
      </p>
    </section>
  </div>

  <section id="vizSection">
    <h2>mmWave visualization &amp; cameras</h2>
    <div class="viz-cam-row">
      <div class="viz-cam-col viz-cam-col--viz">
        <div class="viz-panel">
          <h3>mmWave visualization</h3>
          <div class="row">
            <div style="flex:1;min-width:0">
              <label for="vizSelect">Visualization file</label>
              <select id="vizSelect" aria-label="Visualization HTML file">
                <option value="">— Loading list —</option>
              </select>
            </div>
            <div>
              <button type="button" id="btnVizRefresh" title="Rescan folder">Refresh list</button>
              <button class="primary" id="btnViz">Load</button>
            </div>
          </div>
          <div id="vizFrameHost" class="viz-frame-wrap">
            <div class="viz-frame-scale-inner">
              <iframe id="vizFrame" title="SAR visualization"></iframe>
            </div>
          </div>
        </div>
      </div>
      <div class="cam-box" id="rgbFacePanel">
        <h3>RGB live</h3>
        <img class="live" id="rgbLive" alt="RGB">
        <button type="button" id="btnCapRgb">Capture RGB</button>
        <div id="rgbFaceMatchResult" class="face-match-result" style="display:none" aria-live="polite">
          <div class="face-match-split">
            <div class="face-match-cap-wrap">
              <img id="rgbCapAnnotated" class="face-match-cap-img" alt="Captured frame with face detection">
            </div>
            <div class="face-match-poi-col">
              <div class="poi-match-stack">
                <img id="rgbPoiImg" class="poi-match-img" alt="" style="display:none">
                <div id="rgbPoiOverlay" class="poi-wanted-overlay">Wanted Person Matched</div>
                <div id="rgbNoMatchMsg" class="face-no-match-msg" style="display:none">No Match Found</div>
              </div>
            </div>
          </div>
          <div id="rgbPoiDetail" class="poi-detail-panel" style="display:none" aria-live="polite">
            <h4>
              <span id="rgbPoiDetailTitle">Person record</span>
              <span id="rgbPoiWantedBadge" class="poi-detail-badge poi-detail-badge--wanted" style="display:none"></span>
            </h4>
            <dl class="poi-detail-grid">
              <dt>Full name</dt><dd id="rgbPoiFullName"></dd>
              <dt>Age</dt><dd id="rgbPoiAge"></dd>
              <dt>DOB</dt><dd id="rgbPoiDob"></dd>
              <dt>Crime</dt><dd id="rgbPoiCrime"></dd>
              <dt id="rgbPoiExtraLabel" style="display:none">Extra</dt>
              <dd id="rgbPoiExtra" style="display:none"></dd>
            </dl>
          </div>
        </div>
      </div>
      <div class="cam-box">
        <h3>IR live</h3>
        <img class="live" id="irLive" alt="IR">
        <button type="button" id="btnCapIr">Capture IR</button>
        <div id="irDetectResult" class="face-match-result ir-detect-result-single" style="display:none" aria-live="polite">
          <div class="face-match-cap-wrap">
            <img id="irCap" class="face-match-cap-img" alt="IR frame with gun detection">
          </div>
        </div>
      </div>
    </div>
  </section>

  <script>
    const piBase = () => window.__PI_BASE__ || '';

    async function loadConfig() {
      const r = await fetch('/api/config');
      const j = await r.json();
      window.__PI_BASE__ = j.pi_base_url || '';
      const sn = document.getElementById('scanName');
      if (sn && j.suggested_scan_name) {
        sn.value = j.suggested_scan_name;
      }
    }

    function setScanBar(pct) {
      document.getElementById('scanBar').style.width = Math.min(100, Math.max(0, pct)) + '%';
    }

    const es = new EventSource('/api/scan/stream');
    es.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch (e) { return; }
      if (d.progress) {
        const p = d.progress;
        if (p.phase === 'scan' && p.row != null && p.total) {
          document.getElementById('scanStatus').textContent = 'Scanning row ' + p.row + ' / ' + p.total;
          setScanBar((p.row / p.total) * 100);
        } else if (p.phase === 'scan' && p.status === 'complete') {
          document.getElementById('scanStatus').textContent = 'Scan complete';
          setScanBar(100);
        } else if (p.phase === 'reconstruct') {
          document.getElementById('scanStatus').textContent =
            p.status === 'start' ? 'Reconstructing…' :
            p.status === 'complete' ? 'Reconstruction done' :
            p.status === 'error' ? ('Reconstruction error: ' + (p.message || '')) : JSON.stringify(p);
          if (p.status === 'start') setScanBar(100);
        }
      }
    };

    document.getElementById('btnScan').onclick = async () => {
      const w = parseFloat(document.getElementById('w').value);
      const h = parseFloat(document.getElementById('h').value);
      const scan_name = (document.getElementById('scanName').value || '').trim();
      const r = await fetch('/api/scan/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ width_mm: w, height_mm: h, scan_name: scan_name })
      });
      if (!r.ok) {
        let msg = await r.text();
        try {
          const j = JSON.parse(msg);
          if (j.detail !== undefined) {
            msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
          }
        } catch (e) { /* keep raw */ }
        alert(msg);
        return;
      }
      document.getElementById('scanStatus').textContent = 'Starting…';
      setScanBar(0);
    };

    document.getElementById('btnCancel').onclick = async () => {
      await fetch('/api/scan/cancel', { method: 'POST' });
      document.getElementById('scanStatus').textContent = 'Cancelled';
    };

    async function loadVizList() {
      const sel = document.getElementById('vizSelect');
      try {
        const r = await fetch('/api/viz/list');
        const data = await r.json();
        const files = data.files || [];
        sel.innerHTML = '';
        const opt0 = document.createElement('option');
        opt0.value = '';
        opt0.textContent = files.length ? '— Select a file —' : '— No .html files in visualizations/ —';
        sel.appendChild(opt0);
        for (const f of files) {
          const o = document.createElement('option');
          o.value = f.name;
          o.textContent = f.name;
          sel.appendChild(o);
        }
      } catch (e) {
        sel.innerHTML = '<option value="">— Failed to load list —</option>';
        console.error(e);
      }
    }

    function updateVizFit() {
      const host = document.getElementById('vizFrameHost');
      if (!host || !host.classList.contains('viz-sized')) return;
      const raw = getComputedStyle(host).getPropertyValue('--viz-w').trim();
      const exportW = parseFloat(raw) || 800;
      const col = host.closest('.viz-cam-col--viz');
      if (!col) return;
      const cw = col.clientWidth;
      if (cw <= 0) return;
      const scale = Math.min(1, cw / exportW);
      host.style.setProperty('--viz-scale', String(scale));
    }

    let _vizFitObserverAttached = false;
    function ensureVizFitObserver() {
      if (_vizFitObserverAttached) return;
      const col = document.querySelector('.viz-cam-col--viz');
      if (!col) return;
      const ro = new ResizeObserver(() => updateVizFit());
      ro.observe(col);
      window.addEventListener('resize', updateVizFit);
      _vizFitObserverAttached = true;
    }

    async function loadVizPayload(payload) {
      const r = await fetch('/api/viz/load', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      if (!r.ok) {
        const t = await r.text();
        alert(t || r.status);
        return;
      }
      const j = await r.json();
      document.getElementById('vizSection').classList.add('viz-has-viewer');
      const host = document.getElementById('vizFrameHost');
      const frame = document.getElementById('vizFrame');
      host.classList.remove('viz-sized');
      host.style.removeProperty('--viz-w');
      host.style.removeProperty('--viz-ar-w');
      host.style.removeProperty('--viz-ar-h');
      host.style.removeProperty('--viz-scale');
      if (j.width_px && j.height_px) {
        host.classList.add('viz-sized');
        host.style.setProperty('--viz-w', j.width_px + 'px');
        host.style.setProperty('--viz-ar-w', String(j.width_px));
        host.style.setProperty('--viz-ar-h', String(j.height_px));
        ensureVizFitObserver();
        requestAnimationFrame(() => {
          updateVizFit();
          requestAnimationFrame(updateVizFit);
        });
      }
      frame.onload = () => updateVizFit();
      frame.src = '/api/viz/document/' + j.doc_id + '?t=' + Date.now();
    }

    document.getElementById('btnViz').onclick = async () => {
      const name = document.getElementById('vizSelect').value.trim();
      if (!name) {
        alert('Select a visualization file from the list.');
        return;
      }
      await loadVizPayload({ file: name });
    };

    document.getElementById('btnVizRefresh').onclick = () => loadVizList();

    loadVizList();

    async function refreshCams() {
      await loadConfig();
      const b = piBase();
      const rgb = b + '/api/camera/stream?' + Date.now();
      const ir = b + '/api/ir/stream?' + Date.now();
      console.log('[peekir_dashboard] Live MJPEG (browser -> Pi):', rgb, ir);
      const rEl = document.getElementById('rgbLive');
      const iEl = document.getElementById('irLive');
      rEl.onerror = () => console.error('[peekir_dashboard] RGB stream failed to load:', rgb);
      iEl.onerror = () => console.error('[peekir_dashboard] IR stream failed to load:', ir);
      rEl.src = rgb;
      iEl.src = ir;
    }

    function applyRgbPoiDetailUI(detail, embeddingName) {
      const panel = document.getElementById('rgbPoiDetail');
      const title = document.getElementById('rgbPoiDetailTitle');
      const badge = document.getElementById('rgbPoiWantedBadge');
      const overlay = document.getElementById('rgbPoiOverlay');
      const xLabel = document.getElementById('rgbPoiExtraLabel');
      const xDd = document.getElementById('rgbPoiExtra');
      const fallback = embeddingName ? String(embeddingName) : '';
      if (!detail) {
        panel.style.display = 'none';
        badge.style.display = 'none';
        title.textContent = 'Person record';
        overlay.textContent = fallback ? ('Match: ' + fallback) : 'POI match';
        return;
      }
      panel.style.display = 'block';
      title.textContent = detail.full_name || detail.name || 'Person record';
      document.getElementById('rgbPoiFullName').textContent = detail.full_name || detail.name || '—';
      document.getElementById('rgbPoiAge').textContent = (detail.age !== undefined && detail.age !== null)
        ? String(detail.age) : '—';
      document.getElementById('rgbPoiDob').textContent = (detail.dob !== undefined && detail.dob !== null)
        ? String(detail.dob) : '—';
      document.getElementById('rgbPoiCrime').textContent = (detail.crime !== undefined && detail.crime !== null)
        ? String(detail.crime) : '—';
      if (detail.extra_info) {
        xLabel.style.display = '';
        xDd.style.display = '';
        xDd.textContent = String(detail.extra_info);
      } else {
        xLabel.style.display = 'none';
        xDd.style.display = 'none';
        xDd.textContent = '';
      }
      const want = !!detail.wanted;
      badge.style.display = 'inline-block';
      badge.textContent = want ? 'Wanted' : 'Not wanted';
      badge.className = 'poi-detail-badge ' + (want ? 'poi-detail-badge--wanted' : 'poi-detail-badge--clear');
      const disp = detail.full_name || detail.name || fallback || '';
      overlay.textContent = want ? ('WANTED — ' + disp) : ('Identified — ' + disp);
    }

    document.getElementById('btnCapRgb').onclick = async () => {
      const btn = document.getElementById('btnCapRgb');
      const panel = document.getElementById('rgbFacePanel');
      const resultEl = document.getElementById('rgbFaceMatchResult');
      const capImg = document.getElementById('rgbCapAnnotated');
      const poiImg = document.getElementById('rgbPoiImg');
      const poiOv = document.getElementById('rgbPoiOverlay');
      const noMatch = document.getElementById('rgbNoMatchMsg');
      btn.disabled = true;
      panel.classList.remove('face-alert-flash');
      try {
        const r = await fetch('/api/pi/rgb-face-match');
        if (!r.ok) {
          const t = await r.text();
          alert(t || 'RGB capture / face match failed');
          return;
        }
        const j = await r.json();
        capImg.src = 'data:image/png;base64,' + j.annotated_image_b64;
        resultEl.style.display = 'block';
        if (j.matched) {
          noMatch.style.display = 'none';
          if (j.poi_image_b64) {
            poiImg.src = 'data:image/jpeg;base64,' + j.poi_image_b64;
            poiImg.style.display = 'block';
          } else {
            poiImg.style.display = 'none';
          }
          poiOv.style.display = 'block';
          applyRgbPoiDetailUI(j.poi_detail || null, j.name);
          panel.classList.add('face-alert-flash');
          window.setTimeout(() => panel.classList.remove('face-alert-flash'), 2000);
        } else {
          poiImg.style.display = 'none';
          poiImg.removeAttribute('src');
          poiOv.style.display = 'none';
          document.getElementById('rgbPoiDetail').style.display = 'none';
          noMatch.style.display = 'block';
        }
      } finally {
        btn.disabled = false;
      }
    };
    document.getElementById('btnCapIr').onclick = async () => {
      const btn = document.getElementById('btnCapIr');
      const resultEl = document.getElementById('irDetectResult');
      const img = document.getElementById('irCap');
      btn.disabled = true;
      resultEl.style.display = 'block';
      try {
        const r = await fetch('/api/pi/ir-gun-detect');
        if (!r.ok) {
          const t = await r.text();
          alert(t || 'IR detection failed');
          return;
        }
        const j = await r.json();
        img.src = 'data:image/png;base64,' + j.annotated_image_b64;
      } finally {
        btn.disabled = false;
      }
    };

    function setMetalConnFromKind(kind, payload) {
      const el = document.getElementById('metalConn');
      let css = 'disconnected';
      let text = 'Disconnected';
      const p = payload || {};
      if (kind === 'scanning' || kind === 'connecting') {
        css = 'connecting';
        text = 'Connecting';
      } else if (kind === 'connected') {
        css = 'connected';
        text = 'Connected';
      } else if (kind === 'listening') {
        css = 'connected';
        text = 'Listening';
      } else if (kind === 'disconnected') {
        css = 'disconnected';
        text = 'Disconnected';
      } else if (kind === 'error') {
        css = 'error';
        const msg = p.message ? String(p.message) : '';
        text = msg ? ('Error: ' + msg.slice(0, 120)) : 'Error';
      } else if (kind === 'stopped') {
        css = 'stopped';
        text = 'Stopped';
      } else {
        text = kind;
        css = 'disconnected';
      }
      el.className = 'metal-conn-tag metal-conn-tag--' + css;
      el.textContent = text;
    }

    document.getElementById('btnPair').onclick = async () => {
      setMetalConnFromKind('connecting', {});
      try {
        const r = await fetch('/api/metal/pair', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({})
        });
        if (!r.ok) {
          setMetalConnFromKind('error', { message: await r.text() });
        }
      } catch (e) {
        setMetalConnFromKind('error', { message: String(e) });
      }
    };
    document.getElementById('btnMetalStop').onclick = async () => {
      await fetch('/api/metal/stop', { method: 'POST' });
      document.getElementById('metalPanel').classList.remove('metal-alert-active');
      setMetalConnFromKind('stopped', {});
    };

    const wsMetal = new WebSocket((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws/metal');
    wsMetal.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch (e) { return; }
      if (m.type === 'status') {
        setMetalConnFromKind(m.kind, m.payload || {});
        if (m.kind === 'disconnected' || m.kind === 'error') {
          document.getElementById('metalPanel').classList.remove('metal-alert-active');
        }
      }
      if (m.type === 'notify' && m.data) {
        const d = m.data;
        if (d.battery_percent != null) {
          document.getElementById('metalBat').textContent = d.battery_percent + '%';
        } else if (d.battery_text) {
          document.getElementById('metalBat').textContent = d.battery_text;
        }
        const el = document.getElementById('metalAlert');
        const panel = document.getElementById('metalPanel');
        if (d.metal_high === true) {
          el.textContent = 'METAL: HIGH';
          el.classList.remove('off');
          panel.classList.add('metal-alert-active');
        } else if (d.metal_high === false) {
          el.textContent = 'METAL: LOW';
          el.classList.add('off');
          panel.classList.remove('metal-alert-active');
        }
      }
    };

    loadConfig().then(refreshCams);
  </script>
</body>
</html>
"""


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
