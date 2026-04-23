"""
Simple IR camera viewer + local snapshot saver.

Runs a local web UI that shows the Pi IR MJPEG stream and lets you:
1) Choose a local directory in the UI.
2) Capture the current IR frame and save it to that directory.

Usage:
  python ir_capture_viewer.py

Environment variables:
  PI_BASE_URL       Base URL of the Pi camera server (default: http://10.244.13.117:9000)
  PEEKIR_HOST       Host for this local UI (default: 127.0.0.1)
  PEEKIR_PORT       Port for this local UI (default: 8090)
  PEEKIR_OPEN_BROWSER  Auto-open browser (default: 1)
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


PI_BASE_URL = os.getenv("PI_BASE_URL", "http://10.244.13.117:9000").rstrip("/")
HOST = os.getenv("PEEKIR_HOST", "127.0.0.1")
PORT = int(os.getenv("PEEKIR_PORT", "8090"))
DEFAULT_SAVE_DIR = str((Path.cwd() / "outputs" / "ir_captures").resolve())

app = FastAPI(title="IR Capture Viewer")


class CaptureReq(BaseModel):
    save_dir: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IR Capture Viewer</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #0f1116;
      color: #f5f7ff;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 20px auto;
      padding: 0 16px;
    }}
    .card {{
      background: #171a22;
      border: 1px solid #2a3040;
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 14px;
    }}
    .row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    label {{
      font-size: 14px;
      opacity: 0.9;
      min-width: 140px;
    }}
    input[type="text"] {{
      flex: 1;
      min-width: 280px;
      background: #0f1116;
      color: #f5f7ff;
      border: 1px solid #36415a;
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 14px;
    }}
    button {{
      border: none;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 14px;
      cursor: pointer;
      color: #fff;
      background: #2b7fff;
    }}
    button.secondary {{
      background: #3b4254;
    }}
    button:disabled {{
      opacity: 0.7;
      cursor: not-allowed;
    }}
    #feed {{
      width: 100%;
      max-height: 72vh;
      object-fit: contain;
      background: #000;
      border-radius: 10px;
      border: 1px solid #2a3040;
    }}
    #status {{
      margin-top: 8px;
      font-size: 13px;
      color: #b9c6ff;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row" style="margin-bottom: 10px;">
        <label for="saveDir">Save Directory</label>
        <input id="saveDir" type="text" value="{DEFAULT_SAVE_DIR}" />
      </div>
      <div class="row">
        <button id="btnCapture">Capture IR Image</button>
        <button id="btnRefresh" class="secondary">Refresh Feed</button>
      </div>
      <div id="status">Ready</div>
    </div>
    <div class="card">
      <img id="feed" alt="IR live feed" />
    </div>
  </div>

  <script>
    const PI_BASE_URL = {PI_BASE_URL!r};

    function setStatus(msg) {{
      document.getElementById('status').textContent = msg;
    }}

    function refreshFeed() {{
      const src = PI_BASE_URL + '/api/ir/stream?t=' + Date.now();
      const img = document.getElementById('feed');
      img.onerror = () => setStatus('IR stream failed: ' + src);
      img.onload = () => setStatus('Live feed connected: ' + src);
      img.src = src;
    }}

    document.getElementById('btnRefresh').onclick = refreshFeed;

    document.getElementById('btnCapture').onclick = async () => {{
      const btn = document.getElementById('btnCapture');
      const saveDir = document.getElementById('saveDir').value.trim();
      if (!saveDir) {{
        setStatus('Please enter a save directory.');
        return;
      }}
      btn.disabled = true;
      setStatus('Capturing frame...');
      try {{
        const r = await fetch('/api/capture', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ save_dir: saveDir }})
        }});
        const j = await r.json();
        if (!r.ok || !j.ok) {{
          setStatus('Capture failed: ' + (j.error || r.status));
          return;
        }}
        setStatus('Saved: ' + j.saved_path);
      }} catch (e) {{
        setStatus('Capture failed: ' + String(e));
      }} finally {{
        btn.disabled = false;
      }}
    }};

    refreshFeed();
  </script>
</body>
</html>
"""


@app.post("/api/capture")
async def capture_ir(req: CaptureReq) -> dict:
    save_dir_raw = req.save_dir.strip()
    if not save_dir_raw:
        raise HTTPException(400, "save_dir is required")

    save_dir = Path(save_dir_raw).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    url = f"{PI_BASE_URL}/api/ir/frame"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
    except Exception as exc:
        return {"ok": False, "error": f"Could not reach Pi IR frame endpoint: {exc}"}

    if resp.status_code != 200:
        return {"ok": False, "error": f"Pi returned status {resp.status_code}"}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_path = save_dir / f"ir_{ts}.jpg"
    out_path.write_bytes(resp.content)

    return {"ok": True, "saved_path": str(out_path), "bytes": len(resp.content)}


def _open_browser(url: str, delay_s: float = 1.0) -> None:
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

    browse_host = "127.0.0.1" if HOST in {"0.0.0.0", "::"} else HOST
    if os.getenv("PEEKIR_OPEN_BROWSER", "1").strip().lower() not in {"0", "false", "no", "off"}:
        _open_browser(f"http://{browse_host}:{PORT}/")

    uvicorn.run(app, host=HOST, port=PORT)
