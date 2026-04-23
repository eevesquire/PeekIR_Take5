#!/usr/bin/env python3
"""
Start the Pi camera server over SSH from your laptop.

Runs ``python3 pi_camera_server.py`` on the Pi (venv activated, project dir),
blocking until you press Ctrl+C or the remote process exits. Same SSH style as
``sar_complete.py`` (``bash -lc``, StrictHostKeyChecking, ServerAliveInterval).

Opens the camera UI in your default browser on the laptop by default. Stopping
this script (Ctrl+C, SIGTERM, or closing after ``kill``) terminates the SSH
client so the remote server session ends with the connection.

Usage:
    python run_pi_camera_server.py
    python run_pi_camera_server.py --pi-host 192.168.1.50
    python run_pi_camera_server.py --no-browser

Requires: ``ssh`` on PATH (OpenSSH client). SSH key or agent auth to the Pi.
"""

from __future__ import annotations

import argparse
import atexit
import signal
import shlex
import subprocess
import sys
import threading
import time


def _build_ssh_cmd(pi_user: str, pi_host: str, remote_cmd: str) -> list[str]:
    """Wrap a remote shell command in ``bash -lc`` and an ``ssh`` invocation.

    Uses ``-tt`` so a TTY is allocated (helps deliver Ctrl+C / hangup to the
    remote process when the connection closes).
    """
    target = f"{pi_user}@{pi_host}"
    return [
        "ssh",
        "-tt",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=15",
        target,
        f'bash -lc "{remote_cmd}"',
    ]


def _remote_cd_and_activate(pi_project_dir: str, pi_venv: str) -> str:
    """Shell snippet: cd into project and activate the venv."""
    return f"cd $HOME/{pi_project_dir} && " f"source {pi_venv}/bin/activate"


_ssh_proc: subprocess.Popen | None = None


def _terminate_ssh_if_running() -> None:
    """Kill the local ``ssh`` process so the Pi session ends and the server stops."""
    global _ssh_proc
    p = _ssh_proc
    if p is None or p.poll() is not None:
        return
    try:
        p.terminate()
        p.wait(timeout=10)
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


def _on_exit_signal(signum: int, _frame) -> None:
    _terminate_ssh_if_running()
    code = 128 + signum if signum > 0 else 0
    sys.exit(130 if signum == signal.SIGINT else code)


def _open_browser_delayed(url: str, delay_s: float) -> None:
    def _run() -> None:
        time.sleep(delay_s)
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def main() -> None:
    global _ssh_proc

    parser = argparse.ArgumentParser(
        description="SSH to the Pi and run pi_camera_server.py (blocks until stopped)."
    )
    parser.add_argument(
        "--pi-host",
        default="10.244.13.117",
        help="Raspberry Pi hostname or IP (default %(default)s)",
    )
    parser.add_argument(
        "--pi-user",
        default="peekir",
        help="SSH user on the Pi (default %(default)s)",
    )
    parser.add_argument(
        "--pi-project-dir",
        default="PeekIR_Pi",
        help="Project path on the Pi relative to $HOME (default %(default)s)",
    )
    parser.add_argument(
        "--pi-venv",
        default="venv",
        help="Virtualenv directory inside the project (default %(default)s)",
    )
    parser.add_argument(
        "--ir-device",
        default=None,
        metavar="PATH",
        help="Optional: set IR_CAMERA_DEVICE on the Pi (e.g. /dev/video0). "
        "Omit to use the server default.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port for browser URL (default %(default)s; must match pi_camera_server)",
    )
    parser.add_argument(
        "--browser-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before opening the browser (default %(default)s)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the camera page in a browser on this machine.",
    )
    args = parser.parse_args()

    base = _remote_cd_and_activate(args.pi_project_dir, args.pi_venv)
    if args.ir_device:
        env_prefix = f"IR_CAMERA_DEVICE={shlex.quote(args.ir_device)} "
    else:
        env_prefix = ""
    # exec replaces the shell so the Python server is the session leader; when
    # SSH disconnects, the remote process receives SIGHUP and exits cleanly.
    remote = f"{base} && {env_prefix}exec python3 pi_camera_server.py"

    cmd = _build_ssh_cmd(args.pi_user, args.pi_host, remote)
    url = f"http://{args.pi_host}:{args.port}/"

    print(f"SSH → {args.pi_user}@{args.pi_host}")
    print(f"Camera UI: {url}")
    print("Press Ctrl+C to stop the remote server and disconnect.\n")

    if not args.no_browser:
        _open_browser_delayed(url, args.browser_delay)

    atexit.register(_terminate_ssh_if_running)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_exit_signal)
    signal.signal(signal.SIGINT, _on_exit_signal)

    rc = 1
    try:
        _ssh_proc = subprocess.Popen(cmd)
        rc = _ssh_proc.wait()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    finally:
        _terminate_ssh_if_running()
        _ssh_proc = None

    sys.exit(rc if rc is not None else 1)


if __name__ == "__main__":
    main()
