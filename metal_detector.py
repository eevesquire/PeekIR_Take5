#!/usr/bin/env python3
"""
ESP32 BLE monitor for metal detection + battery status.

Requires:
    pip install bleak

Usage:
    python metal_detector.py
    python metal_detector.py --name ESP32-BLE-MON
    python metal_detector.py --address 0C:4E:A0:4D:B1:2E
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from bleak import BleakClient, BleakScanner

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHARACTERISTIC_UUID_TX = "12345678-1234-5678-1234-56789abcdef2"

DEFAULT_DEVICE_NAME = "ESP32-BLE-MON"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_notification_structured(message: str) -> dict[str, Any]:
    """Parse a BLE notification string for dashboards and callbacks."""
    raw = message.strip()
    out: dict[str, Any] = {
        "raw": raw,
        "metal_high": None,
        "battery_text": None,
        "battery_percent": None,
        "boot": None,
    }
    if raw.startswith("PIN: HIGH"):
        out["metal_high"] = True
    elif raw.startswith("PIN: LOW"):
        out["metal_high"] = False
    elif raw.startswith("BAT:"):
        rest = raw[4:].strip()
        out["battery_text"] = rest
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", rest)
        if m:
            out["battery_percent"] = float(m.group(1))
    elif raw.startswith("BOOT:"):
        out["boot"] = raw[5:].strip()
    return out


def parse_message(message: str) -> str:
    message = message.strip()

    if message.startswith("PIN: HIGH"):
        return f"[{ts()}] Metal detect: HIGH | {message}"
    if message.startswith("PIN: LOW"):
        return f"[{ts()}] Metal detect: LOW  | {message}"
    if message.startswith("BAT:"):
        return f"[{ts()}] Battery: {message[4:].strip()}"
    if message.startswith("BOOT:"):
        return f"[{ts()}] Device: {message}"
    return f"[{ts()}] Notify: {message}"


def _make_notification_handler(
    on_notify: Callable[[dict[str, Any]], None] | None,
    verbose: bool,
) -> Callable:
    def notification_handler(_sender, data: bytearray) -> None:
        try:
            message = data.decode("utf-8", errors="replace")
        except Exception as exc:
            if verbose:
                print(f"[{ts()}] Decode error: {exc}", flush=True)
            if on_notify:
                on_notify({"raw": "", "decode_error": str(exc)})
            return

        structured = parse_notification_structured(message)
        if on_notify:
            on_notify(structured)
        elif verbose:
            print(parse_message(message), flush=True)

    return notification_handler


async def find_device(
    name: str | None,
    address: str | None,
    timeout: float = 10.0,
    *,
    verbose: bool = True,
):
    if address:
        device = await BleakScanner.find_device_by_address(address, timeout=timeout)
        if device is None:
            raise RuntimeError(f"Could not find device with address {address}")
        return device

    if verbose:
        print(f"[{ts()}] Scanning for BLE device...", flush=True)
    devices = await BleakScanner.discover(timeout=timeout)

    if name:
        for d in devices:
            if d.name == name:
                return d

    for d in devices:
        uuids = [u.lower() for u in (d.metadata.get("uuids") or [])]
        if SERVICE_UUID.lower() in uuids:
            return d

    if name:
        for d in devices:
            if d.name and name.lower() in d.name.lower():
                return d

    found = "\n".join(
        f"  - {d.name or '<unknown>'} [{d.address}]"
        for d in devices
    ) or "  <no devices found>"

    raise RuntimeError(f"Could not find matching ESP32 device.\nFound:\n{found}")


async def wait_for_stop(stop_event: asyncio.Event):
    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        pass


async def monitor(
    device_name: str | None,
    address: str | None,
    reconnect_delay: float,
    *,
    stop_event: asyncio.Event | None = None,
    on_notify: Callable[[dict[str, Any]], None] | None = None,
    on_status: Callable[[str, dict[str, Any]], None] | None = None,
    verbose: bool = True,
):
    """Connect, notify, reconnect until ``stop_event`` is set.

    ``on_status(kind, payload)`` receives e.g. (\"scanning\", {}), (\"connecting\", {...}),
    (\"connected\", {...}), (\"disconnected\", {}), (\"error\", {\"message\": ...}).
    ``on_notify`` receives structured dicts from :func:`parse_notification_structured`.
    """
    if stop_event is None:
        stop_event = asyncio.Event()

    notify_handler = _make_notification_handler(on_notify, verbose and on_notify is None)

    def status(kind: str, payload: dict[str, Any] | None = None):
        if on_status:
            on_status(kind, payload or {})
        elif verbose:
            if kind == "scanning":
                print(f"[{ts()}] Scanning for BLE device...", flush=True)

    while not stop_event.is_set():
        try:
            status("scanning", {})
            device = await find_device(device_name, address, verbose=False)
            if verbose and on_status is None:
                print(
                    f"[{ts()}] Connecting to {device.name or '<unknown>'} [{device.address}]...",
                    flush=True,
                )
            status(
                "connecting",
                {"name": device.name, "address": str(device.address)},
            )

            disconnected_event = asyncio.Event()

            def handle_disconnect(_client):
                if verbose and on_status is None:
                    print(f"[{ts()}] Disconnected", flush=True)
                status("disconnected", {})
                disconnected_event.set()

            async with BleakClient(device, disconnected_callback=handle_disconnect) as client:
                if not client.is_connected:
                    raise RuntimeError("Connection failed")

                if verbose and on_status is None:
                    print(f"[{ts()}] Connected", flush=True)
                status(
                    "connected",
                    {"name": device.name, "address": str(device.address)},
                )

                await client.start_notify(CHARACTERISTIC_UUID_TX, notify_handler)
                if verbose and on_status is None:
                    print(f"[{ts()}] Listening for notifications...", flush=True)
                status("listening", {})

                stop_task = asyncio.create_task(wait_for_stop(stop_event))
                disconnect_task = asyncio.create_task(disconnected_event.wait())

                done, pending = await asyncio.wait(
                    [stop_task, disconnect_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                with contextlib.suppress(Exception):
                    await client.stop_notify(CHARACTERISTIC_UUID_TX)

        except asyncio.CancelledError:
            break
        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as exc:
            if stop_event.is_set():
                break
            if on_status:
                on_status("error", {"message": str(exc)})
            elif verbose:
                print(f"[{ts()}] Error: {exc}", flush=True)
                print(f"[{ts()}] Retrying in {reconnect_delay:.1f} seconds...", flush=True)
            try:
                await asyncio.sleep(reconnect_delay)
            except KeyboardInterrupt:
                stop_event.set()
                break


def main():
    parser = argparse.ArgumentParser(
        description="Monitor ESP32 BLE metal detector and battery status"
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_DEVICE_NAME,
        help=f"BLE device name to search for (default: {DEFAULT_DEVICE_NAME})",
    )
    parser.add_argument(
        "--address",
        default=None,
        help="BLE address to connect to directly",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting",
    )
    args = parser.parse_args()

    try:
        asyncio.run(monitor(args.name, args.address, args.reconnect_delay))
    except KeyboardInterrupt:
        print(f"\n[{ts()}] Stopped", flush=True)


if __name__ == "__main__":
    main()
