#!/usr/bin/env python3
"""
calibrate_arm.py - measure everything the arm (Z) and rotation code needs.

Run on the Pi, from the project folder (next to move.py), with the venv active:

    pip install feetech-servo-sdk pyserial      # once
    python3 calibrate_arm.py

Do NOT run move.py (or a scan) at the same time.

What it measures (menu-driven, do them in order the first time):
  1. Servo: finds the USB port and servo ID (and the voltage it runs on)
  2. Servo: torque OFF, you turn the panel by hand -> records FRONT, XMAX-side and XMIN-side readings
  3. Servo: re-centre so FRONT reads 2048 (needed if step 2 warns about the 0/4095 wrap point)
  4. Servo: moves FRONT -> XMAX -> FRONT -> XMIN -> FRONT, times each move and settle (needs 12 V)
  5. Servo: fine-tune each pose with the servo holding, nudging 1 or 5 degrees at a time (needs 12 V)
  6. Arm:   signal test with 12 V OFF -> checks the Pi -> H-bridge wiring (motor doesn't move)
  7. Arm:   short low-power jog -> works out which H-bridge input EXTENDS (needs 12 V)
  8. Arm:   timed full-travel runs at several power levels (you press Enter at the stop)
  9. Arm:   trial of "fast, then slow near the stop" with your chosen numbers
 10. Gantry: X or Y move with the arm attached, via move.py (you measure with a tape)
 11. Side-scan geometry: standoffs, clearances and the Y range (tape measure only)

Without the 12 V supply you can already run: 1, 2, 3 (if the servo answers on USB power),
6 (signal test), 10 (gantry, needs the 24 V supply) and 11 (geometry).

Everything is saved to arm_calibration.json next to this script after every step.

Safety:
  - Keep a hand on the 12 V supply's OUTPUT button (cuts power to the arm motor and the servo).
  - The e-stop button (GPIO 24) also stops the arm motor while this script runs.
  - Ctrl+C at any time stops the motor and exits.
  - Every arm run has a timeout, so the motor cannot run forever.
"""

import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL_FILE = HERE / "arm_calibration.json"

# ---- Pins (BCM numbers, as wired) -----------------------------------------
PIN_RPWM = 18    # physical pin 12 (red)
PIN_LPWM = 19    # physical pin 35 (green)
PIN_R_EN = 20    # physical pin 38 (R_EN, yellow)
PIN_L_EN = 21    # physical pin 40 (L_EN, blue)
PIN_ESTOP = 24   # physical pin 18 (existing e-stop button)
# Not set in code: H-bridge VCC (purple) -> physical pin 4 (5 V), GND (white) -> physical pin 9.
PWM_HZ = 1000

ARM_TIMEOUT_S = 15.0      # hard cap on any single arm run
SERVO_BAUD = 1_000_000
STEPS_PER_REV = 4096      # STS3215: 4096 position steps per turn


# ---- Small helpers --------------------------------------------------------
def load_cal():
    try:
        return json.loads(CAL_FILE.read_text())
    except Exception:
        return {}


def save_cal(cal):
    cal["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    CAL_FILE.write_text(json.dumps(cal, indent=2))
    print(f"  (saved to {CAL_FILE.name})")


def ask(prompt, default=None):
    s = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
    return s if s else ("" if default is None else str(default))


def ask_float(prompt, default=None):
    while True:
        s = ask(prompt, default)
        try:
            return float(s)
        except ValueError:
            print("  Please type a number.")


def yes(prompt):
    return ask(prompt + " (y/n)").lower().startswith("y")


def steps_to_deg(steps):
    return steps * 360.0 / STEPS_PER_REV


def wrap_delta(target, start):
    """Shortest signed turn (steps) from start to target, allowing for the 4095 -> 0 wrap."""
    return (target - start + STEPS_PER_REV // 2) % STEPS_PER_REV - STEPS_PER_REV // 2


def crosses_wrap(target, start):
    """True if a direct move start -> target (position mode) would NOT take the shortest way,
    i.e. the shortest path passes through 0/4095 and the servo would go the long way round."""
    return (target - start) != wrap_delta(target, start)


# ---- Arm motor (BTS7960) --------------------------------------------------
class Arm:
    def __init__(self):
        from gpiozero import PWMOutputDevice, DigitalOutputDevice, Button
        self.a = PWMOutputDevice(PIN_RPWM, frequency=PWM_HZ, initial_value=0)
        self.b = PWMOutputDevice(PIN_LPWM, frequency=PWM_HZ, initial_value=0)
        self.en = [DigitalOutputDevice(p, initial_value=False) for p in (PIN_R_EN, PIN_L_EN)]
        self._Button = Button
        self.estop = None
        self.attach_estop()

    def attach_estop(self):
        try:
            self.estop = self._Button(PIN_ESTOP, pull_up=True, bounce_time=0.05)
            self.estop.when_pressed = self._estop_pressed
        except Exception as e:
            print(f"  Note: e-stop button not available ({e}). Use the 12 V OUTPUT button.")
            self.estop = None

    def release_estop(self):
        if self.estop is not None:
            self.estop.close()
            self.estop = None

    def _estop_pressed(self):
        self.stop()
        print("\n!!! E-STOP pressed: arm motor stopped. Press Enter to continue.")

    def run(self, channel, power):
        """channel 'A' = RPWM side, 'B' = LPWM side; power 0..1."""
        power = max(0.0, min(1.0, power))
        self.a.value = 0
        self.b.value = 0
        for e in self.en:
            e.on()
        (self.a if channel == "A" else self.b).value = power

    def stop(self):
        self.a.value = 0
        self.b.value = 0
        for e in self.en:
            e.off()

    def close(self):
        self.stop()
        for d in (self.a, self.b, *self.en):
            d.close()
        self.release_estop()


def timed_run(arm, channel, power, prompt, slow_after=None, slow_power=None):
    """Run until the user presses Enter (or timeout). Returns elapsed seconds."""
    done = threading.Event()
    timer = threading.Timer(ARM_TIMEOUT_S, lambda: (arm.stop(), done.set(),
                            print(f"\n  Timeout ({ARM_TIMEOUT_S:.0f}s): motor stopped. Press Enter.")))
    slow_timer = None
    if slow_after is not None:
        slow_timer = threading.Timer(slow_after, lambda: None if done.is_set() else arm.run(channel, slow_power))
    print(f"  {prompt}")
    t0 = time.monotonic()
    arm.run(channel, power)
    timer.start()
    if slow_timer:
        slow_timer.start()
    try:
        input("  >>> Press Enter the moment it reaches the stop <<<")
    finally:
        t = time.monotonic() - t0
        arm.stop()
        done.set()
        timer.cancel()
        if slow_timer:
            slow_timer.cancel()
    return t


# ---- Servo (Feetech STS3215) ----------------------------------------------
class _Bus:
    """Talks to Feetech STS servos with whichever API the installed scservo_sdk has.

    Newer feetech-servo-sdk releases only export the generic PacketHandler (the one
    LeRobot uses); older/vendor copies export the sms_sts helper class. Both work here.
    """
    # STS register addresses
    ACC, GOAL_POS, GOAL_SPEED = 41, 42, 46
    TORQUE_ENABLE, PRESENT_POS, MOVING = 40, 56, 66
    PRESENT_VOLTAGE = 62   # 1 byte, units of 0.1 V

    def __init__(self, port):
        import scservo_sdk as scs
        self.ok = scs.COMM_SUCCESS
        self.port = scs.PortHandler(port)
        if not self.port.openPort():
            raise RuntimeError(f"could not open {port}")
        if not self.port.setBaudRate(SERVO_BAUD):
            raise RuntimeError("could not set baud rate")
        if hasattr(scs, "sms_sts"):
            self.pk, self.generic = scs.sms_sts(self.port), False
        else:
            self.pk, self.generic = scs.PacketHandler(0), True   # 0 = STS (little-endian)

    def _c(self, fn, *a):
        return getattr(self.pk, fn)(*((self.port,) + a if self.generic else a))

    def ping(self, sid):
        _model, res, _err = self._c("ping", sid)
        return res == self.ok

    def read_pos(self, sid):
        v, res, _err = self._c("read2ByteTxRx", sid, self.PRESENT_POS)
        if res != self.ok:
            raise RuntimeError("servo read failed")
        return -(v & 0x7FFF) if v & 0x8000 else v

    def read1(self, sid, addr):
        v, res, _err = self._c("read1ByteTxRx", sid, addr)
        return v if res == self.ok else None

    def write1(self, sid, addr, val):
        return self._c("write1ByteTxRx", sid, addr, int(val))

    def write2(self, sid, addr, val):
        return self._c("write2ByteTxRx", sid, addr, int(val) & 0xFFFF)

    def close(self):
        self.port.closePort()


class Servo:
    def __init__(self, port, sid):
        self.bus = _Bus(port)
        self.port_name = port
        self.id = sid

    def reopen(self):
        """Re-open the port, e.g. after the USB board reset and came back (maybe as a new ttyACM)."""
        try:
            self.bus.close()
        except Exception:
            pass
        for _ in range(10):                      # wait up to ~5 s for it to reappear
            ports = find_ports()
            if ports:
                port = self.port_name if self.port_name in ports else ports[0]
                try:
                    self.bus = _Bus(port)
                    if port != self.port_name:
                        print(f"  (servo board came back as {port})")
                    self.port_name = port
                    return True
                except Exception:
                    pass
            time.sleep(0.5)
        return False

    def safe(self, fn, *a, tries=3):
        """Call a bus function, re-opening the port and retrying if the USB link hiccups."""
        last = None
        for _ in range(tries):
            try:
                return fn(*a)
            except Exception as e:
                last = e
                time.sleep(0.2)
                self.reopen()
        raise RuntimeError(f"servo not responding ({type(last).__name__}: {last})")

    @staticmethod
    def scan(port, ids=range(0, 21)):
        bus = _Bus(port)
        try:
            print(f"  (servo SDK mode: {'generic PacketHandler' if bus.generic else 'sms_sts'})")
            return [(i, "STS") for i in ids if bus.ping(i)]
        finally:
            bus.close()

    def pos(self):
        return self.safe(lambda: self.bus.read_pos(self.id))

    def moving(self):
        return bool(self.bus.read1(self.id, _Bus.MOVING))

    def volts(self):
        v = self.bus.read1(self.id, _Bus.PRESENT_VOLTAGE)
        return None if v is None else v / 10.0

    def torque(self, on):
        self.safe(lambda: self.bus.write1(self.id, _Bus.TORQUE_ENABLE, 1 if on else 0))

    def goto(self, target, speed=400, acc=20):
        self.torque(True)
        self.bus.write1(self.id, _Bus.ACC, acc)
        self.bus.write2(self.id, _Bus.GOAL_SPEED, speed)
        self.bus.write2(self.id, _Bus.GOAL_POS, target)

    def close(self):
        self.bus.close()


def find_ports():
    """Prefer the stable /dev/serial/by-id/ names: they stay the same when the board
    resets and Linux renumbers it (e.g. ttyACM0 -> ttyACM1)."""
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    return by_id or sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


# ---- Steps ----------------------------------------------------------------
def step_find_servo(cal):
    print("\n== 1. Find the servo ==")
    ports = find_ports()
    if not ports:
        print("  No USB serial ports found. Check: servo board plugged into the Pi,")
        print("  mode switch on USB, 12 V on, and that the USB cable carries DATA")
        print("  (some phone charger cables are power-only - try another cable).")
        return
    print("  Serial ports:", ", ".join(ports))
    for port in ports:
        found = Servo.scan(port)
        if found:
            sid, model = found[0]
            print(f"  Found servo ID {sid} (model {model}) on {port}")
            cal["servo"] = {**cal.get("servo", {}), "port": port, "id": sid}
            try:
                sv = Servo(port, sid)
                v = sv.volts()
                sv.close()
            except Exception:
                v = None
            if v is not None:
                print(f"  Servo supply voltage: {v:.1f} V")
                if v < 10:
                    print("  -> Below 12 V: the servo's electronics are running, but NOT from the 12 V supply")
                    print("     (probably 5 V fed through from USB). Reading positions (step 2) is fine;")
                    print("     don't run step 3 (moving the panel) until the 12 V supply is connected.")
                cal["servo"]["volts_at_scan"] = v
            save_cal(cal)
            return
    print("  No servo answered. Check 12 V at the board, and that the servo cable is in a")
    print("  TTL-Bus connector (not RS485).")


def need_servo(cal):
    s = cal.get("servo", {})
    if "port" not in s:
        print("  Run step 1 first.")
        return None
    try:
        return Servo(s["port"], s["id"])
    except Exception as e:
        print(f"  Could not open the servo: {e}")
        return None


def step_servo_poses(cal):
    print("\n== 2. Record FRONT, XMAX-side and XMIN-side panel positions (torque off) ==")
    sv = need_servo(cal)
    if not sv:
        return
    try:
        sv.torque(False)
        print("  Torque is OFF. Turn the panel GENTLY by hand.")
        poses = {}
        print("  Poses are named by gantry end, not by the mannequin's left/right:")
        print("    xmax = panel at the MAX-X end, turned to face the mannequin")
        print("    xmin = panel at the MIN-X (home) end, turned to face the mannequin")
        for name, desc in (("front", "facing straight ahead for the normal X/Y scan"),
                           ("xmax", "as it should face the mannequin from the MAX-X end"),
                           ("xmin", "as it should face the mannequin from the MIN-X end")):
            while True:
                input(f"  Hold the panel {desc}, then press Enter...")
                try:
                    vals = [sv.pos() for _ in range(5)]
                    break
                except RuntimeError as e:
                    print(f"  Read failed: {e}")
                    print("  The servo board probably reset while the panel was turned (turning a servo by")
                    print("  hand makes its motor act as a generator, which can upset a USB-powered board).")
                    print("  Turn more slowly. Try again, or press Ctrl+C and wait for the 12 V supply.")
                    sv.torque(False) if sv.reopen() else None
            poses[name] = round(sum(vals) / len(vals))
            print(f"    {name}: {poses[name]}  (~{steps_to_deg(poses[name]):.1f} deg)")
        dl = wrap_delta(poses["xmin"], poses["front"])
        dr = wrap_delta(poses["xmax"], poses["front"])
        print(f"  xmin pose is {dl:+d} steps ({steps_to_deg(dl):+.1f} deg) from front.")
        print(f"  xmax pose is {dr:+d} steps ({steps_to_deg(dr):+.1f} deg) from front.")
        print(f"  Mirror check: the two turns differ by {abs(abs(dl) - abs(dr))} steps "
              f"({steps_to_deg(abs(abs(dl) - abs(dr))):.1f} deg) in size.")
        if (dl > 0) == (dr > 0):
            print("  WARNING: both side poses are on the same side of front. Check the panel directions.")
        if any(crosses_wrap(poses[k], poses["front"]) for k in ("xmin", "xmax")):
            print("  WARNING: a turn crosses the servo's 0/4095 wrap point, so the servo would go the")
            print("  LONG way round (~300 deg). Run the 'Servo: re-centre' step next.")
        lim = ask_float("  Furthest you can turn PAST either side before cables pull tight (deg, 0 if unsure)", 0)
        cal["servo"].update({**poses, "extra_limit_deg": lim, "port": sv.port_name})
        cal["servo"].pop("side", None)
        save_cal(cal)
    finally:
        sv.close()


def step_servo_recenter(cal):
    print("\n== 3. Servo: re-centre so FRONT reads 2048 ==")
    s = cal.get("servo", {})
    if "port" not in s:
        print("  Run step 1 first.")
        return
    print("  This makes the servo treat the panel's FRONT direction as the middle of its range (2048),")
    print("  so turning to either side never crosses the 0/4095 wrap point. It is saved inside the")
    print("  servo and survives power cycles.")
    sv = need_servo(cal)
    if not sv:
        return
    try:
        sv.torque(False)
        input("  Turn the panel by hand (slowly) to face straight FRONT, hold it, then press Enter...")
        before = sv.pos()
        sv.safe(lambda: sv.bus.write1(sv.id, _Bus.TORQUE_ENABLE, 128))   # 128 = "current position becomes 2048"
        time.sleep(0.5)
        sv.torque(False)
        after = sv.pos()
        print(f"  Position before: {before}, after: {after}")
        if abs(after - 2048) > 30:
            print("  The servo did not re-centre (it should now read ~2048). Try once more; if it still")
            print("  fails, tell software (the servo's offset register can be set instead).")
            return
        print("  Re-centred.")
        if "xmax" in s and "xmin" in s and "front" in s:
            for k in ("xmax", "xmin"):
                s[k] = 2048 + wrap_delta(s[k], s["front"])
            s["front"] = 2048
            print(f"  Updated poses from your earlier readings: front 2048, xmax {s['xmax']}, xmin {s['xmin']}.")
            print("  Re-run step 2 to confirm them with fresh readings.")
        else:
            s["front"] = 2048
        s["recentred"] = True
        save_cal(cal)
    finally:
        sv.close()


def step_servo_move(cal):
    print("\n== 4. Servo move + settle timing ==")
    s = cal.get("servo", {})
    if "front" not in s:
        print("  Run step 2 first.")
        return
    sv = need_servo(cal)
    if not sv:
        return
    speed = int(ask_float("  Servo speed (steps/s, ~400 = 35 deg/s)", 400))
    print("  Keep hands and cables clear of the panel.")
    if not yes("  Ready to move the panel?"):
        sv.close()
        return
    results = []
    try:
        for target_name in ("front", "xmax", "front", "xmin", "front"):
            target = s[target_name]
            t0 = time.monotonic()
            sv.goto(target, speed=speed)
            still_since, last = None, None
            while time.monotonic() - t0 < 10:
                p = sv.pos()
                if last is not None and abs(p - last) <= 2 and abs(p - target) <= 15:
                    still_since = still_since or time.monotonic()
                    if time.monotonic() - still_since > 0.25:
                        break
                else:
                    still_since = None
                last = p
                time.sleep(0.02)
            t = time.monotonic() - t0
            err = sv.pos() - target
            print(f"    -> {target_name}: {t:.2f}s, final error {err:+d} steps ({steps_to_deg(err):+.2f} deg)")
            wob = ask_float("    Seconds until the panel visibly stopped wobbling (estimate)", 0.5)
            results.append({"to": target_name, "move_s": round(t, 2), "error_steps": err, "wobble_s": wob})
        cal["servo"]["speed"] = speed
        cal["servo"]["moves"] = results
        save_cal(cal)
    finally:
        sv.close()


def step_servo_finetune(cal):
    print("\n== 5. Servo: fine-tune poses with the servo holding (needs 12 V) ==")
    s = cal.get("servo", {})
    if not all(k in s for k in ("front", "xmax", "xmin")):
        print("  Run step 2 first.")
        return
    sv = need_servo(cal)
    if not sv:
        return
    v = sv.volts()
    if v is not None and v < 10:
        print(f"  Servo supply is only {v:.1f} V. Connect the 12 V supply first.")
        sv.close()
        return
    print("  For each pose the servo moves there and holds. Nudge it until the panel faces exactly right:")
    print("    +  / -   = 1 degree      ++ / --  = 5 degrees      a number = that many degrees (e.g. -2.5)")
    print("    Enter    = keep this position and go to the next pose")
    print("  Keep hands and cables clear while it moves.")
    if not yes("  Ready?"):
        sv.close()
        return
    try:
        for name in ("front", "xmax", "xmin", "front"):
            target = s[name]
            sv.goto(target, speed=300)
            time.sleep(1.5)
            while True:
                ans = input(f"  {name}: {target} ({steps_to_deg(target - 2048):+.1f} deg from centre) > ").strip()
                if not ans:
                    break
                step = {"+": 1, "-": -1, "++": 5, "--": -5}.get(ans)
                if step is None:
                    try:
                        step = float(ans)
                    except ValueError:
                        print("    Type + - ++ -- or a number of degrees.")
                        continue
                target = max(0, min(STEPS_PER_REV - 1, round(target + step * STEPS_PER_REV / 360)))
                sv.goto(target, speed=150)
                time.sleep(0.6)
            s[name] = target
        dl = wrap_delta(s["xmin"], s["front"])
        dr = wrap_delta(s["xmax"], s["front"])
        print(f"  Final: front {s['front']}, xmax {s['xmax']} ({steps_to_deg(dr):+.1f} deg), "
              f"xmin {s['xmin']} ({steps_to_deg(dl):+.1f} deg)")
        s["finetuned"] = True
        save_cal(cal)
    finally:
        sv.close()


def step_arm_signal_test(cal, arm):
    print("\n== 6. Signal test (no 12 V needed) ==")
    print("  Checks the Pi -> H-bridge signal wiring with the 12 V supply OFF or disconnected.")
    print("  The motor will NOT move. Optionally put a multimeter (DC volts) between the H-bridge's")
    print("  GND pin and each signal pin as it is switched on.")
    if not yes("  12 V is off/disconnected and you're ready?"):
        return
    def show(label):
        try:
            out = subprocess.run(["pinctrl", "get", "18-21"], capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            out = "(pinctrl not found)"
        print(f"  [{label}]\n    " + out.replace("\n", "\n    "))
    results = {}
    for ch, name, pin in (("A", "RPWM", 12), ("B", "LPWM", 35)):
        arm.run(ch, 1.0)          # 100% so a multimeter reads a steady ~3.3 V
        show(f"EN pins 38+40 high, {name} (pin {pin}) on")
        ok = ask(f"  Multimeter on {name} reads ~3.3 V? (y / n / skip)", "skip").lower()
        results[name] = ok
        arm.stop()
    show("all off")
    print("  If a pin never changes: check that nothing else (move.py) is running and holding it.")
    print("  If pinctrl shows the pin switching but the H-bridge pin doesn't: check that jumper wire.")
    cal.setdefault("arm", {})["signal_test"] = results
    save_cal(cal)


def step_arm_direction(cal, arm):
    print("\n== 7. Which way is EXTEND? ==")
    print("  Put the arm roughly MID-TRAVEL first (not at a stop).")
    if not yes("  Arm is mid-travel and clear?"):
        return
    print("  Running side A (RPWM) at 40% power for 0.3 s...")
    arm.run("A", 0.4)
    time.sleep(0.3)
    arm.stop()
    ans = ask("  Did the arm EXTEND, RETRACT, or NOT MOVE? (e/r/n)").lower()
    if ans.startswith("n"):
        print("  No movement: check 12 V at B+/B-, the M+/M- wires, the EN wires on pins 38 and 40,")
        print("  and H-bridge GND on pin 9. VCC is on pin 4 (5 V): if everything else checks out,")
        print("  move VCC to pin 1 (3.3 V) so the board reads the Pi's 3.3 V signals reliably.")
        return
    cal["arm"] = {**cal.get("arm", {}), "extend": "A" if ans.startswith("e") else "B"}
    print(f"  Saved: EXTEND = side {cal['arm']['extend']} "
          f"({'RPWM/GPIO18' if cal['arm']['extend'] == 'A' else 'LPWM/GPIO19'})")
    save_cal(cal)


def step_arm_timing(cal, arm):
    print("\n== 8. Full-travel timing ==")
    a = cal.get("arm", {})
    if "extend" not in a:
        print("  Run step 7 first.")
        return
    ext = a["extend"]
    ret = "B" if ext == "A" else "A"
    travel = ask_float("  Measured full travel of the arm (mm; 8 in = 203)", a.get("travel_mm", 203))
    print("  For each run: start at the OPPOSITE stop. Press Enter the instant it reaches the stop.")
    print("  Start low. Only try high power if the team is OK with the arm hitting the stop.")
    runs = a.get("runs", [])
    while True:
        p = ask("  Power % for next run (e.g. 40, 60, 80) or blank to finish", "")
        if not p:
            break
        power = max(5, min(100, float(p))) / 100
        for name, ch in (("extend", ext), ("retract", ret)):
            if not yes(f"  Arm at the {'RETRACTED' if name == 'extend' else 'EXTENDED'} stop, ready to {name.upper()}?"):
                continue
            t = timed_run(arm, ch, power, f"{name} at {power * 100:.0f}%...")
            speed = travel / t if t > 0 else 0
            print(f"    {name} @ {power * 100:.0f}%: {t:.2f}s  (~{speed:.0f} mm/s)")
            hit = ask("    How did it reach the stop? (gentle / ok / hard)", "ok")
            runs.append({"dir": name, "power": power, "time_s": round(t, 2),
                         "mm_per_s": round(speed, 1), "impact": hit})
    cal["arm"].update({"travel_mm": travel, "runs": runs})
    save_cal(cal)
    if runs:
        print("\n  Summary:")
        for r in runs:
            print(f"    {r['dir']:7s} {r['power'] * 100:4.0f}%  {r['time_s']:5.2f}s  {r['mm_per_s']:5.0f} mm/s  {r['impact']}")


def step_arm_profile(cal, arm):
    print("\n== 9. Fast-then-slow trial ==")
    a = cal.get("arm", {})
    if "extend" not in a:
        print("  Run step 7 first.")
        return
    ext = a["extend"]
    ret = "B" if ext == "A" else "A"
    fast = ask_float("  Fast power %", a.get("fast_power", 0.8) * 100) / 100
    fast_t = ask_float("  Seconds at fast power before slowing", a.get("fast_time_s", 2.0))
    slow = ask_float("  Slow power % for the last stretch", a.get("slow_power", 0.35) * 100) / 100
    trials = a.get("profile_trials", [])
    for name, ch in (("extend", ext), ("retract", ret)):
        if not yes(f"  Arm at the {'RETRACTED' if name == 'extend' else 'EXTENDED'} stop, ready to {name.upper()}?"):
            continue
        t = timed_run(arm, ch, fast, f"{name}: {fast * 100:.0f}% for {fast_t}s, then {slow * 100:.0f}%...",
                      slow_after=fast_t, slow_power=slow)
        hit = ask("    How did it reach the stop? (gentle / ok / hard / didn't reach)", "ok")
        print(f"    {name}: {t:.2f}s total")
        trials.append({"dir": name, "fast_power": fast, "fast_time_s": fast_t,
                       "slow_power": slow, "total_s": round(t, 2), "impact": hit})
    if yes("  Use these settings for the real arm code?"):
        a.update({"fast_power": fast, "fast_time_s": fast_t, "slow_power": slow})
    a["profile_trials"] = trials
    cal["arm"] = a
    save_cal(cal)


def step_gantry_y(cal, arm):
    print("\n== 10. Gantry move with the arm attached (uses move.py) ==")
    if not (HERE / "move.py").exists():
        print("  move.py not found next to this script.")
        return
    print("  Y: checks the gantry can lift the arm. X: checks the trip to each end for the side scans.")
    axis = ask("  Axis to test (y / x)", "y").lower()
    pair = ("right", "left") if axis.startswith("x") else ("up", "down")
    dist = ask_float(f"  Distance to move {pair[0]} and back {pair[1]} (mm)", 100)
    speed = ask_float("  Speed (mm/s)", 25)
    print("  Put a tape measure or a mark on the frame to check the real distance.")
    print("  Watch the 24 V supply: note if its C.C. light comes on. Reed switches still stop it at the ends.")
    if not yes("  Ready?"):
        return
    arm.release_estop()          # let move.py's own e-stop monitor use GPIO 24
    rows = cal.get("gantry", [])
    try:
        for d in pair:
            t0 = time.monotonic()
            r = subprocess.run([sys.executable, "move.py", f"{d}={dist:g}", f"speed={speed:g}mms"], cwd=HERE)
            t = time.monotonic() - t0
            meas = ask_float(f"    Measured {d} distance (mm)", dist)
            cc = yes("    Did the 24 V C.C. light come on, or did it strain/skip?")
            rows.append({"dir": d, "commanded_mm": dist, "measured_mm": meas, "speed_mms": speed,
                         "time_s": round(t, 2), "strain": cc, "exit_code": r.returncode})
            print(f"    {d}: commanded {dist:g} mm, measured {meas:g} mm, {t:.1f}s"
                  f"{'  <-- STRAIN' if cc else ''}")
    finally:
        arm.attach_estop()
    cal["gantry"] = rows
    save_cal(cal)


def step_side_geometry(cal):
    print("\n== 11. Side-scan geometry (tape measure, no power needed) ==")
    print("  For each end: gantry at that X end, arm EXTENDED, panel turned to face the mannequin.")
    g = cal.get("side_geometry", {})
    for end in ("xmax", "xmin"):
        print(f"  -- {end.upper()} end --")
        g[end] = {
            "standoff_mm": ask_float("    Panel face to the mannequin's side (mm)", g.get(end, {}).get("standoff_mm", 150)),
            "clear_to_extend": yes("    Arm extends here without touching anything?"),
            "clear_to_rotate_retracted": yes("    Panel can turn here with the arm RETRACTED without touching anything?"),
            "clear_to_rotate_extended": yes("    Panel can turn here with the arm EXTENDED without touching anything?"),
        }
    g["front_standoff_mm"] = ask_float("  Front scan: panel face to the mannequin's front, arm retracted (mm)",
                                       g.get("front_standoff_mm", 323))
    g["y_bottom_mm"] = ask_float("  Side pass: lowest Y position to scan, from Y home (mm)", g.get("y_bottom_mm", 0))
    g["y_top_mm"] = ask_float("  Side pass: highest Y position to scan, from Y home (mm)", g.get("y_top_mm", 300))
    cal["side_geometry"] = g
    save_cal(cal)


# ---- Main -----------------------------------------------------------------
def main():
    cal = load_cal()
    print("Arm + rotation calibration. Keep a hand on the 12 V OUTPUT button.")
    arm = None
    try:
        arm = Arm()
    except Exception as e:
        print(f"Could not set up the H-bridge pins ({e}). Arm steps disabled.")
    menu = [
        ("Servo: find port and ID", lambda: step_find_servo(cal)),
        ("Servo: record FRONT / XMAX / XMIN positions (turn by hand)", lambda: step_servo_poses(cal)),
        ("Servo: re-centre so FRONT = 2048 (avoids the 0/4095 wrap)", lambda: step_servo_recenter(cal)),
        ("Servo: move + settle timing", lambda: step_servo_move(cal)),
        ("Servo: fine-tune poses with the servo holding (needs 12 V)", lambda: step_servo_finetune(cal)),
        ("Arm: signal test (no 12 V needed)", lambda: arm and step_arm_signal_test(cal, arm)),
        ("Arm: which way is extend", lambda: arm and step_arm_direction(cal, arm)),
        ("Arm: full-travel timing", lambda: arm and step_arm_timing(cal, arm)),
        ("Arm: fast-then-slow trial", lambda: arm and step_arm_profile(cal, arm)),
        ("Gantry: X or Y move with arm attached", lambda: arm and step_gantry_y(cal, arm)),
        ("Side-scan geometry (tape measure)", lambda: step_side_geometry(cal)),
        ("Show saved results", lambda: print(json.dumps(cal, indent=2))),
    ]
    try:
        while True:
            print("\n" + "\n".join(f"  {i + 1}. {m[0]}" for i, m in enumerate(menu)) + "\n  q. Quit")
            c = ask("Choose").lower()
            if c == "q":
                break
            if c.isdigit() and 1 <= int(c) <= len(menu):
                menu[int(c) - 1][1]()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if arm:
            arm.close()
        print(f"Results are in {CAL_FILE}")


if __name__ == "__main__":
    main()
