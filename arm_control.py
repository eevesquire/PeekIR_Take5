#!/usr/bin/env python3
"""
arm_control.py - arm (Z, BTS7960 + DC gearmotor) and panel rotation (Feetech STS3215) for the PeekIR gantry.

Lives next to move.py on the Pi. move.py imports it only when an arm/rotation command is used, so the
normal X/Y gantry keeps working even if the arm hardware is unpowered.

Settings come from arm_calibration.json (written by calibrate_arm.py). Anything not measured yet falls
back to the PLACEHOLDER values in DEFAULTS below - search for "PLACEHOLDER" to see them.

State (arm retracted/extended, panel pose) is saved in arm_state.json so it survives move.py restarts
(the coordinator starts a fresh `move.py tcp` for every scan).

Safety rules enforced here:
  - The arm has no position sensor: EXTEND/RETRACT run a timed move into the hard stop
    (fast for fast_time_s, then slow for slow_time_s), then cut power.
  - The panel may only ROTATE while the arm is RETRACTED (turn first, then extend).
  - stop() cuts the arm motor immediately and aborts any arm move in progress (used by the e-stop).
"""

import glob
import json
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL_FILE = HERE / "arm_calibration.json"
STATE_FILE = HERE / "arm_state.json"

# ---- Pins (BCM), as wired --------------------------------------------------
PIN_RPWM = 18     # physical 12, red    - H-bridge side "A"
PIN_LPWM = 19     # physical 35, green  - H-bridge side "B"
PIN_R_EN = 20     # physical 38, yellow
PIN_L_EN = 21     # physical 40, blue
PWM_HZ = 1000

STEPS_PER_REV = 4096
SERVO_BAUD = 1_000_000

DEFAULTS = {
    "arm": {
        "extend": "A",          # PLACEHOLDER: which H-bridge side extends (calibrate_arm step 7)
        "fast_power": 0.8,      # PLACEHOLDER: power for most of the travel (step 9)
        "fast_time_s": 2.0,     # PLACEHOLDER: seconds at fast power (step 9)
        "slow_power": 0.35,     # PLACEHOLDER: gentle power for the last stretch into the stop (step 9)
        "slow_time_s": 3.0,     # PLACEHOLDER: seconds at slow power; must be long enough to always reach the stop
        "max_total_s": 12.0,    # hard cap on any single arm move
    },
    "servo": {
        "port": None,           # auto: /dev/serial/by-id/* then /dev/ttyACM*
        "id": 1,
        "front": 2048,          # measured by hand (recentred); refine with calibrate_arm step 5
        "xmax": 2914,
        "xmin": 974,
        "speed": 300,           # steps/s (~26 deg/s): gentle with the panel's weight
        "acc": 20,
        "tolerance": 20,        # steps (~1.8 deg) counted as "arrived"
        "move_timeout_s": 8.0,
        "settle_s": 0.5,        # PLACEHOLDER: extra wait after arriving for the panel to stop wobbling (step 4)
        "min_volts": 10.0,      # refuse to move without the 12 V supply (it runs at ~5 V from USB)
    },
}


class ArmError(RuntimeError):
    pass


def _deep_merge(base, extra):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update({kk: vv for kk, vv in v.items() if vv is not None})
        elif v is not None:
            out[k] = v
    return out


def load_settings():
    """DEFAULTS overridden by whatever calibrate_arm.py has measured."""
    try:
        cal = json.loads(CAL_FILE.read_text())
    except Exception:
        cal = {}
    s = _deep_merge(DEFAULTS, {"arm": {k: cal.get("arm", {}).get(k) for k in DEFAULTS["arm"]},
                               "servo": {k: cal.get("servo", {}).get(k) for k in DEFAULTS["servo"]}})
    # Slow phase from the fast-then-slow trials: longest observed slow phase + 0.5 s margin.
    trials = [t for t in cal.get("arm", {}).get("profile_trials", [])
              if t.get("impact") not in ("didn't reach",) and t.get("fast_time_s") == s["arm"]["fast_time_s"]]
    if trials and "slow_time_s" not in cal.get("arm", {}):
        s["arm"]["slow_time_s"] = round(max(t["total_s"] - t["fast_time_s"] for t in trials) + 0.5, 2)
    return s


def read_state():
    """Saved arm/panel state without touching any hardware."""
    try:
        st = json.loads(STATE_FILE.read_text())
    except Exception:
        st = {}
    return {"arm": st.get("arm", "unknown"), "pose": st.get("pose", "unknown")}


def _write_state(**kw):
    st = read_state()
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        STATE_FILE.write_text(json.dumps(st))
    except OSError:
        pass


# ---- Servo bus (works with both scservo_sdk variants) ------------------------
def find_servo_ports():
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    return by_id or sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


class _Bus:
    ACC, GOAL_POS, GOAL_SPEED = 41, 42, 46
    TORQUE_ENABLE, PRESENT_POS, MOVING, PRESENT_VOLTAGE = 40, 56, 66, 62

    def __init__(self, port):
        import scservo_sdk as scs
        self.ok = scs.COMM_SUCCESS
        self.port = scs.PortHandler(port)
        if not self.port.openPort() or not self.port.setBaudRate(SERVO_BAUD):
            raise ArmError(f"could not open servo port {port}")
        if hasattr(scs, "sms_sts"):
            self.pk, self.generic = scs.sms_sts(self.port), False
        else:
            self.pk, self.generic = scs.PacketHandler(0), True

    def _c(self, fn, *a):
        return getattr(self.pk, fn)(*((self.port,) + a if self.generic else a))

    def read2(self, sid, addr):
        v, res, _ = self._c("read2ByteTxRx", sid, addr)
        if res != self.ok:
            raise ArmError("servo read failed")
        return v

    def read1(self, sid, addr):
        v, res, _ = self._c("read1ByteTxRx", sid, addr)
        if res != self.ok:
            raise ArmError("servo read failed")
        return v

    def write1(self, sid, addr, val):
        self._c("write1ByteTxRx", sid, addr, int(val))

    def write2(self, sid, addr, val):
        self._c("write2ByteTxRx", sid, addr, int(val) & 0xFFFF)

    def close(self):
        try:
            self.port.closePort()
        except Exception:
            pass


class PanelServo:
    def __init__(self, cfg):
        self.cfg = cfg
        self.id = cfg["id"]
        self.port_name = cfg["port"]
        self.bus = None
        self._open()

    def _open(self):
        ports = find_servo_ports()
        if not ports:
            raise ArmError("servo driver board not found on USB (check its USB cable)")
        port = self.port_name if self.port_name in ports else ports[0]
        self.bus = _Bus(port)
        self.port_name = port

    def _safe(self, fn, tries=3):
        last = None
        for _ in range(tries):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(0.3)
                try:
                    if self.bus:
                        self.bus.close()
                    self._open()
                except Exception:
                    pass
        raise ArmError(f"servo not responding ({last})")

    def pos(self):
        v = self._safe(lambda: self.bus.read2(self.id, _Bus.PRESENT_POS))
        return -(v & 0x7FFF) if v & 0x8000 else v

    def volts(self):
        return self._safe(lambda: self.bus.read1(self.id, _Bus.PRESENT_VOLTAGE)) / 10.0

    def torque(self, on):
        self._safe(lambda: self.bus.write1(self.id, _Bus.TORQUE_ENABLE, 1 if on else 0))

    def goto(self, target, abort=None):
        c = self.cfg
        v = self.volts()
        if v < c["min_volts"]:
            raise ArmError(f"servo supply is {v:.1f} V; switch on the 12 V supply")
        target = int(max(0, min(STEPS_PER_REV - 1, target)))

        def _cmd():
            self.bus.write1(self.id, _Bus.TORQUE_ENABLE, 1)
            self.bus.write1(self.id, _Bus.ACC, c["acc"])
            self.bus.write2(self.id, _Bus.GOAL_SPEED, c["speed"])
            self.bus.write2(self.id, _Bus.GOAL_POS, target)
        self._safe(_cmd)

        t0, still_since, last = time.monotonic(), None, None
        while time.monotonic() - t0 < c["move_timeout_s"]:
            if abort is not None and abort.is_set():
                raise ArmError("rotation aborted")
            p = self.pos()
            if abs(p - target) <= c["tolerance"] and last is not None and abs(p - last) <= 3:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since > 0.25:
                    time.sleep(c["settle_s"])
                    return p
            else:
                still_since = None
            last = p
            time.sleep(0.03)
        raise ArmError(f"panel did not reach {target} (at {last}); blocked, or cables pulling?")

    def close(self):
        if self.bus:
            self.bus.close()


# ---- Arm motor (BTS7960) ----------------------------------------------------
class ArmMotor:
    def __init__(self):
        from gpiozero import PWMOutputDevice, DigitalOutputDevice
        self.a = PWMOutputDevice(PIN_RPWM, frequency=PWM_HZ, initial_value=0)
        self.b = PWMOutputDevice(PIN_LPWM, frequency=PWM_HZ, initial_value=0)
        self.en = [DigitalOutputDevice(p, initial_value=False) for p in (PIN_R_EN, PIN_L_EN)]

    def run(self, side, power):
        self.a.value = 0
        self.b.value = 0
        for e in self.en:
            e.on()
        (self.a if side == "A" else self.b).value = max(0.0, min(1.0, power))

    def stop(self):
        try:
            self.a.value = 0
            self.b.value = 0
            for e in self.en:
                e.off()
        except Exception:
            pass

    def close(self):
        self.stop()
        for d in (self.a, self.b, *self.en):
            try:
                d.close()
            except Exception:
                pass


# ---- Combined -------------------------------------------------------------------
class ArmSystem:
    """Arm + panel servo. The servo is opened lazily, so arm-only use works without it."""

    def __init__(self, estop=None, log=print):
        self.s = load_settings()
        self.estop = estop or (lambda: False)
        self.log = log
        self.motor = ArmMotor()
        self._servo = None
        self._abort = threading.Event()
        self._lock = threading.Lock()

    @property
    def state(self):
        return read_state()

    def servo(self):
        if self._servo is None:
            self._servo = PanelServo(self.s["servo"])
        return self._servo

    # -- arm --------------------------------------------------------------------
    def _arm_move(self, direction):
        a = self.s["arm"]
        side = a["extend"] if direction == "extend" else ("B" if a["extend"] == "A" else "A")
        if self.estop():
            raise ArmError("EMERGENCY STOP is pressed")
        with self._lock:
            self._abort.clear()
            _write_state(arm="moving")
            fast_t = min(a["fast_time_s"], a["max_total_s"])
            slow_t = min(a["slow_time_s"], a["max_total_s"] - fast_t)
            self.log(f"ARM {direction}: {a['fast_power']:.0%} for {fast_t:.1f}s, "
                     f"then {a['slow_power']:.0%} for {slow_t:.1f}s (side {side})")
            try:
                for power, dur in ((a["fast_power"], fast_t), (a["slow_power"], slow_t)):
                    self.motor.run(side, power)
                    end = time.monotonic() + dur
                    while time.monotonic() < end:
                        if self._abort.is_set() or self.estop():
                            raise ArmError("arm move aborted (stop / e-stop)")
                        time.sleep(0.01)
            except BaseException:
                self.motor.stop()
                _write_state(arm="unknown")
                raise
            self.motor.stop()
            _write_state(arm="extended" if direction == "extend" else "retracted")

    def extend(self):
        self._arm_move("extend")

    def retract(self):
        self._arm_move("retract")

    # -- rotation ---------------------------------------------------------------
    def rotate(self, pose):
        if pose not in ("front", "xmax", "xmin"):
            raise ArmError(f"unknown pose '{pose}' (use front, xmax or xmin)")
        if self.state["arm"] != "retracted":
            raise ArmError(f"refusing to rotate: arm is {self.state['arm']} (retract it first)")
        if self.estop():
            raise ArmError("EMERGENCY STOP is pressed")
        self._abort.clear()
        _write_state(pose="moving")
        try:
            p = self.servo().goto(self.s["servo"][pose], abort=self._abort)
        except BaseException:
            _write_state(pose="unknown")
            raise
        _write_state(pose=pose)
        self.log(f"ROTATE {pose}: at {p}")

    # -- helpers ----------------------------------------------------------------
    def park(self):
        """Safe position for X/Y moves and homing: arm retracted, panel facing front."""
        if self.state["arm"] != "retracted":
            self.retract()
        if self.state["pose"] != "front":
            self.rotate("front")

    def stop(self):
        """Cut the arm motor now and abort any arm/rotation in progress (servo keeps holding)."""
        self._abort.set()
        self.motor.stop()

    def close(self):
        self.stop()
        self.motor.close()
        if self._servo is not None:
            self._servo.close()
