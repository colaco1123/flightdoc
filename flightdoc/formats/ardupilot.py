"""ArduPilot dataflash (.bin) logs to ``Flight``."""

from __future__ import annotations

import os
import re
from typing import Dict, List

import numpy as np

from ..model import Event, Flight, spans
from .dataflash import DataflashLog

COPTER_MODES = {
    0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
    5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT", 13: "SPORT",
    14: "FLIP", 15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE", 18: "THROW",
    19: "AVOID_ADSB", 20: "GUIDED_NOGPS", 21: "SMART_RTL", 22: "FLOWHOLD",
    23: "FOLLOW", 24: "ZIGZAG", 25: "SYSTEMID", 26: "AUTOROTATE",
    27: "AUTO_RTL", 28: "TURTLE",
}
PLANE_MODES = {
    0: "MANUAL", 1: "CIRCLE", 2: "STABILIZE", 3: "TRAINING", 4: "ACRO",
    5: "FBWA", 6: "FBWB", 7: "CRUISE", 8: "AUTOTUNE", 10: "AUTO", 11: "RTL",
    12: "LOITER", 13: "TAKEOFF", 14: "AVOID_ADSB", 15: "GUIDED",
    17: "QSTABILIZE", 18: "QHOVER", 19: "QLOITER", 20: "QLAND", 21: "QRTL",
    22: "QAUTOTUNE", 23: "QACRO", 24: "THERMAL", 25: "LOITER_ALT_QLAND",
}

# AP_Logger LogErrorSubsystem
ERR_SUBSYS = {
    1: "main", 2: "radio", 3: "compass", 4: "optical flow",
    5: "radio failsafe", 6: "battery failsafe", 7: "GPS failsafe",
    8: "GCS failsafe", 9: "fence failsafe", 10: "flight mode", 11: "GPS",
    12: "crash check", 13: "flip", 14: "autotune", 15: "parachute",
    16: "EKF check", 17: "EKF failsafe", 18: "barometer", 19: "CPU",
    20: "ADSB failsafe", 21: "terrain", 22: "navigation", 23: "terrain failsafe",
    24: "EKF primary", 25: "thrust loss check", 26: "sensor failsafe",
    27: "leak failsafe", 28: "pilot input", 29: "vibration failsafe",
    30: "internal error", 31: "dead reckoning failsafe",
}
EV_NAMES = {10: "armed", 11: "disarmed", 15: "auto armed", 17: "land complete (maybe)",
            18: "land complete", 28: "not landed", 25: "set home",
            56: "EKF yaw reset", 57: "EKF yaw reset"}

MOTOR_FUNCTIONS = {33 + i: i + 1 for i in range(8)}
MOTOR_FUNCTIONS.update({82 + i: 9 + i for i in range(4)})
THROTTLE_FUNCTION = 70


def _t(arr) -> np.ndarray:
    names = arr.dtype.names
    if "TimeUS" in names:
        return arr["TimeUS"].astype(np.float64) / 1e6
    if "TimeMS" in names:
        return arr["TimeMS"].astype(np.float64) / 1e3
    raise KeyError("message has no timestamp")


def _instance(arr, col: str, value: int = 0):
    """Rows for one sensor instance when the message carries an instance column."""
    if arr is None or col not in arr.dtype.names:
        return arr
    mask = np.asarray(arr[col]) == value
    if not mask.any():
        return arr
    return arr[mask]


def load(path: str) -> List[Flight]:
    log = DataflashLog.open(path)
    return [_to_flight(log, os.path.basename(path))]


def _to_flight(log: DataflashLog, source: str) -> Flight:
    f = Flight(source=source, firmware="ardupilot")
    if log.bad_bytes:
        f.notes.append(f"{log.bad_bytes} unreadable bytes skipped")

    params: Dict[str, float] = {}
    parm = log.read("PARM")
    if parm is not None:
        for name, val in zip(parm["Name"], parm["Value"]):
            params[name.rstrip(b"\0").decode("ascii", "replace")] = float(val)
    f.params = params

    t_ref = None
    def rel(arr):
        return _t(arr) - t_ref

    # Time origin: first timestamp of a common high rate message.
    for name in ("ATT", "IMU", "RCOU", "CTUN", "BAT", "GPS"):
        arr = log.read(name)
        if arr is not None and len(arr):
            t_ref = float(_t(arr)[0])
            break
    if t_ref is None:
        t_ref = 0.0

    # ---- identity ----------------------------------------------------------
    msgs = log.read("MSG")
    messages = []
    if msgs is not None:
        texts = [m.rstrip(b"\0").decode("utf-8", "replace") for m in msgs["Message"]]
        messages = list(zip(rel(msgs), texts))
    for _, text in messages:
        m = re.match(r"(Ardu\w+|Blimp|AntennaTracker)\s+(V[\d.]+[^\s]*)", text)
        if m:
            f.firmware_version = f"{m.group(1)} {m.group(2)}"
            kind = m.group(1).lower()
            f.vehicle = {"arducopter": "copter", "arduplane": "plane",
                         "ardurover": "rover", "ardusub": "sub"}.get(kind, kind)
            break
    for _, text in messages:
        m = re.match(r"Frame:\s*(\S+)", text)
        if m:
            f.meta["frame"] = text.split(":", 1)[1].strip()
            break
    modes = PLANE_MODES if f.vehicle == "plane" else COPTER_MODES

    # ---- attitude and rates ----------------------------------------------
    att = log.read("ATT")
    if att is not None and len(att) > 1:
        f.add("att", rel(att), roll=att["Roll"], pitch=att["Pitch"], yaw=att["Yaw"],
              des_roll=att["DesRoll"], des_pitch=att["DesPitch"], des_yaw=att["DesYaw"])
    rate = log.read("RATE")
    if rate is not None and len(rate) > 1:
        f.add("rate", rel(rate), roll=rate["R"], pitch=rate["P"], yaw=rate["Y"],
              des_roll=rate["RDes"], des_pitch=rate["PDes"], des_yaw=rate["YDes"])

    imu = _instance(log.read("IMU"), "I")
    if imu is not None and len(imu) > 1:
        t = rel(imu)
        r2d = 180.0 / np.pi
        f.add("gyro", t, x=imu["GyrX"] * r2d, y=imu["GyrY"] * r2d, z=imu["GyrZ"] * r2d)
        f.add("accel", t, x=imu["AccX"], y=imu["AccY"], z=imu["AccZ"])

    vibe = _instance(log.read("VIBE"), "IMU")
    if vibe is not None and len(vibe) > 1:
        names = vibe.dtype.names
        if "Clip" in names:
            clip = vibe["Clip"].astype(float)
        else:  # older logs: one counter per IMU
            clip = sum(vibe[c].astype(float) for c in ("Clip0", "Clip1", "Clip2") if c in names)
        f.add("vibe", rel(vibe), x=vibe["VibeX"], y=vibe["VibeY"], z=vibe["VibeZ"], clip=clip)

    # ---- motors ------------------------------------------------------------
    rcou = log.read("RCOU")
    if rcou is not None and len(rcou) > 1:
        chan_to_motor = {}
        for k, v in params.items():
            m = re.fullmatch(r"SERVO(\d+)_FUNCTION", k)
            if m and int(v) in MOTOR_FUNCTIONS:
                chan_to_motor[int(m.group(1))] = MOTOR_FUNCTIONS[int(v)]
        if not chan_to_motor and f.vehicle == "plane":
            for k, v in params.items():
                m = re.fullmatch(r"SERVO(\d+)_FUNCTION", k)
                if m and int(v) == THROTTLE_FUNCTION:
                    chan_to_motor[int(m.group(1))] = 1
        if not chan_to_motor:
            chan_to_motor = {1: 1, 2: 2, 3: 3, 4: 4}
        lo = params.get("MOT_PWM_MIN", 0) or params.get("RC3_MIN", 1000) or 1000
        hi = params.get("MOT_PWM_MAX", 0) or params.get("RC3_MAX", 2000) or 2000
        cols = {}
        for ch, motor in sorted(chan_to_motor.items(), key=lambda kv: kv[1]):
            name = f"C{ch}"
            if name in rcou.dtype.names:
                cols[f"m{motor}"] = np.clip((rcou[name].astype(float) - lo) / max(hi - lo, 1), 0, 1.05)
        if cols:
            f.add("motors", rel(rcou), **cols)
            f.meta["motor_range"] = f"{lo:.0f}..{hi:.0f}"

    esc = log.read("ESC")
    if esc is not None and "RPM" in esc.dtype.names and "Instance" in esc.dtype.names:
        inst = np.asarray(esc["Instance"])
        t_all = rel(esc)
        base = inst == inst.min()
        t = t_all[base]
        cols = {}
        for k in np.unique(inst):
            sel = inst == k
            cols[f"m{int(k) + 1}"] = np.interp(t, t_all[sel], esc["RPM"][sel].astype(float))
        f.add("erpm", t, **cols)

    ctun = log.read("CTUN")
    if ctun is not None and len(ctun) > 1:
        t = rel(ctun)
        names = ctun.dtype.names
        if "ThO" in names:
            tho = ctun["ThO"].astype(float)
            if np.nanmax(tho) > 1.5:  # plane logs percent
                tho = tho / 100.0
            f.add("throttle", t, thr=tho)
        if "Alt" in names:
            f.add("alt", t, alt=ctun["Alt"].astype(float))
    if not f.has("alt"):
        baro = _instance(log.read("BARO"), "I")
        if baro is not None and len(baro) > 1:
            f.add("alt", rel(baro), alt=baro["Alt"].astype(float))

    # ---- power -----------------------------------------------------------
    bat = _instance(log.read("BAT"), "Inst")
    if bat is None:
        bat = log.read("CURR")
    if bat is not None and len(bat) > 1 and "Volt" in bat.dtype.names:
        cols = dict(volt=bat["Volt"].astype(float))
        if "Curr" in bat.dtype.names:
            cols["curr"] = bat["Curr"].astype(float)
        f.add("battery", rel(bat), **cols)
    powr = log.read("POWR")
    if powr is not None and len(powr) > 1 and "Vcc" in powr.dtype.names:
        f.add("power", rel(powr), vcc=powr["Vcc"].astype(float))

    # ---- sensors -----------------------------------------------------------
    mag = _instance(log.read("MAG"), "I")
    if mag is not None and len(mag) > 1:
        f.add("mag", rel(mag), x=mag["MagX"], y=mag["MagY"], z=mag["MagZ"])

    gps = _instance(log.read("GPS"), "I")
    if gps is not None and len(gps) > 1:
        names = gps.dtype.names
        cols = dict(fix=gps["Status"], nsats=gps["NSats"], hdop=gps["HDop"])
        if "Lat" in names:
            cols.update(lat=gps["Lat"], lon=gps["Lng"], alt=gps["Alt"])
        if "Spd" in names:
            cols["spd"] = gps["Spd"]
        f.add("gps", rel(gps), **cols)

    ekf = None
    for name in ("XKF4", "NKF4"):
        ekf = log.read(name)
        if ekf is not None and len(ekf) > 1:
            break
    if ekf is not None and len(ekf) > 1:
        names = ekf.dtype.names
        if "C" in names:
            core = np.asarray(ekf["C"])
            if "PI" in names:
                sel = core == np.asarray(ekf["PI"])
                if sel.sum() < 2:
                    sel = core == core.min()
            else:
                sel = core == core.min()
            ekf = ekf[sel]
        mag_ratio = ekf["SM"] if "SM" in names else np.maximum.reduce(
            [ekf[c] for c in ("SMX", "SMY", "SMZ") if c in names])
        f.add("ekf", rel(ekf), vel=ekf["SV"], pos=ekf["SP"], hgt=ekf["SH"], mag=mag_ratio)

    rssi = log.read("RSSI")
    if rssi is not None and len(rssi) > 1:
        r = rssi["RXRSSI"].astype(float)
        if np.nanmax(r) > 1.5:
            r = r / 255.0
        f.add("rc", rel(rssi), rssi=np.clip(r, 0, 1))

    pm = log.read("PM")
    if pm is not None and len(pm) > 1:
        names = pm.dtype.names
        cols = {}
        if "Load" in names:
            cols["load"] = pm["Load"].astype(float) / 1000.0
        if "NLon" in names and "NL" in names:
            nl = np.maximum(pm["NL"].astype(float), 1)
            cols["slow"] = pm["NLon"].astype(float) / nl
        elif "NLon" in names and "NLoop" in names:
            nl = np.maximum(pm["NLoop"].astype(float), 1)
            cols["slow"] = pm["NLon"].astype(float) / nl
        if cols:
            f.add("cpu", rel(pm), **cols)

    # ---- events --------------------------------------------------------------
    for t, text in messages:
        low = text.lower()
        level = "info"
        if any(w in low for w in ("crash", "failsafe", "error", "fail", "thrust loss",
                                  "lost", "bad", "emergency", "glitch", "clipping")):
            level = "error" if any(w in low for w in ("crash", "thrust loss", "failsafe")) else "warning"
        f.events.append(Event(float(t), "message", text, level))

    mode = log.read("MODE")
    if mode is not None:
        col = "ModeNum" if "ModeNum" in mode.dtype.names else "Mode"
        last = None
        for t, num in zip(rel(mode), mode[col]):
            if int(num) != last:
                f.events.append(Event(float(t), "mode", modes.get(int(num), f"mode {int(num)}")))
            last = int(num)

    err = log.read("ERR")
    if err is not None:
        for t, sub, code in zip(rel(err), err["Subsys"], err["ECode"]):
            sub, code = int(sub), int(code)
            name = ERR_SUBSYS.get(sub, f"subsystem {sub}")
            if code == 0:
                f.events.append(Event(float(t), "error", f"{name}: resolved", "info"))
                continue
            text = f"{name}: error {code}"
            if sub == 12:
                text = "crash check: " + ("crash detected" if code == 1 else "loss of control")
            elif sub == 25:
                text = "thrust loss check: potential thrust loss"
            elif sub == 11 and code == 2:
                text = "GPS: glitch"
            elif sub == 16 and code == 2:
                text = "EKF check: bad variance"
            elif sub in (5, 6, 7, 8, 9, 17, 20, 23, 26, 29, 31):
                text = f"{name} triggered"
            f.events.append(Event(float(t), "error", text, "error"))

    ev = log.read("EV")
    if ev is not None:
        for t, i in zip(rel(ev), ev["Id"]):
            i = int(i)
            if i in EV_NAMES and i not in (10, 11):
                f.events.append(Event(float(t), "event", EV_NAMES[i]))

    # ---- armed spans -------------------------------------------------------
    arm = log.read("ARM")
    changes = []
    if arm is not None and len(arm):
        changes = [(float(t), bool(s)) for t, s in zip(rel(arm), arm["ArmState"])]
    elif ev is not None:
        changes = [(float(t), int(i) == 10) for t, i in zip(rel(ev), ev["Id"]) if int(i) in (10, 11)]
    changes.sort()
    end_time = 0.0
    for df in f.signals.values():
        end_time = max(end_time, float(df["t"].iloc[-1]))
    f.duration = end_time
    start = None
    armed = []
    if changes and not changes[0][1]:
        # Logging usually starts at arming, so the first record we see can be
        # the disarm: the vehicle was armed from the start of the log.
        start = 0.0
    for t, state in changes:
        if state and start is None:
            start = t
            f.events.append(Event(t, "arm", "armed"))
        elif not state and start is not None:
            armed.append((start, t))
            start = None
            f.events.append(Event(t, "disarm", "disarmed"))
    if start is not None:
        armed.append((start, end_time))
        f.ended_armed = True
        f.notes.append("log ends while still armed")
    if not changes and f.has("motors"):
        # No arming records at all. With the default LOG_DISARMED=0 the log
        # file is opened by arming, so the ARM record can be missing; fall
        # back to the motors spinning.
        m = f.signals["motors"]
        mt = m["t"].to_numpy()
        spinning = m[f.motor_columns()].max(axis=1).to_numpy() > 0.05
        armed = spans(spinning, mt, min_len=1.0, max_gap=2.0)
        if armed and end_time - armed[-1][1] < 1.0 and spinning[-1]:
            f.ended_armed = True
            f.notes.append("log ends with the motors still spinning")
    f.armed = armed
    f.events.sort(key=lambda e: e.t)
    return f
