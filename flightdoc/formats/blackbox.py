"""Turn decoded Betaflight / INAV blackbox sessions into ``Flight`` objects."""

from __future__ import annotations

import os
import re
from typing import List, Optional

import numpy as np

from ..model import Event, Flight
from .bbdecode import (EV_DISARM, EV_FLIGHT_MODE, EV_INFLIGHT_ADJ,
                       EV_LOGGING_RESUME, EV_SYNC_BEEP, BlackboxSession,
                       decode_file)

# Betaflight 4.x box ids (boxId_e). Mode flags in the log are box bits, not
# the internal flight mode flags.
BF_MODES = [
    "ARM", "ANGLE", "HORIZON", "MAG", "HEADFREE", "PASSTHRU", "FAILSAFE",
    "GPSRESCUE", "ANTIGRAVITY", "HEADADJ", "CAMSTAB", "BEEPERON",
    "LEDLOW", "CALIB", "OSD", "TELEMETRY", "SERVO1", "SERVO2", "SERVO3",
    "BLACKBOX", "AIRMODE", "3D", "FPVANGLEMIX", "BLACKBOXERASE", "CAMERA1",
    "CAMERA2", "CAMERA3", "FLIPOVERAFTERCRASH", "PREARM", "BEEPGPSCOUNT",
    "VTXPITMODE", "PARALYZE", "USER1", "USER2", "USER3", "USER4", "PIDAUDIO",
    "ACROTRAINER", "VTXCONTROLDISABLE", "LAUNCHCONTROL",
]

INAV_MODES = [
    "ARM", "ANGLE", "HORIZON", "NAVALTHOLD", "HEADINGHOLD", "HEADFREE",
    "HEADADJ", "CAMSTAB", "NAVRTH", "NAVPOSHOLD", "MANUAL", "BEEPERON",
    "LEDLOW", "LIGHTS", "NAVLAUNCH", "OSD", "TELEMETRY", "BLACKBOX",
    "FAILSAFE", "NAVWP", "AIRMODE", "HOMERESET", "GCSNAV", "KILLSWITCH",
    "SURFACE", "FLAPERON", "TURNASSIST", "AUTOTRIM", "AUTOTUNE", "CAMERA1",
    "CAMERA2", "CAMERA3", "OSDALT1", "OSDALT2", "OSDALT3", "NAVCOURSEHOLD",
    "BRAKING", "USER1", "USER2", "FPVANGLEMIX", "LOITERDIRCHN",
    "MSPRCOVERRIDE", "PREARM", "TURTLE", "NAVCRUISE", "AUTOLEVEL",
    "PLANWPMISSION", "SOARING", "USER3", "USER4", "CHANGEMISSION",
    "BEEPERMUTE", "MULTIFUNCTION", "MIXERPROFILE", "MIXERTRANSITION",
    "ANGLEHOLD",
]

BF_DISARM = {
    0: "arming disabled", 1: "failsafe", 2: "throttle timeout", 3: "sticks",
    4: "switch", 5: "crash protection", 6: "runaway takeoff",
    7: "GPS rescue", 8: "serial command",
}
INAV_DISARM = {
    0: "none", 1: "timeout", 2: "sticks", 3: "3D switch", 4: "switch",
    5: "kill switch", 6: "failsafe", 7: "navigation", 8: "landing",
}

# failsafePhase values that mean the craft is actually in failsafe
BF_FAILSAFE_ACTIVE = {1: "RX loss detected", 2: "landing", 3: "landed",
                      6: "GPS rescue"}
INAV_FAILSAFE_ACTIVE = {1: "RX loss detected", 2: "RX loss idle",
                        3: "return to home", 4: "landing", 5: "landed"}


def _flags(value: int, names: List[str]) -> str:
    on = [names[i] for i in range(len(names)) if value & (1 << i)]
    return "|".join(on) if on else "none"


def _version(rev: str):
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", rev)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def _ints(text: str) -> List[int]:
    out = []
    for part in (text or "").split(","):
        try:
            out.append(int(part.strip()))
        except ValueError:
            pass
    return out


def _vbat_scale(firmware: str, version, raw: np.ndarray) -> Optional[float]:
    """Volts per raw vbat unit. Betaflight 4.x and INAV 2.x+ log centivolts."""
    if firmware == "betaflight" and version >= (4, 0, 0):
        return 0.01
    if firmware == "inav" and version >= (2, 0, 0):
        return 0.01
    good = raw[raw > 0]
    if good.size == 0:
        return None
    med = float(np.median(good))
    # Older firmware logged decivolts. A 1S whoop in centivolts is ~400,
    # a 6S pack in decivolts is ~250, so the ranges barely overlap.
    return 0.01 if med > 300 else 0.1


def _classify(rev: str) -> str:
    low = rev.lower()
    if low.startswith("inav"):
        return "inav"
    return "betaflight"  # Betaflight, EmuFlight, Rotorflight, Cleanflight share the format


def load(path: str) -> List[Flight]:
    size = os.path.getsize(path)
    # Flash chips are 2^n bytes. A dump that is exactly that size and ends
    # without an end-of-log marker ran out of space; that is not a fault.
    flash_full = size >= (1 << 20) and size & (size - 1) == 0
    sessions = decode_file(path)
    flights = []
    for k, sess in enumerate(sessions):
        if len(sess.main) < 50:
            continue  # arm/disarm blips with no flight in them
        fl = _to_flight(sess, os.path.basename(path))
        if flash_full and k == len(sessions) - 1 and fl.ended_armed:
            fl.meta["storage_full"] = "yes"
            fl.meta["file_size"] = str(size)
            fl.notes.append("log stopped because the flash chip was full")
        flights.append(fl)
    return flights


def _to_flight(s: BlackboxSession, source: str) -> Flight:
    h = s.headers
    rev = h.get("Firmware revision", "")
    fw = _classify(rev)
    ver = _version(rev)
    f = Flight(source=source, firmware=fw, firmware_version=rev.split(" (")[0],
               session=s.index)
    f.meta.update({k: h[k] for k in ("Craft name", "Board information",
                                     "Firmware date", "looptime",
                                     "motorOutput", "debug_mode") if k in h})
    if s.corrupt_frames:
        f.notes.append(f"{s.corrupt_frames} corrupt frames skipped while decoding")

    time_us = s.column("time").astype(np.int64)
    t0 = int(time_us[0])
    t = (time_us - t0) / 1e6
    f.duration = float(t[-1])

    def col(name):
        c = s.column(name)
        return None if c is None else c.astype(float)

    # ---- rates (deg/s) and tracking ------------------------------------
    gyro = [col(f"gyroADC[{i}]") for i in range(3)]
    if all(g is not None for g in gyro):
        sp = [col(f"setpoint[{i}]") for i in range(3)]
        if any(x is None for x in sp):
            sp = [col(f"axisRate[{i}]") for i in range(3)]  # INAV
        cols = dict(roll=gyro[0], pitch=gyro[1], yaw=gyro[2])
        if all(x is not None for x in sp):
            cols.update(des_roll=sp[0], des_pitch=sp[1], des_yaw=sp[2])
        f.add("rate", t, **cols)
        f.add("gyro", t, x=gyro[0], y=gyro[1], z=gyro[2])

    acc = [col(f"accSmooth[{i}]") for i in range(3)]
    acc_1g = (_ints(h.get("acc_1G", "")) or [0])[0]
    if all(a is not None for a in acc) and acc_1g > 0:
        k = 9.80665 / acc_1g
        f.add("accel", t, x=acc[0] * k, y=acc[1] * k, z=acc[2] * k)

    att = [col(f"attitude[{i}]") for i in range(3)]
    if all(a is not None for a in att):  # INAV, decidegrees
        f.add("att", t, roll=att[0] / 10, pitch=att[1] / 10, yaw=att[2] / 10)

    # ---- motors ---------------------------------------------------------
    motors = []
    i = 0
    while s.column(f"motor[{i}]") is not None:
        motors.append(col(f"motor[{i}]"))
        i += 1
    if motors:
        lo, hi = None, None
        mo = _ints(h.get("motorOutput", ""))
        if len(mo) >= 2 and mo[1] > mo[0]:
            lo, hi = mo
        else:
            lo = (_ints(h.get("minthrottle", "")) or [1000])[0]
            hi = (_ints(h.get("maxthrottle", "")) or [2000])[0]
        span = max(hi - lo, 1)
        f.add("motors", t, **{f"m{j + 1}": np.clip((m - lo) / span, 0, 1.05)
                              for j, m in enumerate(motors)})
        f.meta["motor_range"] = f"{lo}..{hi}"
    erpm = []
    i = 0
    while s.column(f"eRPM[{i}]") is not None:
        erpm.append(col(f"eRPM[{i}]"))
        i += 1
    if erpm:
        f.add("erpm", t, **{f"m{j + 1}": e for j, e in enumerate(erpm)})

    servos = [n for n in s.main_names if n.startswith("servo[")]
    if len(motors) == 1 or (servos and len(motors) < 3):
        f.vehicle = "plane"
    elif len(motors) >= 3:
        f.vehicle = {3: "tricopter", 4: "quad", 6: "hexa", 8: "octo"}.get(len(motors), "multirotor")

    # ---- throttle -------------------------------------------------------
    thr = col("setpoint[3]")
    if thr is not None:
        f.add("throttle", t, thr=np.clip(thr / 1000.0, 0, 1))
    else:
        rc = col("rcCommand[3]")
        if rc is not None:
            f.add("throttle", t, thr=np.clip((rc - 1000) / 1000.0, 0, 1))

    # ---- battery --------------------------------------------------------
    vb = col("vbatLatest")
    if vb is None:
        vb = col("vbat")
    cur = col("amperageLatest")
    if cur is None:
        cur = col("amperage")
    if vb is not None and np.any(vb > 0):
        scale = _vbat_scale(fw, ver, vb)
        if scale and np.median(vb) * scale < 2.5:
            scale = None  # no voltage sensor wired up, the ADC just reads noise
        if scale:
            cols = dict(volt=vb * scale)
            if cur is not None and np.any(cur != 0):
                cols["curr"] = cur * 0.01
            f.add("battery", t, **cols)

    rssi = col("rssi")
    baro = col("BaroAlt")
    if baro is not None and np.any(baro != 0):
        f.add("alt", t, alt=(baro - baro[: max(1, min(len(baro), 50))].mean()) / 100.0)
    elif s.column("navPos[2]") is not None:
        z = col("navPos[2]")
        f.add("alt", t, alt=(z - z[0]) / 100.0)

    mag = [col(f"magADC[{i}]") for i in range(3)]
    if all(m is not None for m in mag) and np.any(np.abs(mag[0]) > 0):
        f.add("mag", t, x=mag[0], y=mag[1], z=mag[2])

    # ---- slow frames: failsafe and link state ---------------------------
    rc_cols = {}
    if rssi is not None and np.any(rssi > 0):
        rc_cols["rssi"] = np.clip(rssi / 1023.0, 0, 1)
    f_active = INAV_FAILSAFE_ACTIVE if fw == "inav" else BF_FAILSAFE_ACTIVE
    if s.slow and "failsafePhase" in s.slow_names:
        fi = s.slow_names.index("failsafePhase")
        ri = s.slow_names.index("rxSignalReceived") if "rxSignalReceived" in s.slow_names else -1
        st = np.array([max(ts - t0, 0) / 1e6 for ts, _ in s.slow])
        phase = np.array([v[fi] for _, v in s.slow])
        rx = np.array([v[ri] if ri >= 0 else 1 for _, v in s.slow])
        if not np.any(rx == 1):
            rx = np.ones_like(rx)  # some 4.x builds never fill this in
        fs = np.array([1.0 if (p in f_active or r == 0) else 0.0 for p, r in zip(phase, rx)])
        # hold the last slow-frame value at main-frame rate
        idx = np.clip(np.searchsorted(st, t, side="right") - 1, 0, len(st) - 1)
        rc_cols["failsafe"] = fs[idx]
        last = 0
        for ts, p, r in zip(st, phase, rx):
            if p != last and p in f_active:
                f.events.append(Event(float(ts), "failsafe",
                                      f"failsafe: {f_active[p]}", "error"))
            last = p
        for ts, r, prev_r in zip(st[1:], rx[1:], rx[:-1]):
            if r == 0 and prev_r != 0:
                f.events.append(Event(float(ts), "failsafe", "RX signal lost", "error"))
    if rc_cols:
        f.add("rc", t, **rc_cols)

    # ---- GPS ------------------------------------------------------------
    if s.gps and "GPS_numSat" in s.gps_names:
        names = s.gps_names
        arr = np.array([v for _, v in s.gps], dtype=float)
        gt = np.array([(ts - t0) / 1e6 for ts, _ in s.gps])
        g = dict(nsats=arr[:, names.index("GPS_numSat")])
        if "GPS_coord[0]" in names:
            g["lat"] = arr[:, names.index("GPS_coord[0]")] / 1e7
            g["lon"] = arr[:, names.index("GPS_coord[1]")] / 1e7
        if "GPS_hdop" in names:
            g["hdop"] = arr[:, names.index("GPS_hdop")] / 100.0
        if "GPS_fixType" in names:
            g["fix"] = arr[:, names.index("GPS_fixType")]
        if "GPS_speed" in names:
            g["spd"] = arr[:, names.index("GPS_speed")] / 100.0
        keep = (gt >= -1) & (gt <= f.duration + 1)
        f.add("gps", gt[keep], **{k: v[keep] for k, v in g.items()})

    # ---- events -----------------------------------------------------------
    modes = INAV_MODES if fw == "inav" else BF_MODES
    disarms = INAV_DISARM if fw == "inav" else BF_DISARM
    disarm_t = None
    for ts, ev, data in s.events:
        et = (ts - t0) / 1e6
        if ev == EV_DISARM:
            reason = disarms.get(data.get("reason"), f"code {data.get('reason')}")
            lvl = "error" if data.get("reason") in ((1, 5, 6) if fw != "inav" else (6,)) else "info"
            f.events.append(Event(et, "disarm", f"disarmed ({reason})", lvl))
            if disarm_t is None:
                disarm_t = et
        elif ev == EV_FLIGHT_MODE:
            f.events.append(Event(et, "mode", _flags(data["flags"], modes)))
        elif ev == EV_LOGGING_RESUME:
            # Written at the start of most logs too, so only interesting mid-flight.
            if et > 1.0:
                f.events.append(Event(et, "message", "logging resumed after a gap", "warning"))
        elif ev == EV_INFLIGHT_ADJ:
            f.events.append(Event(et, "message",
                                  f"in-flight adjustment {data['function']} = {data['value']}"))
        elif ev == EV_SYNC_BEEP:
            f.events.append(Event(et, "message", "sync beep"))
    f.events.sort(key=lambda e: e.t)

    end = disarm_t if disarm_t is not None else f.duration
    f.armed = [(0.0, max(0.0, min(end, f.duration)))]
    f.ended_armed = not s.clean_end and disarm_t is None
    if f.ended_armed:
        f.notes.append("log ends without a disarm or end-of-log marker")
    return f
