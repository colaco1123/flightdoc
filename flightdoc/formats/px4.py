"""PX4 ULog (.ulg) files to ``Flight``. Uses pyulog for the container format."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..model import Event, Flight, spans

NAV_STATES = {
    0: "MANUAL", 1: "ALTCTL", 2: "POSCTL", 3: "AUTO_MISSION", 4: "AUTO_LOITER",
    5: "AUTO_RTL", 6: "AUTO_RCRECOVER", 7: "AUTO_RTGS", 8: "AUTO_LANDENGFAIL",
    9: "AUTO_LANDGPSFAIL", 10: "ACRO", 11: "UNUSED", 12: "DESCEND",
    13: "TERMINATION", 14: "OFFBOARD", 15: "STAB", 16: "RATTITUDE",
    17: "AUTO_TAKEOFF", 18: "AUTO_LAND", 19: "AUTO_FOLLOW_TARGET",
    20: "AUTO_PRECLAND", 21: "ORBIT", 22: "AUTO_VTOL_TAKEOFF",
}
# failsafe_flags fields worth reporting when they trip in flight
FAILSAFE_FLAGS = {
    "manual_control_signal_lost": "failsafe: RC signal lost",
    "gcs_connection_lost": "failsafe: ground station link lost",
    "gnss_lost": "failsafe: GNSS lost",
    "global_position_invalid": "failsafe: position estimate lost",
    "local_altitude_invalid": "failsafe: altitude estimate lost",
    "battery_warning": "failsafe: battery warning",
    "battery_low_remaining_time": "failsafe: battery flight time low",
    "battery_unhealthy": "failsafe: battery unhealthy",
    "fd_critical_failure": "failure detector: attitude failure",
    "fd_motor_failure": "failure detector: motor failure",
    "fd_imbalanced_prop": "failure detector: imbalanced prop",
    "fd_alt_loss": "failure detector: altitude loss",
    "geofence_breached": "failsafe: geofence breached",
    "wind_limit_exceeded": "failsafe: wind limit exceeded",
}

LOG_LEVELS = {0: "error", 1: "error", 2: "error", 3: "error", 4: "warning",
              5: "info", 6: "info", 7: "info"}


def _euler(q0, q1, q2, q3):
    roll = np.degrees(np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2)))
    return roll, pitch, yaw


def load(path: str) -> List[Flight]:
    from pyulog import ULog  # imported lazily so the other formats work without it

    ulog = ULog(path)
    return [_to_flight(ulog, os.path.basename(path))]


def _topic(ulog, name: str, multi: int = 0) -> Optional[Dict[str, np.ndarray]]:
    for d in ulog.data_list:
        if d.name == name and d.multi_id == multi:
            return d.data
    return None


def _first(ulog, *names):
    for n in names:
        d = _topic(ulog, n)
        if d is not None and len(d["timestamp"]) > 1:
            return n, d
    return None, None


def _to_flight(ulog, source: str) -> Flight:
    f = Flight(source=source, firmware="px4")
    info = ulog.msg_info_dict
    rel = info.get("ver_sw_release")
    if isinstance(rel, int) and rel > 0:
        # 0xMMmmppTT: major, minor, patch, type (255 = release)
        f.firmware_version = f"PX4 v{rel >> 24}.{(rel >> 16) & 0xFF}.{(rel >> 8) & 0xFF}"
        if rel & 0xFF != 0xFF:
            f.firmware_version += "-dev"
    else:
        f.firmware_version = "PX4 " + str(info.get("ver_sw", ""))[:12]
    f.meta.update({k: str(info[k]) for k in ("sys_name", "ver_hw", "sys_uuid") if k in info})
    f.params = {k: float(v) for k, v in ulog.initial_parameters.items()}
    t0 = ulog.start_timestamp

    def t(d):
        return (d["timestamp"].astype(np.float64) - t0) / 1e6

    # ---- attitude / rates ----------------------------------------------------
    att = _topic(ulog, "vehicle_attitude")
    if att is not None:
        r, p, y = _euler(att["q[0]"], att["q[1]"], att["q[2]"], att["q[3]"])
        cols = dict(roll=r, pitch=p, yaw=y)
        sp = _topic(ulog, "vehicle_attitude_setpoint")
        if sp is not None:
            if "q_d[0]" in sp:
                dr, dp, dy = _euler(sp["q_d[0]"], sp["q_d[1]"], sp["q_d[2]"], sp["q_d[3]"])
            else:
                dr, dp, dy = (np.degrees(sp[k]) for k in ("roll_body", "pitch_body", "yaw_body"))
            ta, ts = t(att), t(sp)
            cols.update(des_roll=np.interp(ta, ts, dr), des_pitch=np.interp(ta, ts, dp),
                        des_yaw=np.interp(ta, ts, dy))
        f.add("att", t(att), **cols)

    name, rates = _first(ulog, "vehicle_angular_velocity", "vehicle_attitude")
    if rates is not None:
        if name == "vehicle_angular_velocity":
            rr = [np.degrees(rates[f"xyz[{i}]"]) for i in range(3)]
        else:
            rr = [np.degrees(rates[k]) for k in ("rollspeed", "pitchspeed", "yawspeed")]
        cols = dict(roll=rr[0], pitch=rr[1], yaw=rr[2])
        rsp = _topic(ulog, "vehicle_rates_setpoint")
        if rsp is not None:
            tr, ts = t(rates), t(rsp)
            cols.update(des_roll=np.interp(tr, ts, np.degrees(rsp["roll"])),
                        des_pitch=np.interp(tr, ts, np.degrees(rsp["pitch"])),
                        des_yaw=np.interp(tr, ts, np.degrees(rsp["yaw"])))
        f.add("rate", t(rates), **cols)

    sc = _topic(ulog, "sensor_combined")
    if sc is not None:
        ts = t(sc)
        f.add("gyro", ts, **{a: np.degrees(sc[f"gyro_rad[{i}]"]) for i, a in enumerate("xyz")})
        f.add("accel", ts, **{a: sc[f"accelerometer_m_s2[{i}]"] for i, a in enumerate("xyz")})
        if "magnetometer_ga[0]" in sc:
            f.add("mag", ts, **{a: sc[f"magnetometer_ga[{i}]"] for i, a in enumerate("xyz")})
    if not f.has("mag"):
        name, mag = _first(ulog, "vehicle_magnetometer", "sensor_mag")
        if mag is not None:
            key = "magnetometer_ga" if "magnetometer_ga[0]" in mag else None
            if key:
                f.add("mag", t(mag), **{a: mag[f"{key}[{i}]"] for i, a in enumerate("xyz")})
            elif "x" in mag:
                f.add("mag", t(mag), x=mag["x"], y=mag["y"], z=mag["z"])

    # ---- motors --------------------------------------------------------------
    am = _topic(ulog, "actuator_motors")
    if am is not None:
        cols = {}
        for i in range(12):
            k = f"control[{i}]"
            if k in am and np.any(np.isfinite(am[k])) and np.nanmax(np.abs(am[k])) > 0:
                cols[f"m{i + 1}"] = np.nan_to_num(am[k], nan=0.0)
        if cols:
            f.add("motors", t(am), **cols)
    if not f.has("motors"):
        ao = _topic(ulog, "actuator_outputs")
        if ao is not None:
            n = int(np.max(ao["noutputs"])) if "noutputs" in ao else 8
            lo = f.params.get("PWM_MAIN_MIN", f.params.get("PWM_MIN", 1000.0))
            hi = f.params.get("PWM_MAIN_MAX", f.params.get("PWM_MAX", 2000.0))
            cols = {}
            for i in range(min(n, 8)):
                v = ao.get(f"output[{i}]")
                if v is None or np.nanmax(v) < 900:  # unused channel
                    continue
                cols[f"m{i + 1}"] = np.clip((v - lo) / max(hi - lo, 1), 0, 1.05)
            if cols:
                f.add("motors", t(ao), **cols)
                f.meta["motor_range"] = f"{lo:.0f}..{hi:.0f}"

    name, thr = _first(ulog, "vehicle_thrust_setpoint", "vehicle_attitude_setpoint")
    if thr is not None:
        if "xyz[2]" in thr:
            f.add("throttle", t(thr), thr=np.clip(-thr["xyz[2]"], 0, 1))
        elif "thrust_body[2]" in thr:
            f.add("throttle", t(thr), thr=np.clip(-thr["thrust_body[2]"], 0, 1))
        elif "thrust" in thr:
            f.add("throttle", t(thr), thr=np.clip(thr["thrust"], 0, 1))

    # ---- power -----------------------------------------------------------------
    bat = _topic(ulog, "battery_status")
    if bat is not None:
        v = bat.get("voltage_v", bat.get("voltage_filtered_v"))
        if v is not None and np.nanmax(v) > 0:
            cols = dict(volt=v)
            if "current_a" in bat:
                cols["curr"] = np.clip(bat["current_a"], 0, None)
            f.add("battery", t(bat), **cols)
    sp = _topic(ulog, "system_power")
    if sp is not None:
        v5 = sp.get("voltage5v_v", sp.get("voltage5V_v"))
        if v5 is not None:
            f.add("power", t(sp), vcc=v5)

    # ---- navigation ------------------------------------------------------------
    name, gps = _first(ulog, "sensor_gps", "vehicle_gps_position", "sensor_gnss")
    if gps is not None:
        def pick(*keys):
            for k in keys:
                if k in gps:
                    return gps[k].astype(float)
            return None
        lat, lon = pick("lat", "latitude"), pick("lon", "longitude")
        cols = {}
        if pick("fix_type") is not None:
            cols["fix"] = pick("fix_type")
        if pick("satellites_used") is not None:
            cols["nsats"] = pick("satellites_used")
        if lat is not None and lon is not None:
            # older releases log degrees * 1e7 as integers, newer ones degrees
            scale = 1e-7 if np.nanmax(np.abs(lat)) > 1000 else 1.0
            cols.update(lat=lat * scale, lon=lon * scale)
        if pick("hdop") is not None:
            cols["hdop"] = pick("hdop")
        alt = pick("alt", "altitude_msl")
        if alt is not None:
            cols["alt"] = alt / 1000.0 if np.nanmax(np.abs(alt)) > 20000 else alt
        spd = pick("vel_m_s", "ground_speed")
        if spd is not None:
            cols["spd"] = spd
        if cols:
            f.add("gps", t(gps), **cols)

    lp = _topic(ulog, "vehicle_local_position")
    if lp is not None and "z" in lp:
        f.add("alt", t(lp), alt=-(lp["z"] - lp["z"][0]))

    es = _topic(ulog, "estimator_status")
    if es is not None:
        # field names moved between releases (mag_test_ratio became hdg_test_ratio)
        names = {"vel": ("vel_test_ratio",), "pos": ("pos_test_ratio",),
                 "hgt": ("hgt_test_ratio",), "mag": ("mag_test_ratio", "hdg_test_ratio")}
        cols = {}
        for key, options in names.items():
            for opt in options:
                if opt in es:
                    cols[key] = np.nan_to_num(es[opt].astype(float), nan=0.0)
                    break
        if cols:
            f.add("ekf", t(es), **cols)

    rc = _topic(ulog, "input_rc")
    if rc is not None:
        cols = {}
        if "rssi" in rc:
            r = rc["rssi"].astype(float)
            cols["rssi"] = np.clip(r / 100.0, 0, 1) if np.nanmax(r) > 1 else r
        if "rc_lost" in rc:
            cols["failsafe"] = rc["rc_lost"].astype(float)
        if cols:
            f.add("rc", t(rc), **cols)

    cpu = _topic(ulog, "cpuload")
    if cpu is not None:
        f.add("cpu", t(cpu), load=cpu["load"])

    # ---- status, arming and events ----------------------------------------------
    vs = _topic(ulog, "vehicle_status")
    end_time = max((float(df["t"].iloc[-1]) for df in f.signals.values()), default=0.0)
    f.duration = end_time
    if vs is not None:
        ts = t(vs)
        armed = vs["arming_state"] == 2
        f.armed = spans(armed, ts)
        if f.armed and armed[-1]:
            f.armed[-1] = (f.armed[-1][0], end_time)
            f.ended_armed = True
            f.notes.append("log ends while still armed")
        last_nav = None
        for tt, nav in zip(ts, vs["nav_state"]):
            if nav != last_nav:
                f.events.append(Event(float(tt), "mode", NAV_STATES.get(int(nav), f"nav {int(nav)}")))
                last_nav = nav
        prev_arm = False
        for tt, a in zip(ts, armed):
            if a != prev_arm:
                f.events.append(Event(float(tt), "arm" if a else "disarm", "armed" if a else "disarmed"))
                prev_arm = a
        if "failsafe" in vs:
            fs = vs["failsafe"].astype(bool)
            for a, b in spans(fs, ts):
                f.events.append(Event(a, "failsafe", "failsafe active", "warning"))
        if "vehicle_type" in vs:
            vt = int(np.bincount(vs["vehicle_type"].astype(int)).argmax())
            f.vehicle = {1: "multirotor", 2: "plane", 3: "rover"}.get(vt, "")
            if "is_vtol" in vs and np.any(vs["is_vtol"]):
                f.vehicle = "vtol"
        elif "is_rotary_wing" in vs:
            f.vehicle = "multirotor" if np.any(vs["is_rotary_wing"]) else "plane"

    flags = _topic(ulog, "failsafe_flags")
    if flags is not None:
        ft = t(flags)
        armed_now = f.is_armed(ft)
        for key, text in FAILSAFE_FLAGS.items():
            if key not in flags:
                continue
            on = flags[key].astype(bool)
            rising = np.flatnonzero(on[1:] & ~on[:-1] & armed_now[1:]) + 1
            for i in rising[:5]:
                f.events.append(Event(float(ft[i]), "failsafe", text, "error"))

    if not f.armed and f.has("motors"):
        # Old logs (or ones without vehicle_status): treat spinning motors as armed.
        m = f.signals["motors"]
        spinning = m[f.motor_columns()].max(axis=1).to_numpy() > 0.05
        f.armed = spans(spinning, m["t"].to_numpy(), min_len=1.0, max_gap=2.0)

    for m in ulog.logged_messages:
        lvl = int(m.log_level)
        if lvl >= 48:  # stored as the ASCII digit '0'..'7'
            lvl -= 48
        level = LOG_LEVELS.get(lvl, "info")
        f.events.append(Event((m.timestamp - t0) / 1e6, "message", m.message, level))
    f.events.sort(key=lambda e: e.t)
    return f
