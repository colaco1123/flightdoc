"""CSV files written by ``blackbox_decode`` (Betaflight or INAV flavour).

Prefer the raw .bbl/.bfl/.txt log when you have it: the CSV loses the header
(motor output range, accelerometer scale, firmware version) and the event
stream, so some checks have less to work with.
"""

from __future__ import annotations

import os
import re
from typing import List

import numpy as np
import pandas as pd

from ..model import Flight


def _clean(name: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", name.strip())


def load(path: str) -> List[Flight]:
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    units = {_clean(c): c for c in df.columns}
    df.columns = [_clean(c) for c in df.columns]
    if "time" not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: no 'time' column, not a blackbox_decode CSV")

    f = Flight(source=os.path.basename(path), firmware="betaflight",
               firmware_version="unknown (decoded CSV)")
    f.notes.append("loaded from a decoded CSV; header and events are not available")
    if "navState" in df.columns or "attitude[0]" in df.columns:
        f.firmware = "inav"

    us = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    if "(s)" in units.get("time", ""):
        us = us * 1e6
    ok = np.isfinite(us)
    df = df[ok]
    t = (us[ok] - us[ok][0]) / 1e6
    f.duration = float(t[-1])

    def num(c):
        return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) if c in df.columns else None

    g = [num(f"gyroADC[{i}]") for i in range(3)]
    if all(x is not None for x in g):
        sp = [num(f"setpoint[{i}]") for i in range(3)]
        if any(x is None for x in sp):
            sp = [num(f"axisRate[{i}]") for i in range(3)]
        cols = dict(roll=g[0], pitch=g[1], yaw=g[2])
        if all(x is not None for x in sp):
            cols.update(des_roll=sp[0], des_pitch=sp[1], des_yaw=sp[2])
        f.add("rate", t, **cols)
        f.add("gyro", t, x=g[0], y=g[1], z=g[2])

    motors = []
    while f"motor[{len(motors)}]" in df.columns:
        motors.append(num(f"motor[{len(motors)}]"))
    if motors:
        allm = np.concatenate(motors)
        hi = 2047 if np.nanmax(allm) > 2000 else 2000
        lo = 48 if np.nanmin(allm) < 900 else 1000
        f.add("motors", t, **{f"m{i + 1}": np.clip((m - lo) / (hi - lo), 0, 1.05)
                              for i, m in enumerate(motors)})
        f.vehicle = "plane" if len(motors) == 1 else "multirotor"

    thr = num("setpoint[3]")
    if thr is not None:
        f.add("throttle", t, thr=np.clip(thr / 1000.0, 0, 1))
    elif num("rcCommand[3]") is not None:
        f.add("throttle", t, thr=np.clip((num("rcCommand[3]") - 1000) / 1000.0, 0, 1))

    vb = num("vbatLatest") if "vbatLatest" in df.columns else num("vbat")
    if vb is not None and np.nanmax(vb) > 0:
        if "(V)" not in units.get("vbatLatest", units.get("vbat", "")):
            vb = vb * 0.01
        cols = dict(volt=vb)
        cur = num("amperageLatest") if "amperageLatest" in df.columns else num("amperage")
        if cur is not None and np.nanmax(np.abs(cur)) > 0:
            if "(A)" not in units.get("amperageLatest", units.get("amperage", "")):
                cur = cur * 0.01
            cols["curr"] = cur
        f.add("battery", t, **cols)

    f.armed = [(0.0, f.duration)]
    return [f]
