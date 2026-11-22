"""Firmware-independent view of a flight log.

Every loader (ArduPilot, PX4, Betaflight, INAV) fills in a ``Flight``. The
checks only ever look at this structure, so a check written once works for
every firmware that logs the signals it needs.

Signals are stored as small pandas DataFrames keyed by a short name. Every
frame has a ``t`` column in seconds from the start of the log. Columns that a
firmware does not log are simply absent; checks test for them with
``flight.has("battery", "curr")``.

Canonical signals and units
---------------------------
att      roll, pitch, yaw, des_roll, des_pitch, des_yaw      deg
rate     roll, pitch, yaw, des_roll, des_pitch, des_yaw      deg/s
gyro     x, y, z  (raw, high rate if available)               deg/s
accel    x, y, z  (raw, high rate if available)               m/s^2
vibe     x, y, z and/or total (ArduPilot style vibration), clip  m/s^2, count
sticks   roll, pitch, yaw  pilot stick deflection           -1..1
motors   m1 .. mN normalised command, 0 = idle/min, 1 = max
erpm     m1 .. mN  motor speed telemetry (relative units)
throttle thr  pilot or controller throttle demand             0..1
battery  volt, curr                                           V, A
power    vcc  flight controller supply rail                   V
mag      x, y, z  magnetometer, any consistent unit
gps      fix, nsats, hdop, lat, lon, alt, spd                 -, -, -, deg, deg, m, m/s
ekf      vel, pos, hgt, mag  innovation test ratios (1.0 = fail)
alt      alt  altitude above start                            m
rc       rssi (0..1), failsafe (0/1)
cpu      load (0..1), slow (fraction of overrun loops)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Event:
    t: float
    kind: str          # "mode", "arm", "disarm", "error", "message", "failsafe", ...
    text: str
    level: str = "info"  # "info", "warning", "error"

    def __repr__(self) -> str:  # short form helps when debugging checks
        return f"<{self.kind} @{self.t:.2f}s {self.text!r}>"


@dataclass
class Flight:
    source: str                      # file name
    firmware: str                    # "ardupilot", "px4", "betaflight", "inav"
    firmware_version: str = ""
    vehicle: str = ""                # "copter", "plane", "quad", ...
    session: int = 0                 # index when a file holds several logs
    duration: float = 0.0
    signals: Dict[str, pd.DataFrame] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)
    # (start, end) spans in seconds where the vehicle was armed
    armed: List[Tuple[float, float]] = field(default_factory=list)
    # True when the log stopped without a disarm (power cut, impact, full card)
    ended_armed: bool = False
    notes: List[str] = field(default_factory=list)

    # ---- access helpers -------------------------------------------------
    def has(self, name: str, *cols: str) -> bool:
        df = self.signals.get(name)
        if df is None or len(df) < 2:
            return False
        return all(c in df.columns for c in cols)

    def get(self, name: str) -> Optional[pd.DataFrame]:
        return self.signals.get(name)

    def add(self, name: str, t, **cols) -> None:
        """Store a signal. Rows with a non-finite or repeated timestamp are dropped."""
        t = np.asarray(t, dtype=float)
        if t.size < 2:
            return
        data = {"t": t}
        for k, v in cols.items():
            v = np.asarray(v, dtype=float)
            if v.shape != t.shape:
                raise ValueError(f"{name}.{k}: length {v.shape} does not match time {t.shape}")
            data[k] = v
        df = pd.DataFrame(data)
        df = df[np.isfinite(df["t"])]
        df = df.sort_values("t", kind="stable").drop_duplicates("t", keep="last")
        self.signals[name] = df.reset_index(drop=True)

    def motor_columns(self, name: str = "motors") -> List[str]:
        df = self.signals.get(name)
        if df is None:
            return []
        cols = [c for c in df.columns if c.startswith("m") and c[1:].isdigit()]
        return sorted(cols, key=lambda c: int(c[1:]))

    def is_armed(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        mask = np.zeros(t.shape, dtype=bool)
        for a, b in self.armed:
            mask |= (t >= a) & (t <= b)
        return mask

    def flying(self, t: np.ndarray) -> np.ndarray:
        """Best guess at "in the air": armed and motors well above idle."""
        t = np.asarray(t, dtype=float)
        mask = self.is_armed(t)
        if self.has("motors"):
            m = self.signals["motors"]
            cols = self.motor_columns()
            if cols:
                mean = m[cols].mean(axis=1).to_numpy()
                spinning = np.interp(t, m["t"].to_numpy(), mean) > 0.2
                mask &= spinning
        return mask

    def first_armed(self) -> float:
        return self.armed[0][0] if self.armed else 0.0

    def events_of(self, *kinds: str) -> List[Event]:
        return [e for e in self.events if e.kind in kinds]

    def summary(self) -> str:
        sig = ", ".join(f"{k}({len(v)})" for k, v in sorted(self.signals.items()))
        return (f"{self.source} [{self.firmware} {self.firmware_version}] "
                f"{self.duration:.1f}s, signals: {sig}")


def window(df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """Rows of ``df`` with t in [t0, t1]."""
    t = df["t"].to_numpy()
    lo = np.searchsorted(t, t0, side="left")
    hi = np.searchsorted(t, t1, side="right")
    return df.iloc[lo:hi]


def spans(mask: np.ndarray, t: np.ndarray, min_len: float = 0.0,
          max_gap: float = 0.0) -> List[Tuple[float, float]]:
    """Turn a boolean mask sampled at times ``t`` into (start, end) spans.

    Spans separated by less than ``max_gap`` seconds are merged, then spans
    shorter than ``min_len`` are dropped.
    """
    mask = np.asarray(mask, dtype=bool)
    t = np.asarray(t, dtype=float)
    if mask.size == 0 or not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    out: List[Tuple[float, float]] = []
    for s, e in zip(starts, ends):
        a, b = float(t[s]), float(t[e])
        if out and a - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [(a, b) for a, b in out if b - a >= min_len]


def sample_rate(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 3:
        return 0.0
    dt = np.median(np.diff(t))
    return 1.0 / dt if dt > 0 else 0.0
