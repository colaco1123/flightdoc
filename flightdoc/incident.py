"""Find the moment a flight went wrong.

Most of the value of a crash report is in pointing at *when* things broke, so
everything else can be judged by how close it is to that moment. We look, in
order of trust, for:

1. the flight controller saying so (ArduPilot crash check, Betaflight crash
   protection disarm, PX4 "free fall" / "attitude failure" messages)
2. an impact: an acceleration spike while armed, usually followed by
   tumbling or the log stopping
3. the vehicle tipping past a sane attitude and staying there
4. the log simply stopping while the motors were spinning
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from .checks.base import Incident, fmt_t
from .model import Flight, sample_rate

G = 9.80665

CRASH_TEXT = re.compile(
    r"crash|free ?fall|attitude failure|flight termination|parachute|"
    r"loss of control|flip detected", re.I)


def find_incident(f: Flight) -> Optional[Incident]:
    candidates: List[Incident] = []

    # 1. the autopilot's own verdict
    for e in f.events:
        if e.kind in ("error", "message", "disarm") and CRASH_TEXT.search(e.text) and \
                e.level != "info":
            if e.kind == "disarm" and "crash" not in e.text and "runaway" not in e.text:
                continue
            candidates.append(Incident(e.t, "crash", f"flight controller reported: {e.text}",
                                       [f"{fmt_t(e.t)} {e.text}"], 0.95))
            break
        if e.kind == "disarm" and "runaway" in e.text:
            candidates.append(Incident(e.t, "crash", "runaway takeoff protection disarmed the craft",
                                       [f"{fmt_t(e.t)} {e.text}"], 0.9))
            break

    impact = _impact(f)
    if impact:
        candidates.append(impact)
    tip = _tipped_over(f)
    if tip:
        candidates.append(tip)
    cut = _log_cut(f)
    if cut:
        candidates.append(cut)
    stuck = _stuck(f)
    if stuck:
        candidates.append(stuck)

    if not candidates:
        return None
    # Prefer the most trusted signal, but if two agree on the time merge
    # their evidence so the report shows both.
    candidates.sort(key=lambda c: -c.confidence)
    best = candidates[0]
    for other in candidates[1:]:
        # only fold in what led up to it, not the aftermath
        if -5.0 < other.t - best.t < 1.0:
            best.evidence.extend(x for x in other.evidence if x not in best.evidence)
            best.confidence = min(0.99, best.confidence + 0.05)
    # An earlier, independent symptom (tipping over before an impact) moves
    # the start of the incident back so we look at what led to it.
    earliest = min(c.t for c in candidates if abs(c.t - best.t) < 10.0)
    if earliest < best.t:
        best.evidence.append(f"first sign of trouble at {fmt_t(earliest)}")
        best.t = earliest
    return best


def _impact(f: Flight) -> Optional[Incident]:
    if not f.has("accel", "x", "y", "z"):
        return None
    a = f.signals["accel"]
    t = a["t"].to_numpy()
    armed = f.is_armed(t)
    if armed.sum() < 10:
        return None
    mag = np.sqrt(a["x"] ** 2 + a["y"] ** 2 + a["z"] ** 2).to_numpy()
    # Heavily filtered accelerometers (Betaflight logs accSmooth) under-read
    # impacts, so the threshold is modest and we ask for a second symptom.
    # Freestyle and racing quads pull 5 g and 1000 deg/s on purpose, so for
    # those firmwares only the craft stopping right after counts as proof.
    acro_craft = f.firmware in ("betaflight", "inav") or f.vehicle == "plane"
    spike = np.abs(mag - G) > 4.0 * G
    spike &= armed
    if not spike.any():
        return None
    # ignore spikes in the first second (arming jolt, hand launch)
    spike &= t > (f.first_armed() + 1.0)
    if not spike.any():
        return None
    idx = np.flatnonzero(spike)
    for i in idx:
        ti = t[i]
        after = (t > ti) & (t < ti + 3.0)
        confirm = []
        strong = []  # confirmations that count even for acro craft
        if mag[i] > 8.0 * G and not acro_craft:
            confirm.append("an 8 g+ spike does not happen in normal flight")
        ends_soon = f.duration - ti < 3.0 and f.meta.get("storage_full") != "yes"
        if ends_soon or (f.armed and f.armed[-1][1] < f.duration - 0.5
                         and abs(f.armed[-1][1] - ti) < 3.0):
            strong.append("log ends or craft disarms within 3 s")
        if f.has("gyro") and not acro_craft:
            g = f.signals["gyro"]
            gt = g["t"].to_numpy()
            sel = (gt > ti - 0.5) & (gt < ti + 1.0)
            if sel.any():
                peak = np.abs(g.loc[sel, ["x", "y", "z"]].to_numpy()).max()
                if peak > 500:
                    confirm.append(f"rotation rate peaks at {peak:.0f} deg/s")
        if f.has("att", "roll", "pitch") and not acro_craft:
            at = f.signals["att"]
            sel = (at["t"] > ti - 2.0) & (at["t"] < ti + 1.0)
            if sel.any():
                tilt = np.maximum(at.loc[sel, "roll"].abs(), at.loc[sel, "pitch"].abs()).max()
                if tilt > 60:
                    confirm.append(f"tilted {tilt:.0f} deg around the spike")
        if f.has("motors") and after.any():
            m = f.signals["motors"]
            mt = m["t"].to_numpy()
            selm = (mt > ti + 0.3) & (mt < ti + 3.0)
            # Stopped and staying stopped: the pilot cut throttle or the
            # props are jammed in the grass.
            if selm.any() and m.loc[selm, f.motor_columns()].max(axis=1).quantile(0.9) < 0.12:
                strong.append("motors stop right after")
        if strong or confirm:
            return Incident(float(ti), "crash",
                            f"impact of {mag[i] / G:.1f} g at {fmt_t(ti)}",
                            [f"acceleration spike of {mag[i] / G:.1f} g"] + strong + confirm,
                            0.8 if strong else 0.65)
    return None


def _tipped_over(f: Flight) -> Optional[Incident]:
    if not f.has("att", "roll", "pitch"):
        return None
    a = f.signals["att"]
    t = a["t"].to_numpy()
    flying = f.flying(t)
    tilt = np.maximum(np.abs(a["roll"].to_numpy()), np.abs(a["pitch"].to_numpy()))
    # A fixed wing or a quad doing flips legitimately goes past 90 deg, so we
    # only call it when it is sustained and followed by the end of the flight.
    over = (tilt > 75) & flying
    if not over.any():
        return None
    fs = sample_rate(t) or 10.0
    n = max(1, int(0.7 * fs))
    run = np.convolve(over.astype(int), np.ones(n, dtype=int), mode="same") >= n
    if not run.any():
        return None
    i = int(np.flatnonzero(run)[0])
    ti = float(t[i])
    end = f.armed[-1][1] if f.armed else f.duration
    modes = [e for e in f.events if e.kind == "mode" and e.t <= ti]
    acro = bool(modes) and re.search(r"ACRO|FLIP|SPORT|MANUAL|AIRMODE", modes[-1].text) is not None
    if f.vehicle == "plane" or acro:
        # Aerobatics are legitimate here; only count it if the flight ends soon after.
        if end - ti > 15.0:
            return None
        conf = 0.5
    else:
        conf = 0.75 if end - ti < 15.0 else 0.6
    return Incident(ti, "loss_of_control",
                    f"tilted past 75 deg at {fmt_t(ti)}",
                    [f"roll/pitch beyond 75 deg for at least 0.7 s from {fmt_t(ti)}",
                     f"flight ends {end - ti:.1f} s later"], conf)


def _log_cut(f: Flight) -> Optional[Incident]:
    if not f.ended_armed or not f.has("motors"):
        return None
    m = f.signals["motors"]
    cols = f.motor_columns()
    tail = m[m["t"] > f.duration - 1.0]
    if tail.empty or tail[cols].mean().mean() < 0.15:
        return None
    if f.meta.get("storage_full") == "yes":
        return None
    return Incident(f.duration, "log_cut",
                    f"log stops at {fmt_t(f.duration)} with motors still running",
                    ["no disarm and no end-of-log marker",
                     f"average motor output in the last second: {tail[cols].mean().mean():.0%}"],
                    0.7)


def _stuck(f: Flight) -> Optional[Incident]:
    from .checks.motors import stuck_spans
    found = stuck_spans(f)
    if not found or f.vehicle == "plane":
        return None
    a, b = found[0]
    return Incident(a, "loss_of_control",
                    f"craft stopped moving at {fmt_t(a)} while the motors kept fighting",
                    [f"motors far apart with almost no rotation from {fmt_t(a)} to {fmt_t(b)}"],
                    0.55)
