"""Motor output: failures, saturation, imbalance and thrust headroom."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..model import Flight, spans
from .base import (Finding, Incident, Panel, Plot, check, flight_window, fmt_t,
                   lowpass, near, sample_rate)

AP_MOTOR_DOC = ("ArduPilot: diagnosing problems with logs (motor outputs)",
                "https://ardupilot.org/copter/docs/common-diagnosing-problems-using-logs.html")
AP_THRUST_LOSS = ("ArduPilot: thrust loss and yaw imbalance warnings",
                  "https://ardupilot.org/copter/docs/thrust_loss_yaw_imbalance.html")

HIGH = 0.9
LOW = 0.2


def _smooth(m, fs):
    # Only smooth fast logs; at 10 Hz (ArduPilot RCOU default) a filter eats the
    # short peaks we are looking for.
    if fs < 50:
        return m
    return m.apply(lambda c: lowpass(c.to_numpy(), fs, 10.0))


@check("motor_failure")
def motor_failure(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """One motor pinned at maximum while another is cut back hard.

    When a motor or prop stops producing thrust the attitude controller asks
    that motor for everything it has and throttles the opposite one down to
    keep the frame level. That pattern, held for more than a moment, is the
    classic signature of a lost motor, ESC or prop. Planes are skipped.
    """
    cols = f.motor_columns()
    if len(cols) < 3 or f.vehicle == "plane":
        return []
    m = flight_window(f, f.signals["motors"])
    if len(m) < 10:
        return []
    t = m["t"].to_numpy()
    fs = sample_rate(t)
    sm = _smooth(m[cols], fs)
    arr = sm.to_numpy()
    best = None
    for i, c in enumerate(cols):
        others = np.delete(arr, i, axis=1)
        pinned = (arr[:, i] >= HIGH) & (others.min(axis=1) <= LOW)
        for a, b in spans(pinned, t, min_len=0.3, max_gap=0.3):
            if inc is not None and a > inc.t + 2.0:
                continue  # thrashing on the ground after the crash
            # was this motor behaving before? (rules out a crash already in progress)
            before = (t > a - 5) & (t < a - 0.5)
            normal_before = before.any() and np.abs(arr[before, i] - arr[before].mean(axis=1)).mean() < 0.12
            score = min(b - a, 3.0) * (2.0 if normal_before else 1.0) * (2.0 if near(a, inc, 20) else 1.0)
            if best is None or score > best[0]:
                best = (score, i, a, b, normal_before)
    if best is None:
        return []
    _, i, a, b, normal_before = best
    sel = (t >= a) & (t <= b)
    motor = cols[i]
    low_motor = cols[int(np.argmin(arr[sel].mean(axis=0)))]
    ev = [f"motor {motor[1:]} held at {arr[sel, i].mean():.0%} from {fmt_t(a)} to {fmt_t(b)} "
          f"({b - a:.1f} s)",
          f"motor {low_motor[1:]} cut to {arr[sel, cols.index(low_motor)].mean():.0%} at the same time "
          "to keep the frame level"]
    if normal_before:
        ev.append("the motors were balanced just before, so this started suddenly")
    thrust_msgs = [e for e in f.events if "thrust loss" in e.text.lower() and abs(e.t - a) < 10]
    if thrust_msgs:
        ev.append(f"flight controller warned: '{thrust_msgs[0].text}' at {fmt_t(thrust_msgs[0].t)}")
    sev = "critical" if (near(a, inc, 30) or thrust_msgs or b - a > 2.0) else "warning"
    conf = 0.9 if (normal_before and (thrust_msgs or near(a, inc, 30))) else 0.7 if normal_before else 0.5
    if f.has("erpm"):
        e = f.signals["erpm"]
        if motor in e:
            es = e[(e["t"] >= a) & (e["t"] <= b)]
            other = [c for c in e.columns if c.startswith("m") and c != motor]
            if len(es) and other:
                ratio = es[motor].mean() / max(es[other].mean().mean(), 1e-6)
                ev.append(f"motor {motor[1:]} speed telemetry at {ratio:.0%} of the others while commanded to max")
                if ratio < 0.7:
                    conf = min(0.97, conf + 0.1)
    return [Finding(
        check="motor_failure",
        title=f"Motor {motor[1:]} lost thrust",
        severity=sev,
        summary=(f"Motor {motor[1:]} was driven to full power while motor {low_motor[1:]} was cut back: "
                 f"the controller was fighting a loss of thrust on motor {motor[1:]} "
                 "(failed motor, ESC, prop, or a loose prop nut)"),
        confidence=conf, evidence=ev,
        advice=[f"Inspect motor {motor[1:]} and its prop: spin it by hand, check the bell, "
                "bearings and windings for damage or burning",
                f"Check the ESC for motor {motor[1:]} and its solder joints and connectors",
                "Make sure the prop was not on the wrong motor (direction) and that the prop nut "
                "was tight; a prop that slips loses thrust only under load",
                "Confirm the motor numbering and direction match the frame setup"],
        t0=a, t1=b,
        plot=Plot([Panel("motors", cols, "motor output", hlines=[(1.0, "max")], ylim=(-0.05, 1.1))],
                  t0=a - 8, t1=b + 4),
        refs=[AP_THRUST_LOSS, AP_MOTOR_DOC],
        data={"motor": int(motor[1:]), "pinned_s": b - a},
    )]


def stuck_spans(f: Flight, min_spread: float = 0.5, max_rate: float = 100.0,
                min_len: float = 1.5):
    """Spans where the motors are wildly unequal but the craft hardly moves.

    In the air, a big spread between motors always produces rotation. If the
    gyro stays quiet, the craft is resting on something (the ground, a tree)
    while the controller winds its I-term up against it.
    """
    cols = f.motor_columns()
    if len(cols) < 3 or not f.has("gyro"):
        return []
    g = f.signals["gyro"]
    t = g["t"].to_numpy()
    fs = sample_rate(t) or 1.0
    motion = g[["x", "y", "z"]].abs().max(axis=1).rolling(
        max(1, int(fs * 0.5)), center=True, min_periods=1).max().to_numpy()
    m = f.signals["motors"]
    arr = m[cols].to_numpy()
    spread = np.interp(t, m["t"].to_numpy(), arr.max(axis=1) - arr.min(axis=1))
    mean = np.interp(t, m["t"].to_numpy(), arr.mean(axis=1))
    stuck = (motion < max_rate) & (spread > min_spread) & (mean > 0.08) & f.is_armed(t)
    return spans(stuck, t, min_len=min_len, max_gap=0.5)


@check("takeoff")
def takeoff(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """Craft that cannot lift one corner: wrong motor order, direction or props."""
    if f.vehicle == "plane":
        return []
    found = stuck_spans(f)
    if not found:
        return []
    cols = f.motor_columns()
    m = f.signals["motors"]
    # when did the pilot first ask for lift?
    t_up = None
    if f.has("throttle"):
        th = f.signals["throttle"]
        up = th["t"][th["thr"] > 0.1]
        t_up = float(up.iloc[0]) if len(up) else None
    if t_up is None:
        up = m["t"][m[cols].mean(axis=1) > 0.15]
        t_up = float(up.iloc[0]) if len(up) else 0.0
    a, b = found[0]
    sel = (m["t"] >= a) & (m["t"] <= b)
    avg = m.loc[sel, cols].mean()
    hi, lo = avg.idxmax(), avg.idxmin()
    at_takeoff = a - t_up < 20.0
    ev = [f"from {fmt_t(a)} to {fmt_t(b)} motor {hi[1:]} averaged {avg[hi]:.0%} while "
          f"motor {lo[1:]} sat at {avg[lo]:.0%}", "the gyro barely moved, so the craft was "
          "resting on something rather than flying"]
    if len(found) > 1:
        ev.append(f"{len(found)} such episodes, {sum(y - x for x, y in found):.0f} s in total")
    if at_takeoff:
        return [Finding(
            check="takeoff", title="Could not take off level", severity="critical",
            summary=(f"{a - t_up:.0f} s after throttle-up the controller was driving motor {hi[1:]} "
                     f"to {avg[hi]:.0%} with motor {lo[1:]} at idle and the craft still would not "
                     "lift that side. On a fresh build or after a rebuild this is nearly always "
                     "motor order, motor direction or props on the wrong motors."),
            confidence=0.8, evidence=ev,
            advice=["Check motor order against the mixer diagram in the configurator "
                    "(motor tab, spin each motor individually)",
                    "Check each motor spins the right way (props-in or props-out as configured)",
                    "Check each prop is the right hand for its motor",
                    "Check the flight controller orientation (board alignment) matches how it is "
                    "mounted"],
            t0=a, t1=b,
            plot=Plot([Panel("motors", cols, "motor output", ylim=(-0.05, 1.1)),
                       Panel("gyro", ["x", "y", "z"], "deg/s", labels=["roll", "pitch", "yaw"])],
                      t0=max(0.0, t_up - 3), t1=b + 3),
            refs=[("Betaflight: motor direction and props",
                   "https://www.betaflight.com/docs/wiki/getting-started/setup-guide")],
            data={"motor_high": int(hi[1:]), "motor_low": int(lo[1:])},
        )]
    return [Finding(
        check="takeoff", title="Armed and stuck on the ground", severity="info",
        summary=(f"From {fmt_t(a)} the craft sat still while the motors fought it "
                 f"(motor {hi[1:]} at {avg[hi]:.0%}): usually the aftermath of a crash, "
                 "with the craft upside down or caught in something while still armed"),
        confidence=0.6, cause=False, evidence=ev, t0=a, t1=b,
        advice=["Disarm immediately after a crash; set up crash recovery / turtle mode on a switch"],
        plot=Plot([Panel("motors", cols, "motor output", ylim=(-0.05, 1.1))], t0=a - 5, t1=b + 2),
    )]


@check("thrust_headroom")
def thrust_headroom(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    cols = f.motor_columns()
    if not cols:
        return []
    m = flight_window(f, f.signals["motors"])
    if len(m) < 20:
        return []
    t = m["t"].to_numpy()
    arr = m[cols].to_numpy()
    mean = arr.mean(axis=1)
    out: List[Finding] = []
    if f.vehicle != "plane" and len(cols) >= 3:
        hover = float(np.median(mean))
        all_high = (arr.min(axis=1) > 0.9)
        sat = spans(all_high, t, min_len=0.5, max_gap=0.2)
        sat_time = sum(b - a for a, b in sat)
        ev = [f"median motor output in flight: {hover:.0%}"]
        sev = "ok"
        if hover > 0.75:
            sev = "critical" if hover > 0.85 else "warning"
            ev.append("a well set up multirotor hovers at 30-60% so it has room to correct")
        if sat:
            ev.append(f"all motors at full output for {sat_time:.1f} s in total, "
                      f"longest from {fmt_t(max(sat, key=lambda s: s[1] - s[0])[0])}")
            if sat_time > 2.0 or any(near(a, inc, 15) for a, _ in sat):
                sev = "critical" if any(near(a, inc, 15) for a, _ in sat) else max(sev, "warning", key=_rank)
        if sev != "ok":
            near_inc = [s for s in sat if near(s[0], inc, 15)]
            t0 = near_inc[0][0] if near_inc else (sat[0][0] if sat else None)
            out.append(Finding(
                check="thrust_headroom",
                title="Not enough thrust headroom",
                severity=sev,
                summary=(f"The motors were near their limit (median {hover:.0%}"
                         + (f", all at full power for {sat_time:.1f} s" if sat else "") + "): "
                         "the craft is too heavy, the battery too weak, or the props too small"),
                confidence=0.7, evidence=ev,
                advice=["Reduce take-off weight or move to a higher-KV / bigger prop setup",
                        "Check the battery can deliver the current: a sagging pack looks like "
                        "lost thrust late in the flight",
                        "On ArduPilot, set MOT_THST_HOVER sensibly and check MOT_SPIN_MAX"],
                t0=t0, t1=None,
                plot=Plot([Panel("motors", cols, "motor output", ylim=(-0.05, 1.1))]),
                refs=[AP_MOTOR_DOC], data={"hover": hover, "saturated_s": sat_time}))
    return out


def _rank(sev: str) -> int:
    return {"ok": 0, "info": 1, "warning": 2, "critical": 3}[sev]


@check("motor_balance")
def motor_balance(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """Steady-state differences between motors in calm flight."""
    cols = f.motor_columns()
    if len(cols) < 4 or f.vehicle == "plane":
        return []
    m = flight_window(f, f.signals["motors"])
    if len(m) < 50:
        return []
    t = m["t"].to_numpy()
    arr = m[cols].to_numpy()
    calm = np.ones(len(m), dtype=bool)
    if f.has("rate", "roll", "pitch"):
        r = f.signals["rate"]
        rt = r["t"].to_numpy()
        rr = np.interp(t, rt, np.abs(r["roll"].to_numpy())) + np.interp(t, rt, np.abs(r["pitch"].to_numpy()))
        calm &= rr < 30
    calm &= (arr.max(axis=1) < 0.9) & (arr.min(axis=1) > 0.05)
    if inc is not None:
        calm &= t < inc.t - 3
    for a, b in stuck_spans(f):  # sitting on the ground fighting is not calm flight
        calm &= (t < a) | (t > b)
    if calm.sum() < 50:
        return []
    avg = arr[calm].mean(axis=0)
    overall = avg.mean()
    if overall <= 0.05:
        return []
    dev = (avg - overall) / overall
    hi_i, lo_i = int(np.argmax(dev)), int(np.argmin(dev))
    spread = float(avg[hi_i] - avg[lo_i])
    if spread < 0.08:
        return []
    sev = "warning" if spread >= 0.12 else "info"
    ev = [", ".join(f"motor {c[1:]} {a:.0%}" for c, a in zip(cols, avg)) + " (average in calm flight)",
          f"motor {cols[hi_i][1:]} works {dev[hi_i]:+.0%} harder than average, "
          f"motor {cols[lo_i][1:]} {dev[lo_i]:+.0%}"]
    adv = ["Check the centre of gravity: the hardest working motor sits under the heavy side",
           f"Compare motor {cols[hi_i][1:]} with the others: a weak motor, bent shaft or damaged "
           "prop makes the controller push it harder",
           "If diagonal pairs differ (both CW motors high), the motors are twisted on the arms "
           "and the copter is fighting a yaw imbalance"]
    return [Finding(
        check="motor_balance",
        title="Motors working unevenly",
        severity=sev,
        summary=(f"Motor {cols[hi_i][1:]} runs {spread:.0%} of full output higher than "
                 f"motor {cols[lo_i][1:]} in calm flight"),
        confidence=0.55, evidence=ev, advice=adv,
        plot=Plot([Panel("motors", cols, "motor output", ylim=(-0.05, 1.1))]),
        refs=[AP_THRUST_LOSS],
        data={"spread": spread},
    )]


@check("esc_desync")
def esc_desync(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """Motor commanded hard but its RPM telemetry collapses: ESC desync."""
    if not f.has("erpm") or not f.motor_columns():
        return []
    e = f.signals["erpm"]
    m = f.signals["motors"]
    ecols = [c for c in e.columns if c.startswith("m") and c in m.columns]
    if len(ecols) < 2:
        return []
    t = e["t"].to_numpy()
    flying = f.flying(t)
    if flying.sum() < 20:
        return []
    cmd = np.column_stack([np.interp(t, m["t"].to_numpy(), m[c].to_numpy()) for c in ecols])
    rpm = e[ecols].to_numpy().astype(float)
    ref = np.median(rpm[flying], axis=0)
    ref[ref <= 0] = np.nan
    if np.all(np.isnan(ref)):
        return []  # no usable RPM telemetry (SITL logs zeros)
    rel = rpm / ref
    with np.errstate(all="ignore"):
        others = np.nanmedian(rel, axis=1) if np.isfinite(rel).any() else np.zeros(len(rel))
    hits = []
    for i, c in enumerate(ecols):
        bad = flying & (cmd[:, i] > 0.4) & (rel[:, i] < 0.25 * others) & (others > 0.5)
        for a, b in spans(bad, t, min_len=0.05, max_gap=0.05):
            hits.append((a, b, c))
    if not hits:
        return []
    a, b, c = min(hits, key=lambda h: (0 if near(h[0], inc, 20) else 1, h[0]))
    return [Finding(
        check="esc_desync",
        title=f"Motor {c[1:]} RPM dropped out (ESC desync?)",
        severity="critical" if near(a, inc, 20) else "warning",
        summary=(f"Motor {c[1:]} was commanded hard but its RPM telemetry collapsed at {fmt_t(a)}: "
                 "typical of an ESC desync, a seized motor or a lost prop"),
        confidence=0.75,
        evidence=[f"{len(hits)} dropout(s); first relevant at {fmt_t(a)} lasting {b - a:.2f} s"],
        advice=["Raise the ESC's demag compensation / lower timing (BLHeli_S/32) or update the "
                "ESC firmware", "Check the motor for debris or a damaged bell",
                "Avoid very aggressive throttle punches on low-KV setups"],
        t0=a, t1=b,
        plot=Plot([Panel("erpm", ecols, "motor speed"), Panel("motors", ecols, "command", ylim=(-0.05, 1.1))],
                  t0=a - 3, t1=b + 3),
    )]
