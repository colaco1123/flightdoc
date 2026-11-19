"""Estimator health, GPS and compass."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from ..model import Flight, spans
from .base import (Finding, Incident, Panel, Plot, check, flight_window, fmt_t,
                   near)

AP_EKF_DOC = ("ArduPilot: EKF failsafe",
              "https://ardupilot.org/copter/docs/ekf-inav-failsafe.html")
AP_COMPASS_DOC = ("ArduPilot: compass/motor calibration (compassmot)",
                  "https://ardupilot.org/copter/docs/common-compass-setup-advanced.html")
AP_GPS_DOC = ("ArduPilot: GPS glitch protection (archived page)",
              "https://ardupilot.org/copter/docs/gps-failsafe-glitch-protection.html")
PX4_EKF_DOC = ("PX4: EKF2 log analysis",
               "https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html")

EKF_PARTS = {"vel": "velocity (usually GPS)", "pos": "horizontal position (usually GPS)",
             "hgt": "height (barometer, or vibration upsetting it)",
             "mag": "heading (compass)"}


@check("ekf")
def ekf(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    if not f.has("ekf"):
        return []
    e = flight_window(f, f.signals["ekf"])
    if len(e) < 10:
        return []
    t = e["t"].to_numpy()
    parts = [p for p in EKF_PARTS if p in e]
    worst, worst_val = None, 0.0
    for p in parts:
        val = float(np.nanpercentile(e[p].to_numpy(), 99.5))
        if val > worst_val:
            worst, worst_val = p, val
    if worst is None:
        return []
    bad = spans(e[worst].to_numpy() > 1.0, t, min_len=0.5, max_gap=1.0)
    warn = spans(e[worst].to_numpy() > 0.8, t, min_len=1.0, max_gap=1.0)
    ekf_events = [x for x in f.events if ("ekf" in x.text.lower() or "glitch" in x.text.lower()
                                          or "lane switch" in x.text.lower())
                  and x.level != "info"]
    lane = [x for x in f.events if "lane switch" in x.text.lower() or "primary changed" in x.text.lower()]
    if bad:
        sev = "critical" if any(near(a, inc, 20) for a, _ in bad) or len(bad) > 2 else "warning"
    elif warn or ekf_events:
        sev = "warning"
    else:
        sev = "ok"
    ev = [f"worst innovation test ratio: {worst} at {worst_val:.2f} "
          "(1.0 means the estimator is rejecting that sensor)"]
    for p in parts:
        if p != worst:
            ev.append(f"{p}: {np.nanpercentile(e[p].to_numpy(), 99.5):.2f}")
    if bad:
        ev.append(f"{worst} above 1.0 for {sum(b - a for a, b in bad):.1f} s, first at {fmt_t(bad[0][0])}")
    if lane:
        ev.append(f"{len(lane)} EKF lane/primary switches")
    for x in ekf_events[:3]:
        ev.append(f"{fmt_t(x.t)} {x.text}")
    if sev == "ok":
        return [Finding("ekf", "Position estimate was consistent", "ok",
                        f"All estimator test ratios stayed below 0.8 (worst {worst} {worst_val:.2f})",
                        evidence=ev, confidence=0.7,
                        plot=Plot([Panel("ekf", parts, "test ratio", hlines=[(1.0, "reject")])]))]
    t0 = bad[0][0] if bad else (warn[0][0] if warn else (ekf_events[0].t if ekf_events else None))
    return [Finding(
        check="ekf",
        title=f"Estimator rejected {worst} data",
        severity=sev,
        summary=(f"The EKF stopped trusting its {EKF_PARTS[worst]} measurements "
                 f"(test ratio {worst_val:.1f}). Position-holding modes become unreliable when "
                 "this happens."),
        confidence=0.75, evidence=ev,
        advice={"mag": ["Calibrate the compass away from metal, then run compass/motor "
                        "interference calibration", "Move the compass away from power wires and ESCs"],
                "hgt": ["Shield the barometer from prop wash and light with open-cell foam",
                        "Fix Z-axis vibration, which corrupts the height estimate"],
                "vel": ["Improve GPS reception: mount the antenna high, away from the VTX and "
                        "other noisy electronics", "Wait for more satellites before take-off"],
                "pos": ["Improve GPS reception: mount the antenna high, away from the VTX and "
                        "other noisy electronics", "Wait for more satellites before take-off"]}[worst],
        t0=t0, t1=bad[-1][1] if bad else None,
        plot=Plot([Panel("ekf", parts, "test ratio", hlines=[(1.0, "reject"), (0.8, "warn")])]),
        refs=[AP_EKF_DOC] if f.firmware == "ardupilot" else [PX4_EKF_DOC],
        data={"worst_ratio": worst_val, "worst_part": worst},
    )]


@check("gps")
def gps(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    if not f.has("gps", "nsats"):
        return []
    g = flight_window(f, f.signals["gps"])
    if len(g) < 5:
        return []
    t = g["t"].to_numpy()
    ns = g["nsats"].to_numpy()
    ev = [f"satellites in flight: {int(np.nanmin(ns))} to {int(np.nanmax(ns))} "
          f"(median {np.nanmedian(ns):.0f})"]
    problems = []
    if "hdop" in g and np.nanmax(g["hdop"].to_numpy()) > 0:
        hd = g["hdop"].to_numpy()
        ev.append(f"HDOP median {np.nanmedian(hd):.2f}, worst {np.nanmax(hd):.2f}")
        if np.nanmedian(hd) > 2.0:
            problems.append("poor geometry (HDOP above 2)")
    if np.nanmedian(ns) < 6:
        problems.append("few satellites")
    lost = []
    if "fix" in g:
        fix = g["fix"].to_numpy()
        need = 3 if f.firmware in ("ardupilot", "px4") else 2
        lost = spans(fix < need, t, min_len=0.5)
        if lost:
            problems.append(f"3D fix lost {len(lost)} time(s)")
            ev.append(f"fix lost at {', '.join(fmt_t(a) for a, _ in lost[:3])}")
    jumps = []
    if "lat" in g and "lon" in g:
        lat = np.radians(g["lat"].to_numpy())
        lon = np.radians(g["lon"].to_numpy())
        ok = (np.abs(lat) > 1e-6) | (np.abs(lon) > 1e-6)
        if ok.sum() > 5:
            lat, lon, tt = lat[ok], lon[ok], t[ok]
            d = 6371000 * np.hypot(np.diff(lat), np.diff(lon) * np.cos(lat[:-1]))
            dt = np.maximum(np.diff(tt), 1e-3)
            speed = d / dt
            spd = g["spd"].to_numpy()[ok][1:] if "spd" in g else np.zeros_like(speed)
            jump = (speed > 3 * spd + 15) & (d > 10)
            jumps = list(tt[1:][jump])
            if jumps:
                problems.append(f"position jumped {len(jumps)} time(s)")
                ev.append(f"largest jump {d[jump].max():.0f} m at {fmt_t(tt[1:][jump][np.argmax(d[jump])])}")
    # A receiver that dies outright just stops publishing
    gaps = []
    ft = t[f.flying(t)]
    if ft.size > 2:
        d_t = np.diff(ft)
        for k in np.flatnonzero(d_t > 2.0):
            a, b = float(ft[k]), float(ft[k + 1])
            # only a dropout if we were flying the whole time, not landed between flights
            probe = np.linspace(a, b, 9)[1:-1]
            if f.flying(probe).all():
                gaps.append((a, b))
    allt = f.signals["gps"]["t"].to_numpy()
    flying_end = f.armed[-1][1] if f.armed else f.duration
    if allt.size and flying_end - allt[-1] > 3.0 and f.flying(np.array([allt[-1]]))[0]:
        gaps.append((float(allt[-1]), float(flying_end)))
    if gaps:
        problems.append("GPS data stopped in flight")
        ev.append(f"no GPS data from {fmt_t(gaps[0][0])} for {gaps[0][1] - gaps[0][0]:.0f} s")
    lost_ev = [x for x in f.events if re.search(r"gnss lost|gps.*(lost|fail)|position estimate lost",
                                                x.text, re.I)]
    for x in lost_ev[:2]:
        ev.append(f"{fmt_t(x.t)} {x.text}")
    glitch = [x for x in f.events if "glitch" in x.text.lower()]
    if glitch:
        problems.append("flight controller reported a GPS glitch")
        ev.append(f"{fmt_t(glitch[0].t)} {glitch[0].text}")
    if not problems:
        return [Finding("gps", "GPS was healthy", "ok",
                        f"{np.nanmedian(ns):.0f} satellites with no glitches", evidence=ev,
                        confidence=0.7, plot=Plot([Panel("gps", ["nsats"], "satellites")]))]
    t0 = min([x for x in [jumps[0] if jumps else None, lost[0][0] if lost else None,
                          glitch[0].t if glitch else None, gaps[0][0] if gaps else None]
               if x is not None], default=None)
    sev = ("critical" if (t0 is not None and near(t0, inc, 30)) or len(jumps) > 2 or gaps
           else "warning")
    cols = ["nsats"] + (["hdop"] if "hdop" in g else [])
    return [Finding(
        check="gps", title="GPS problems", severity=sev,
        summary="GPS: " + ", ".join(problems),
        confidence=0.7 if (jumps or glitch or gaps) else 0.5, evidence=ev,
        advice=["Mount the GPS on a mast away from the VTX, camera and power wiring",
                "Wait for a solid fix (10+ satellites, HDOP under 1.5) before take-off",
                "If the craft lurched in a GPS mode, switch to a non-GPS mode (ALT_HOLD, "
                "ANGLE) as soon as it misbehaves"],
        t0=t0, t1=max(b for _, b in gaps) if gaps else None,
        plot=Plot([Panel("gps", cols, "satellites / HDOP")]),
        refs=[AP_GPS_DOC] if f.firmware == "ardupilot" else [],
        data={"median_sats": float(np.nanmedian(ns))},
    )]


@check("compass")
def compass(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """Magnetic field that moves with throttle or current: motor interference."""
    if not f.has("mag", "x", "y", "z"):
        return []
    m = flight_window(f, f.signals["mag"])
    if len(m) < 50:
        return []
    t = m["t"].to_numpy()
    field = np.sqrt(m["x"] ** 2 + m["y"] ** 2 + m["z"] ** 2).to_numpy()
    if np.nanmedian(field) <= 0:
        return []
    load_name, load = None, None
    if f.has("battery", "curr") and np.nanmax(f.signals["battery"]["curr"]) > 1.0:
        b = f.signals["battery"]
        load = np.interp(t, b["t"].to_numpy(), b["curr"].to_numpy())
        load_name = "current"
    elif f.has("throttle"):
        th = f.signals["throttle"]
        load = np.interp(t, th["t"].to_numpy(), th["thr"].to_numpy())
        load_name = "throttle"
    if load is None or np.ptp(load) <= 0:
        return []
    # Like ArduPilot's compassmot: fit each body axis against load. Turning
    # the craft rotates the earth field through the axes too, but that is not
    # correlated with current, so it averages out of the slope.
    xyz = m[["x", "y", "z"]].to_numpy()
    A = np.column_stack([load, np.ones_like(load)])
    coef, *_ = np.linalg.lstsq(A, xyz, rcond=None)
    slope_vec = coef[0]
    span_load = float(np.nanpercentile(load, 95) - np.nanpercentile(load, 5))
    base = float(np.nanmedian(field))
    pct = float(np.linalg.norm(slope_vec) * span_load / base * 100) if base > 0 else 0.0
    fit = A @ coef
    resid = xyz - fit
    explained = 1.0 - resid.var(axis=0).sum() / max(xyz.var(axis=0).sum(), 1e-9)
    corr = float(np.sqrt(max(explained, 0.0)))
    ev = [f"magnetometer reading shifts by {pct:.0f}% of the earth field across the "
          f"{load_name} range ({load_name} explains {explained:.0%} of the variation)"]
    switches = [x for x in f.events if "switching to compass" in x.text.lower()]
    yaw_resets = [x for x in f.events if "yaw reset" in x.text.lower()]
    if switches:
        ev.append(f"autopilot switched between compasses {len(switches)} times")
    if yaw_resets:
        ev.append(f"{len(yaw_resets)} EKF yaw resets")
    if pct > 60 and abs(corr) > 0.5:
        sev = "critical"
    elif (pct > 30 and abs(corr) > 0.4) or len(switches) > 4:
        sev = "warning"
    else:
        return [Finding("compass", "Compass not affected by motors", "ok",
                        f"Field strength varies {pct:.0f}% with {load_name}", evidence=ev,
                        confidence=0.6)]
    return [Finding(
        check="compass",
        title="Compass interference from motors/power wiring",
        severity=sev,
        summary=(f"The magnetic field reading moves {pct:.0f}% with {load_name}: the compass is "
                 "picking up current in nearby wires. This causes toilet-bowling and "
                 "fly-aways in GPS modes."),
        confidence=0.75 if abs(corr) > 0.6 else 0.55, evidence=ev,
        advice=["Move the compass (usually in the GPS puck) further from the battery leads, "
                "ESCs and power distribution", "Twist battery and motor power wires together",
                "Run compass/motor calibration (COMPASS_MOT on ArduPilot) and keep "
                "interference under 30%"],
        plot=Plot([Panel("mag", ["x", "y", "z"], "magnetic field")]
                  + ([Panel("battery", ["curr"], "amps")] if load_name == "current"
                     else [Panel("throttle", ["thr"], "throttle")])),
        refs=[AP_COMPASS_DOC],
        data={"interference_pct": pct, "corr": corr},
    )]
