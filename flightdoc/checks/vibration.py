"""Vibration (accelerometer) and gyro noise."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..model import Flight
from .base import (Finding, Incident, Panel, Plot, check, dominant_peak,
                   flight_window, highpass, lowpass, near, psd, sample_rate,
                   spans)

AP_VIBE_DOC = ("ArduPilot: measuring vibration",
               "https://ardupilot.org/copter/docs/common-measuring-vibration.html")
AP_ISOLATION_DOC = ("ArduPilot: vibration damping",
                    "https://ardupilot.org/copter/docs/common-vibration-damping.html")
PX4_VIBE_DOC = ("PX4: vibration isolation",
                "https://docs.px4.io/main/en/assembly/vibration_isolation.html")
BF_FILTER_DOC = ("Betaflight: filtering and noise",
                 "https://www.betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide")

VIBE_WARN = 30.0   # m/s^2, ArduPilot's "probably a problem" level
VIBE_BAD = 60.0    # m/s^2, "almost certainly a problem"


def _ap_style_vibe(acc: np.ndarray, fs: float) -> np.ndarray:
    """Vibration metric in the spirit of ArduPilot's VIBE: the RMS of what is
    left after removing the slow (<5 Hz) part of the accelerometer signal."""
    hp = highpass(acc, fs, 5.0)
    return np.sqrt(lowpass(hp ** 2, fs, 2.0).clip(min=0))


@check("vibration")
def vibration(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    if f.has("vibe", "x", "y", "z"):
        v = flight_window(f, f.signals["vibe"])
        if len(v) < 5:
            return []
        t = v["t"].to_numpy()
        level = v[["x", "y", "z"]].max(axis=1).to_numpy()
        axis_peak = v[["x", "y", "z"]].max()
        clip = v["clip"].to_numpy() if "clip" in v else np.zeros_like(level)
        clips = float(clip.max() - clip.min()) if clip.size else 0.0
        source = "VIBE"
    elif f.has("accel", "x", "y", "z") and f.firmware in ("px4", "ardupilot"):
        a = flight_window(f, f.signals["accel"])
        fs = sample_rate(a["t"].to_numpy())
        if len(a) < 200 or fs < 100:
            return []
        t = a["t"].to_numpy()
        comps = {ax: _ap_style_vibe(a[ax].to_numpy(), fs) for ax in "xyz"}
        level = np.maximum.reduce([comps[ax] for ax in "xyz"])
        axis_peak = {ax: float(comps[ax].max()) for ax in "xyz"}
        # PX4 and ArduPilot IMUs clip around 16 g
        clips = float((np.abs(a[["x", "y", "z"]].to_numpy()) > 15.5 * 9.81).any(axis=1).sum())
        source = "accelerometer"
        # keep the derived metric so the report can plot it like VIBE
        f.add("vibe", t, x=comps["x"], y=comps["y"], z=comps["z"])
    else:
        return []

    if level.size == 0:
        return []
    worst_axis = max("xyz", key=lambda ax: float(axis_peak[ax]))
    p95 = float(np.percentile(level, 95))
    frac_warn = float((level > VIBE_WARN).mean())
    frac_bad = float((level > VIBE_BAD).mean())
    bad_spans = spans(level > VIBE_WARN, t, min_len=1.0, max_gap=1.0)

    # ArduPilot's guidance: a few clips (under ~100, e.g. hard landings) are
    # fine; a count that keeps climbing through the flight is not.
    if frac_bad > 0.05 or (clips > 500 and p95 > VIBE_WARN):
        sev = "critical"
    elif p95 > VIBE_WARN or frac_warn > 0.1 or clips > 100:
        sev = "warning"
    else:
        sev = "ok"

    t0 = bad_spans[0][0] if bad_spans else None
    t1 = bad_spans[-1][1] if bad_spans else None
    if inc and bad_spans:
        close = [s for s in bad_spans if near(s[0], inc, 30) or near(s[1], inc, 30)]
        if close:
            t0, t1 = close[0][0], close[-1][1]

    ev = [f"95th percentile vibration {p95:.0f} m/s/s (above {VIBE_WARN:.0f} is a problem, "
          f"above {VIBE_BAD:.0f} is severe)",
          f"worst axis: {worst_axis.upper()} at {float(axis_peak[worst_axis]):.0f} m/s/s peak",
          f"{frac_warn:.0%} of the flight above {VIBE_WARN:.0f} m/s/s"]
    if clips:
        ev.append(f"accelerometer clipped {clips:.0f} times in flight (sensor saturated)")
    advice = []
    if sev != "ok":
        advice = [
            "Check props for chips, cracks or imbalance and balance them",
            "Check motor bearings and bells by hand for roughness or play",
            "Soft-mount the flight controller and make sure no wires tug on it",
            "Tighten arms and frame screws; a loose arm resonates",
        ]
        if worst_axis == "z":
            advice.insert(0, "Z-axis vibration dominates: this upsets altitude hold "
                             "first, and a copter with bad Z vibration can climb away "
                             "or fall when the EKF loses its height estimate")
    finding = Finding(
        check="vibration",
        title="High vibration" if sev != "ok" else "Vibration within limits",
        severity=sev,
        summary=(f"Vibration from the {source} data is {'severe' if sev == 'critical' else 'high'} "
                 f"({p95:.0f} m/s/s typical peak, worst on {worst_axis.upper()})"
                 if sev != "ok" else f"Vibration stayed below {VIBE_WARN:.0f} m/s/s ({p95:.0f} typical)"),
        confidence=0.85 if source == "VIBE" else 0.65,
        evidence=ev, advice=advice, t0=t0, t1=t1,
        plot=Plot([Panel("vibe", ["x", "y", "z"], "vibration, m/s/s",
                         hlines=[(VIBE_WARN, "warn"), (VIBE_BAD, "bad")])]),
        refs=[AP_VIBE_DOC, AP_ISOLATION_DOC] if f.firmware == "ardupilot" else [PX4_VIBE_DOC, AP_VIBE_DOC],
        data={"vibe_p95": p95, "clips": clips, "frac_above_30": frac_warn},
    )
    return [finding]


@check("gyro_noise")
def gyro_noise(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """High frequency noise left in the (filtered) gyro signal.

    This is the main vibration check for Betaflight and INAV logs, where the
    accelerometer is logged after heavy filtering.
    """
    if not f.has("gyro", "x", "y", "z"):
        return []
    g = flight_window(f, f.signals["gyro"])
    if len(g) < 500:
        return []
    fs = sample_rate(g["t"].to_numpy())
    if fs < 300:
        return []  # too slow to see motor noise
    # Ignore the hardest manoeuvres: we want the noise floor, not the flips.
    noise = {}
    peaks = {}
    for ax, name in (("x", "roll"), ("y", "pitch")):
        x = g[ax].to_numpy()
        hf = highpass(x, fs, 50.0)
        noise[name] = float(np.sqrt(np.median(hf ** 2)))
        fr, pw = psd(x, fs)
        peaks[name] = dominant_peak(fr, pw, 50.0, fs / 2 * 0.95)
    worst = max(noise, key=noise.get)
    level = noise[worst]
    if level > 12:
        sev = "critical"
    elif level > 5:
        sev = "warning"
    else:
        sev = "ok"
    freq, share = peaks[worst]
    ev = [f"gyro noise above 50 Hz: roll {noise['roll']:.1f} deg/s RMS, "
          f"pitch {noise['pitch']:.1f} deg/s RMS (filtered gyro, as logged)"]
    if freq and share > 0.15:
        ev.append(f"strongest noise peak around {freq:.0f} Hz on {worst}")
    advice = []
    if sev != "ok":
        advice = ["Replace or balance props and check motors for bent shafts or bad bearings",
                  "Look for a loose stack, frame arm or camera mount near that frequency",
                  "On Betaflight, enable bidirectional DShot and the RPM filter if you have not"]
        if freq and share > 0.15:
            advice.append(f"A narrow peak near {freq:.0f} Hz is usually a frame resonance or "
                          "motor noise; a notch filter there helps once the source is fixed")
    return [Finding(
        check="gyro_noise",
        title="Noisy gyro" if sev != "ok" else "Gyro noise is low",
        severity=sev,
        summary=(f"{level:.1f} deg/s of high frequency noise reaches the PID loop on {worst}"
                 if sev != "ok" else f"Gyro noise is low ({level:.1f} deg/s RMS above 50 Hz)"),
        confidence=0.6,
        evidence=ev, advice=advice,
        plot=Plot([Panel("gyro", ["x", "y"], "deg/s", labels=["roll", "pitch"], kind="spectrum")],
                  title="Gyro spectrum in flight"),
        refs=[BF_FILTER_DOC] if f.firmware in ("betaflight", "inav") else [PX4_VIBE_DOC],
        data={"noise_dps": level, "peak_hz": freq or 0.0},
    )]
