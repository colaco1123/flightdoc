"""Attitude control: oscillation and loss of control."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..model import Flight, spans
from .base import (Finding, Incident, Panel, Plot, bandpass, check, fmt_t,
                   flight_window, near, psd, rolling_rms, sample_rate)

AP_TUNE_DOC = ("ArduPilot: tuning process instructions",
               "https://ardupilot.org/copter/docs/tuning-process-instructions.html")
PX4_TUNE_DOC = ("PX4: multicopter PID tuning",
                "https://docs.px4.io/main/en/config_mc/pid_tuning_guide_multicopter.html")
BF_TUNE_DOC = ("Betaflight: PID tuning guide",
               "https://www.betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide")


def _tune_ref(f: Flight):
    return {"ardupilot": AP_TUNE_DOC, "px4": PX4_TUNE_DOC}.get(f.firmware, BF_TUNE_DOC)


@check("oscillation")
def oscillation(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """Sustained oscillation that the pilot did not command.

    We band-pass the rate tracking error between 4 and 40 Hz and look for
    stretches where it is both large and concentrated at one frequency. A
    pilot's stick input is broadband and tracked; a P/D gain that is too high
    rings at a single frequency and shows up as error.
    """
    if not f.has("rate", "roll", "pitch"):
        return []
    r = flight_window(f, f.signals["rate"])
    if len(r) < 100:
        return []
    t = r["t"].to_numpy()
    fs = sample_rate(t)
    if fs < 40:
        return []
    hi = min(40.0, fs / 2 * 0.8)
    # Acro quads track hard manoeuvres with tens of deg/s of error as a
    # matter of course, so their bar is higher than for a camera platform.
    acro = f.firmware in ("betaflight", "inav")
    best = None
    for ax in ("roll", "pitch", "yaw"):
        if ax not in r:
            continue
        x = r[ax].to_numpy()
        err = x - r[f"des_{ax}"].to_numpy() if f"des_{ax}" in r else x
        band = bandpass(err, fs, 4.0, hi)
        level = rolling_rms(band, fs * 0.5)
        limit = (60.0 if acro else 25.0) * (1.4 if ax == "yaw" else 1.0)
        min_share = 0.3 if acro else 0.12
        for a, b in spans(level > limit, t, min_len=0.6, max_gap=0.3):
            if inc is not None and a > inc.t - 0.5:
                continue  # the crash itself rings every axis; not a cause
            sel = (t >= a) & (t <= b)
            fr, pw = psd(band[sel], fs, seg_seconds=min(0.5, (b - a) / 2))
            in_band = (fr >= 4) & (fr <= hi)
            if not in_band.any() or pw[in_band].sum() <= 0:
                continue
            k = np.argmax(pw[in_band])
            peak_f = float(fr[in_band][k])
            share = float(pw[in_band][k] / pw[in_band].sum())
            amp = float(level[sel].max())
            if share < min_share:
                continue  # broadband: turbulence or hard flying, not a tune problem
            score = amp * (b - a) * (3 if near(a, inc, 15) else 1)
            if best is None or score > best[0]:
                best = (score, ax, a, b, peak_f, amp, share)
    if best is None:
        return []
    _, ax, a, b, peak_f, amp, share = best
    sev = "critical" if near(a, inc, 15) or amp > 120 else "warning"
    low_freq = peak_f < 10
    advice = (["Low frequency oscillation (under 10 Hz) usually means P or I is too high, "
               "or the frame/arms flex", "Reduce P on that axis by 20-30% and test again"]
              if low_freq else
              ["Higher frequency oscillation points at D too high or noise reaching D: "
               "lower D and check filtering", "Check for a loose flight controller stack, "
               "arm or motor screws"])
    if f.firmware == "ardupilot":
        advice.append("AutoTune on a calm day gives a sane starting point")
    return [Finding(
        check="oscillation",
        title=f"{ax.capitalize()} oscillation at {peak_f:.0f} Hz",
        severity=sev,
        summary=(f"The {ax} axis oscillated at about {peak_f:.0f} Hz for {b - a:.1f} s "
                 f"(up to {amp:.0f} deg/s of tracking error) without pilot input to match"),
        confidence=0.6 if share > 0.25 else 0.45,
        evidence=[f"oscillation from {fmt_t(a)} to {fmt_t(b)}",
                  f"{share:.0%} of the error energy sits at {peak_f:.0f} Hz"],
        advice=advice, t0=a, t1=b,
        plot=Plot([Panel("rate", [ax, f"des_{ax}"] if f"des_{ax}" in r else [ax],
                         "deg/s", labels=["actual", "target"])], t0=a - 2, t1=b + 2),
        refs=[_tune_ref(f)],
        data={"freq_hz": peak_f, "amp_dps": amp},
    )]


@check("attitude_control")
def attitude_control(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """Large, sustained gap between the attitude asked for and the attitude flown."""
    if f.vehicle == "plane" and inc is None:
        return []  # manual and acro flying on a wing breaks this measure
    if f.has("att", "roll", "des_roll"):
        a = flight_window(f, f.signals["att"])
        if len(a) < 20:
            return []
        t = a["t"].to_numpy()
        err = np.maximum(np.abs(a["roll"] - a["des_roll"]).to_numpy(),
                         np.abs(a["pitch"] - a["des_pitch"]).to_numpy())
        limit, unit, sig, cols = 30.0, "deg", "att", ["roll", "des_roll", "pitch", "des_pitch"]
    elif f.has("rate", "roll", "des_roll"):
        a = flight_window(f, f.signals["rate"])
        if len(a) < 100:
            return []
        t = a["t"].to_numpy()
        err = np.maximum(np.abs(a["roll"] - a["des_roll"]).to_numpy(),
                         np.abs(a["pitch"] - a["des_pitch"]).to_numpy())
        limit, unit, sig, cols = 250.0, "deg/s", "rate", ["roll", "des_roll", "pitch", "des_pitch"]
    else:
        return []
    bad = spans(err > limit, t, min_len=0.5 if unit == "deg" else 0.25, max_gap=0.2)
    if not bad:
        return []
    near_inc = [s for s in bad if near(s[0], inc, 15, 2)]
    a0, a1 = (near_inc or bad)[0]
    peak = float(err[(t >= a0) & (t <= a1)].max())
    sev = "critical" if near_inc else "warning"
    return [Finding(
        check="attitude_control",
        title="Lost attitude control",
        severity=sev,
        summary=(f"The craft stopped following its attitude target at {fmt_t(a0)} "
                 f"(error up to {peak:.0f} {unit}). This is what a crash looks like from "
                 "the inside; the other findings explain why it happened."),
        confidence=0.85, cause=False,
        evidence=[f"{len(bad)} episode(s) with error above {limit:.0f} {unit}",
                  f"worst from {fmt_t(a0)} to {fmt_t(a1)}"],
        t0=a0, t1=a1,
        plot=Plot([Panel(sig, cols, unit, labels=["roll", "roll target", "pitch", "pitch target"])],
                  t0=a0 - 6, t1=a1 + 3),
    )]
