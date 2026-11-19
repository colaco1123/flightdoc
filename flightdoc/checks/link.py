"""Radio link, failsafes, CPU load and what the flight controller itself said."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from ..model import Flight, spans
from .base import (Finding, Incident, Panel, Plot, check, fmt_t, flight_window,
                   near)

AP_RADIO_DOC = ("ArduPilot: radio failsafe",
                "https://ardupilot.org/copter/docs/radio-failsafe.html")
BF_FAILSAFE_DOC = ("Betaflight: failsafe",
                   "https://www.betaflight.com/docs/wiki/guides/current/Failsafe")
PX4_FAILSAFE_DOC = ("PX4: safety and failsafes",
                    "https://docs.px4.io/main/en/config/safety.html")


RADIO_TEXT = re.compile(r"\b(radio|rc|rx)\b|signal lost|throttle failsafe", re.I)


@check("radio")
def radio(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    events = [e for e in f.events if e.level != "info" and RADIO_TEXT.search(e.text)]
    ev = []
    t0 = None
    if events:
        t0 = events[0].t
        ev += [f"{fmt_t(e.t)} {e.text}" for e in events[:4]]
    if f.has("rc", "failsafe"):
        rc = flight_window(f, f.signals["rc"])
        lost = spans(rc["failsafe"].to_numpy() > 0.5, rc["t"].to_numpy(), min_len=0.1)
        if lost:
            t0 = min(t0, lost[0][0]) if t0 is not None else lost[0][0]
            ev.append(f"link reported lost {len(lost)} time(s), "
                      f"{sum(b - a for a, b in lost):.1f} s in total")
    low_rssi = None
    if f.has("rc", "rssi"):
        rc = flight_window(f, f.signals["rc"])
        r = rc["rssi"].to_numpy()
        if np.nanmax(r) > 0.05:
            low = spans(r < 0.2, rc["t"].to_numpy(), min_len=0.5)
            if low:
                low_rssi = low[0][0]
                ev.append(f"RSSI under 20% for {sum(b - a for a, b in low):.1f} s, "
                          f"first at {fmt_t(low[0][0])}")
                t0 = low_rssi if t0 is None else min(t0, low_rssi)
    if not ev:
        return []
    sev = "critical" if near(t0, inc, 20) else "warning"
    refs = {"ardupilot": [AP_RADIO_DOC], "px4": [PX4_FAILSAFE_DOC]}.get(f.firmware, [BF_FAILSAFE_DOC])
    plot = None
    if f.has("rc"):
        cols = [c for c in ("rssi", "failsafe") if c in f.signals["rc"]]
        plot = Plot([Panel("rc", cols, "link")])
    return [Finding(
        check="radio", title="Radio link lost", severity=sev,
        summary=f"The RC link dropped at {fmt_t(t0)} and a failsafe kicked in",
        confidence=0.8 if events else 0.6, evidence=ev,
        advice=["Check receiver antenna placement and damage; keep them away from carbon "
                "and the VTX antenna", "Do a range check before flying",
                "Check what the failsafe action is set to (RTL / land / drop) and that it "
                "suits where you fly"],
        t0=t0, plot=plot, refs=refs,
    )]


@check("cpu")
def cpu(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    if not f.has("cpu"):
        return []
    c = flight_window(f, f.signals["cpu"])
    if len(c) < 3:
        return []
    ev, sev = [], "ok"
    if "load" in c:
        load = c["load"].to_numpy()
        p95 = float(np.nanpercentile(load, 95))
        ev.append(f"CPU load 95th percentile {p95:.0%}")
        if p95 > 0.95:
            sev = "warning"
    if "slow" in c:
        slow = float(np.nanmax(c["slow"].to_numpy()))
        ev.append(f"worst share of overrun loops {slow:.1%}")
        if slow > 0.1:
            sev = "warning"
    if sev == "ok":
        return []
    return [Finding("cpu", "Flight controller overloaded", sev,
                    "The flight controller was running out of processing time, which delays "
                    "control loops and logging",
                    evidence=ev, confidence=0.5,
                    advice=["Lower the logging rate or disable unused features",
                            "Lower the loop rate on slower boards (F4 class)"])]


@check("fc_messages")
def fc_messages(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """Errors the flight controller logged near the incident, as a timeline."""
    errs = [e for e in f.events if e.level == "error" and e.kind != "failsafe"]
    if inc is not None:
        errs = [e for e in errs if inc.t - 30 <= e.t <= inc.t + 5]
    if not errs:
        return []
    # collapse repeats
    seen, lines = {}, []
    for e in errs:
        seen.setdefault(e.text, []).append(e.t)
    for text, ts in sorted(seen.items(), key=lambda kv: kv[1][0])[:10]:
        lines.append(f"{fmt_t(ts[0])} {text}" + (f" (x{len(ts)})" if len(ts) > 1 else ""))
    return [Finding("fc_messages", "Flight controller error messages", "info",
                    f"The flight controller logged {len(errs)} error(s)"
                    + (" around the incident" if inc else ""),
                    evidence=lines, confidence=0.9, cause=False, t0=errs[0].t)]


@check("mode_change")
def mode_change(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """A mode change just before things went wrong is worth knowing about."""
    if inc is None:
        return []
    modes = [e for e in f.events if e.kind == "mode" and inc.t - 5 <= e.t <= inc.t + 0.5]
    if not modes:
        return []
    m = modes[-1]
    prev = [e for e in f.events if e.kind == "mode" and e.t < m.t]
    frm = f" from {prev[-1].text}" if prev else ""
    return [Finding("mode_change", f"Mode changed to {m.text} right before the incident", "info",
                    f"Mode switched{frm} to {m.text} {inc.t - m.t:.1f} s before the incident. "
                    "If nobody flipped a switch, a failsafe did.",
                    evidence=[f"{fmt_t(e.t)} {e.text}" for e in modes],
                    confidence=0.6, cause=False, t0=m.t)]
