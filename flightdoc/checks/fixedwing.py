"""Fixed wing launches and take-offs that end before the aircraft is flying."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..model import Flight
from .base import Finding, Incident, Panel, Plot, check, fmt_t, sample_rate

INAV_LAUNCH_DOC = ("INAV: fixed wing launch mode",
                   "https://github.com/iNavFlight/inav/blob/master/docs/Navigation.md")
AP_TAKEOFF_DOC = ("ArduPilot: automatic takeoff (Plane)",
                  "https://ardupilot.org/plane/docs/automatic-takeoff.html")


def _at(df, col, t):
    tt = df["t"].to_numpy()
    return float(np.interp(t, tt, df[col].to_numpy()))


@check("launch")
def launch(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """A fixed wing that hits something before it ever climbs away.

    Looks at the seconds before the impact and reports what the pilot and the
    aircraft were doing: how long the throttle took to come up, whether it
    rolled off or pitched up hard, and what the sticks asked for.
    """
    if f.vehicle != "plane" or inc is None or inc.kind not in ("crash", "loss_of_control"):
        return []
    if not f.has("alt", "alt") or not f.has("throttle", "thr"):
        return []
    start = f.first_armed()
    alt = f.signals["alt"]
    before = alt[(alt["t"] >= start) & (alt["t"] <= inc.t)]
    if before.empty:
        return []
    # Propwash over the barometer throws spikes of 10-20 m on a ground roll,
    # so judge the climb on a half-second running median.
    n = max(1, int((sample_rate(before["t"].to_numpy()) or 10) * 0.5))
    smooth = before["alt"].rolling(n, center=True, min_periods=1).median()
    climb = float(smooth.max() - smooth.iloc[: max(1, len(smooth) // 20)].mean())
    if climb > 5.0:
        return []  # it flew; whatever happened is not a launch problem
    thr = f.signals["throttle"]
    th = thr[(thr["t"] >= start) & (thr["t"] <= inc.t)]
    if th.empty or th["thr"].max() < 0.2:
        return []

    t_up = float(th["t"][th["thr"] > 0.1].iloc[0]) if (th["thr"] > 0.1).any() else start
    full = th["t"][th["thr"] > 0.8]
    t_full = float(full.iloc[0]) if len(full) else None
    pre = th[th["t"] < (t_full if t_full is not None else inc.t) - 0.5]
    partial = float(pre["thr"].quantile(0.9)) if len(pre) else 0.0

    factors, ev = [], []
    ev.append(f"never climbed more than {max(climb, 0):.1f} m before the impact at {fmt_t(inc.t)}")
    ev.append(f"throttle came up at {fmt_t(t_up)} and was held around {partial:.0%} "
              f"for {((t_full or inc.t) - t_up):.1f} s")

    if t_full is None:
        factors.append("never reached full power")
        ev.append(f"throttle peaked at {th['thr'].max():.0%}: the aircraft never had full thrust")
    elif inc.t - t_full < 1.5:
        factors.append("full power came too late")
        ev.append(f"full throttle only {inc.t - t_full:.1f} s before the impact")

    rolled = pitched = None
    if f.has("att", "roll", "pitch"):
        a = f.signals["att"]
        w = a[(a["t"] >= inc.t - 2.0) & (a["t"] <= inc.t)]  # after the hit is just tumbling
        if len(w):
            i = int(np.argmax(np.abs(w["roll"].to_numpy())))
            rolled = float(w["roll"].iloc[i])
            pitched = float(w["pitch"].max())
            if abs(rolled) > 45:
                side = "left" if rolled < 0 else "right"  # positive roll is right wing down
                factors.append(f"rolled off to the {side}")
                ev.append(f"banked {abs(rolled):.0f} deg {side} at {fmt_t(float(w['t'].iloc[i]))}, "
                          "within 2 s of the impact")
            if pitched > 35:
                factors.append("nose-high")
                ev.append(f"nose up to {pitched:.0f} deg just before the impact")
        after = a[a["t"] > inc.t + 0.3]
        if len(after) and np.abs(after["roll"].iloc[-1]) > 120:
            ev.append("came to rest inverted")

    if f.has("sticks", "pitch"):
        s = f.signals["sticks"]
        w = s[(s["t"] >= inc.t - 2.0) & (s["t"] <= inc.t)]
        if len(w) and np.abs(w["pitch"]).max() > 0.6:
            factors.append("large pitch input at low speed")
            ev.append(f"pitch stick at {np.abs(w['pitch']).max():.0%} of full travel in the last 2 s")

    if f.has("battery", "volt"):
        b = f.signals["battery"]
        w = b[(b["t"] >= inc.t - 2.0) & (b["t"] <= inc.t + 0.5)]
        if len(w) > 2:
            text = f"battery {w['volt'].max():.2f} V to {w['volt'].min():.2f} V at full power"
            if "curr" in w:
                text += f", {w['curr'].max():.0f} A peak"
            ev.append(text)

    launch_mode = any(e.kind == "mode" and "NAVLAUNCH" in e.text for e in f.events) or \
        any(e.kind == "mode" and "TAKEOFF" in e.text for e in f.events)

    title = "Launch failed"
    if factors:
        title += ": " + ", ".join(factors[:2])
    advice = []
    if any("power" in x for x in factors):
        advice.append("Bring the throttle to full before releasing the aircraft or starting the "
                      "ground roll: a wing needs its full thrust while it is slowest")
    if any("rolled" in x for x in factors):
        advice.append("A roll-off right after power comes on is usually a tip stall (too slow, "
                      "too nose-high) or torque: on a twin, make sure the props counter-rotate; "
                      "on a single motor, add a little opposite aileron or right thrust")
    if "nose-high" in factors or any("pitch input" in x for x in factors):
        advice.append("Launch level or a few degrees nose-up and let it build speed before "
                      "pulling: hauling back at low speed stalls the wing")
    advice.append("Check the CG against the design's recommended range; a tail-heavy wing "
                  "pitches up and stalls on launch")
    if f.firmware == "inav" and not launch_mode:
        advice.append("Use INAV's NAV LAUNCH mode: it holds the motor until the throw is "
                      "detected, then climbs out at a set angle and throttle")
    elif f.firmware == "ardupilot" and not launch_mode:
        advice.append("Use ArduPilot's TAKEOFF mode or an auto-launch so the climb-out is flown "
                      "for you")

    cols = [c for c in ("thr",) if c in thr]
    panels = [Panel("throttle", cols, "throttle", ylim=(-0.05, 1.05))]
    if f.has("att", "roll"):
        panels.append(Panel("att", ["roll", "pitch"], "attitude (deg)"))
    panels.append(Panel("alt", ["alt"], "altitude (m)"))
    return [Finding(
        check="launch", title=title, severity="critical",
        summary=("The aircraft hit the ground before it ever climbed away"
                 + (": " + "; ".join(factors) + "." if factors else ".")),
        confidence=0.75 if factors else 0.55, evidence=ev, advice=advice,
        t0=t_up, t1=inc.t,
        plot=Plot(panels, t0=max(0.0, t_up - 1), t1=inc.t + 1.5),
        refs=[INAV_LAUNCH_DOC] if f.firmware == "inav" else [AP_TAKEOFF_DOC],
        data={"climb_m": climb, "partial_throttle": partial,
              "full_throttle_lead_s": (inc.t - t_full) if t_full is not None else -1.0},
    )]
