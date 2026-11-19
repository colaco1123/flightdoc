"""Battery, flight controller supply, and logs that stop mid-air."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..model import Flight, spans
from .base import (Finding, Incident, Panel, Plot, check, flight_window, fmt_t,
                   lowpass, sample_rate)

AP_BATT_DOC = ("ArduPilot: battery failsafe",
               "https://ardupilot.org/copter/docs/failsafe-battery.html")
AP_POWER_DOC = ("ArduPilot: powering the autopilot",
                "https://ardupilot.org/copter/docs/common-powermodule-landingpage.html")
BF_BATT_DOC = ("Betaflight: battery monitoring",
               "https://www.betaflight.com/docs/wiki/guides/current/Battery")

CELL_FULL = 4.2


def resting_voltage(volts: np.ndarray) -> float:
    """Best guess at the unloaded pack voltage: the highest reading early on.

    Taking the very first sample is wrong for planes and quads that log from
    the moment they punch out at full throttle.
    """
    v = np.asarray(volts, dtype=float)
    v = v[np.isfinite(v) & (v > 2.5)]
    if v.size == 0:
        return 0.0
    head = v[: max(10, v.size // 5)]
    return float(np.percentile(head, 99))


def estimate_cells(volts: np.ndarray) -> int:
    """Cell count from the resting voltage at the start of the flight."""
    start = resting_voltage(volts)
    if start <= 0:
        return 0
    # A pack can start anywhere from storage (3.8 V) to LiHV full (4.35 V)
    best, best_err = 1, 9.0
    for n in range(1, 15):
        per = start / n
        err = 0.0 if 3.5 <= per <= 4.4 else min(abs(per - 3.5), abs(per - 4.4))
        if err < best_err - 1e-9:
            best, best_err = n, err
    return best


@check("battery")
def battery(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    if not f.has("battery", "volt"):
        return []
    b_all = f.signals["battery"]
    b = flight_window(f, b_all)
    if len(b) < 10:
        return []
    t = b["t"].to_numpy()
    fs = sample_rate(t)
    v = b["volt"].to_numpy()
    v_s = lowpass(v, fs, 1.0) if fs > 4 else v
    cells = estimate_cells(b_all["volt"].to_numpy())
    if cells == 0:
        return []
    per = v_s / cells
    start = resting_voltage(b_all["volt"].to_numpy()) / cells
    vmin = float(per.min())
    i_min = int(np.argmin(per))
    low = spans(per < 3.3, t, min_len=2.0, max_gap=0.5)
    very_low = spans(per < 3.0, t, min_len=0.5, max_gap=0.5)
    ev = [f"{cells}S pack (estimated), {start:.2f} V/cell at take-off, "
          f"lowest {vmin:.2f} V/cell at {fmt_t(t[i_min])}"]
    fs_events = [e for e in f.events if "batt" in e.text.lower() and e.level != "info"]
    for e in fs_events[:2]:
        ev.append(f"{fmt_t(e.t)} flight controller: {e.text}")

    r_cell = None
    if "curr" in b and np.nanmax(b["curr"].to_numpy()) > 1.0:
        cur = b["curr"].to_numpy()
        ok = np.isfinite(cur) & np.isfinite(v)
        if ok.sum() > 50 and np.ptp(cur[ok]) > 2.0:
            # Sag against current over short windows gives the pack resistance
            # without the slow discharge curve getting in the way.
            seg = max(10, int(fs * 5)) if fs > 0 else 50
            slopes = []
            for k in range(0, ok.sum() - seg, seg):
                cc, vv = cur[ok][k:k + seg], v[ok][k:k + seg]
                if np.ptp(cc) > 3.0:
                    slopes.append(np.polyfit(cc, vv, 1)[0])
            if slopes:
                r_pack = -float(np.median(slopes))
                if 0 < r_pack < 1.0:
                    r_cell = r_pack / cells
                    ev.append(f"pack resistance about {r_pack * 1000:.0f} mOhm "
                              f"({r_cell * 1000:.1f} mOhm per cell) from voltage sag vs current")
            ev.append(f"peak current {np.nanmax(cur):.0f} A, average {np.nanmean(cur):.1f} A")

    sev = "ok"
    title = "Battery voltage healthy"
    summary = f"Battery held up: lowest {vmin:.2f} V/cell under load"
    t0 = None
    if very_low or vmin < 3.0:
        sev = "critical"
        t0 = very_low[0][0] if very_low else float(t[i_min])
        title = "Battery voltage collapsed"
        summary = (f"Cell voltage fell to {vmin:.2f} V under load: the pack was empty, "
                   "damaged or could not supply the current")
    elif low:
        sev = "warning"
        t0 = low[0][0]
        title = "Battery ran low"
        summary = f"Cells spent {sum(b2 - a for a, b2 in low):.0f} s below 3.3 V under load"
    if start < 3.8 and sev == "ok":
        sev = "info"
        title = "Took off on a partly discharged battery"
        summary = f"Resting voltage at take-off was only {start:.2f} V/cell"
    if r_cell is not None and r_cell > 0.012 and sev == "ok":
        sev = "info"
        title = "Battery sags a lot under load"
        summary = f"About {r_cell * 1000:.0f} mOhm per cell: an ageing or undersized pack"
    if fs_events and sev in ("ok", "info"):
        sev = "warning"
        title = "Battery failsafe triggered"
        summary = f"The flight controller raised a battery warning at {fmt_t(fs_events[0].t)}"
        t0 = fs_events[0].t
    if inc and sev in ("warning", "critical") and t0 is not None:
        # a battery that dies right at the incident is a prime suspect
        at_inc = per[np.searchsorted(t, min(inc.t, t[-1])) - 1] if t[0] < inc.t else per[0]
        ev.append(f"{at_inc:.2f} V/cell at the moment of the incident")
        if at_inc < 3.3:
            sev = "critical"
    advice = []
    if sev != "ok":
        advice = ["Land at 3.5 V/cell resting (about 3.3 V/cell under load) and set the "
                  "low-voltage warning and failsafe accordingly",
                  "Measure each cell's internal resistance; retire packs that are puffy or "
                  "well above their siblings",
                  "Check the pack's C rating against the peak current in this log"]
    return [Finding(
        check="battery", title=title, severity=sev, summary=summary,
        confidence=0.75, evidence=ev, advice=advice, t0=t0,
        plot=Plot([Panel("battery", ["volt"], "volts", scale=1.0,
                         hlines=[(3.3 * cells, "3.3 V/cell"), (3.0 * cells, "3.0 V/cell")])]
                  + ([Panel("battery", ["curr"], "amps")] if "curr" in b else [])),
        refs=[AP_BATT_DOC] if f.firmware == "ardupilot" else [BF_BATT_DOC],
        data={"cells": cells, "min_cell_v": vmin, "start_cell_v": start,
              "r_cell_mohm": (r_cell or 0) * 1000},
    )]


@check("board_power")
def board_power(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """Flight controller supply rail (ArduPilot POWR.Vcc, PX4 5 V rail)."""
    if not f.has("power", "vcc"):
        return []
    p = f.signals["power"]
    pa = p[f.is_armed(p["t"].to_numpy())] if f.armed else p
    if len(pa) < 5:
        return []
    vcc = pa["vcc"].to_numpy()
    t = pa["t"].to_numpy()
    if np.nanmax(vcc) < 3.0:
        return []  # not measured (SITL reports 0 or a constant)
    rng = float(np.nanmax(vcc) - np.nanmin(vcc))
    vmin = float(np.nanmin(vcc))
    ev = [f"supply rail {np.nanmedian(vcc):.2f} V typical, {vmin:.2f} V lowest, "
          f"{rng:.2f} V of variation while armed"]
    sev = "ok"
    if vmin < 4.3:
        sev = "critical"
    elif vmin < 4.6 or rng > 0.3:
        sev = "warning"
    t0 = float(t[int(np.argmin(vcc))])
    if sev == "ok":
        return [Finding("board_power", "Flight controller power is stable", "ok",
                        f"Supply rail stayed between {vmin:.2f} and {np.nanmax(vcc):.2f} V",
                        evidence=ev, confidence=0.7,
                        plot=Plot([Panel("power", ["vcc"], "volts")]))]
    return [Finding(
        check="board_power",
        title="Flight controller supply dipped",
        severity=sev,
        summary=(f"The flight controller's supply fell to {vmin:.2f} V at {fmt_t(t0)}: "
                 "a brownout risk that can reboot the board or upset sensors"),
        confidence=0.7, evidence=ev,
        advice=["Power the flight controller from a dedicated BEC or power module, not "
                "shared with servos or a VTX",
                "Add a capacitor on the BEC output and check connectors and solder joints",
                "Make sure servos are not drawing from the same rail"],
        t0=t0,
        plot=Plot([Panel("power", ["vcc"], "volts", hlines=[(4.6, "low"), (4.3, "brownout")])]),
        refs=[AP_POWER_DOC],
    )]


@check("log_cut")
def log_cut(f: Flight, inc: Optional[Incident]) -> List[Finding]:
    """The log stopped while the motors were still spinning."""
    if not f.ended_armed:
        return []
    cols = f.motor_columns()
    if not cols:
        return []
    m = f.signals["motors"]
    tail = m[m["t"] > f.duration - 1.0]
    if tail.empty:
        return []
    out = float(tail[cols].mean().mean())
    if out < 0.15:
        return []
    if f.meta.get("storage_full") == "yes":
        return [Finding("log_cut", "Log storage filled up", "info",
                        "The log ends mid-flight because the onboard flash was full, "
                        "not because of a fault",
                        evidence=[f"file is exactly {f.meta.get('file_size', '?')} bytes, "
                                  "the size of the flash chip"],
                        advice=["Erase the flash before flying, or lower the logging rate"],
                        confidence=0.6, cause=False)]
    ev = [f"last record at {fmt_t(f.duration)}, motors averaging {out:.0%}",
          "no disarm and no end-of-log marker"]
    v_hint = ""
    if f.has("battery", "volt"):
        b = f.signals["battery"]
        last = b[b["t"] > f.duration - 3.0]
        if len(last) > 2:
            drop = float(last["volt"].iloc[0] - last["volt"].iloc[-1])
            ev.append(f"battery voltage in the last 3 s: {last['volt'].iloc[0]:.2f} V to "
                      f"{last['volt'].iloc[-1]:.2f} V")
            if drop > 0.5:
                v_hint = " The battery voltage was falling fast right before the cut."
    if f.has("power", "vcc"):
        p = f.signals["power"]
        last = p[p["t"] > f.duration - 3.0]
        if len(last) > 2 and last["vcc"].max() > 3:
            ev.append(f"FC supply in the last 3 s: {last['vcc'].min():.2f}-{last['vcc'].max():.2f} V")
    return [Finding(
        check="log_cut",
        title="Power was lost in flight",
        severity="critical",
        summary=("The log stops mid-air with the motors running: the flight controller lost "
                 "power or rebooted (battery disconnect, brownout, broken lead) or the "
                 "craft hit something hard enough to cut power." + v_hint),
        confidence=0.7, evidence=ev,
        advice=["Check the battery connector, lead and solder joints for heat damage",
                "Check that the battery strap held and the XT60/XT30 did not pull out",
                "Check the flight controller's BEC and supply wiring for a short",
                "If the craft was flying low, this can also be the moment of impact"],
        t0=f.duration - 1.0, t1=f.duration,
        plot=Plot([Panel("motors", cols, "motor output", ylim=(-0.05, 1.1))]
                  + ([Panel("battery", ["volt"], "volts")] if f.has("battery", "volt") else []),
                  t0=f.duration - 15, t1=f.duration + 0.5),
    )]
