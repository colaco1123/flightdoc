"""Matplotlib figures for the report. Everything returns PNG bytes."""

from __future__ import annotations

import io
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..checks.base import Finding, Incident, Plot, psd  # noqa: E402
from ..model import Flight, sample_rate, window  # noqa: E402

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2",
           "#db2777", "#4b5563"]
INK = "#1f2937"
MUTED = "#9ca3af"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#d1d5db", "axes.labelcolor": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280", "axes.grid": True,
    "grid.color": "#eef0f3", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 110,
})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _decimate(df, max_points: int = 6000):
    step = max(1, len(df) // max_points)
    return df.iloc[::step]


def _mark(ax, f: Flight, inc: Optional[Incident], t0: float, t1: float):
    if inc is not None and t0 <= inc.t <= t1:
        ax.axvline(inc.t, color="#dc2626", lw=1.2, ls="--", alpha=0.8)


def finding_plot(f: Flight, fd: Finding, inc: Optional[Incident]) -> Optional[bytes]:
    p: Optional[Plot] = fd.plot
    if p is None:
        return None
    panels = [pn for pn in p.panels if f.has(pn.signal)]
    if not panels:
        return None
    t0 = p.t0 if p.t0 is not None else 0.0
    t1 = p.t1 if p.t1 is not None else f.duration
    t0, t1 = max(t0, -1.0), min(t1, f.duration + 1.0)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.2, 1.9 * len(panels) + 0.4),
                             sharex=panels[0].kind != "spectrum", squeeze=False)
    for ax, pn in zip(axes[:, 0], panels):
        df = f.signals[pn.signal]
        cols = [c for c in pn.columns if c in df.columns]
        if pn.kind == "spectrum":
            sel = df[f.flying(df["t"].to_numpy())]
            if len(sel) < 256:
                sel = df
            fs = sample_rate(sel["t"].to_numpy())
            for k, c in enumerate(cols):
                fr, pw = psd(sel[c].to_numpy(), fs)
                ax.semilogy(fr, pw, color=PALETTE[k % len(PALETTE)], lw=1,
                            label=(pn.labels[k] if pn.labels else c))
            ax.set_xlabel("frequency (Hz)")
            ax.set_ylabel("power")
            ax.legend(loc="upper right", ncol=len(cols))
            continue
        w = _decimate(window(df, t0, t1))
        for k, c in enumerate(cols):
            if pn.labels and k < len(pn.labels):
                label = pn.labels[k]
            elif pn.signal in ("motors", "erpm"):
                label = f"motor {c[1:]}"
            else:
                label = c
            style = "--" if c.startswith("des_") else "-"
            ax.plot(w["t"], w[c] * pn.scale, style, lw=1.0, color=PALETTE[k % len(PALETTE)],
                    label=label)
        for y, label in pn.hlines:
            ax.axhline(y, color=MUTED, lw=0.9, ls=":")
            ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=7, color="#6b7280")
        if fd.t0 is not None:
            end = fd.t1 if fd.t1 is not None and fd.t1 > fd.t0 else fd.t0 + 0.05 * (t1 - t0)
            ax.axvspan(fd.t0, end, color="#fde68a", alpha=0.35, lw=0)
        _mark(ax, f, inc, t0, t1)
        if pn.ylim:
            ax.set_ylim(*pn.ylim)
        ax.set_ylabel(pn.ylabel)
        ax.set_xlim(t0, t1)
        if len(cols) > 1:
            ax.legend(loc="upper left", ncol=min(len(cols), 6))
    if panels[0].kind != "spectrum":
        axes[-1, 0].set_xlabel("time (s)")
    if p.title:
        axes[0, 0].set_title(p.title, loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    return _png(fig)


def overview_plot(f: Flight, inc: Optional[Incident]) -> bytes:
    """Altitude/throttle, motors and attitude for the whole flight, with events."""
    rows = []
    if f.has("alt"):
        rows.append(("alt", ["alt"], "altitude (m)"))
    elif f.has("throttle"):
        rows.append(("throttle", ["thr"], "throttle"))
    if f.motor_columns():
        rows.append(("motors", f.motor_columns(), "motors"))
    if f.has("att", "roll"):
        rows.append(("att", ["roll", "pitch"], "attitude (deg)"))
    elif f.has("rate", "roll"):
        rows.append(("rate", ["roll", "pitch", "yaw"], "rate (deg/s)"))
    if f.has("battery", "volt"):
        rows.append(("battery", ["volt"], "battery (V)"))
    if not rows:
        fig, ax = plt.subplots(figsize=(8.2, 1.2))
        ax.text(0.5, 0.5, "no plottable signals", ha="center", va="center")
        ax.axis("off")
        return _png(fig)
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.2, 1.35 * len(rows) + 0.5),
                             sharex=True, squeeze=False)
    for ax, (sig, cols, label) in zip(axes[:, 0], rows):
        df = _decimate(f.signals[sig], 4000)
        for k, c in enumerate(cols):
            ax.plot(df["t"], df[c], lw=0.8, color=PALETTE[k % len(PALETTE)],
                    label=(f"motor {c[1:]}" if sig == "motors" else c))
        for a, b in f.armed:
            ax.axvspan(a, b, color="#f3f4f6", lw=0, zorder=-1)
        for e in f.events:
            if e.kind == "mode":
                ax.axvline(e.t, color="#a5b4fc", lw=0.6)
            elif e.level == "error":
                ax.axvline(e.t, color="#fca5a5", lw=0.6)
        _mark(ax, f, inc, -1, f.duration + 1)
        ax.set_ylabel(label)
        if len(cols) > 1 and len(cols) <= 8:
            ax.legend(loc="upper left", ncol=len(cols))
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 0].set_xlim(0, max(f.duration, 1.0))
    fig.tight_layout()
    return _png(fig)
