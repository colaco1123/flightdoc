"""Shared types and signal helpers for the checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sps

from ..model import Flight, sample_rate, spans, window

SEVERITY_WEIGHT = {"critical": 1.0, "warning": 0.55, "info": 0.2, "ok": 0.0}


@dataclass
class Panel:
    """One subplot: a signal and the columns to draw from it."""
    signal: str
    columns: Sequence[str]
    ylabel: str = ""
    labels: Optional[Sequence[str]] = None
    hlines: Sequence[Tuple[float, str]] = ()
    ylim: Optional[Tuple[float, float]] = None
    scale: float = 1.0
    kind: str = "line"          # "line" or "spectrum"


@dataclass
class Plot:
    panels: List[Panel]
    t0: Optional[float] = None
    t1: Optional[float] = None
    title: str = ""


@dataclass
class Finding:
    check: str
    title: str
    severity: str                       # critical / warning / info / ok
    summary: str
    confidence: float = 0.7
    evidence: List[str] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)
    t0: Optional[float] = None
    t1: Optional[float] = None
    cause: bool = True                  # False for symptoms such as "lost control"
    plot: Optional[Plot] = None
    refs: List[Tuple[str, str]] = field(default_factory=list)
    score: float = 0.0
    data: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check, "title": self.title, "severity": self.severity,
            "summary": self.summary, "confidence": round(self.confidence, 2),
            "score": round(self.score, 3), "cause": self.cause,
            "time": None if self.t0 is None else [round(self.t0, 2),
                                                   round(self.t1 if self.t1 is not None else self.t0, 2)],
            "evidence": self.evidence, "advice": self.advice,
            "data": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.data.items()},
            "references": [{"title": a, "url": b} for a, b in self.refs],
        }


@dataclass
class Incident:
    """The moment things went wrong, if the log shows one."""
    t: float
    kind: str           # crash, loss_of_control, log_cut, failsafe
    description: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.7


CheckFn = Callable[[Flight, Optional[Incident]], List[Finding]]
REGISTRY: List[Tuple[str, CheckFn]] = []


def check(name: str):
    def wrap(fn: CheckFn) -> CheckFn:
        REGISTRY.append((name, fn))
        return fn
    return wrap


# ---------------------------------------------------------------- helpers
def flight_window(f: Flight, df, pad: float = 0.0):
    """Rows of ``df`` recorded while flying (armed with motors spinning)."""
    t = df["t"].to_numpy()
    mask = f.flying(t)
    if not mask.any():
        mask = f.is_armed(t)
    return df[mask]


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    if fs <= 0 or cutoff >= fs / 2 or len(x) < 3 * (order + 1) * 3:
        return np.asarray(x, dtype=float)
    b, a = sps.butter(order, cutoff / (fs / 2))
    return sps.filtfilt(b, a, np.asarray(x, dtype=float))


def highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    return np.asarray(x, dtype=float) - lowpass(x, fs, cutoff, order)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> np.ndarray:
    hi = min(hi, fs / 2 * 0.9)
    if fs <= 0 or lo >= hi or len(x) < 30:
        return np.zeros_like(np.asarray(x, dtype=float))
    b, a = sps.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, np.asarray(x, dtype=float))


def rolling_rms(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    x2 = np.asarray(x, dtype=float) ** 2
    kernel = np.ones(n) / n
    return np.sqrt(np.convolve(x2, kernel, mode="same"))


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, dtype=float)
    n = max(1, min(int(n), len(x)))
    if n == 1:
        return x
    pad = n // 2
    xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return sliding_window_view(xp, n).max(axis=1)


def psd(x: np.ndarray, fs: float, seg_seconds: float = 0.5):
    nper = int(min(len(x), max(64, fs * seg_seconds)))
    return sps.welch(np.asarray(x, dtype=float) - np.mean(x), fs=fs, nperseg=nper)


def dominant_peak(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float):
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        return None, 0.0
    i = np.argmax(power[sel])
    total = power[sel].sum()
    return float(freqs[sel][i]), float(power[sel][i] / total) if total > 0 else 0.0


def near(t: Optional[float], incident: Optional[Incident], before: float = 15.0,
         after: float = 2.0) -> bool:
    if t is None or incident is None:
        return False
    return incident.t - before <= t <= incident.t + after


def fmt_t(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


__all__ = [
    "Finding", "Incident", "Panel", "Plot", "check", "REGISTRY", "SEVERITY_WEIGHT",
    "flight_window", "lowpass", "highpass", "bandpass", "rolling_rms", "rolling_max",
    "psd", "dominant_peak", "near", "fmt_t", "sample_rate", "spans", "window",
]
