"""Run every check on a flight and rank the results."""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

from .checks import REGISTRY
from .checks.base import SEVERITY_WEIGHT, Finding, Incident, fmt_t
from .incident import find_incident
from .model import Flight


def _lc(title: str) -> str:
    """Lower-case the first word unless it is an acronym (GPS, EKF, RC)."""
    first = title.split(" ", 1)[0]
    return title if first.isupper() else title[0].lower() + title[1:]


@dataclass
class Diagnosis:
    flight: Flight
    incident: Optional[Incident]
    causes: List[Finding] = field(default_factory=list)      # ranked, most likely first
    symptoms: List[Finding] = field(default_factory=list)
    passed: List[Finding] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)          # checks that crashed

    @property
    def verdict(self) -> str:
        top = self.top
        if not self.flight.armed:
            return "The vehicle was never armed in this log, so there is no flight to diagnose."
        if self.incident is None:
            if top is None:
                return "No incident and no problems found in this log."
            return f"No crash in this log, but: {_lc(top.title)}."
        if top is None:
            return (f"{self.incident.description[0].upper()}{self.incident.description[1:]}; "
                    "no clear cause in the logged data.")
        return f"Most likely cause: {_lc(top.title)}."

    @property
    def top(self) -> Optional[Finding]:
        real = [c for c in self.causes if c.severity in ("critical", "warning")]
        return real[0] if real else None

    def to_dict(self) -> dict:
        f = self.flight
        return {
            "file": f.source,
            "session": f.session,
            "firmware": f.firmware,
            "firmware_version": f.firmware_version,
            "vehicle": f.vehicle,
            "duration_s": round(f.duration, 2),
            "verdict": self.verdict,
            "incident": None if self.incident is None else {
                "time_s": round(self.incident.t, 2),
                "kind": self.incident.kind,
                "description": self.incident.description,
                "evidence": self.incident.evidence,
                "confidence": round(self.incident.confidence, 2),
            },
            "causes": [c.to_dict() for c in self.causes],
            "symptoms": [s.to_dict() for s in self.symptoms],
            "passed": [p.to_dict() for p in self.passed],
            "notes": f.notes,
            "check_errors": self.errors,
        }


def _relevance(fd: Finding, inc: Optional[Incident]) -> float:
    if inc is None:
        return 1.0
    if fd.t0 is None:
        return 0.7  # whole-flight condition (vibration, weak battery): still relevant
    start, end = fd.t0, fd.t1 if fd.t1 is not None else fd.t0
    if start <= inc.t + 2 and end >= inc.t - 20:
        # the closer it starts before the incident, the more it explains it
        lead = inc.t - start
        return 1.0 if -2 <= lead <= 20 else 0.85
    if end < inc.t - 20:
        return 0.45  # happened earlier and the craft kept flying
    return 0.25      # after the incident: a consequence, not a cause


def rank(findings: List[Finding], inc: Optional[Incident]) -> List[Finding]:
    for fd in findings:
        fd.score = SEVERITY_WEIGHT[fd.severity] * fd.confidence * _relevance(fd, inc)
    return sorted(findings, key=lambda x: -x.score)


def diagnose(flight: Flight, checks=None) -> Diagnosis:
    inc = find_incident(flight)
    found: List[Finding] = []
    errors = []
    for name, fn in REGISTRY:
        if checks and name not in checks:
            continue
        try:
            found.extend(fn(flight, inc))
        except Exception as exc:  # one broken check must not sink the report
            errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
            if os.environ.get("FLIGHTDOC_DEBUG"):
                traceback.print_exc()
    passed = [x for x in found if x.severity == "ok"]
    issues = [x for x in found if x.severity != "ok"]
    _link_root_causes(issues)
    causes = rank([x for x in issues if x.cause], inc)
    symptoms = rank([x for x in issues if not x.cause], inc)
    return Diagnosis(flight, inc, causes, symptoms, passed, errors)


# EKF innovation trouble is usually downstream of a sensor problem that has
# its own check. When both fire, credit the sensor.
EKF_ROOTS = {"mag": "compass", "pos": "gps", "vel": "gps", "hgt": "vibration"}


def _link_root_causes(issues: List[Finding]) -> None:
    by_check = {x.check: x for x in issues}
    ekf = by_check.get("ekf")
    if ekf is None:
        return
    root = by_check.get(EKF_ROOTS.get(str(ekf.data.get("worst_part")), ""))
    if root is None or root.severity not in ("warning", "critical"):
        return
    ekf.confidence *= 0.6
    ekf.evidence.append(f"consistent with: {root.title.lower()}")
    root.confidence = min(0.95, root.confidence + 0.1)
    root.evidence.append(f"the estimator rejected {ekf.data.get('worst_part')} data as a result")
    if ekf.severity == "critical" and root.severity == "warning":
        root.severity = "critical"


def analyze(path: str, checks=None) -> List[Diagnosis]:
    from .formats import load
    return [diagnose(f, checks) for f in load(path)]


def text_report(d: Diagnosis) -> str:
    f = d.flight
    lines = [f"{f.source}" + (f" (log {f.session + 1})" if f.session else "")
             + f"  {f.firmware_version or f.firmware}  {f.vehicle}  {fmt_t(f.duration)}",
             "", d.verdict, ""]
    if d.incident:
        lines.append(f"Incident at {fmt_t(d.incident.t)}: {d.incident.description}")
        lines += [f"  - {e}" for e in d.incident.evidence]
        lines.append("")
    if d.causes:
        lines.append("Likely causes, most likely first:")
        for i, c in enumerate(d.causes, 1):
            when = f" @{fmt_t(c.t0)}" if c.t0 is not None else ""
            lines.append(f"{i:2d}. [{c.severity}] {c.title}{when}  (score {c.score:.2f})")
            lines.append(f"      {c.summary}")
            for e in c.evidence[:4]:
                lines.append(f"      - {e}")
        lines.append("")
    if d.symptoms:
        lines.append("What happened:")
        for s in d.symptoms:
            when = f" @{fmt_t(s.t0)}" if s.t0 is not None else ""
            lines.append(f"  - {s.title}{when}: {s.summary}")
        lines.append("")
    if d.passed:
        lines.append("Looked fine: " + ", ".join(_lc(p.title) for p in d.passed))
    for n in f.notes:
        lines.append(f"note: {n}")
    for e in d.errors:
        lines.append(f"check failed: {e}")
    return "\n".join(lines)
