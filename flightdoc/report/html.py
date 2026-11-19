"""Single-file HTML report. Images are embedded, so it can be emailed or attached to a forum post."""

from __future__ import annotations

import base64
import datetime as dt
import html
from typing import List

from .. import __version__
from ..analysis import Diagnosis
from ..checks.base import Finding, fmt_t
from .plots import finding_plot, overview_plot

CSS = """
:root{--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--bg:#f9fafb;--crit:#b91c1c;--warn:#b45309;
--info:#1d4ed8;--ok:#15803d}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg)}
main{max-width:920px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:36px 0 12px}
.meta{color:var(--muted);font-size:13px}
.verdict{background:#fff;border:1px solid var(--line);border-left:5px solid var(--ink);border-radius:8px;
padding:16px 20px;margin:20px 0;font-size:17px}
.verdict.crit{border-left-color:var(--crit)}.verdict.warn{border-left-color:var(--warn)}
.verdict.ok{border-left-color:var(--ok)}
.verdict small{display:block;color:var(--muted);font-size:13px;margin-top:6px}
.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin:14px 0}
.card h3{margin:0 0 6px;font-size:16px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.tag{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;padding:2px 7px;
border-radius:4px;color:#fff}
.tag.critical{background:var(--crit)}.tag.warning{background:var(--warn)}.tag.info{background:var(--info)}
.tag.ok{background:var(--ok)}
.rank{color:var(--muted);font-weight:400;font-size:13px}
.card p{margin:6px 0}
.card ul{margin:6px 0 6px 18px;padding:0}
.card li{margin:2px 0}
.label{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;
margin-top:10px}
img{max-width:100%;display:block;margin:10px 0 4px;border-radius:4px}
.refs a{color:var(--info)}
.passed{color:var(--muted);font-size:14px}
table{border-collapse:collapse;font-size:13px;width:100%}
td{border-top:1px solid var(--line);padding:4px 8px;vertical-align:top}
td:first-child{color:var(--muted);white-space:nowrap;width:70px}
footer{margin-top:48px;color:var(--muted);font-size:12px}
"""


def _img(png: bytes, alt: str) -> str:
    return (f'<img alt="{html.escape(alt)}" src="data:image/png;base64,'
            f'{base64.b64encode(png).decode()}">')


def _list(items: List[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in items) + "</ul>"


def _card(d: Diagnosis, fd: Finding, rank: int = 0, plots: bool = True) -> str:
    out = ['<div class="card">']
    when = f' <span class="rank">at {fmt_t(fd.t0)}</span>' if fd.t0 is not None else ""
    num = f'<span class="rank">#{rank}</span>' if rank else ""
    out.append(f'<h3>{num}<span class="tag {fd.severity}">{fd.severity}</span>'
               f'{html.escape(fd.title)}{when}</h3>')
    out.append(f"<p>{html.escape(fd.summary)}</p>")
    if fd.evidence:
        out.append('<div class="label">Evidence</div>' + _list(fd.evidence))
    if plots:
        png = finding_plot(d.flight, fd, d.incident)
        if png:
            out.append(_img(png, fd.title))
    if fd.advice:
        out.append('<div class="label">What to check</div>' + _list(fd.advice))
    if fd.refs:
        links = ", ".join(f'<a href="{html.escape(u)}">{html.escape(t)}</a>' for t, u in fd.refs)
        out.append(f'<p class="refs"><small>Read more: {links}</small></p>')
    out.append("</div>")
    return "\n".join(out)


def _section(d: Diagnosis) -> str:
    f = d.flight
    top = d.top
    cls = "ok"
    if top is not None:
        cls = "crit" if top.severity == "critical" else "warn"
    elif d.incident is not None:
        cls = "warn"
    parts = [f"<h1>{html.escape(f.source)}"
             + (f" &middot; log {f.session + 1}" if f.session else "") + "</h1>",
             f'<div class="meta">{html.escape(f.firmware_version or f.firmware)} &middot; '
             f'{html.escape(f.vehicle or "unknown vehicle")} &middot; {fmt_t(f.duration)} logged'
             + (f" &middot; {html.escape(f.meta['frame'])}" if "frame" in f.meta else "")
             + "</div>"]
    sub = ""
    if d.incident is not None:
        sub = f"<small>{html.escape(d.incident.description[0].upper() + d.incident.description[1:])}</small>"
    parts.append(f'<div class="verdict {cls}">{html.escape(d.verdict)}{sub}</div>')
    parts.append(_img(overview_plot(f, d.incident), "flight overview"))
    if d.incident is not None:
        parts.append('<div class="card"><h3>Incident</h3>'
                     f"<p>{html.escape(d.incident.description)}</p>"
                     + _list(d.incident.evidence) + "</div>")
    if d.causes:
        parts.append("<h2>Likely causes</h2>")
        parts.append('<p class="meta">Ranked by severity, how sure the check is, and how close '
                     "in time it is to the incident.</p>")
        for i, c in enumerate(d.causes, 1):
            parts.append(_card(d, c, i))
    if d.symptoms:
        parts.append("<h2>What happened</h2>")
        for s in d.symptoms:
            parts.append(_card(d, s))
    if d.passed:
        parts.append("<h2>Looked fine</h2>")
        parts.append('<p class="passed">' + "; ".join(
            f"{html.escape(p.title)} ({html.escape(p.summary)})" for p in d.passed) + "</p>")
    timeline = [e for e in f.events if e.kind != "message" or e.level != "info"]
    if timeline:
        rows = "".join(f"<tr><td>{fmt_t(e.t)}</td><td>{html.escape(e.text)}</td></tr>"
                       for e in timeline[:200])
        parts.append("<h2>Event log</h2><table>" + rows + "</table>")
    if f.notes or d.errors:
        parts.append("<h2>Notes</h2>" + _list(f.notes + [f"check failed: {e}" for e in d.errors]))
    return "\n".join(parts)


def render(diagnoses: List[Diagnosis]) -> str:
    body = "\n<hr>\n".join(_section(d) for d in diagnoses) if diagnoses else \
        "<p>No flights found in this file.</p>"
    title = diagnoses[0].flight.source if diagnoses else "flightdoc"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (f"<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)} - flightdoc</title><style>{CSS}</style></head>"
            f"<body><main>{body}<footer>flightdoc {__version__}, generated {stamp}. "
            "Automated diagnosis is a starting point: confirm it against the craft "
            "before you fly again.</footer></main></body></html>")
