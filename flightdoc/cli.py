"""Command line entry point: ``flightdoc LOG [LOG ...]``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .analysis import analyze, text_report
from .formats import UnsupportedLog


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="flightdoc",
        description="Diagnose drone crashes from ArduPilot, PX4, Betaflight and INAV logs.")
    ap.add_argument("logs", nargs="+", help=".bin, .ulg, .bbl, .bfl, .txt or blackbox_decode .csv")
    ap.add_argument("-o", "--html", metavar="FILE",
                    help="write an HTML report (with one log, FILE; with several, a directory)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--session", type=int, metavar="N",
                    help="only analyse log N in files that hold several (1-based)")
    ap.add_argument("--version", action="version", version=f"flightdoc {__version__}")
    args = ap.parse_args(argv)

    results = []
    status = 0
    for path in args.logs:
        try:
            diags = analyze(path)
        except (UnsupportedLog, FileNotFoundError, ValueError) as exc:
            print(f"flightdoc: {exc}", file=sys.stderr)
            status = 2
            continue
        if args.session:
            diags = [d for d in diags if d.flight.session == args.session - 1]
        if not diags:
            print(f"flightdoc: {path}: no flights found", file=sys.stderr)
            status = max(status, 1)
            continue
        results.append((path, diags))
        if args.html:
            from .report import render
            if len(args.logs) == 1:
                out = args.html
            else:
                os.makedirs(args.html, exist_ok=True)
                out = os.path.join(args.html, os.path.basename(path) + ".html")
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(render(diags))
            print(f"report written to {out}", file=sys.stderr)

    if args.json:
        payload = [d.to_dict() for _, diags in results for d in diags]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))
    else:
        first = True
        for _, diags in results:
            for d in diags:
                if not first:
                    print("\n" + "-" * 72 + "\n")
                print(text_report(d))
                first = False
    return status


if __name__ == "__main__":
    sys.exit(main())
