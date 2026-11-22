"""Assemble the static site: page, worker, wheels and sample logs.

    python web/build_site.py site

Used by the Pages workflow; handy locally too.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

SAMPLES = [
    "tests/data/motor_failure.bin.gz",
    "tests/data/compass_interference.bin.gz",
    "tests/data/px4_gps_loss.ulg.gz",
    "tests/data/blackbox/crashing-LOG00002.BFL.gz",
]


def main(out):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    for name in ("index.html", "worker.js", "fc.js"):
        shutil.copy(os.path.join(root, "web", name), out)
    for s in SAMPLES:
        shutil.copy(os.path.join(root, s), os.path.join(out, "samples"))
    wheels = os.path.join(out, "wheels")
    os.makedirs(wheels, exist_ok=True)
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(pip + ["wheel", "--no-deps", "-q", "-w", wheels, root], check=True)
    subprocess.run(pip + ["download", "--no-deps", "-q", "--only-binary=:all:", "-d", wheels, "pyulog"],
                   check=True)
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(wheels, "*.whl")))
    with open(os.path.join(out, "wheel.json"), "w") as fh:
        json.dump({"wheels": [f"wheels/{w}" for w in found]}, fh)
    print("site built:", out, found)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "site")
