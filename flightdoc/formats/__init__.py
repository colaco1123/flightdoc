"""Log loaders. ``load(path)`` picks the right one from the file contents."""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile
from typing import List

from ..model import Flight

ULOG_MAGIC = b"ULog\x01\x12\x35"


class UnsupportedLog(ValueError):
    pass


def _head(path: str, n: int = 64 * 1024) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(n)


def sniff(path: str) -> str:
    head = _head(path)
    if head.startswith(ULOG_MAGIC):
        return "px4"
    if b"H Product:Blackbox flight data recorder" in head:
        return "blackbox"
    if head[:2] == b"\xa3\x95" or b"\xa3\x95\x80" in head[:4096]:
        return "ardupilot"
    if os.path.splitext(path)[1].lower() == ".csv":
        return "blackbox_csv"
    raise UnsupportedLog(f"{os.path.basename(path)}: not an ArduPilot .bin, PX4 .ulg "
                         "or Betaflight/INAV blackbox log")


def load(path: str) -> List[Flight]:
    """Load every flight session in a log file (gzip-compressed files work too)."""
    if _head(path, 2) == b"\x1f\x8b":
        name = os.path.basename(path)
        name = name[:-3] if name.lower().endswith(".gz") else name
        with tempfile.TemporaryDirectory() as tmp:
            plain = os.path.join(tmp, name)
            with gzip.open(path, "rb") as src, open(plain, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return load(plain)

    kind = sniff(path)
    if kind == "px4":
        from . import px4
        return px4.load(path)
    if kind == "ardupilot":
        from . import ardupilot
        return ardupilot.load(path)
    if kind == "blackbox":
        from . import blackbox
        return blackbox.load(path)
    from . import bbcsv
    return bbcsv.load(path)
