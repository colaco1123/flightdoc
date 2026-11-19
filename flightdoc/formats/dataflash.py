"""Minimal reader for ArduPilot DataFlash ``.bin`` logs.

The format is self-describing: ``FMT`` messages (id 128) declare each message
type's id, length, name, field types and field names. Every message starts
with the two sync bytes 0xA3 0x95 followed by its type id.

We scan the file once to find the offset of every message, then pull out each
wanted type in one go with numpy, which is far faster than decoding messages
one by one in Python.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

HEAD1, HEAD2 = 0xA3, 0x95
FMT_ID = 128

# DataFlash type character -> (numpy dtype, scale applied after reading)
TYPES = {
    "a": ("<i2", None, 32),    # int16[32]
    "b": ("i1", None, 1),
    "B": ("u1", None, 1),
    "h": ("<i2", None, 1),
    "H": ("<u2", None, 1),
    "i": ("<i4", None, 1),
    "I": ("<u4", None, 1),
    "f": ("<f4", None, 1),
    "d": ("<f8", None, 1),
    "n": ("S4", None, 1),
    "N": ("S16", None, 1),
    "Z": ("S64", None, 1),
    "c": ("<i2", 0.01, 1),
    "C": ("<u2", 0.01, 1),
    "e": ("<i4", 0.01, 1),
    "E": ("<u4", 0.01, 1),
    "L": ("<i4", 1e-7, 1),
    "M": ("u1", None, 1),
    "q": ("<i8", None, 1),
    "Q": ("<u8", None, 1),
    "g": ("<f2", None, 1),
}


class DataflashError(Exception):
    pass


class Format:
    def __init__(self, mid: int, length: int, name: str, fmt: str, columns: List[str]):
        self.id = mid
        self.length = length
        self.name = name
        self.fmt = fmt
        self.columns = columns
        fields = []
        self.scales = {}
        for ch, col in zip(fmt, columns):
            if ch not in TYPES:
                raise DataflashError(f"{name}: unknown field type {ch!r}")
            dt, scale, count = TYPES[ch]
            fields.append((col, dt, (count,)) if count > 1 else (col, dt))
            if scale is not None:
                self.scales[col] = scale
        dtype = np.dtype(fields)
        # Some firmware pads messages; trust the declared length.
        if dtype.itemsize != length - 3:
            pad = length - 3 - dtype.itemsize
            if pad < 0:
                raise DataflashError(f"{name}: declared length {length} too short for {fmt}")
            fields.append(("_pad", f"V{pad}"))
            dtype = np.dtype(fields)
        self.dtype = dtype


class DataflashLog:
    def __init__(self, data: bytes):
        self.data = data
        self.formats: Dict[int, Format] = {}
        self.by_name: Dict[str, Format] = {}
        self.offsets: Dict[int, List[int]] = {}
        self.bad_bytes = 0
        self._scan()

    @classmethod
    def open(cls, path: str) -> "DataflashLog":
        with open(path, "rb") as fh:
            return cls(fh.read())

    def _add_format(self, payload: bytes) -> None:
        mid, length = payload[0], payload[1]
        name = payload[2:6].rstrip(b"\0").decode("ascii", "replace")
        fmt = payload[6:22].rstrip(b"\0").decode("ascii", "replace")
        cols = payload[22:86].rstrip(b"\0").decode("ascii", "replace").split(",")
        try:
            f = Format(mid, length, name, fmt, cols)
        except (DataflashError, TypeError, ValueError):
            return
        self.formats[mid] = f
        self.by_name[name] = f

    def _scan(self) -> None:
        d = self.data
        n = len(d)
        pos = 0
        lengths = {FMT_ID: 89}
        offsets = self.offsets
        while pos + 3 <= n:
            if d[pos] != HEAD1 or d[pos + 1] != HEAD2:
                nxt = d.find(b"\xa3\x95", pos + 1)
                if nxt < 0:
                    self.bad_bytes += n - pos
                    break
                self.bad_bytes += nxt - pos
                pos = nxt
                continue
            mid = d[pos + 2]
            length = lengths.get(mid)
            if length is None or pos + length > n:
                # unknown type or truncated tail: resync one byte on
                self.bad_bytes += 1
                pos += 1
                continue
            if mid == FMT_ID:
                self._add_format(d[pos + 3:pos + length])
                fmt = self.formats.get(d[pos + 3])
                if fmt is not None:
                    lengths[fmt.id] = fmt.length
            else:
                offsets.setdefault(mid, []).append(pos)
            pos += length
        if not self.formats:
            raise DataflashError("no FMT messages found; not an ArduPilot dataflash log")

    def names(self) -> List[str]:
        return sorted(f.name for f in self.formats.values() if f.id in self.offsets)

    def columns(self, name: str) -> List[str]:
        f = self.by_name.get(name)
        return list(f.columns) if f else []

    def read(self, name: str) -> Optional[np.ndarray]:
        """All messages of one type as a numpy structured array (scaled copy)."""
        f = self.by_name.get(name)
        if f is None or f.id not in self.offsets:
            return None
        offs = np.asarray(self.offsets[f.id], dtype=np.int64) + 3
        size = f.length - 3
        buf = np.frombuffer(self.data, dtype=np.uint8)
        cols = np.arange(size, dtype=np.int64)[None, :]
        raw = np.empty(len(offs), dtype=f.dtype)
        rows = raw.view(np.uint8).reshape(len(offs), size)
        step = 50_000  # bound the temporary index array
        for i in range(0, len(offs), step):
            chunk = offs[i:i + step]
            rows[i:i + len(chunk)] = buf[chunk[:, None] + cols]
        if not f.scales:
            return raw
        out = {}
        for col in f.columns:
            v = raw[col]
            out[col] = v * f.scales[col] if col in f.scales else v
        return _Columns(out, len(raw))

    def strings(self, name: str, col: str) -> List[str]:
        arr = self.read(name)
        if arr is None:
            return []
        return [x.rstrip(b"\0").decode("utf-8", "replace") for x in arr[col]]


class _Columns:
    """Dict-of-arrays that looks enough like a structured array for our use."""

    def __init__(self, cols: Dict[str, np.ndarray], n: int):
        self._cols = cols
        self._n = n
        self.dtype = np.dtype([(k, np.asarray(v).dtype) for k, v in cols.items()])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._cols[key]
        return _Columns({k: v[key] for k, v in self._cols.items()},
                        len(next(iter(self._cols.values()))[key]) if self._cols else 0)

    def __len__(self):
        return self._n
