"""Decoder for Betaflight / INAV / Cleanflight / EmuFlight blackbox logs.

The format is documented in Betaflight's ``docs/Blackbox.md`` and implemented
by the reference ``blackbox_decode`` tool from blackbox-tools. This is an
independent Python implementation of the same rules: header lines describe
each frame type (field names, predictors, encodings), then the data section is
a stream of frames that each start with a one-letter marker:

    I  intra frame (keyframe, full values)
    P  inter frame (deltas against previous I/P frames)
    S  slow frame  (flight mode flags, failsafe phase, ...)
    G  GPS frame,  H  GPS home frame
    E  event       (sync beep, disarm, flight mode change, log end)

A frame is only accepted when the byte after it is another frame marker (or
the end of the log), the same heuristic blackbox_decode uses to recover from
corruption on SD cards and flash chips.

The test suite checks this decoder against blackbox_decode output on real
Betaflight and INAV logs.
"""

from __future__ import annotations

import struct
from array import array
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG_START = b"H Product:Blackbox flight data recorder by Nicholas Sherlock\n"
MARKERS = frozenset(b"IPGHES")
MAX_FRAME_LEN = 256
MAX_TIME_JUMP_US = 10_000_000
MAX_ITER_JUMP = 5000

# predictors
P_ZERO, P_PREV, P_STRAIGHT, P_AVG2, P_MINTHR, P_MOTOR0, P_INC = 0, 1, 2, 3, 4, 5, 6
P_HOME, P_1500, P_VBATREF, P_LASTTIME, P_MINMOTOR = 7, 8, 9, 10, 11
P_HOME1 = 256  # second half of a home coordinate pair, rewritten at load time

# encodings
E_SVB, E_UVB, E_NEG14, E_EDELTA_U, E_EDELTA_S = 0, 1, 3, 4, 5
E_TAG8_8SVB, E_TAG2_3S32, E_TAG8_4S16, E_NULL = 6, 7, 8, 9
E_EGAMMA_U, E_EGAMMA_S = 10, 11

# event ids
EV_SYNC_BEEP = 0
EV_AUTOTUNE_START, EV_AUTOTUNE_RESULT, EV_AUTOTUNE_TARGETS = 10, 11, 12
EV_INFLIGHT_ADJ, EV_LOGGING_RESUME, EV_DISARM = 13, 14, 15
EV_GTUNE_RESULT = 20
EV_FLIGHT_MODE = 30
EV_LOG_END = 255


class BlackboxError(Exception):
    pass


@dataclass
class FrameDef:
    names: List[str] = field(default_factory=list)
    signed: List[int] = field(default_factory=list)
    predictor: List[int] = field(default_factory=list)
    encoding: List[int] = field(default_factory=list)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            return -1


@dataclass
class BlackboxSession:
    """One armed session from a blackbox file (files often hold several)."""

    index: int
    headers: Dict[str, str]
    main_names: List[str]
    main: np.ndarray                     # shape (frames, fields), int64
    slow_names: List[str] = field(default_factory=list)
    slow: List[Tuple[int, List[int]]] = field(default_factory=list)    # (time_us, values)
    gps_names: List[str] = field(default_factory=list)
    gps: List[Tuple[int, List[int]]] = field(default_factory=list)
    events: List[Tuple[int, int, dict]] = field(default_factory=list)  # (time_us, id, data)
    corrupt_frames: int = 0
    clean_end: bool = False              # saw an "End of log" event

    def column(self, name: str) -> Optional[np.ndarray]:
        try:
            return self.main[:, self.main_names.index(name)]
        except ValueError:
            return None

    @property
    def firmware(self) -> str:
        return self.headers.get("Firmware revision", "")


def find_sessions(data: bytes) -> List[Tuple[int, int]]:
    """Byte ranges of every log inside ``data``."""
    starts = []
    pos = data.find(LOG_START)
    while pos >= 0:
        starts.append(pos)
        pos = data.find(LOG_START, pos + len(LOG_START))
    return [(s, starts[i + 1] if i + 1 < len(starts) else len(data))
            for i, s in enumerate(starts)]


def _int_list(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        try:
            out.append(int(part))
        except ValueError:
            out.append(0)
    return out


def _s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def _c_div2(v: int) -> int:
    # C integer division truncates toward zero, Python's // floors
    return -((-v) >> 1) if v < 0 else v >> 1


class _Stream:
    __slots__ = ("d", "pos", "end", "eof", "bitbuf", "bitcount")

    def __init__(self, data: bytes, start: int, end: int):
        self.d = data
        self.pos = start
        self.end = end
        self.eof = False
        self.bitbuf = 0
        self.bitcount = 0

    def byte(self) -> int:
        if self.pos >= self.end:
            self.eof = True
            return 0
        b = self.d[self.pos]
        self.pos += 1
        return b

    def uvb(self) -> int:
        result = 0
        shift = 0
        for _ in range(5):
            if self.pos >= self.end:
                self.eof = True
                return 0
            c = self.d[self.pos]
            self.pos += 1
            result |= (c & 0x7F) << shift
            if c < 128:
                return result & 0xFFFFFFFF
            shift += 7
        return 0  # more than 5 bytes: corrupt

    def svb(self) -> int:
        v = self.uvb()
        return (v >> 1) ^ -(v & 1)

    # Elias codes are bit-packed; only very old logs use them.
    def bit(self) -> int:
        if self.bitcount == 0:
            if self.pos >= self.end:
                self.eof = True
                return 0
            self.bitbuf = self.d[self.pos]
            self.pos += 1
            self.bitcount = 8
        self.bitcount -= 1
        return (self.bitbuf >> self.bitcount) & 1

    def bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v

    def align(self) -> None:
        self.bitcount = 0

    def elias_delta_u32(self) -> int:
        n = 0
        while n <= 32 and self.bit() == 0:
            if self.eof:
                return 0
            n += 1
        if self.eof or n > 32:
            return 0
        length = ((1 << n) | self.bits(n)) - 1
        if length > 32:
            return 0
        result = (1 << length) | self.bits(length)
        if result == 0xFFFFFFFF:
            return 0xFFFFFFFE if self.bit() == 0 else 0xFFFFFFFF
        return result - 1

    def elias_gamma_u32(self) -> int:
        n = 0
        while n <= 32 and self.bit() == 0:
            if self.eof:
                return 0
            n += 1
        if self.eof or n > 32:
            return 0
        # n zeros, the 1 we just consumed, then n low bits
        result = (1 << n) | self.bits(n)
        if result == 0xFFFFFFFF:
            return 0xFFFFFFFE if self.bit() == 0 else 0xFFFFFFFF
        return result - 1


def _zigzag(v: int) -> int:
    return (v >> 1) ^ -(v & 1)


def _sext(v: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (v & (sign - 1)) - (v & sign)


def _read_tag2_3s32(s: _Stream) -> List[int]:
    lead = s.byte()
    sel = lead >> 6
    if sel == 0:
        return [_sext((lead >> 4) & 3, 2), _sext((lead >> 2) & 3, 2), _sext(lead & 3, 2)]
    if sel == 1:
        b = s.byte()
        return [_sext(lead & 0xF, 4), _sext(b >> 4, 4), _sext(b & 0xF, 4)]
    if sel == 2:
        b1 = s.byte()
        b2 = s.byte()
        return [_sext(lead & 0x3F, 6), _sext(b1 & 0x3F, 6), _sext(b2 & 0x3F, 6)]
    out = []
    for _ in range(3):
        width = lead & 3
        if width == 0:
            out.append(_sext(s.byte(), 8))
        elif width == 1:
            b1 = s.byte(); b2 = s.byte()
            out.append(_sext(b1 | (b2 << 8), 16))
        elif width == 2:
            b1 = s.byte(); b2 = s.byte(); b3 = s.byte()
            out.append(_sext(b1 | (b2 << 8) | (b3 << 16), 24))
        else:
            b1 = s.byte(); b2 = s.byte(); b3 = s.byte(); b4 = s.byte()
            out.append(_sext(b1 | (b2 << 8) | (b3 << 16) | (b4 << 24), 32))
        lead >>= 2
    return out


def _read_tag8_4s16_v1(s: _Stream) -> List[int]:
    sel = s.byte()
    out = [0, 0, 0, 0]
    i = 0
    while i < 4:
        kind = sel & 3
        if kind == 1:
            c = s.byte()
            out[i] = _sext(c & 0xF, 4)
            i += 1
            sel >>= 2
            if i < 4:
                out[i] = _sext(c >> 4, 4)
        elif kind == 2:
            out[i] = _sext(s.byte(), 8)
        elif kind == 3:
            c1 = s.byte(); c2 = s.byte()
            out[i] = _sext(c1 | (c2 << 8), 16)
        sel >>= 2
        i += 1
    return out


def _read_tag8_4s16_v2(s: _Stream) -> List[int]:
    sel = s.byte()
    out = [0, 0, 0, 0]
    nibble = False
    buf = 0
    for i in range(4):
        kind = sel & 3
        if kind == 1:
            if not nibble:
                buf = s.byte()
                out[i] = _sext(buf >> 4, 4)
                nibble = True
            else:
                out[i] = _sext(buf & 0xF, 4)
                nibble = False
        elif kind == 2:
            if not nibble:
                out[i] = _sext(s.byte(), 8)
            else:
                c1 = (buf << 4) & 0xFF
                buf = s.byte()
                out[i] = _sext(c1 | (buf >> 4), 8)
        elif kind == 3:
            if not nibble:
                c1 = s.byte(); c2 = s.byte()
                out[i] = _sext((c1 << 8) | c2, 16)
            else:
                c1 = s.byte(); c2 = s.byte()
                out[i] = _sext(((buf << 12) | (c1 << 4) | (c2 >> 4)) & 0xFFFF, 16)
                buf = c2
        sel >>= 2
    return out


def _read_tag8_8svb(s: _Stream, count: int) -> List[int]:
    if count == 1:
        return [s.svb()]
    header = s.byte()
    out = []
    for _ in range(8):
        out.append(s.svb() if header & 1 else 0)
        header >>= 1
    return out


class _Parser:
    def __init__(self, data: bytes, start: int, end: int, index: int):
        self.data = data
        self.start = start
        self.end = end
        self.index = index
        self.headers: Dict[str, str] = {}
        self.defs: Dict[str, FrameDef] = {c: FrameDef() for c in "IPGHS"}
        self.data_version = 2
        self.i_interval = 32
        self.p_num = 1
        self.p_denom = 1
        self.minthrottle = 1150
        self.motor_low = 1150
        self.vbatref = 4095

    # ---------------------------------------------------------------- header
    def _header_line(self, line: str) -> None:
        if ":" not in line:
            return
        name, value = line.split(":", 1)
        self.headers[name] = value
        if name.startswith("Field "):
            ftype = name[6]
            fdef = self.defs.setdefault(ftype, FrameDef())
            if name.endswith(" name"):
                fdef.names = value.split(",")
                if ftype == "I":
                    self.defs["P"].names = fdef.names
            elif name.endswith(" signed"):
                fdef.signed = _int_list(value)
                if ftype == "I":
                    self.defs["P"].signed = fdef.signed
            elif name.endswith(" predictor"):
                fdef.predictor = _int_list(value)
            elif name.endswith(" encoding"):
                fdef.encoding = _int_list(value)
        elif name == "I interval":
            self.i_interval = max(1, _int_list(value)[0])
        elif name == "P interval":
            # "1/2" in older logs; newer Betaflight writes the denominator only
            if "/" in value:
                num, den = value.split("/", 1)
                self.p_num, self.p_denom = int(num or 1), int(den or 1)
            else:
                self.p_num, self.p_denom = 1, max(1, _int_list(value)[0])
        elif name == "Data version":
            self.data_version = _int_list(value)[0]
        elif name == "minthrottle":
            self.minthrottle = _int_list(value)[0]
            self.motor_low = self.minthrottle
        elif name == "motorOutput":
            self.motor_low = _int_list(value)[0]
        elif name == "vbatref":
            self.vbatref = _int_list(value)[0]

    def _read_headers(self, s: _Stream) -> bool:
        d = self.data
        while s.pos < s.end:
            c = d[s.pos]
            if c == ord("H") and s.pos + 1 < s.end and d[s.pos + 1] == ord(" "):
                nl = d.find(b"\n", s.pos, min(s.end, s.pos + 4096))
                if nl < 0:
                    s.pos += 1
                    continue
                raw = d[s.pos + 2:nl]
                if b"\0" not in raw:
                    self._header_line(raw.decode("latin-1"))
                s.pos = nl + 1
            elif c in MARKERS and c != ord("H"):
                return True
            else:
                s.pos += 1  # garbage before the first frame
        return False

    # ----------------------------------------------------------------- frames
    def _compile(self, ftype: str):
        fdef = self.defs[ftype]
        n = len(fdef.names)
        pred = (fdef.predictor + [0] * n)[:n]
        enc = (fdef.encoding + [0] * n)[:n]
        signed = (fdef.signed + [0] * n)[:n]
        known_pred = {P_ZERO, P_PREV, P_STRAIGHT, P_AVG2, P_MINTHR, P_MOTOR0, P_INC,
                      P_HOME, P_1500, P_VBATREF, P_LASTTIME, P_MINMOTOR}
        known_enc = {E_SVB, E_UVB, E_NEG14, E_EDELTA_U, E_EDELTA_S, E_TAG8_8SVB,
                     E_TAG2_3S32, E_TAG8_4S16, E_NULL, E_EGAMMA_U, E_EGAMMA_S}
        for i in range(n):
            if pred[i] not in known_pred:
                raise BlackboxError(f"{fdef.names[i]}: unsupported predictor {pred[i]}")
            if enc[i] not in known_enc:
                raise BlackboxError(f"{fdef.names[i]}: unsupported encoding {enc[i]}")
        if ftype == "G":
            for i in range(1, n):
                if pred[i - 1] == P_HOME and pred[i] == P_HOME:
                    pred[i] = P_HOME1
        # Group consecutive fields that share a packed encoding.
        ops = []
        i = 0
        while i < n:
            e = enc[i]
            if pred[i] == P_INC:
                ops.append(("inc", [i], None))
                i += 1
            elif e == E_TAG8_4S16:
                ops.append(("t4", list(range(i, min(i + 4, n))), None))
                i += 4
            elif e == E_TAG2_3S32:
                ops.append(("t3", list(range(i, min(i + 3, n))), None))
                i += 3
            elif e == E_TAG8_8SVB:
                j = i + 1
                while j < i + 8 and j < n and enc[j] == E_TAG8_8SVB:
                    j += 1
                ops.append(("t8", list(range(i, j)), None))
                i = j
            else:
                ops.append(("one", [i], e))
                i += 1
        return n, pred, signed, ops

    def parse(self) -> BlackboxSession:
        s = _Stream(self.data, self.start, self.end)
        if not self._read_headers(s):
            raise BlackboxError("log has a header but no data frames")
        idef = self.defs["I"]
        if not idef.names:
            raise BlackboxError("log is missing its field definitions")

        compiled = {k: self._compile(k) for k in "IPGHS" if self.defs[k].names}
        main_names = idef.names
        nfields = len(main_names)
        it_idx = idef.index("loopIteration")
        time_idx = idef.index("time")
        if it_idx < 0 or time_idx < 0:
            raise BlackboxError("log has no loopIteration/time fields")
        motor0 = idef.index("motor[0]")
        gdef = self.defs["G"]
        hdef = self.defs["H"]
        home_idx = (hdef.index("GPS_home[0]"), hdef.index("GPS_home[1]"))
        g_time_idx = gdef.index("time")

        # flat int64 buffer: a list of Python int lists costs ~4x the memory
        rows = array("q")
        nrows = 0
        prev: Optional[List[int]] = None
        prev2: Optional[List[int]] = None
        valid = False
        last_iter: Optional[int] = None
        last_time = -1
        rollover = [0, 0]
        last_gps_time = -1
        home = [0, 0]
        slow: List[Tuple[int, List[int]]] = []
        gps: List[Tuple[int, List[int]]] = []
        events: List[Tuple[int, int, dict]] = []
        corrupt = 0
        clean_end = False

        d = self.data
        v1 = self.data_version < 2
        i_int, p_num, p_den = self.i_interval, self.p_num, self.p_denom

        def should_have(idx: int) -> bool:
            return (idx % i_int + p_num - 1) % p_den < p_num

        def decode(ftype: str, cur: List[int], pv, pv2, skipped: int) -> None:
            n, pred, signed, ops = compiled[ftype]
            for kind, idxs, enc in ops:
                if kind == "inc":
                    i = idxs[0]
                    cur[i] = skipped + 1 + (pv[i] if pv is not None else 0)
                    continue
                if kind == "one":
                    s.align()
                    if enc == E_SVB:
                        raw = [s.svb()]
                    elif enc == E_UVB:
                        raw = [s.uvb()]
                    elif enc == E_NEG14:
                        raw = [-_sext(s.uvb() & 0x3FFF, 14)]
                    elif enc == E_NULL:
                        raw = [0]
                    elif enc == E_EDELTA_U:
                        raw = [s.elias_delta_u32()]
                    elif enc == E_EDELTA_S:
                        raw = [_zigzag(s.elias_delta_u32())]
                    elif enc == E_EGAMMA_U:
                        raw = [s.elias_gamma_u32()]
                    else:  # E_EGAMMA_S; _compile rejects anything else
                        raw = [_zigzag(s.elias_gamma_u32())]
                else:
                    s.align()
                    if kind == "t4":
                        raw = _read_tag8_4s16_v1(s) if v1 else _read_tag8_4s16_v2(s)
                    elif kind == "t3":
                        raw = _read_tag2_3s32(s)
                    else:
                        raw = _read_tag8_8svb(s, len(idxs))
                for k, i in enumerate(idxs):
                    v = raw[k]
                    p = pred[i]
                    if p == P_PREV:
                        if pv is not None:
                            v += pv[i]
                    elif p == P_STRAIGHT:
                        if pv is not None:
                            v += 2 * pv[i] - pv2[i]
                    elif p == P_AVG2:
                        if pv is not None:
                            v += _c_div2(pv[i] + pv2[i])
                    elif p == P_MINMOTOR:
                        v += self.motor_low
                    elif p == P_MINTHR:
                        v += self.minthrottle
                    elif p == P_1500:
                        v += 1500
                    elif p == P_MOTOR0:
                        if motor0 >= 0:
                            v += cur[motor0]
                    elif p == P_VBATREF:
                        v += self.vbatref
                    elif p == P_LASTTIME:
                        if prev is not None:
                            v += prev[time_idx]
                    elif p == P_HOME:
                        v += home[0]
                    elif p == P_HOME1:
                        v += home[1]
                    if kind == "one":
                        v = _s32(v) if signed[i] else v & 0xFFFFFFFF
                    cur[i] = v
            s.align()

        def fix_time(t: int, which: int, last: int) -> int:
            if last >= 0:
                if (t & 0xFFFFFFFF) < (last & 0xFFFFFFFF) and \
                        ((t - last) & 0xFFFFFFFF) < MAX_TIME_JUMP_US:
                    rollover[which] += 1 << 32
            return (t & 0xFFFFFFFF) + rollover[which]

        def parse_event():
            ev = s.byte()
            data: dict = {}
            if ev == EV_SYNC_BEEP:
                data["time"] = s.uvb() + rollover[0]
            elif ev == EV_AUTOTUNE_START:
                for k in ("phase", "cycle", "p", "i", "d"):
                    data[k] = s.byte()
            elif ev == EV_AUTOTUNE_RESULT:
                for k in ("flags", "p", "i", "d"):
                    data[k] = s.byte()
            elif ev == EV_AUTOTUNE_TARGETS:
                for _ in range(8):
                    s.byte()
            elif ev == EV_GTUNE_RESULT:
                data["axis"] = s.byte()
                data["gyro_avg"] = s.svb()
                s.byte(); s.byte()
            elif ev == EV_INFLIGHT_ADJ:
                fn = s.byte()
                data["function"] = fn & 0x7F
                if fn > 127:
                    raw = bytes(s.byte() for _ in range(4))
                    data["value"] = struct.unpack("<f", raw)[0]
                else:
                    data["value"] = s.svb()
            elif ev == EV_LOGGING_RESUME:
                data["iteration"] = s.uvb()
                data["time"] = s.uvb() + rollover[0]
            elif ev == EV_DISARM:
                data["reason"] = s.uvb()
            elif ev == EV_FLIGHT_MODE:
                data["flags"] = s.uvb()
                data["last_flags"] = s.uvb()
            elif ev == EV_LOG_END:
                msg = bytes(s.byte() for _ in range(11))
                if not msg.startswith(b"End of log"):
                    return None
            else:
                return None
            return ev, data

        pos_limit = self.end
        while True:
            if s.pos >= pos_limit:
                break
            frame_start = s.pos
            marker = d[s.pos]
            s.pos += 1
            s.eof = False
            if marker not in MARKERS or chr(marker) not in compiled and marker != ord("E"):
                valid = False
                continue
            ftype = chr(marker)

            result = None
            if ftype == "I":
                cur = [0] * nfields
                decode("I", cur, prev, None, 0)
            elif ftype == "P":
                skipped = 0
                if last_iter is not None:
                    k = last_iter + 1
                    while not should_have(k) and skipped < MAX_ITER_JUMP:
                        skipped += 1
                        k += 1
                cur = [0] * nfields
                decode("P", cur, prev if valid else None, prev2 if valid else None, skipped)
            elif ftype in "GHS":
                cur = [0] * len(self.defs[ftype].names)
                decode(ftype, cur, None, None, 0)
            else:
                result = parse_event()

            if ftype == "E" and result is not None and result[0] == EV_LOG_END and not s.eof:
                events.append((max(last_time, 0), EV_LOG_END, {}))
                clean_end = True
                break

            frame_end = s.pos
            at_end = s.pos >= pos_limit
            next_ok = at_end and not s.eof or (not at_end and d[s.pos] in MARKERS)
            if frame_end - frame_start > MAX_FRAME_LEN or not next_ok or s.eof:
                corrupt += 1
                valid = False
                prev = prev2 = None
                s.pos = frame_start + 1
                continue

            if ftype in "IP":
                cur[time_idx] = fix_time(cur[time_idx], 0, last_time)
                it = cur[it_idx] & 0xFFFFFFFF
                tm = cur[time_idx]
                if ftype == "I":
                    ok = last_iter is None or (
                        last_iter <= it < last_iter + MAX_ITER_JUMP
                        and last_time <= tm < last_time + MAX_TIME_JUMP_US)
                    valid = ok
                    if not ok:
                        prev = prev2 = None
                elif valid:
                    ok = (last_iter <= it < last_iter + MAX_ITER_JUMP
                          and last_time <= tm < last_time + MAX_TIME_JUMP_US)
                    if not ok:
                        valid = False
                        prev = prev2 = None
                if valid:
                    last_iter = it
                    last_time = tm
                    rows.extend(cur)
                    nrows += 1
                    if ftype == "I":
                        prev = prev2 = cur
                    else:
                        prev2, prev = prev, cur
            elif not valid and last_iter is not None:
                # Side frames (S/G/H/E) carry no checksum or timestamp check of
                # their own. Once the main stream has lost sync we are usually
                # reading stale flash contents after the real end of the log,
                # so trust them only while the main stream is in sync.
                continue
            elif ftype == "H":
                if home_idx[0] >= 0:
                    home[0] = cur[home_idx[0]]
                if home_idx[1] >= 0:
                    home[1] = cur[home_idx[1]]
            elif ftype == "G":
                if g_time_idx >= 0:
                    cur[g_time_idx] = fix_time(cur[g_time_idx], 1, last_gps_time)
                    last_gps_time = cur[g_time_idx]
                gps.append((cur[g_time_idx] if g_time_idx >= 0 else max(last_time, 0), cur))
            elif ftype == "S":
                slow.append((max(last_time, 0), cur))
            elif ftype == "E":
                if result is None:
                    continue
                ev, data = result
                if ev == EV_LOGGING_RESUME:
                    last_iter = data["iteration"]
                    last_time = data["time"]
                events.append((data.get("time", max(last_time, 0)), ev, data))

        main = np.frombuffer(rows, dtype=np.int64).reshape(nrows, nfields).copy()
        return BlackboxSession(
            index=self.index,
            headers=self.headers,
            main_names=list(main_names),
            main=main,
            slow_names=list(self.defs["S"].names),
            slow=slow,
            gps_names=list(gdef.names),
            gps=gps,
            events=events,
            corrupt_frames=corrupt,
            clean_end=clean_end,
        )


def decode_bytes(data: bytes) -> List[BlackboxSession]:
    """Decode every session in a blackbox file. Sessions that fail are skipped."""
    sessions = []
    for i, (a, b) in enumerate(find_sessions(data)):
        try:
            sessions.append(_Parser(data, a, b, i).parse())
        except BlackboxError:
            continue
    return sessions


def decode_file(path: str) -> List[BlackboxSession]:
    with open(path, "rb") as fh:
        return decode_bytes(fh.read())

