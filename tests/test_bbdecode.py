import gzip
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from flightdoc.formats import bbdecode as bb
from conftest import data


def stream(bs: bytes):
    return bb._Stream(bytes(bs), 0, len(bs))


def test_varints():
    assert stream([0x7F]).uvb() == 127
    assert stream([0x80, 0x01]).uvb() == 128
    assert stream([0xFF, 0xFF, 0xFF, 0xFF, 0x0F]).uvb() == 0xFFFFFFFF
    # zig-zag: 0, -1, 1, -2, 2 ...
    assert [stream([v]).svb() for v in (0, 1, 2, 3, 4)] == [0, -1, 1, -2, 2]


def test_truncated_varint_sets_eof():
    s = stream([0x80])
    assert s.uvb() == 0 and s.eof


def test_tag2_3s32():
    # selector 0: three 2-bit fields in one byte: 01 11 10 -> 1, -1, -2
    assert bb._read_tag2_3s32(stream([0b00_01_11_10])) == [1, -1, -2]
    # selector 1: 4-bit fields
    assert bb._read_tag2_3s32(stream([0b01_00_0111, 0xF8])) == [7, -1, -8]
    # selector 3, field widths 8/16/8 bits
    lead = 0b11_00_01_00
    assert bb._read_tag2_3s32(stream([lead, 0xFF, 0x34, 0x12, 0x05])) == [-1, 0x1234, 5]


def test_tag8_4s16_v2():
    # field types (low bits first): 4-bit, 4-bit, 8-bit, zero
    sel = 0b00_10_01_01
    assert bb._read_tag8_4s16_v2(stream([sel, 0x3F, 0x80])) == [3, -1, -128, 0]


def test_tag8_8svb_header_selects_fields():
    # header 0b101: fields 0 and 2 present
    assert bb._read_tag8_8svb(stream([0b101, 0x02, 0x03]), 3)[:3] == [1, 0, -2]


def test_c_style_halving_rounds_toward_zero():
    assert bb._c_div2(-3) == -1 and bb._c_div2(3) == 1


def test_no_sessions_in_garbage():
    assert bb.decode_bytes(b"\x00" * 1000) == []


def _expected():
    with open(data("blackbox", "expected.json")) as fh:
        return json.load(fh)


@pytest.mark.parametrize("name", sorted(json.load(open(os.path.join(
    os.path.dirname(__file__), "data", "blackbox", "expected.json"))).keys()))
def test_real_logs_decode_to_reference(name):
    """Row counts and column sums recorded after cross-checking with blackbox_decode."""
    sessions = bb.decode_bytes(gzip.open(data("blackbox", name)).read())
    exp = _expected()[name]
    assert len(sessions) == len(exp)
    for s, e in zip(sessions, exp):
        assert len(s.main) == e["rows"]
        assert len(s.main_names) == e["fields"]
        for col, total in e["sums"].items():
            assert int(s.column(col).sum()) == total, col


def test_truncation_mid_frame_is_survivable():
    raw = gzip.open(data("blackbox", "btfl_001.bbl.gz")).read()
    full = bb.decode_bytes(raw)
    cut = bb.decode_bytes(raw[: len(raw) - 777])
    assert len(cut[-1].main) <= len(full[-1].main)
    assert len(cut[-1].main) > 0.9 * len(full[-1].main)


@pytest.mark.skipif(shutil.which("blackbox_decode") is None,
                    reason="blackbox_decode not installed")
def test_matches_blackbox_decode(tmp_path):
    import io
    import pandas as pd
    raw = tmp_path / "log.bfl"
    raw.write_bytes(gzip.open(data("blackbox", "LOG00037.BFL.gz")).read())
    s = bb.decode_file(str(raw))[0]
    out = subprocess.run(["blackbox_decode", "--stdout", "--unit-vbat", "raw",
                          "--unit-amperage", "raw", "--unit-flags", "raw", str(raw)],
                         capture_output=True, check=True)
    ref = pd.read_csv(io.BytesIO(out.stdout), skipinitialspace=True)
    ref.columns = [c.strip().split(" (")[0] for c in ref.columns]
    mine = pd.DataFrame(s.main, columns=s.main_names)
    joined = mine.merge(ref, on="time", suffixes=("", "_ref"))
    assert len(joined) == len(ref)
    for c in mine.columns:
        if c + "_ref" in joined and c != "loopIteration":
            assert np.array_equal(joined[c].to_numpy(), joined[c + "_ref"].to_numpy().astype(np.int64)), c
