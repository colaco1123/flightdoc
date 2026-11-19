import gzip

import numpy as np
import pytest

from flightdoc.formats.dataflash import DataflashLog, DataflashError
from conftest import data


@pytest.fixture(scope="module")
def raw():
    return gzip.open(data("healthy.bin.gz")).read()


def test_reads_formats_and_messages(raw):
    log = DataflashLog(raw)
    assert {"ATT", "RCOU", "GPS", "PARM"} <= set(log.names())
    att = log.read("ATT")
    assert len(att) > 100
    assert np.all(np.diff(att["TimeUS"].astype(np.int64)) >= 0)


def test_matches_pymavlink(raw, tmp_path):
    mavutil = pytest.importorskip("pymavlink.mavutil")
    path = tmp_path / "log.bin"
    path.write_bytes(raw)
    m = mavutil.mavlink_connection(str(path))
    ref = []
    while True:
        msg = m.recv_match(type="GPS")
        if msg is None:
            break
        ref.append((msg.TimeUS, msg.Lat, msg.Lng, msg.Alt, msg.HDop))
    ref = np.array(ref)
    gps = DataflashLog(raw).read("GPS")
    mine = np.column_stack([gps["TimeUS"], gps["Lat"], gps["Lng"], gps["Alt"], gps["HDop"]])
    assert mine.shape == ref.shape
    assert np.allclose(mine, ref)


def test_survives_garbage_and_truncation(raw):
    damaged = raw[:200_000] + b"\x00\xa3\x95\xff" * 50 + raw[200_000:-13]
    log = DataflashLog(damaged)
    assert log.bad_bytes > 0
    assert len(log.read("ATT")) > 100


def test_rejects_non_dataflash():
    with pytest.raises(DataflashError):
        DataflashLog(b"hello world" * 100)
