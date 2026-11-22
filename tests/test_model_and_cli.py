import json

import numpy as np

from flightdoc.cli import main
from flightdoc.checks.power import estimate_cells
from flightdoc.model import spans
from conftest import data


def test_spans_basic():
    t = np.arange(10.0)
    m = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1], bool)
    assert spans(m, t) == [(1.0, 2.0), (5.0, 7.0), (9.0, 9.0)]
    assert spans(m, t, min_len=1.5) == [(5.0, 7.0)]
    assert spans(m, t, max_gap=3.0) == [(1.0, 9.0)]
    assert spans(np.zeros(5, bool), t[:5]) == []


def test_cell_count():
    assert estimate_cells(np.full(50, 16.8)) == 4
    assert estimate_cells(np.full(50, 25.1)) == 6
    assert estimate_cells(np.full(50, 4.35)) == 1   # 1S LiHV
    assert estimate_cells(np.full(50, 22.6)) == 6   # 6S at storage voltage


def test_cli_json(capsys):
    assert main([data("radio_failsafe.bin.gz"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["firmware"] == "ardupilot"
    assert out["causes"][0]["check"] == "radio"


def test_cli_html(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert main([data("motor_failure.bin.gz"), "-o", str(out)]) == 0
    html = out.read_text()
    assert "Motor 1 lost thrust" in html and "data:image/png;base64" in html


def test_cli_rejects_unknown_file(tmp_path, capsys):
    junk = tmp_path / "notes.txt"
    junk.write_text("not a log")
    assert main([str(junk)]) == 2


def _two_flights(gps_t):
    from flightdoc.model import Flight
    f = Flight(source="synthetic", firmware="ardupilot", vehicle="copter")
    t = np.arange(0, 100, 0.1)
    m = np.where((t < 40) | (t > 60), 0.5, 0.0)
    f.add("motors", t, m1=m, m2=m, m3=m, m4=m)
    n = gps_t.size
    f.add("gps", gps_t, nsats=np.full(n, 12.0), fix=np.full(n, 3.0), hdop=np.full(n, 0.8))
    f.armed = [(0.0, 40.0), (60.0, 100.0)]
    f.duration = 100.0
    return f


def test_landing_between_flights_is_not_a_gps_dropout():
    from flightdoc.checks.navigation import gps
    f = _two_flights(np.concatenate([np.arange(0, 40, 0.2), np.arange(60, 100, 0.2)]))
    assert [x.severity for x in gps(f, None)] == ["ok"]


def test_gps_dropout_in_flight_is_caught():
    from flightdoc.checks.navigation import gps
    f = _two_flights(np.concatenate([np.arange(0, 20, 0.2), np.arange(30, 100, 0.2)]))
    found = gps(f, None)
    assert found[0].severity in ("warning", "critical")
    assert any("no GPS data" in e for e in found[0].evidence)


def test_never_armed_log_says_so():
    from flightdoc.analysis import diagnose
    from flightdoc.model import Flight
    f = Flight(source="bench", firmware="px4")
    t = np.arange(0, 10, 0.1)
    f.add("battery", t, volt=np.full(t.size, 16.8))
    f.duration = 10.0
    assert "never armed" in diagnose(f).verdict


def test_failed_hand_launch_is_explained():
    """Throttle held low, punched at the last moment, rolls off, hits the ground."""
    from flightdoc.analysis import diagnose
    from flightdoc.model import Flight
    f = Flight(source="synthetic", firmware="inav", vehicle="plane")
    t = np.arange(0, 9, 0.01)
    thr = np.where(t < 7.5, np.where(t > 1.0, 0.35, 0.0), 1.0)
    f.add("throttle", t, thr=thr)
    f.add("motors", t, m1=thr, m2=thr)
    roll = np.where(t > 7.6, -70.0, 0.0)
    f.add("att", t, roll=roll, pitch=np.full(t.size, 10.0), yaw=np.zeros(t.size))
    alt = np.where(t > 7.0, 1.5, 0.0) + np.where((t > 5) & (t < 5.05), 18.0, 0.0)  # a baro spike
    f.add("alt", t, alt=alt)
    acc = np.zeros((t.size, 3)); acc[:, 2] = 9.81
    acc[(t > 7.88) & (t < 7.9), 0] = 60.0
    f.add("accel", t, x=acc[:, 0], y=acc[:, 1], z=acc[:, 2])
    f.armed = [(0.0, 8.0)]
    f.duration = 9.0
    d = diagnose(f)
    assert d.causes and d.causes[0].check == "launch"
    assert "full power came too late" in d.causes[0].title
    assert "rolled off to the left" in d.causes[0].title
