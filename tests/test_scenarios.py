"""End-to-end: every fixture has one known fault, and it has to come out on top.

The ArduPilot and PX4 logs were flown in SITL with tools/sitl_scenarios.py and
tools/px4_sih_scenarios.py, then trimmed with tools/trim_log.py. The blackbox
logs are real flights (see tests/data/blackbox/README.md).
"""

import pytest

SERIOUS = ("warning", "critical")


def top_checks(diags, n=1):
    d = diags[0]
    return [c.check for c in d.causes if c.severity in SERIOUS][:n]


@pytest.mark.parametrize("fixture, expected", [
    ("motor_failure.bin.gz", "motor_failure"),
    ("high_vibration.bin.gz", "vibration"),
    ("compass_interference.bin.gz", "compass"),
    ("gps_glitch.bin.gz", "gps"),
    ("radio_failsafe.bin.gz", "radio"),
    ("px4_gps_loss.ulg.gz", "gps"),
    ("px4_battery_drain.ulg.gz", "battery"),
])
def test_known_fault_ranks_first(diagnose_fixture, fixture, expected):
    assert top_checks(diagnose_fixture(fixture)) == [expected]


def test_power_loss_names_battery_and_power_cut(diagnose_fixture):
    d = diagnose_fixture("power_loss.bin.gz")[0]
    assert d.incident is not None and d.incident.kind == "log_cut"
    assert {"battery", "log_cut"} <= set(top_checks([d], 2))


def test_motor_failure_details(diagnose_fixture):
    d = diagnose_fixture("motor_failure.bin.gz")[0]
    assert d.incident is not None and d.incident.kind == "crash"
    top = d.causes[0]
    assert top.data["motor"] == 1
    assert top.severity == "critical"
    # it happened right before the impact
    assert d.incident.t - 5 < top.t0 < d.incident.t


@pytest.mark.parametrize("fixture", ["healthy.bin.gz", "px4_healthy.ulg.gz"])
def test_healthy_flights_are_clean(diagnose_fixture, fixture):
    d = diagnose_fixture(fixture)[0]
    assert d.incident is None
    assert [c.title for c in d.causes if c.severity in SERIOUS] == []
    assert d.errors == []


def test_compass_is_credited_over_ekf(diagnose_fixture):
    d = diagnose_fixture("compass_interference.bin.gz")[0]
    order = [c.check for c in d.causes]
    assert order.index("compass") < order.index("ekf")


def test_takeoff_with_wrong_motor_setup(diagnose_fixture):
    d = diagnose_fixture("blackbox", "crashing-LOG00002.BFL.gz")[0]
    assert d.flight.firmware == "betaflight"  # EmuFlight logs use the same format
    assert top_checks([d]) == ["takeoff"]
    assert d.causes[0].data["motor_high"] == 1


@pytest.mark.parametrize("fixture", ["LOG00037.BFL.gz", "btfl_001.bbl.gz"])
def test_normal_betaflight_flights_have_no_false_alarm(diagnose_fixture, fixture):
    for d in diagnose_fixture("blackbox", fixture):
        assert d.incident is None, d.incident
        assert top_checks([d]) == []


def test_inav_plane_loads(diagnose_fixture):
    d = diagnose_fixture("blackbox", "inav-LOG00004-first3MB.TXT.gz")[0]
    f = d.flight
    assert f.firmware == "inav" and f.vehicle == "plane"
    assert f.has("gps", "nsats") and f.has("att", "roll") and f.has("battery", "curr")
    assert d.errors == []
    # the fixture is the first 3 MB of a flash dump, so it stops mid-flight
    assert top_checks([d]) == ["log_cut"]


def test_no_check_crashes_on_any_fixture(diagnose_fixture):
    import glob
    import os
    from conftest import DATA
    for path in sorted(glob.glob(os.path.join(DATA, "**", "*.gz"), recursive=True)):
        rel = os.path.relpath(path, DATA).split(os.sep)
        for d in diagnose_fixture(*rel):
            assert d.errors == [], (path, d.errors)
