#!/usr/bin/env python3
"""Shrink a log to the messages flightdoc reads, for use as a test fixture.

    python tools/trim_log.py big.bin small.bin
    python tools/trim_log.py big.ulg small.ulg

ArduPilot logs keep their FMT/PARM/MSG records and the message types listed
in KEEP_AP. PX4 logs keep the topics in KEEP_PX4 (pyulog does the rewrite).
"""

import sys

KEEP_AP = {
    "FMT", "FMTU", "UNIT", "MULT", "PARM", "MSG", "MODE", "ERR", "EV", "ARM",
    "ATT", "RATE", "RCOU", "VIBE", "BAT", "CURR", "POWR", "MAG", "GPS", "XKF4",
    "NKF4", "CTUN", "BARO", "PM", "RSSI", "ESC", "IMU",
}
KEEP_PX4 = [
    "vehicle_attitude", "vehicle_attitude_setpoint", "vehicle_angular_velocity",
    "vehicle_rates_setpoint", "sensor_combined", "vehicle_magnetometer",
    "actuator_motors", "actuator_outputs", "vehicle_thrust_setpoint",
    "battery_status", "system_power", "sensor_gps", "vehicle_gps_position",
    "sensor_gnss", "vehicle_local_position", "estimator_status", "input_rc",
    "cpuload", "vehicle_status", "failsafe_flags",
]


def trim_dataflash(src, dst):
    with open(src, "rb") as fh:
        data = fh.read()
    names, lengths = {}, {128: 89}
    out = bytearray()
    pos, n = 0, len(data)
    while pos + 3 <= n:
        if data[pos] != 0xA3 or data[pos + 1] != 0x95:
            pos += 1
            continue
        mid = data[pos + 2]
        length = lengths.get(mid)
        if length is None or pos + length > n:
            pos += 1
            continue
        if mid == 128:
            body = data[pos + 3:pos + length]
            lengths[body[0]] = body[1]
            names[body[0]] = body[2:6].rstrip(b"\0").decode("ascii", "replace")
            out += data[pos:pos + length]
        elif names.get(mid) in KEEP_AP:
            out += data[pos:pos + length]
        pos += length
    with open(dst, "wb") as fh:
        fh.write(out)


def trim_ulog(src, dst):
    from pyulog import ULog
    ULog(src, message_name_filter_list=KEEP_PX4).write_ulog(dst)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    if src.lower().endswith(".ulg"):
        trim_ulog(src, dst)
    else:
        trim_dataflash(src, dst)


if __name__ == "__main__":
    main()
