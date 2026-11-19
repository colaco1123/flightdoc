#!/usr/bin/env python3
"""Fly ArduCopter SITL through a set of failures and keep the dataflash logs.

These logs are the ground truth for flightdoc's ArduPilot checks: each one has
a single known fault, so the tests can assert that the right cause comes out
on top.

    python tools/sitl_scenarios.py --ardupilot ~/ardupilot --out tests/data/sitl

Needs a built SITL binary (``./waf configure --board sitl && ./waf copter``)
and pymavlink. Each scenario takes one to three minutes of wall time.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time

from pymavlink import mavutil

HOME = "-35.363261,149.165230,584,353"


class Sitl:
    def __init__(self, ap_root: str, workdir: str, speedup: int):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        binary = os.path.join(ap_root, "build/sitl/bin/arducopter")
        defaults = os.path.join(ap_root, "Tools/autotest/default_params/copter.parm")
        self.proc = subprocess.Popen(
            [binary, "--model", "quad", "--speedup", str(speedup), "-w",
             "--defaults", defaults, "--home", HOME, "-I0"],
            cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        self.m = mavutil.mavlink_connection("tcp:127.0.0.1:5760", retries=60)
        self.m.wait_heartbeat(timeout=60)
        self.m.mav.request_data_stream_send(self.m.target_system, self.m.target_component,
                                            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

    # ------------------------------------------------------------- helpers
    def say(self, text):
        print(f"  [{time.strftime('%H:%M:%S')}] {text}", flush=True)

    def param(self, name: str, value: float):
        for _ in range(5):
            self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                      name.encode(), float(value),
                                      mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=3)
            if msg and msg.param_id.rstrip("\x00") == name:
                return
        self.say(f"warning: could not confirm {name}={value}")

    def mode(self, name: str):
        self.m.set_mode(self.m.mode_mapping()[name])
        self.drain(1)

    def drain(self, seconds: float):
        """Read messages for a while, printing statustexts."""
        end = time.time() + seconds
        while time.time() < end:
            msg = self.m.recv_match(blocking=True, timeout=0.5)
            if msg is not None and msg.get_type() == "STATUSTEXT":
                self.say("FC: " + msg.text)

    def armed(self) -> bool:
        hb = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        return bool(hb and hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def alt(self) -> float:
        msg = self.m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=3)
        return msg.relative_alt / 1000.0 if msg else float("nan")

    def wait_ready(self, timeout=180):
        """Wait for a 3D fix and an EKF origin, then arm in GUIDED."""
        self.say("waiting for GPS and EKF")
        end = time.time() + timeout
        while time.time() < end:
            msg = self.m.recv_match(type="GPS_RAW_INT", blocking=True, timeout=2)
            if msg and msg.fix_type >= 3:
                break
        self.drain(15)
        self.mode("GUIDED")
        while time.time() < end:
            self.m.arducopter_arm()
            self.drain(2)
            if self.armed():
                self.say("armed")
                return
        raise RuntimeError("vehicle never became armable")

    def takeoff(self, alt=15.0):
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                                     0, 0, 0, 0, 0, 0, alt)
        end = time.time() + 60
        while time.time() < end:
            if self.alt() > alt * 0.9:
                self.say(f"at {alt:.0f} m")
                return
        self.say("takeoff timed out")

    def goto_offset(self, north: float, east: float, alt: float = 15.0):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b110111111000,
            north, east, -alt, 0, 0, 0, 0, 0, 0, 0, 0)

    def wait_disarm(self, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            msg = self.m.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.get_type() == "STATUSTEXT":
                self.say("FC: " + msg.text)
            if msg.get_type() == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                if not msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    return True
        return False

    def force_disarm(self):
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                     0, 21196, 0, 0, 0, 0, 0)
        self.drain(3)

    def stop(self, hard: bool = False):
        if hard:
            self.proc.send_signal(signal.SIGKILL)
        else:
            self.drain(3)  # let the logger flush
            self.proc.terminate()
        self.proc.wait(timeout=30)

    def newest_log(self):
        logs = sorted(glob.glob(os.path.join(self.workdir, "logs", "*.BIN")), key=os.path.getmtime)
        return logs[-1] if logs else None


# ---------------------------------------------------------------- scenarios
def healthy(s: Sitl):
    s.wait_ready(); s.takeoff(15)
    s.goto_offset(40, 0); s.drain(15)
    s.goto_offset(40, 40); s.drain(15)
    s.goto_offset(0, 0); s.drain(15)
    s.mode("LAND"); s.wait_disarm(90)


def motor_failure(s: Sitl):
    s.wait_ready(); s.takeoff(20)
    s.drain(10)
    s.say("killing motor 1")
    s.param("SIM_ENGINE_MUL", 0.0)
    s.param("SIM_ENGINE_FAIL", 1)
    if not s.wait_disarm(40):
        s.force_disarm()


def high_vibration(s: Sitl):
    for n in ("SIM_ACC1_RND", "SIM_ACC2_RND", "SIM_ACC3_RND"):
        s.param(n, 0)
    s.wait_ready(); s.takeoff(15)
    s.drain(8)
    s.say("adding accelerometer vibration")
    # SITL only injects ACCn_RND as vibration when a vibration frequency is set
    for n, hz in (("SIM_VIB_FREQ_X", 137), ("SIM_VIB_FREQ_Y", 143), ("SIM_VIB_FREQ_Z", 151)):
        s.param(n, hz)
    for n in ("SIM_ACC1_RND", "SIM_ACC2_RND", "SIM_ACC3_RND"):
        s.param(n, 70)
    s.goto_offset(30, 0); s.drain(25)
    s.mode("LAND")
    if not s.wait_disarm(90):
        s.force_disarm()


def compass_interference(s: Sitl):
    s.wait_ready(); s.takeoff(15)
    s.say("adding motor-current magnetic interference")
    s.param("SIM_MAG_MOT_X", 25)
    s.param("SIM_MAG_MOT_Y", -20)
    s.param("SIM_MAG_MOT_Z", 30)
    for n, e in ((40, 0), (40, 40), (0, 40), (0, 0)):
        s.goto_offset(n, e, 15 + (n + e) / 8); s.drain(12)
    s.mode("LAND")
    if not s.wait_disarm(90):
        s.force_disarm()


def gps_glitch(s: Sitl):
    s.wait_ready(); s.takeoff(15)
    s.goto_offset(20, 0); s.drain(10)
    s.say("GPS glitch")
    s.param("SIM_GPS1_GLTCH_X", 0.0006)
    s.param("SIM_GPS1_GLTCH_Y", 0.0004)
    s.drain(20)
    s.param("SIM_GPS1_GLTCH_X", 0)
    s.param("SIM_GPS1_GLTCH_Y", 0)
    s.drain(10)
    s.mode("LAND")
    if not s.wait_disarm(90):
        s.force_disarm()


def radio_failsafe(s: Sitl):
    s.param("FS_THR_ENABLE", 1)
    s.wait_ready(); s.takeoff(15)
    s.goto_offset(30, 30); s.drain(12)
    s.say("cutting RC link")
    s.param("SIM_RC_FAIL", 1)
    if not s.wait_disarm(120):
        s.force_disarm()


def power_loss(s: Sitl):
    s.wait_ready(); s.takeoff(15)
    s.goto_offset(20, 0); s.drain(10)
    s.say("sagging battery, then cutting power")
    for v in (11.8, 11.0, 10.4, 9.9):
        s.param("SIM_BATT_VOLTAGE", v)
        s.drain(3)
    return "kill"


SCENARIOS = {
    "healthy": healthy,
    "motor_failure": motor_failure,
    "high_vibration": high_vibration,
    "compass_interference": compass_interference,
    "gps_glitch": gps_glitch,
    "radio_failsafe": radio_failsafe,
    "power_loss": power_loss,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ardupilot", required=True, help="ArduPilot checkout with a SITL build")
    ap.add_argument("--out", default="sitl_logs")
    ap.add_argument("--speedup", type=int, default=4)
    ap.add_argument("only", nargs="*", help="scenario names (default: all)")
    args = ap.parse_args()

    names = args.only or list(SCENARIOS)
    os.makedirs(args.out, exist_ok=True)
    for name in names:
        print(f"== {name}", flush=True)
        work = os.path.join("/tmp", f"sitl_{name}")
        shutil.rmtree(work, ignore_errors=True)
        s = Sitl(args.ardupilot, work, args.speedup)
        try:
            result = SCENARIOS[name](s)
            s.stop(hard=(result == "kill"))
        except Exception as exc:  # keep going with the next scenario
            print(f"  failed: {exc}", file=sys.stderr)
            s.stop(hard=True)
        log = s.newest_log()
        if log:
            dest = os.path.join(args.out, f"{name}.bin")
            shutil.copy(log, dest)
            print(f"  saved {dest} ({os.path.getsize(dest) // 1024} KiB)")
        else:
            print("  no log produced", file=sys.stderr)


if __name__ == "__main__":
    main()
