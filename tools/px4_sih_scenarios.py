#!/usr/bin/env python3
"""Fly PX4's headless SIH quad simulator through a few failures and keep the logs.

    python tools/px4_sih_scenarios.py --px4 ~/PX4-Autopilot --out sitl_logs

Needs ``make px4_sitl_default`` to have been run in the PX4 checkout. SIH
(simulation-in-hardware) runs the vehicle physics inside PX4 itself, so no
Gazebo or other simulator is required.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import signal
import subprocess
import time

from pymavlink import mavutil

M = mavutil.mavlink


class Px4:
    def __init__(self, px4_root: str, work: str):
        build = os.path.join(px4_root, "build/px4_sitl_default")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        env = dict(os.environ, PX4_SYS_AUTOSTART="10040", PX4_SIM_MODEL="sihsim_quadx",
                   HEADLESS="1")
        self.work = work
        self.proc = subprocess.Popen(
            [os.path.join(build, "bin/px4"), "-d", os.path.join(build, "etc")],
            cwd=work, env=env, stdin=subprocess.DEVNULL,
            stdout=open(os.path.join(work, "px4.out"), "w"), stderr=subprocess.STDOUT)
        self.m = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
        self.m.wait_heartbeat(timeout=60)
        self.say("connected")

    def say(self, text):
        print(f"  [{time.strftime('%H:%M:%S')}] {text}", flush=True)

    def drain(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            msg = self.m.recv_match(blocking=True, timeout=0.5)
            if msg is not None and msg.get_type() == "STATUSTEXT":
                self.say("FC: " + msg.text)

    def cmd(self, command, *params):
        p = list(params) + [0] * (7 - len(params))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     command, 0, *p)

    def param(self, name, value, int_type=True):
        ptype = M.MAV_PARAM_TYPE_INT32 if int_type else M.MAV_PARAM_TYPE_REAL32
        if int_type:
            import struct
            value = struct.unpack("f", struct.pack("i", int(value)))[0]
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                  name.encode(), value, ptype)
        self.drain(1)

    def armed(self):
        hb = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        return bool(hb and hb.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)

    def alt(self):
        msg = self.m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=3)
        return msg.relative_alt / 1000.0 if msg else float("nan")

    def takeoff(self, alt=10.0):
        self.drain(10)  # let the estimator settle
        self.param("MIS_TAKEOFF_ALT", alt, int_type=False)
        end = time.time() + 120
        nan = float("nan")
        while time.time() < end:
            self.cmd(M.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.drain(1)
            # param 7 is AMSL; NaN means "use MIS_TAKEOFF_ALT above home"
            self.cmd(M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, nan, nan, nan, nan)
            self.drain(2)
            if self.armed():
                break
        while time.time() < end:
            if self.alt() > alt * 0.9:
                self.say(f"at {alt:.0f} m")
                return
        self.say("takeoff timed out")

    def wait_disarm(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            msg = self.m.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.get_type() == "STATUSTEXT":
                self.say("FC: " + msg.text)
            if msg.get_type() == "HEARTBEAT" and msg.type != M.MAV_TYPE_GCS:
                if not msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED:
                    return True
        return False

    def land(self):
        self.cmd(M.MAV_CMD_NAV_LAND)
        if not self.wait_disarm(90):
            self.cmd(M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196)
            self.drain(2)

    def stop(self):
        self.drain(2)
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def newest_log(self):
        logs = sorted(glob.glob(os.path.join(self.work, "**", "*.ulg"), recursive=True),
                      key=os.path.getmtime)
        return logs[-1] if logs else None


def healthy(p: Px4):
    p.takeoff(10)
    p.drain(20)
    p.land()


def gps_loss(p: Px4):
    # SIH ignores injected motor failures (the physics never sees them), but
    # sensor failures go through the normal driver path.
    p.param("SYS_FAILURE_EN", 1)
    p.takeoff(15)
    p.drain(10)
    p.say("GPS off")
    # MAV_CMD_INJECT_FAILURE: unit 4 = GPS, type 1 = off
    p.cmd(420, 4, 1, 0)
    if not p.wait_disarm(90):
        p.land()


def battery_drain(p: Px4):
    p.param("SIM_BAT_MIN_PCT", 0, int_type=False)
    p.param("SIM_BAT_DRAIN", 40, int_type=False)
    p.takeoff(15)
    if not p.wait_disarm(120):
        p.land()


SCENARIOS = {"healthy": healthy, "gps_loss": gps_loss, "battery_drain": battery_drain}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--px4", required=True)
    ap.add_argument("--out", default="sitl_logs")
    ap.add_argument("only", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.only or list(SCENARIOS):
        print(f"== px4 {name}", flush=True)
        p = Px4(args.px4, f"/tmp/px4_{name}")
        try:
            SCENARIOS[name](p)
        finally:
            p.stop()
        log = p.newest_log()
        if log:
            dest = os.path.join(args.out, f"px4_{name}.ulg")
            shutil.copy(log, dest)
            print(f"  saved {dest} ({os.path.getsize(dest) // 1024} KiB)")
        else:
            print("  no log produced")


if __name__ == "__main__":
    main()
