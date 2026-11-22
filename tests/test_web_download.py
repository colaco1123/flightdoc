"""The web page's USB download code (web/fc.js), run under Node against the
MSP emulator in tools/. Skipped when Node is not installed."""

import gzip
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

from conftest import data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("firmware, fixture", [
    ("betaflight", "LOG00037.BFL.gz"),
    ("inav", "inav-LOG00004-first3MB.TXT.gz"),
])
def test_msp_flash_download_is_byte_exact(tmp_path, firmware, fixture):
    log = tmp_path / fixture[:-3]
    log.write_bytes(gzip.open(data("blackbox", fixture)).read())
    port = _free_port()
    emu = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "msp_emulator.py"),
                            "--firmware", firmware, "--port", str(port), "--drop", "0.03", str(log)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        run = subprocess.run(["node", os.path.join(ROOT, "tests", "web", "download.mjs"), str(port), str(log)],
                             capture_output=True, text=True, timeout=600)
        assert run.returncode == 0, run.stdout + run.stderr
        assert "identical: true" in run.stdout
    finally:
        emu.kill()
