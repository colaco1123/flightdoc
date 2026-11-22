#!/usr/bin/env python3
"""A fake Betaflight / INAV flight controller that serves a blackbox file over MSP.

Used to test the web page's USB download code without hardware. The reply
formats follow the firmware source:

  Betaflight src/main/msp/msp.c  mspFcDataFlashReadCommand,
                                 serializeDataflashReadReply,
                                 serializeDataflashSummaryReply
  INAV       src/main/fc/fc_msp.c  (same function names)
  both       src/main/msp/msp_serial.c  mspSerialEncode (jumbo frames)

    python tools/msp_emulator.py --firmware inav --port 5762 LOG00004.TXT

Options let it split writes and drop replies so the client's buffering and
retry logic get exercised.
"""

import argparse
import random
import socket
import struct
import threading

OUTBUF = 4096 + 16            # MSP_PORT_DATAFLASH_BUFFER_SIZE + MSP_PORT_DATAFLASH_INFO_SIZE
INFO = 16                     # MSP_PORT_DATAFLASH_INFO_SIZE
FLASH_SIZE = 16 * 1024 * 1024


class Fc:
    def __init__(self, firmware, data, drop):
        self.inav = firmware == "inav"
        self.flash = bytearray(b"\xff" * FLASH_SIZE)
        self.flash[: len(data)] = data
        self.used = len(data)
        self.drop = drop

    def reply(self, cmd, payload):
        if cmd == 1:      # MSP_API_VERSION
            return bytes([0, 2, 5])
        if cmd == 2:      # MSP_FC_VARIANT
            return b"INAV" if self.inav else b"BTFL"
        if cmd == 3:      # MSP_FC_VERSION
            return bytes([9, 0, 1] if self.inav else [4, 5, 1])
        if cmd == 4:      # MSP_BOARD_INFO
            return b"HGF4" + bytes([0, 0, 2])
        if cmd == 70:     # MSP_DATAFLASH_SUMMARY
            flags = 1 if self.inav else 3      # INAV: ready only; BF: supported | ready
            return struct.pack("<BIII", flags, FLASH_SIZE // 65536, FLASH_SIZE, self.used)
        if cmd == 79:     # MSP_SDCARD_SUMMARY: no card
            return struct.pack("<BBBII", 0, 0, 0, 0, 0)
        if cmd == 71:
            return self.read(payload)
        return None

    def read(self, src):
        address = struct.unpack_from("<I", src, 0)[0]
        if self.inav:
            length = struct.unpack_from("<H", src, 4)[0] if len(src) == 6 else 128
            length = min(length, OUTBUF)
            length = min(length, FLASH_SIZE - address)
            return struct.pack("<I", address) + bytes(self.flash[address:address + length])
        legacy = len(src) < 6
        length = 128 if legacy else struct.unpack_from("<H", src, 4)[0]
        allow_compression = (len(src) >= 7 and src[6] != 0)
        if allow_compression:
            raise ValueError("test client asked for Huffman compression")
        read_len = min(length, OUTBUF - INFO)
        read_len = max(0, min(read_len, FLASH_SIZE - address))
        data = bytes(self.flash[address:address + read_len])
        if legacy:
            return struct.pack("<I", address) + data + b"\0" * (length - len(data))
        return struct.pack("<IHB", address, len(data), 0) + data


def encode(cmd, data, error=False):
    head = bytearray(b"$M" + (b"!" if error else b">"))
    if len(data) >= 255:
        body = bytes([255, cmd]) + struct.pack("<H", len(data))
    else:
        body = bytes([len(data), cmd])
    crc = 0
    for b in body + data:
        crc ^= b
    return bytes(head) + body + data + bytes([crc])


def serve(conn, fc, rng):
    buf = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return
        buf += chunk
        while True:
            i = buf.find(b"$M<")
            if i < 0 or len(buf) - i < 6:
                break
            size, cmd = buf[i + 3], buf[i + 4]
            if len(buf) - i < 6 + size:
                break
            payload = bytes(buf[i + 5:i + 5 + size])
            crc = size ^ cmd
            for b in payload:
                crc ^= b
            ok = crc == buf[i + 5 + size]
            del buf[: i + 6 + size]
            if not ok:
                continue
            if cmd == 71 and rng.random() < fc.drop:
                continue  # lost on the wire
            data = fc.reply(cmd, payload)
            frame = encode(cmd, b"", error=True) if data is None else encode(cmd, data)
            # deliver in random pieces like a USB CDC link does
            pos = 0
            while pos < len(frame):
                n = rng.randint(1, 700)
                conn.sendall(frame[pos:pos + n])
                pos += n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--firmware", choices=["betaflight", "inav"], default="betaflight")
    ap.add_argument("--port", type=int, default=5762)
    ap.add_argument("--drop", type=float, default=0.0, help="fraction of flash replies to lose")
    args = ap.parse_args()
    with open(args.log, "rb") as fh:
        fc = Fc(args.firmware, fh.read(), args.drop)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"fake {args.firmware} on 127.0.0.1:{args.port}, {fc.used} bytes of log", flush=True)
    rng = random.Random(7)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn, fc, rng), daemon=True).start()


if __name__ == "__main__":
    main()
