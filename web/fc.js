// Talk to a flight controller over a byte stream and pull its logs off.
//
// Betaflight, INAV and their forks speak MSP on USB; the blackbox log lives
// on an onboard flash chip and is read with MSP_DATAFLASH_READ.
// ArduPilot and PX4 speak MAVLink; logs are fetched with the LOG_REQUEST_*
// messages, which both autopilots implement.
//
// The transport is anything with `write(Uint8Array)` and a way to deliver
// incoming bytes to `link.feed(Uint8Array)`: Web Serial in the browser, a TCP
// socket in the tests.

const MSP = {
  API_VERSION: 1, FC_VARIANT: 2, FC_VERSION: 3, BOARD_INFO: 4,
  DATAFLASH_SUMMARY: 70, DATAFLASH_READ: 71, SDCARD_SUMMARY: 79, REBOOT: 68,
};
const JUMBO = 255;
const READ_BLOCK = 4096;          // both firmwares size their MSP buffer for this

const MAV = {
  HEARTBEAT: 0, COMMAND_LONG: 76, LOG_REQUEST_LIST: 117, LOG_ENTRY: 118,
  LOG_REQUEST_DATA: 119, LOG_DATA: 120, LOG_REQUEST_END: 122,
  AUTOPILOT_VERSION: 148, STATUSTEXT: 253,
};
// CRC seeds from the MAVLink message definitions (pymavlink common dialect)
const CRC_EXTRA = { 0: 50, 76: 152, 117: 128, 118: 56, 119: 116, 120: 134, 122: 203, 148: 178, 253: 83 };
const MAV_LEN = { 0: 9, 76: 33, 117: 6, 118: 14, 119: 12, 120: 97, 122: 2, 148: 78, 253: 54 };
const AUTOPILOT = { 3: "ardupilot", 12: "px4" };
const MAV_TYPE = { 1: "plane", 2: "quad", 4: "helicopter", 10: "rover", 12: "sub", 13: "hexa", 14: "octo", 15: "tricopter", 19: "vtol", 20: "vtol", 21: "vtol", 22: "vtol" };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const ascii = (bytes) => String.fromCharCode(...bytes).replace(/\0+$/, "");

function x25(crc, byte) {
  let t = byte ^ (crc & 0xff);
  t = (t ^ (t << 4)) & 0xff;
  return ((crc >> 8) ^ (t << 8) ^ (t << 3) ^ (t >> 4)) & 0xffff;
}

export class Link {
  constructor(write) {
    this.write = write;
    this.buf = new Uint8Array(0);
    this.mspWaiters = [];
    this.mavListeners = new Set();
    this.seq = 0;
  }

  feed(chunk) {
    const merged = new Uint8Array(this.buf.length + chunk.length);
    merged.set(this.buf);
    merged.set(chunk, this.buf.length);
    this.buf = merged;
    this._parse();
  }

  // ---------------------------------------------------------------- parsing
  _parse() {
    let i = 0;
    const b = this.buf;
    while (i < b.length) {
      const c = b[i];
      if (c === 0x24) {                       // '$' MSP v1
        if (b.length - i < 5) break;
        if (b[i + 1] !== 0x4d || (b[i + 2] !== 0x3e && b[i + 2] !== 0x21)) { i++; continue; }
        const isError = b[i + 2] === 0x21;
        let size = b[i + 3];
        const cmd = b[i + 4];
        let hdr = 5;
        if (size === JUMBO) {
          if (b.length - i < 7) break;
          size = b[i + 5] | (b[i + 6] << 8);
          hdr = 7;
        }
        if (b.length - i < hdr + size + 1) break;
        let crc = 0;
        for (let k = i + 3; k < i + hdr + size; k++) crc ^= b[k];
        if (crc !== b[i + hdr + size]) { i++; continue; }
        this._mspDone(cmd, b.slice(i + hdr, i + hdr + size), isError);
        i += hdr + size + 1;
      } else if (c === 0xfd || c === 0xfe) {  // MAVLink v2 / v1
        const v2 = c === 0xfd;
        const hdr = v2 ? 10 : 6;
        if (b.length - i < hdr) break;
        const len = b[i + 1];
        const signed = v2 && (b[i + 2] & 1);
        const total = hdr + len + 2 + (signed ? 13 : 0);
        if (b.length - i < total) break;
        const msgid = v2 ? (b[i + 7] | (b[i + 8] << 8) | (b[i + 9] << 16)) : b[i + 5];
        const extra = CRC_EXTRA[msgid];
        if (extra === undefined) { i += total; continue; }   // a message we do not use
        let crc = 0xffff;
        for (let k = i + 1; k < i + hdr + len; k++) crc = x25(crc, b[k]);
        crc = x25(crc, extra);
        const got = b[i + hdr + len] | (b[i + hdr + len + 1] << 8);
        if (crc !== got) { i++; continue; }
        // MAVLink 2 trims trailing zero bytes; put them back
        const payload = new Uint8Array(Math.max(len, MAV_LEN[msgid] || len));
        payload.set(b.slice(i + hdr, i + hdr + len));
        const sys = v2 ? b[i + 5] : b[i + 3];
        const comp = v2 ? b[i + 6] : b[i + 4];
        for (const fn of [...this.mavListeners]) fn(msgid, new DataView(payload.buffer), sys, comp);
        i += total;
      } else {
        i++;
      }
    }
    this.buf = this.buf.slice(i);
  }

  // ------------------------------------------------------------------- MSP
  _mspDone(cmd, payload, isError) {
    const idx = this.mspWaiters.findIndex((w) => w.cmd === cmd);
    if (idx < 0) return;
    const [w] = this.mspWaiters.splice(idx, 1);
    clearTimeout(w.timer);
    if (isError) w.reject(new Error(`flight controller rejected MSP command ${cmd}`));
    else w.resolve(payload);
  }

  msp(cmd, data = [], timeout = 800) {
    const payload = Uint8Array.from(data);
    const frame = new Uint8Array(6 + payload.length);
    frame.set([0x24, 0x4d, 0x3c, payload.length, cmd]);
    frame.set(payload, 5);
    let crc = payload.length ^ cmd;
    for (const x of payload) crc ^= x;
    frame[frame.length - 1] = crc;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.mspWaiters = this.mspWaiters.filter((w) => w.timer !== timer);
        reject(new Error(`no reply to MSP command ${cmd}`));
      }, timeout);
      this.mspWaiters.push({ cmd, resolve, reject, timer });
      this.write(frame);
    });
  }

  // --------------------------------------------------------------- MAVLink
  mav(msgid, payload, sys = 255, comp = 190) {
    const len = payload.length;
    const frame = new Uint8Array(10 + len + 2);
    frame.set([0xfd, len, 0, 0, this.seq++ & 0xff, sys, comp, msgid & 0xff, (msgid >> 8) & 0xff, msgid >> 16]);
    frame.set(payload, 10);
    let crc = 0xffff;
    for (let k = 1; k < 10 + len; k++) crc = x25(crc, frame[k]);
    crc = x25(crc, CRC_EXTRA[msgid]);
    frame[10 + len] = crc & 0xff;
    frame[11 + len] = crc >> 8;
    this.write(frame);
  }

  onMav(fn) {
    this.mavListeners.add(fn);
    return () => this.mavListeners.delete(fn);
  }

  waitMav(msgid, timeout, filter = () => true) {
    return new Promise((resolve, reject) => {
      const off = this.onMav((id, dv, sys, comp) => {
        if (id === msgid && filter(dv, sys, comp)) { off(); clearTimeout(t); resolve({ dv, sys, comp }); }
      });
      const t = setTimeout(() => { off(); reject(new Error(`no MAVLink message ${msgid}`)); }, timeout);
    });
  }
}

// ===================================================================== detect
export async function detect(link, say = () => {}) {
  say("Looking for MSP (Betaflight / INAV) or MAVLink (ArduPilot / PX4)");
  // skip other ground stations: MAV_TYPE_GCS (6) or MAV_AUTOPILOT_INVALID (8)
  const heartbeat = link.waitMav(MAV.HEARTBEAT, 3000, (dv) => dv.getUint8(4) !== 6 && dv.getUint8(5) !== 8)
    .catch(() => null);
  let api = null;
  for (let attempt = 0; attempt < 3 && !api; attempt++) {
    api = await link.msp(MSP.API_VERSION, [], 700).catch(() => null);
  }
  if (api) {
    const variant = ascii(await link.msp(MSP.FC_VARIANT));
    const v = await link.msp(MSP.FC_VERSION);
    const board = await link.msp(MSP.BOARD_INFO).catch(() => null);
    const names = { BTFL: "Betaflight", INAV: "INAV", EMUF: "EmuFlight", RTFL: "Rotorflight", CLFL: "Cleanflight", QUIC: "QuickSilver" };
    return {
      protocol: "msp",
      variant,
      firmware: names[variant] || variant,
      version: `${v[0]}.${v[1]}.${v[2]}`,
      board: board ? ascii(board.slice(0, 4)) : "",
      api: `${api[1]}.${api[2]}`,
    };
  }
  const hb = await heartbeat;
  if (!hb) throw new Error("No flight controller answered. Check the cable, close the configurator / ground station, and try again.");
  const autopilot = AUTOPILOT[hb.dv.getUint8(5)] || `autopilot ${hb.dv.getUint8(5)}`;
  const info = {
    protocol: "mavlink", firmware: autopilot === "ardupilot" ? "ArduPilot" : autopilot === "px4" ? "PX4" : autopilot,
    autopilot, vehicle: MAV_TYPE[hb.dv.getUint8(4)] || "", sys: hb.sys, comp: hb.comp, version: "",
  };
  // ask for AUTOPILOT_VERSION (MAV_CMD_REQUEST_MESSAGE = 512)
  const cmd = new DataView(new ArrayBuffer(33));
  cmd.setFloat32(0, 148, true);
  cmd.setUint16(28, 512, true);
  cmd.setUint8(30, info.sys);
  cmd.setUint8(31, info.comp);
  const ver = link.waitMav(MAV.AUTOPILOT_VERSION, 1500).catch(() => null);
  link.mav(MAV.COMMAND_LONG, new Uint8Array(cmd.buffer));
  const v = await ver;
  if (v) {
    const sw = v.dv.getUint32(16, true);
    info.version = `${sw >>> 24}.${(sw >> 16) & 0xff}.${(sw >> 8) & 0xff}`;
  }
  return info;
}

// =================================================================== MSP logs
export async function mspStorage(link, info) {
  const s = new DataView((await link.msp(MSP.DATAFLASH_SUMMARY)).buffer);
  const flags = s.getUint8(0);
  // INAV reports only "ready"; Betaflight reports bit 1 "supported" too
  const total = s.byteLength >= 13 ? s.getUint32(5, true) : 0;
  const used = s.byteLength >= 13 ? s.getUint32(9, true) : 0;
  const flash = { ready: (flags & 1) === 1, total, used };
  let sd = null;
  try {
    const d = await link.msp(MSP.SDCARD_SUMMARY);
    if (d.length >= 3 && (d[0] & 1)) sd = { state: d[1] };
  } catch (e) { /* older firmware */ }
  return { flash, sd, hasFlash: total > 0 };
}

export async function mspDownload(link, info, used, progress = () => {}) {
  const out = new Uint8Array(used);
  const inav = info.variant === "INAV";
  let block = READ_BLOCK;
  let addr = 0;
  let failures = 0;      // consecutive failures at the current address
  let good = 0;          // consecutive successes, to grow the block back
  const started = Date.now();
  while (addr < used) {
    const want = Math.min(block, used - addr);
    // INAV honours the length only when the request is exactly 6 bytes;
    // Betaflight takes a 7th byte to turn compression off.
    const req = [addr & 0xff, (addr >> 8) & 0xff, (addr >> 16) & 0xff, addr >>> 24, want & 0xff, want >> 8];
    if (!inav) req.push(0);
    let reply;
    try {
      reply = await link.msp(MSP.DATAFLASH_READ, req, 1000);
    } catch (e) {
      if (++failures > 8) throw new Error("the flight controller stopped answering flash reads");
      // a single lost reply is just retried; repeated ones suggest the link
      // cannot carry big frames, so step the block size down
      if (failures >= 2 && block > 256) block = block >> 1;
      good = 0;
      continue;
    }
    const dv = new DataView(reply.buffer, reply.byteOffset, reply.byteLength);
    const at = dv.getUint32(0, true);
    if (at !== addr) continue;  // stale reply from a retried request
    let data;
    if (inav) {
      data = reply.subarray(4);
    } else {
      const n = dv.getUint16(4, true);
      if (dv.getUint8(6) !== 0) throw new Error("flight controller sent compressed data we did not ask for");
      data = reply.subarray(7, 7 + n);
    }
    if (data.length === 0) break;            // end of the volume
    out.set(data.subarray(0, Math.min(data.length, used - addr)), addr);
    addr += data.length;
    failures = 0;
    if (++good >= 16 && block < READ_BLOCK) { block = block << 1; good = 0; }
    const secs = (Date.now() - started) / 1000;
    progress(addr / used, `${(addr / 1048576).toFixed(1)} of ${(used / 1048576).toFixed(1)} MB` +
             (secs > 1 ? `, ${(addr / 1024 / secs).toFixed(0)} KB/s` : ""));
  }
  return out.subarray(0, Math.min(addr, used));
}

// Betaflight can reboot into USB mass storage so the SD card shows up as a drive.
export function mspRebootToMassStorage(link) {
  return link.msp(MSP.REBOOT, [2], 500).catch(() => null);
}

// =============================================================== MAVLink logs
export async function mavListLogs(link, info, say = () => {}, timeout = 12000) {
  // ArduPilot ignores log requests until its logger is up, which is a few
  // seconds after boot (exactly when people plug in USB), so keep asking.
  const logs = new Map();
  let expected = null;
  return new Promise((resolve, reject) => {
    let quiet, retry;
    const started = Date.now();
    const finish = (err) => {
      offLog(); offText(); clearTimeout(quiet); clearInterval(retry);
      err ? reject(err) : resolve([...logs.values()].sort((a, b) => a.id - b.id));
    };
    const offText = link.onMav((id, dv) => {
      if (id !== MAV.STATUSTEXT) return;
      const text = ascii(new Uint8Array(dv.buffer, 1, 50));
      if (/disarm for log download/i.test(text)) finish(new Error("Disarm the vehicle to download logs."));
      else if (/log download in progress/i.test(text)) say("The autopilot is still busy with an earlier download; retrying");
    });
    const offLog = link.onMav((id, dv) => {
      if (id !== MAV.LOG_ENTRY) return;
      clearInterval(retry);
      const e = { time: dv.getUint32(0, true), size: dv.getUint32(4, true), id: dv.getUint16(8, true) };
      expected = dv.getUint16(10, true);
      if (expected === 0) return finish();
      if (e.size > 0) logs.set(e.id, e);
      if (logs.size >= expected) return finish();
      clearTimeout(quiet);
      quiet = setTimeout(() => finish(), 1500);  // missing entries: settle for what came
    });
    const ask = () => {
      if (Date.now() - started > timeout) return finish(new Error("The autopilot did not answer the log list request."));
      const p = new DataView(new ArrayBuffer(6));
      p.setUint16(0, 0, true);
      p.setUint16(2, 0xffff, true);
      p.setUint8(4, info.sys);
      p.setUint8(5, info.comp);
      link.mav(MAV.LOG_REQUEST_LIST, new Uint8Array(p.buffer));
    };
    ask();
    retry = setInterval(ask, 2000);
  });
}

export async function mavDownload(link, info, entry, progress = () => {}) {
  const size = entry.size;
  const out = new Uint8Array(size);
  const chunks = Math.ceil(size / 90);
  const have = new Uint8Array(chunks);
  let got = 0;
  const started = Date.now();
  let lastData = Date.now();

  const request = (ofs, count) => {
    const p = new DataView(new ArrayBuffer(12));
    p.setUint32(0, ofs, true);
    p.setUint32(4, count, true);
    p.setUint16(8, entry.id, true);
    p.setUint8(10, info.sys);
    p.setUint8(11, info.comp);
    link.mav(MAV.LOG_REQUEST_DATA, new Uint8Array(p.buffer));
  };
  const off = link.onMav((id, dv) => {
    if (id !== MAV.LOG_DATA || dv.getUint16(4, true) !== entry.id) return;
    const ofs = dv.getUint32(0, true);
    const n = dv.getUint8(6);
    const k = Math.floor(ofs / 90);
    if (ofs % 90 !== 0 || k >= chunks || have[k] || n === 0) return;
    out.set(new Uint8Array(dv.buffer, 7, Math.min(n, size - ofs)), ofs);
    have[k] = 1;
    got++;
    lastData = Date.now();
  });

  try {
    request(0, size);
    let stalls = 0;
    while (got < chunks) {
      await sleep(250);
      const secs = (Date.now() - started) / 1000;
      progress(got / chunks, `${(got * 90 / 1048576).toFixed(1)} of ${(size / 1048576).toFixed(1)} MB` +
               (secs > 1 ? `, ${(got * 90 / 1024 / secs).toFixed(0)} KB/s` : ""));
      if (Date.now() - lastData < 700) continue;
      // the stream went quiet with gaps left: ask again for the first hole
      if (++stalls > 40) throw new Error("the autopilot stopped sending log data");
      let first = have.indexOf(0);
      if (first < 0) break;
      let last = first;
      while (last < chunks && !have[last] && last - first < 1000) last++;
      request(first * 90, Math.min((last - first) * 90, size - first * 90));
      lastData = Date.now();
    }
  } finally {
    off();
    const p = new Uint8Array([info.sys, info.comp]);
    link.mav(MAV.LOG_REQUEST_END, p);
  }
  return out;
}
