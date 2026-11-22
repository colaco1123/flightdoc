// Download a log through web/fc.js over TCP and compare it with a reference file.
//   node tests/web/download.mjs <port> <reference-file> [mavlink-log-index]
import net from "node:net";
import fs from "node:fs";
import { Link, detect, mspStorage, mspDownload, mavListLogs, mavDownload } from "../../web/fc.js";

const [port, reference, which] = process.argv.slice(2);
const sock = net.connect(Number(port), "127.0.0.1");
await new Promise((r, j) => { sock.once("connect", r); sock.once("error", j); });
const link = new Link((bytes) => sock.write(bytes));
sock.on("data", (d) => link.feed(new Uint8Array(d)));

const info = await detect(link, (m) => console.log("..", m));
console.log("detected", JSON.stringify(info));
let data;
const t0 = Date.now();
let last = 0;
const progress = (f, text) => { if (f - last >= 0.25 || f === 1) { console.log(`   ${(f * 100).toFixed(0)}% ${text}`); last = f; } };
if (info.protocol === "msp") {
  const st = await mspStorage(link, info);
  console.log("storage", JSON.stringify(st));
  data = await mspDownload(link, info, st.flash.used, progress);
} else {
  const logs = await mavListLogs(link, info, (m) => console.log("..", m));
  console.log("logs", JSON.stringify(logs));
  const entry = which ? logs.find((l) => l.id === Number(which)) : logs[logs.length - 1];
  data = await mavDownload(link, info, entry, progress);
}
const ref = fs.readFileSync(reference);
const same = Buffer.compare(Buffer.from(data), ref) === 0;
console.log(`got ${data.length} bytes in ${((Date.now() - t0) / 1000).toFixed(1)}s; reference ${ref.length}; identical: ${same}`);
sock.destroy();
process.exit(same ? 0 : 1);
