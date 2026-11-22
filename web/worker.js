// Runs flightdoc inside Pyodide, off the main thread so the page stays responsive.

const DEFAULT_PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/";
let ready = null;

function say(stage, text) {
  self.postMessage({ type: "progress", stage, text });
}

async function boot(wheels, pyodideUrl) {
  const base = pyodideUrl || DEFAULT_PYODIDE;
  say("load", "Loading Python in your browser (about 30 MB the first time, cached after that)");
  // recent Pyodide releases only run in module workers
  const { loadPyodide } = await import(base + "pyodide.mjs");
  const py = await loadPyodide({ indexURL: base });
  say("load", "Loading numpy, pandas, scipy and matplotlib");
  await py.loadPackage(["numpy", "pandas", "scipy", "matplotlib", "micropip"]);
  say("load", "Installing flightdoc");
  py.globals.set("wheel_urls", py.toPy(wheels));
  await py.runPythonAsync(`
import micropip
# flightdoc and pyulog are served next to the page, so nothing comes from PyPI
await micropip.install(list(wheel_urls), deps=False)
import flightdoc
from flightdoc import analyze
from flightdoc.report import render
`);
  return py;
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      ready = boot(msg.wheels, msg.pyodideUrl);
      await ready;
      self.postMessage({ type: "ready" });
      return;
    }
    if (msg.type === "analyze") {
      const py = await ready;
      say("run", `Reading ${msg.name}`);
      // keep the original name so the report shows it; strip anything path-like
      const safe = msg.name.replace(/[\\/]/g, "_");
      const dir = "/tmp/in";
      try { py.FS.mkdir(dir); } catch (e) { /* exists */ }
      const path = `${dir}/${safe}`;
      py.FS.writeFile(path, new Uint8Array(msg.bytes));
      say("run", "Decoding the log and running the checks (a big blackbox log can take a minute)");
      py.globals.set("log_path", path);
      const out = await py.runPythonAsync(`
import json
diags = analyze(log_path)
summary = [{"verdict": d.verdict,
            "session": d.flight.session,
            "firmware": d.flight.firmware_version or d.flight.firmware,
            "duration": round(d.flight.duration, 1),
            "incident": d.incident is not None,
            "causes": [{"title": c.title, "severity": c.severity} for c in d.causes[:5]]}
           for d in diags]
json.dumps({"html": render(diags), "summary": summary})
`);
      py.FS.unlink(path);
      self.postMessage({ type: "result", name: msg.name, ...JSON.parse(out) });
    }
  } catch (err) {
    const text = String(err && err.message ? err.message : err);
    // Python tracebacks are long; the last line is the useful one
    const lines = text.trim().split("\n");
    self.postMessage({ type: "error", text: lines[lines.length - 1], detail: text });
  }
};
