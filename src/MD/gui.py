"""Web-based GUI for configuring alloy simulation parameters.

Starts a local HTTP server, opens the browser, and writes config.json on save.
"""

import json
import os
import threading
import tkinter as tk
from tkinter import filedialog
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

from element import ELEMENTS
from viz import ELEMENT_COLORS, get_color

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    "composition": {"Cu": 0.5, "Al": 0.5},
    "box_size_m": 5e-9,
    "temperature": 300.0,
    "total_steps": 1000,
    "thermo_interval": 10,
    "dump_interval": 50,
}

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alloy Simulation Config</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    display: flex; justify-content: center; align-items: flex-start;
    min-height: 100vh; padding: 40px 20px;
  }
  .container {
    width: 100%; max-width: 640px;
    background: #16213e; border: 1px solid #0f3460;
    border-radius: 12px; padding: 32px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  h1 {
    font-size: 1.4rem; font-weight: 600; text-align: center;
    color: #e94560; margin-bottom: 24px; letter-spacing: 0.5px;
  }
  .section {
    background: #1a1a2e; border: 1px solid #0f3460;
    border-radius: 8px; padding: 20px; margin-bottom: 20px;
  }
  .section-title {
    font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
    color: #e94560; letter-spacing: 1px; margin-bottom: 16px;
  }
  /* Element rows */
  .element-row {
    display: grid; grid-template-columns: 50px 24px 1fr 120px 20px auto;
    align-items: center; padding: 6px 0; gap: 0 8px;
    border-bottom: 1px solid rgba(15,52,96,0.5);
  }
  .pct-suffix { font-size: 0.85rem; color: #888; }
  .fill-btn {
    padding: 4px 8px; font-size: 0.75rem; font-weight: 600;
    background: #0f3460; color: #c0c0c0; border: 1px solid #1a4a8a;
    border-radius: 4px; cursor: pointer; white-space: nowrap;
    transition: background 0.2s, color 0.2s;
  }
  .fill-btn:hover { background: #1a4a8a; color: #fff; }
  .fill-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .element-row:last-of-type { border-bottom: none; }
  .color-swatch {
    width: 16px; height: 16px; border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.15);
  }
  .element-label {
    font-weight: 600; font-size: 0.95rem;
  }
  /* Toggle switch */
  .toggle { position: relative; width: 40px; height: 22px; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle .slider {
    position: absolute; inset: 0; background: #333;
    border-radius: 22px; cursor: pointer; transition: background 0.2s;
  }
  .toggle .slider::before {
    content: ''; position: absolute; width: 16px; height: 16px;
    left: 3px; bottom: 3px; background: #888;
    border-radius: 50%; transition: transform 0.2s, background 0.2s;
  }
  .toggle input:checked + .slider { background: #0f3460; }
  .toggle input:checked + .slider::before {
    transform: translateX(18px);
    background: var(--el-color, #e94560);
  }
  /* Fraction input */
  .fraction-input {
    width: 100%; padding: 6px 10px; background: #0d1b2a;
    border: 1px solid #0f3460; border-radius: 6px;
    color: #e0e0e0; font-size: 0.9rem; text-align: right;
    transition: border-color 0.2s;
  }
  .fraction-input:focus { outline: none; border-color: #e94560; }
  .fraction-input:disabled { opacity: 0.3; cursor: not-allowed; }
  /* Total indicator */
  .total-bar {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 12px; padding: 10px 14px;
    background: #0d1b2a; border-radius: 6px;
    font-weight: 600; font-size: 0.95rem;
  }
  .total-bar.valid { border: 1px solid #2ecc71; color: #2ecc71; }
  .total-bar.invalid { border: 1px solid #e94560; color: #e94560; }
  /* Param fields */
  .param-row {
    display: grid; grid-template-columns: 1fr 160px;
    align-items: center; padding: 6px 0;
  }
  .param-label { font-size: 0.9rem; color: #c0c0c0; }
  .param-input {
    width: 100%; padding: 6px 10px; background: #0d1b2a;
    border: 1px solid #0f3460; border-radius: 6px;
    color: #e0e0e0; font-size: 0.9rem; text-align: right;
    transition: border-color 0.2s;
  }
  .param-input:focus { outline: none; border-color: #e94560; }
  /* 3D Cube */
  .cube-section { display: flex; justify-content: center; padding: 20px 0; }
  .scene {
    width: 140px; height: 140px;
    perspective: 500px;
  }
  .cube {
    width: 100%; height: 100%;
    position: relative;
    transform-style: preserve-3d;
    animation: spin 12s linear infinite;
  }
  @keyframes spin {
    from { transform: rotateX(-25deg) rotateY(0deg); }
    to   { transform: rotateX(-25deg) rotateY(360deg); }
  }
  .cube-face {
    position: absolute; width: 140px; height: 140px;
    overflow: hidden; backface-visibility: visible;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .cube-face.front  { transform: translateZ(70px); }
  .cube-face.back   { transform: rotateY(180deg) translateZ(70px); }
  .cube-face.left   { transform: rotateY(-90deg) translateZ(70px); }
  .cube-face.right  { transform: rotateY(90deg)  translateZ(70px); }
  .cube-face.top    { transform: rotateX(90deg)  translateZ(70px); }
  .cube-face.bottom { transform: rotateX(-90deg) translateZ(70px); }
  .cube-band { width: 100%; }
  /* Save button */
  .save-btn {
    display: block; width: 100%; padding: 14px;
    background: linear-gradient(135deg, #e94560, #c23152);
    border: none; border-radius: 8px; color: #fff;
    font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    letter-spacing: 0.5px;
  }
  .save-btn:hover { opacity: 0.9; }
  .save-btn:active { transform: scale(0.98); }
  .save-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  /* Feedback */
  .feedback {
    text-align: center; margin-top: 12px; font-size: 0.9rem;
    min-height: 1.2em;
  }
  .feedback.success { color: #2ecc71; }
  .feedback.error { color: #e94560; }
</style>
</head>
<body>
<div class="container">
  <h1>Alloy Simulation Configuration</h1>

  <div class="section">
    <div class="section-title">Composition</div>
    <div id="elements"></div>
    <div class="total-bar invalid" id="totalBar">
      <span>Total</span><span id="totalValue">0.0000</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Composition Preview</div>
    <div class="cube-section">
      <div class="scene">
        <div class="cube" id="cube">
          <div class="cube-face front"></div>
          <div class="cube-face back"></div>
          <div class="cube-face left"></div>
          <div class="cube-face right"></div>
          <div class="cube-face top"></div>
          <div class="cube-face bottom"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Simulation Parameters</div>
    <div class="param-row">
      <span class="param-label">Box size (m)</span>
      <input class="param-input" id="box_size_m" type="text" value="5e-9">
    </div>
    <div class="param-row">
      <span class="param-label">Temperature (K)</span>
      <input class="param-input" id="temperature" type="text" value="300">
    </div>
    <div class="param-row">
      <span class="param-label">Total steps</span>
      <input class="param-input" id="total_steps" type="text" value="1000">
    </div>
    <div class="param-row">
      <span class="param-label">Thermo interval</span>
      <input class="param-input" id="thermo_interval" type="text" value="10">
    </div>
    <div class="param-row">
      <span class="param-label">Dump interval</span>
      <input class="param-input" id="dump_interval" type="text" value="50">
    </div>
  </div>

  <button class="save-btn" id="saveBtn">Save As...</button>
  <div class="feedback" id="feedback"></div>
</div>

<script>
let ELEMENTS = [];
let ELEMENT_COLORS = {};

const elContainer = document.getElementById('elements');
const totalBar = document.getElementById('totalBar');
const totalValue = document.getElementById('totalValue');
const feedback = document.getElementById('feedback');

function buildElementRows() {
  elContainer.innerHTML = '';
  ELEMENTS.forEach(sym => {
    const row = document.createElement('div');
    row.className = 'element-row';
    const color = ELEMENT_COLORS[sym] || '#888';
    row.innerHTML = `
      <label class="toggle">
        <input type="checkbox" data-el="${sym}">
        <span class="slider"></span>
      </label>
      <span class="color-swatch" style="background:${color}"></span>
      <span class="element-label" style="color:${color}">${sym}</span>
      <input class="fraction-input" data-el-frac="${sym}" type="text" value="0" disabled>
      <span class="pct-suffix">%</span>
      <button class="fill-btn" data-fill="${sym}" disabled>Fill</button>
    `;
    elContainer.appendChild(row);

    const cb = row.querySelector('input[type=checkbox]');
    const frac = row.querySelector('.fraction-input');
    const fillBtn = row.querySelector('.fill-btn');
    const sliderDot = row.querySelector('.slider');
    cb.addEventListener('change', () => {
      frac.disabled = !cb.checked;
      fillBtn.disabled = !cb.checked;
      sliderDot.style.setProperty('--el-color', cb.checked ? color : '');
      updateTotal();
    });
    frac.addEventListener('input', updateTotal);
    fillBtn.addEventListener('click', () => fillRest(sym));
  });
}

function updateTotal() {
  let total = 0;
  ELEMENTS.forEach(sym => {
    const cb = document.querySelector(`input[data-el="${sym}"]`);
    const frac = document.querySelector(`input[data-el-frac="${sym}"]`);
    if (cb && cb.checked) {
      const v = parseFloat(frac.value);
      if (!isNaN(v)) total += v;
    }
  });
  totalValue.textContent = total.toFixed(2) + '%';
  const valid = Math.abs(total - 100) < 1e-6;
  totalBar.className = 'total-bar ' + (valid ? 'valid' : 'invalid');
  updateCube();
}

// Load existing config
fetch('/config').then(r => r.json()).then(cfg => {
  ELEMENTS = cfg.available_elements || [];
  ELEMENT_COLORS = cfg.colors || {};
  buildElementRows();

  if (cfg.composition) {
    Object.entries(cfg.composition).forEach(([sym, val]) => {
      const cb = document.querySelector(`input[data-el="${sym}"]`);
      const frac = document.querySelector(`input[data-el-frac="${sym}"]`);
      if (cb && frac) {
        cb.checked = true;
        frac.disabled = false;
        frac.value = val * 100;
        const slider = cb.closest('.element-row').querySelector('.slider');
        slider.style.setProperty('--el-color', ELEMENT_COLORS[sym] || '#888');
      }
    });
  }
  if (cfg.box_size_m !== undefined) document.getElementById('box_size_m').value = cfg.box_size_m;
  if (cfg.temperature !== undefined) document.getElementById('temperature').value = cfg.temperature;
  if (cfg.total_steps !== undefined) document.getElementById('total_steps').value = cfg.total_steps;
  if (cfg.thermo_interval !== undefined) document.getElementById('thermo_interval').value = cfg.thermo_interval;
  if (cfg.dump_interval !== undefined) document.getElementById('dump_interval').value = cfg.dump_interval;
  updateTotal();
}).catch(() => {});

// Save
document.getElementById('saveBtn').addEventListener('click', () => {
  const composition = {};
  let total = 0;
  ELEMENTS.forEach(sym => {
    const cb = document.querySelector(`input[data-el="${sym}"]`);
    const frac = document.querySelector(`input[data-el-frac="${sym}"]`);
    if (cb.checked) {
      const v = parseFloat(frac.value);
      if (isNaN(v)) { showFeedback(`Invalid percentage for ${sym}`, true); return; }
      if (v > 0) composition[sym] = v / 100;
      total += v;
    }
  });

  if (Object.keys(composition).length === 0) {
    showFeedback('Select at least one element', true); return;
  }
  if (Math.abs(total - 100) > 1e-6) {
    showFeedback(`Percentages sum to ${total.toFixed(2)}%, must equal 100%`, true); return;
  }

  const params = ['box_size_m','temperature','total_steps','thermo_interval','dump_interval'];
  const config = { composition };
  for (const key of params) {
    const v = parseFloat(document.getElementById(key).value);
    if (isNaN(v)) { showFeedback(`Invalid value for ${key}`, true); return; }
    if (['total_steps','thermo_interval','dump_interval'].includes(key)) {
      config[key] = Math.round(v);
    } else {
      config[key] = v;
    }
  }

  const btn = document.getElementById('saveBtn');
  btn.disabled = true;
  btn.textContent = 'Saving...';

  fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(config)
  }).then(r => {
    if (!r.ok) return r.json().then(d => { throw new Error(d.error || 'Save failed'); });
    return r.json();
  }).then(d => {
    if (d.status === 'cancelled') {
      showFeedback('Save cancelled — pick a file to save.', true);
      btn.disabled = false;
      btn.textContent = 'Save As...';
    } else {
      showFeedback('Configuration saved!', false);
      btn.disabled = false;
      btn.textContent = 'Save As...';
    }
  }).catch(err => {
    showFeedback(err.message, true);
    btn.disabled = false;
    btn.textContent = 'Save As...';
  });
});

function fillRest(targetSym) {
  let otherTotal = 0;
  ELEMENTS.forEach(sym => {
    if (sym === targetSym) return;
    const cb = document.querySelector(`input[data-el="${sym}"]`);
    const frac = document.querySelector(`input[data-el-frac="${sym}"]`);
    if (cb && cb.checked) {
      const v = parseFloat(frac.value);
      if (!isNaN(v)) otherTotal += v;
    }
  });
  const remainder = Math.max(0, 100 - otherTotal);
  const frac = document.querySelector(`input[data-el-frac="${targetSym}"]`);
  frac.value = parseFloat(remainder.toFixed(4));
  updateTotal();
}

function updateCube() {
  // Collect enabled elements and their percentage values
  const bands = [];
  ELEMENTS.forEach(sym => {
    const cb = document.querySelector(`input[data-el="${sym}"]`);
    const frac = document.querySelector(`input[data-el-frac="${sym}"]`);
    if (cb && cb.checked) {
      const v = parseFloat(frac.value);
      if (!isNaN(v) && v > 0) bands.push({ color: ELEMENT_COLORS[sym] || '#888', pct: v });
    }
  });
  // Normalize to fill the face if total != 100
  const sum = bands.reduce((s, b) => s + b.pct, 0) || 1;
  // Build HTML bands for each face
  let bandsHTML = '';
  bands.forEach(b => {
    const h = (b.pct / sum) * 140;
    bandsHTML += `<div class="cube-band" style="height:${h}px;background:${b.color}"></div>`;
  });
  if (bands.length === 0) {
    bandsHTML = `<div class="cube-band" style="height:140px;background:#0d1b2a"></div>`;
  }
  document.querySelectorAll('.cube-face').forEach(face => { face.innerHTML = bandsHTML; });
}

function showFeedback(msg, isError) {
  feedback.textContent = msg;
  feedback.className = 'feedback ' + (isError ? 'error' : 'success');
}
</script>
</body>
</html>"""


def load_config():
    """Load existing config.json if present, else return defaults."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULTS.copy()


class Handler(BaseHTTPRequestHandler):
    """Request handler for the configuration GUI."""

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass

    def do_GET(self):
        if self.path == "/":
            self._respond(200, "text/html", HTML_PAGE.encode())
        elif self.path == "/config":
            config = load_config()
            # Dynamic elements from EAM directory
            available = sorted(ELEMENTS.keys())
            colors = {}
            for sym in available:
                rgb = get_color(sym)
                colors[sym] = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"

            config["available_elements"] = available
            config["colors"] = colors
            data = json.dumps(config).encode()
            self._respond(200, "application/json", data)
        else:
            self._respond(404, "text/plain", b"Not Found")

    def do_POST(self):
        if self.path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                config = json.loads(body)
                # Open tkinter Save As dialog
                root = tk.Tk()
                root.withdraw()
                root.lift()
                root.attributes("-topmost", True)
                save_path = filedialog.asksaveasfilename(
                    parent=root,
                    title="Save Configuration As",
                    initialdir=os.path.dirname(CONFIG_PATH),
                    initialfile="config.json",
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                )
                root.destroy()
                if not save_path:
                    self._respond(200, "application/json", b'{"status":"cancelled"}')
                    return
                with open(save_path, "w") as f:
                    json.dump(config, f, indent=4)
                self._respond(200, "application/json", b'{"status":"ok"}')
            except (json.JSONDecodeError, OSError) as exc:
                msg = json.dumps({"error": str(exc)}).encode()
                self._respond(500, "application/json", msg)
        else:
            self._respond(404, "text/plain", b"Not Found")

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = 8471
    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Opening configuration GUI at {url}")
    webbrowser.open(url)
    server.serve_forever()
    print("Configuration saved. Continuing...")


if __name__ == "__main__":
    main()
