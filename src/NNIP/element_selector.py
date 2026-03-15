"""Web-based GUI for selecting elements from available pseudopotentials.

Starts a local HTTP server, opens the browser, and returns the list of
selected element symbols when the user clicks 'Continue'.
"""

import json
import os
import re
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

PSEUDO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "pseudopotentials"
)

# Map UPF filenames to element symbols
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)[\._-]")

# Colors for element display (subset matching the alloy GUI palette)
_COLORS = {
    "Al": "#7A9CC6", "Au": "#FFD700", "Cr": "#8B8682", "Cu": "#B87333",
    "Fe": "#A0522D", "Mg": "#9ACD32", "Mn": "#FF69B4", "Mo": "#B0C4DE",
    "Si": "#DAA520", "Ti": "#BDB76B", "Zn": "#7F7F7F", "B": "#FFB266",
    "C": "#808080", "H": "#FFFFFF", "N": "#4169E1", "O": "#FF0000",
    "Pb": "#575961", "Rh": "#E8E8E8", "S": "#FFFF00",
}

# Result container — filled by the server handler, read by select_elements()
_result = {"elements": None, "server": None}


def scan_pseudopotentials(pseudo_dir=None):
    """Return sorted list of element symbols available in pseudo_dir."""
    d = pseudo_dir or PSEUDO_DIR
    elements = set()
    if not os.path.isdir(d):
        return []
    for fname in os.listdir(d):
        if fname.lower().endswith((".upf", ".van", ".gth", ".bhs")):
            m = _ELEMENT_RE.match(fname)
            if m:
                elements.add(m.group(1))
    return sorted(elements)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MEAM Element Selector</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    display: flex; justify-content: center; align-items: flex-start;
    min-height: 100vh; padding: 40px 20px;
  }
  .container {
    width: 100%; max-width: 540px;
    background: #16213e; border: 1px solid #0f3460;
    border-radius: 12px; padding: 32px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  h1 {
    font-size: 1.4rem; font-weight: 600; text-align: center;
    color: #e94560; margin-bottom: 8px; letter-spacing: 0.5px;
  }
  .subtitle {
    text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 24px;
  }
  .section {
    background: #1a1a2e; border: 1px solid #0f3460;
    border-radius: 8px; padding: 20px; margin-bottom: 20px;
  }
  .section-title {
    font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
    color: #e94560; letter-spacing: 1px; margin-bottom: 16px;
  }
  .element-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
    gap: 10px;
  }
  .el-card {
    display: flex; flex-direction: column; align-items: center;
    padding: 12px 8px; border-radius: 8px;
    border: 2px solid #0f3460; cursor: pointer;
    transition: border-color 0.2s, background 0.2s, transform 0.1s;
    user-select: none;
  }
  .el-card:hover { background: rgba(15,52,96,0.5); transform: scale(1.05); }
  .el-card.selected { border-color: var(--el-color, #e94560); background: rgba(233,69,96,0.1); }
  .el-symbol { font-size: 1.3rem; font-weight: 700; }
  .el-name { font-size: 0.7rem; color: #888; margin-top: 2px; }
  .count-bar {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 12px; padding: 10px 14px;
    background: #0d1b2a; border-radius: 6px;
    font-weight: 600; font-size: 0.95rem;
    border: 1px solid #0f3460; color: #888;
  }
  .count-bar.has-selection { border-color: #2ecc71; color: #2ecc71; }
  .continue-btn {
    display: block; width: 100%; padding: 14px; margin-top: 20px;
    background: linear-gradient(135deg, #e94560, #c23152);
    border: none; border-radius: 8px; color: #fff;
    font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    letter-spacing: 0.5px;
  }
  .continue-btn:hover { opacity: 0.9; }
  .continue-btn:active { transform: scale(0.98); }
  .continue-btn:disabled { opacity: 0.5; cursor: not-allowed; }
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
  <h1>MEAM Element Selector</h1>
  <p class="subtitle">Select elements for DFT reference + MEAM potential generation</p>

  <div class="section">
    <div class="section-title">Available Elements</div>
    <div class="element-grid" id="grid"></div>
    <div class="count-bar" id="countBar">
      <span>Selected</span><span id="countValue">0 elements</span>
    </div>
  </div>

  <button class="continue-btn" id="continueBtn" disabled>Continue with Selection</button>
  <div class="feedback" id="feedback"></div>
</div>

<script>
let ELEMENTS = [];
let COLORS = {};
let selected = new Set();

fetch('/elements').then(r => r.json()).then(data => {
  ELEMENTS = data.elements;
  COLORS = data.colors;
  buildGrid();
});

function buildGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  ELEMENTS.forEach(sym => {
    const card = document.createElement('div');
    card.className = 'el-card';
    const color = COLORS[sym] || '#888';
    card.style.setProperty('--el-color', color);
    card.innerHTML = `
      <span class="el-symbol" style="color:${color}">${sym}</span>
    `;
    card.addEventListener('click', () => {
      if (selected.has(sym)) {
        selected.delete(sym);
        card.classList.remove('selected');
      } else {
        selected.add(sym);
        card.classList.add('selected');
      }
      updateCount();
    });
    grid.appendChild(card);
  });
}

function updateCount() {
  const n = selected.size;
  document.getElementById('countValue').textContent = n + ' element' + (n !== 1 ? 's' : '');
  const bar = document.getElementById('countBar');
  bar.className = 'count-bar' + (n > 0 ? ' has-selection' : '');
  document.getElementById('continueBtn').disabled = (n === 0);
}

document.getElementById('continueBtn').addEventListener('click', () => {
  if (selected.size === 0) {
    showFeedback('Select at least one element', true);
    return;
  }
  const btn = document.getElementById('continueBtn');
  btn.disabled = true;
  btn.textContent = 'Processing...';

  fetch('/select', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({elements: Array.from(selected).sort()})
  }).then(r => r.json()).then(d => {
    showFeedback('Selection submitted! You can close this tab.', false);
  }).catch(err => {
    showFeedback(err.message, true);
    btn.disabled = false;
    btn.textContent = 'Continue with Selection';
  });
});

function showFeedback(msg, isError) {
  const fb = document.getElementById('feedback');
  fb.textContent = msg;
  fb.className = 'feedback ' + (isError ? 'error' : 'success');
}
</script>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):
    """HTTP handler for the element selector GUI."""

    def log_message(self, format, *args):
        """Log requests so we can see what the browser is doing."""
        print(f"  [HTTP] {format % args}", flush=True)

    def do_GET(self):
        print(f"  [GET] {self.path}", flush=True)
        if self.path == "/":
            self._respond(200, "text/html", HTML_PAGE.encode())
        elif self.path == "/elements":
            elements = scan_pseudopotentials()
            print(f"  [GET /elements] Found {len(elements)} elements: {elements}", flush=True)
            colors = {sym: _COLORS.get(sym, "#888") for sym in elements}
            data = json.dumps({"elements": elements, "colors": colors}).encode()
            self._respond(200, "application/json", data)
        else:
            print(f"  [GET] 404: {self.path}", flush=True)
            self._respond(404, "text/plain", b"Not Found")

    def do_POST(self):
        print(f"  [POST] {self.path}", flush=True)
        if self.path == "/select":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            print(f"  [POST /select] Raw body: {raw}", flush=True)
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [POST /select] JSON parse error: {exc}", flush=True)
                self._respond(400, "application/json", b'{"error":"bad json"}')
                return
            _result["elements"] = body.get("elements", [])
            print(f"  [POST /select] Stored elements: {_result['elements']}", flush=True)
            self._respond(200, "application/json", b'{"status":"ok"}')
            print("  [POST /select] Response sent, scheduling shutdown...", flush=True)
            threading.Thread(target=self._shutdown, daemon=True).start()
        else:
            print(f"  [POST] 404: {self.path}", flush=True)
            self._respond(404, "text/plain", b"Not Found")

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _shutdown(self):
        print("  [SHUTDOWN] Shutting down HTTP server...", flush=True)
        server = _result.get("server")
        if server:
            server.shutdown()
            print("  [SHUTDOWN] Server stopped.", flush=True)
        else:
            print("  [SHUTDOWN] ERROR: No server reference found!", flush=True)


def select_elements(pseudo_dir=None):
    """Open the element selector GUI and return the list of selected symbols.

    Args:
        pseudo_dir: path to pseudopotentials directory (default: project pseudopotentials/)

    Returns:
        list[str] — sorted element symbols selected by the user
    """
    if pseudo_dir:
        global PSEUDO_DIR
        PSEUDO_DIR = pseudo_dir

    # Scan first to report what's available
    available = scan_pseudopotentials()
    print(f"Available pseudopotentials: {available}", flush=True)

    port = 8472
    print(f"Starting HTTP server on port {port}...", flush=True)
    server = HTTPServer(("127.0.0.1", port), _Handler)
    _result["server"] = server
    _result["elements"] = None

    url = f"http://127.0.0.1:{port}"
    print(f"Opening element selector at {url}", flush=True)
    webbrowser.open(url)
    print("Waiting for element selection (server.serve_forever)...", flush=True)
    server.serve_forever()
    print("serve_forever() returned.", flush=True)

    elements = _result["elements"] or []
    print(f"Selected elements: {elements}", flush=True)
    return elements


if __name__ == "__main__":
    selected = select_elements()
    print(f"Final selection: {selected}")
