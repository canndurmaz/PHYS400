"""Desktop entry point: Flask + native pywebview window.

This is the file PyInstaller freezes into the AppImage. At runtime it:

1.  Starts the Flask app from :mod:`app` on a background thread, bound
    to an ephemeral localhost port so multiple instances can coexist
    and we never collide with a developer's already-running dev server.
2.  Opens a native pywebview window pointing at that URL. On Linux this
    uses WebKit2GTK (already on every modern desktop distro); on macOS
    it uses the system WebKit and on Windows it uses Edge WebView2.
3.  Blocks on ``webview.start()``; when the user closes the window,
    that call returns and the process exits cleanly (the Flask thread
    is a daemon, so it dies with the main thread).

The Flask server is bound to ``127.0.0.1`` — never reachable from the
network — because this is a single-user desktop app, not a service.
"""

from __future__ import annotations

import socket
import sys
import threading
import time

import webview

from app import app as flask_app


def _pick_port() -> int:
    """Ask the kernel for an unused ephemeral port and return it.

    We bind a temporary socket to port 0 (let the OS pick), read what
    it gave us, then close it. There's a tiny window where another
    process could grab the same port before Flask binds — in practice
    not worth worrying about for a single-user desktop app.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_server(host: str, port: int, timeout_s: float = 10.0) -> None:
    """Block until the Flask socket is accepting connections.

    pywebview will happily load a 'connection refused' page if we open
    the window before Flask is ready, so we poll the port first. The
    Flask boot is ~200 ms in the packaged app (no TF, ONNX models lazy-
    loaded), so this loop almost always exits on the first iteration.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.05)
    raise RuntimeError(f"Flask did not come up on {host}:{port} within {timeout_s}s")


def _serve(host: str, port: int) -> None:
    """Run the Flask app in this thread. Reloader OFF (required when not
    in the main thread); debug OFF (we ship release artifacts, not stack
    traces, to end users)."""
    flask_app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


def main() -> int:
    host = "127.0.0.1"
    port = _pick_port()

    server_thread = threading.Thread(
        target=_serve, args=(host, port), daemon=True, name="flask-serve",
    )
    server_thread.start()
    _wait_for_server(host, port)

    # The window title also becomes the AppImage window class on Linux,
    # which is what desktop environments use to group + label icons.
    webview.create_window(
        title="Alloy Elastic Surrogate",
        url=f"http://{host}:{port}/",
        width=1280,
        height=860,
        min_size=(960, 640),
        # Keep the resize handle on so the user can adapt to their screen;
        # the layout is already responsive via the existing CSS grid.
        resizable=True,
    )
    # gui=None lets pywebview auto-pick: GTK on Linux, Cocoa on macOS,
    # EdgeChromium on Windows. Explicit selection only matters when you
    # need to override the default on a system with multiple backends.
    webview.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
