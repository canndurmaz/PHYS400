# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the alloy-surrogate desktop app.

Builds a one-file self-extracting binary. Entry point is the pywebview
launcher in ``app_desktop.py``. The Flask UI, templates, static assets,
ensemble ONNX checkpoints, and ground-truth data are all bundled in;
TensorFlow is deliberately excluded (the ONNX runtime is enough) to keep
the binary an order of magnitude smaller.

One-file mode trades ~1-2 s of cold-start (the bootloader extracts the
bundle to /tmp/_MEI<pid>/ on each launch) for a true single-file
artifact — no AppImage wrapper, no _internal/ sidecar directory, just
one executable the user can ``chmod +x`` and run on any modern Linux.

Build with:
    pyinstaller --noconfirm alloy-surrogate.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

here = os.path.abspath(SPECPATH)

# Files that need to land at the bundle root so app.py finds them via
# ``sys._MEIPASS``. Tuples are (source path, destination path inside bundle).
datas = [
    (os.path.join(here, "templates"), "templates"),
    (os.path.join(here, "static"),    "static"),
    (os.path.join(here, "results.json"),    "."),
    (os.path.join(here, "nn_metrics.json"), "."),
]
# Glob in all ONNX ensemble members. .keras files are NOT bundled — we
# ship the runtime format only.
for name in sorted(os.listdir(here)):
    if name.startswith("alloy_model") and name.endswith(".onnx"):
        datas.append((os.path.join(here, name), "."))

# pywebview's platform backends are imported dynamically via gi; PyInstaller
# can't trace them statically, so we name them explicitly. We include all
# three platforms because including them is cheap (~tens of KB of stubs).
hiddenimports = [
    "webview.platforms.gtk",
    "webview.platforms.cocoa",
    "webview.platforms.edgechromium",
    # setuptools >= 60 vendors a stack of helper packages through
    # ``pkg_resources.extern``; PyInstaller's static analyser misses them
    # and the bundled app dies at startup with ImportError. Name them all
    # explicitly. setuptools 82 (the version in this venv) needs at least
    # jaraco.* + more_itertools + platformdirs + packaging + backports.
    "jaraco.text",
    "jaraco.context",
    "jaraco.functools",
    "more_itertools",
    "platformdirs",
    "packaging",
    "autocommand",
    "typing_extensions",
    "backports.tarfile",
]
# onnxruntime ships a C extension whose submodules PyInstaller occasionally
# misses; let the hook collector enumerate them all.
hiddenimports += collect_submodules("onnxruntime")

# Pull in any data files onnxruntime ships (ONNX schema, etc.).
datas += collect_data_files("onnxruntime")

# Modules to *exclude*. TensorFlow is the big one — predict_from_model
# only falls back to it when no .onnx exists, and we make sure the bundle
# always contains .onnx, so TF would just be dead weight. The rest are
# false positives that PyInstaller's static analysis hauls in via deep
# transitive deps but the runtime never actually executes.
excludes = [
    # Training-only
    "tensorflow", "tensorflow.python", "tf2onnx", "keras",
    "matplotlib", "tabulate",
    # Not used by the runtime; pulled in by pandas/SQLAlchemy/etc. webs
    "scipy", "pandas", "django", "sqlalchemy",
    "ml_dtypes", "jedi", "onnx", "lxml", "pytz",
    "tables", "h5py", "cryptography",
    "sklearn", "joblib", "numba",
    # Other pywebview backends we never select
    "PIL", "PySide6", "PyQt5", "PyQt6",
    # Tcl/Tk, IPython, Jupyter, debuggers — none of this runs in the app
    "tkinter", "_tkinter",
    "IPython", "jupyter", "jupyter_client", "jupyter_core",
    "notebook", "nbformat", "nbconvert",
    "pydevd", "pydevd_plugins", "_pydevd_bundle", "_pydevd_frame_eval",
    "zmq", "tornado",
    "test", "tests", "unittest",
]


a = Analysis(
    [os.path.join(here, "app_desktop.py")],
    pathex=[here],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# One-file: pack binaries + zipped pure-Python + data files into the EXE
# itself. The bootloader extracts everything to ``sys._MEIPASS`` (a
# ``/tmp/_MEI<pid>`` dir) on each launch.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="alloy-surrogate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX-compressing libpython sometimes breaks loaders
    upx_exclude=[],
    runtime_tmpdir=None, # default /tmp/_MEI<pid>; let the OS pick
    console=False,       # no terminal window pops up on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# No COLLECT() in one-file mode — everything is inside the EXE above.
