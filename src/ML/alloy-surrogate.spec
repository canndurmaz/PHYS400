# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the alloy-surrogate desktop app.

Builds a one-folder bundle (faster startup than one-file; AppImage wraps
it into a single distributable file anyway). Entry point is the pywebview
launcher in ``app_desktop.py``. The Flask UI, templates, static assets,
ensemble ONNX checkpoints, and ground-truth data are all bundled in;
TensorFlow is deliberately excluded (the ONNX runtime is enough) to keep
the binary an order of magnitude smaller.

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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="alloy-surrogate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX-compressing libpython sometimes breaks loaders
    console=False,       # no terminal window pops up on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="alloy-surrogate",
)
