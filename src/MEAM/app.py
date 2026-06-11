"""Flask UI for the optimized-MEAM on-demand MD predictor.

Companion to src/ML/app.py: instead of fast NN inference, this app
launches a real LAMMPS job (subprocess, async) and returns the elastic
constants from a fresh strain sweep against EAM/optimized/.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import threading
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_file

# Re-use the 12-element basis and the MD defaults from existing modules
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "ML"))
sys.path.insert(0, os.path.join(_REPO, "src", "MD"))
from model_constants import ALL_ELEMENTS        # noqa: E402
from config import _DEFAULTS as _MD_DEFAULTS    # noqa: E402

# Strip non-knob keys (e.g. "composition") so validate_payload only sees
# numeric simulation parameters.
_KNOB_KEYS = {"box_size_m", "temperature", "total_steps",
               "thermo_interval", "dump_interval"}
_DEFAULTS = {k: v for k, v in _MD_DEFAULTS.items() if k in _KNOB_KEYS}

from jobs import JobStore, RunCache, cache_key, validate_payload, ValidationError

_log = logging.getLogger(__name__)

_OPTIMIZED_LIBRARY = os.path.join(_REPO, "EAM", "optimized",
                                  "optimized_library_AlCoCrCuFeMgMnMoNiSiTiZn.meam")
_OPTIMIZED_PARAMS  = os.path.join(_REPO, "EAM", "optimized",
                                  "optimized_AlCoCrCuFeMgMnMoNiSiTiZn.meam")
_RUNNER = os.path.join(_THIS, "runner.py")
_PY = os.path.join(_REPO, "phys", "bin", "python3")
_DEFAULT_RUNS_PATH = os.path.join(_THIS, "runs.json")
_RENDERS_DIR = os.path.join(_THIS, "renders")


def _meam_mtime() -> float:
    """max(mtime) over the two optimized MEAM files. Test override via env."""
    override = os.environ.get("MEAM_FAKE_MEAM_MTIME")
    if override is not None:
        return float(override)
    return max(os.path.getmtime(_OPTIMIZED_LIBRARY),
               os.path.getmtime(_OPTIMIZED_PARAMS))


_STORE: Optional[JobStore] = None
_CACHE: Optional[RunCache] = None
_POOL: Optional[ThreadPoolExecutor] = None


def _store() -> JobStore:
    assert _STORE is not None, "create_app() must be called first"
    return _STORE


def _cache() -> RunCache:
    assert _CACHE is not None, "create_app() must be called first"
    return _CACHE


def _run_subprocess(spec: dict, job_id: str, store: JobStore, cache: RunCache,
                    cache_k: str, meam_mt: float) -> None:
    """Run runner.py in a child process, drain its JSON-line stdout into
    the JobStore. Called from the pool worker (separate process)."""
    store.mark_running(job_id)
    env = {**os.environ,
           "LD_LIBRARY_PATH": os.path.expanduser("~/.local/lib")
                              + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    proc = subprocess.Popen(
        [_PY, _RUNNER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, env=env, bufsize=1,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps(spec))
    proc.stdin.close()

    final: dict | None = None
    error: dict | None = None
    for raw in proc.stdout:
        raw = raw.rstrip("\n")
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            store.append_log(job_id, raw)
            continue
        t = obj.get("type")
        if t == "log":
            store.append_log(job_id, obj.get("line", ""))
        elif t == "thermo":
            store.append_thermo(job_id, {k: obj[k] for k in
                                          ("step", "pxx", "pyy", "pzz")})
        elif t == "result":
            final = obj
        elif t == "error":
            error = obj
    rc = proc.wait()

    if final is not None:
        result = {k: final[k] for k in
                  ("C11_GPa", "C12_GPa", "E_GPa", "nu",
                   "physical", "physical_reason")}
        store.mark_done(job_id, result, render_path=final.get("render_path"))
        cache.put(cache_k, {
            "composition": spec["composition"],
            "knobs": spec["knobs"],
            "do_viz": spec.get("do_viz", False),
            "result": result,
            "render_path": final.get("render_path"),
            "completed_at": store.get(job_id)["finished_at"],
            "meam_mtime": meam_mt,
        })
    elif error is not None:
        store.mark_error(job_id, error.get("message", "unknown"),
                         error.get("traceback", ""))
    else:
        store.mark_error(job_id, f"runner exited {rc} with no result",
                         "")


def create_app(use_pool: bool = True) -> Flask:
    global _STORE, _CACHE, _POOL
    _STORE = JobStore()
    _CACHE = RunCache(os.environ.get("MEAM_RUNS_PATH", _DEFAULT_RUNS_PATH))
    _POOL = ThreadPoolExecutor(max_workers=1) if use_pool else None

    if os.environ.get("MEAM_SKIP_BOOT_CHECK") != "1":
        for p in (_OPTIMIZED_LIBRARY, _OPTIMIZED_PARAMS):
            if not os.path.isfile(p):
                raise RuntimeError(
                    f"Optimized MEAM not found at {p}. "
                    f"Run ./src/NNIP/run_pipeline.sh first."
                )

    app = Flask(
        __name__,
        template_folder=os.path.join(_THIS, "templates"),
        static_folder=os.path.join(_THIS, "static"),
    )

    @app.route("/")
    def index():
        return render_template("index.html",
                               elements=ALL_ELEMENTS, defaults=_DEFAULTS)

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True) or {}
        try:
            comp_norm, knobs, do_viz = validate_payload(
                payload, ALL_ELEMENTS, _DEFAULTS,
            )
        except ValidationError as exc:
            return jsonify(error=str(exc)), 400

        mtime = _meam_mtime()
        k = cache_key(comp_norm, knobs, do_viz, mtime)
        hit = _cache().get(k)
        if hit is not None:
            rp = hit.get("render_path")
            render_available = bool(rp) and os.path.isfile(rp)
            # If a render was requested but the cached run has none (earlier
            # OVITO failure, or the PNG was deleted), fall through and re-run.
            if not do_viz or render_available:
                return jsonify(status="done", cached=True, cache_key=k,
                               result=hit["result"],
                               render_available=render_available)

        render_path = (os.path.join(_RENDERS_DIR, f"{k}.png")
                       if do_viz else None)
        spec = {"composition": comp_norm, "knobs": knobs,
                "do_viz": do_viz, "render_output_path": render_path}
        job_id = _store().create(comp_norm, knobs, do_viz, cache_key=k)
        if do_viz:
            os.makedirs(_RENDERS_DIR, exist_ok=True)
        if _POOL is None:
            # Test mode: drain inline in a thread so tests don't block forever
            t = threading.Thread(
                target=_run_subprocess,
                args=(spec, job_id, _store(), _cache(), k, mtime),
                daemon=True,
            )
            t.start()
        else:
            _POOL.submit(
                _run_subprocess,
                spec, job_id, _store(), _cache(), k, mtime,
            )
        return jsonify(job_id=job_id, status="queued", cached=False)

    @app.get("/api/jobs/<job_id>")
    def api_job(job_id: str):
        job = _store().get(job_id)
        if job is None:
            return jsonify(error="unknown job_id"), 404
        # deque -> list for JSON
        job["log_lines"] = list(job["log_lines"])[-50:]
        rp = job.get("render_path")
        job["render_available"] = bool(rp) and os.path.isfile(rp)
        # don't leak filesystem paths to the browser (cache_key stays)
        job.pop("render_path", None)
        return jsonify(job)

    @app.get("/api/renders/<cache_k>")
    def api_render(cache_k: str):
        # Validate: hex sha1, 40 chars -- prevents path traversal
        if len(cache_k) != 40 or not all(c in "0123456789abcdef"
                                          for c in cache_k):
            return jsonify(error="bad cache key"), 400
        path = os.path.join(_RENDERS_DIR, f"{cache_k}.png")
        if not os.path.isfile(path):
            return jsonify(error="no render available"), 404
        return send_file(path, mimetype="image/png")

    @app.get("/api/jobs")
    def api_recent():
        recents = []
        for j in _store().recent(limit=20):
            recents.append({k: j[k] for k in
                            ("id", "status", "submitted_at", "composition")})
        return jsonify(jobs=recents)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5001, debug=False)
