"""In-memory JobStore + on-disk RunCache for the MEAM Flask app.

No Flask, no LAMMPS imports — safe to unit-test under plain pytest.
"""

from __future__ import annotations

import hashlib
import json


def cache_key(
    composition: dict,
    knobs: dict,
    do_viz: bool,
    meam_mtime: float,
) -> str:
    """SHA-1 hex over a sorted-key JSON dump.

    Including ``meam_mtime`` means re-running the NN pipeline's Stage 4
    (which rewrites the optimized MEAM files) invalidates every cached
    result automatically. ``do_viz`` is part of the key so a "viz on"
    request doesn't collide with a previous "viz off" cached entry whose
    payload lacks a render_path.
    """
    payload = json.dumps(
        {
            "composition": {k: composition[k] for k in sorted(composition)},
            "knobs": {k: knobs[k] for k in sorted(knobs)},
            "do_viz": bool(do_viz),
            "meam_mtime": float(meam_mtime),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


import logging
import os
import tempfile

_log = logging.getLogger(__name__)


class RunCache:
    """Single-file JSON cache for completed MD runs.

    Atomic temp+rename write avoids leaving a half-written runs.json if the
    process is killed mid-write. On parse failure at load time, the cache
    starts empty and logs a warning — the worst case is one extra recompute.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._data = loaded
                else:
                    _log.warning("runs.json is not an object; starting empty")
            except (OSError, json.JSONDecodeError) as exc:
                _log.warning("runs.json unreadable (%s); starting empty", exc)

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, payload: dict) -> None:
        self._data[key] = payload
        dir_ = os.path.dirname(self._path) or "."
        fd, tmp = tempfile.mkstemp(prefix="runs.", suffix=".json.tmp", dir=dir_)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


import threading
import uuid
from collections import deque
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    """Thread-safe in-memory table of job records.

    Mutations are serialized under a single lock; reads return shallow
    copies so callers can iterate without holding the lock. The returned
    log_lines (deque) and thermo (list) are tail-only views — callers
    must NOT mutate them; treat them as read-only snapshots.

    All mutator methods (mark_running, mark_done, mark_error, append_log,
    append_thermo) raise KeyError if the job_id is unknown.
    """

    LOG_TAIL = 200   # keep last N stdout lines per job

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []      # insertion order, newest at end
        self._lock = threading.Lock()

    def create(self, composition: dict, knobs: dict, do_viz: bool,
               cache_key: str | None = None) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "submitted_at": _now_iso(),
                "started_at": None,
                "finished_at": None,
                "log_lines": deque(maxlen=self.LOG_TAIL),
                "thermo": [],
                "result": None,
                "error": None,
                "render_path": None,
                "cache_key": cache_key,
                "composition": dict(composition),
                "knobs": dict(knobs),
                "do_viz": bool(do_viz),
            }
            self._order.append(job_id)
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return None if j is None else dict(j)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = _now_iso()

    def mark_done(self, job_id: str, result: dict, render_path: str | None) -> None:
        with self._lock:
            j = self._jobs[job_id]
            j["status"] = "done"
            j["result"] = result
            j["render_path"] = render_path
            j["finished_at"] = _now_iso()

    def mark_error(self, job_id: str, message: str, traceback: str) -> None:
        with self._lock:
            j = self._jobs[job_id]
            j["status"] = "error"
            j["error"] = {"message": message, "traceback": traceback}
            j["finished_at"] = _now_iso()

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            self._jobs[job_id]["log_lines"].append(line)

    def append_thermo(self, job_id: str, record: dict) -> None:
        with self._lock:
            self._jobs[job_id]["thermo"].append(record)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            return [dict(self._jobs[i]) for i in ids]


class ValidationError(ValueError):
    """Bad form payload — propagated to HTTP 400."""


def validate_payload(
    payload: dict,
    all_elements: list[str],
    defaults: dict,
) -> tuple[dict, dict, bool]:
    """Return (composition_normalised, knobs_with_defaults, do_viz).

    Mirrors the validation rules in ``src/ML/app.py``'s ``/api/predict``,
    plus rejects elements outside the 12-element basis (the optimized MEAM
    covers exactly those; anything else can't be assigned a LAMMPS type).
    """
    raw_comp = payload.get("composition")
    if not isinstance(raw_comp, dict) or not raw_comp:
        raise ValidationError("Provide a non-empty 'composition' object.")

    allowed = set(all_elements)
    comp: dict[str, float] = {}
    for el, val in raw_comp.items():
        if el not in allowed:
            raise ValidationError(
                f"Unsupported element {el!r}. Supported: "
                + ", ".join(all_elements)
            )
        if val in (None, "", "null"):
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            raise ValidationError(f"Fraction for {el} is not a number: {val!r}")
        if f < 0:
            raise ValidationError(f"Fraction for {el} is negative: {f}")
        if f > 0:
            comp[el] = f

    if not comp:
        raise ValidationError("At least one element must have a positive fraction.")

    total = sum(comp.values())
    if total <= 0:
        raise ValidationError("Composition sums to zero.")
    comp_norm = {el: f / total for el, f in comp.items()}

    raw_knobs = payload.get("knobs") or {}
    if not isinstance(raw_knobs, dict):
        raise ValidationError("'knobs' must be an object if provided.")
    knobs = dict(defaults)
    for k, v in raw_knobs.items():
        if k not in defaults:
            raise ValidationError(f"Unknown knob {k!r}. Allowed: {sorted(defaults)}")
        try:
            knobs[k] = float(v) if isinstance(defaults[k], float) else int(v)
        except (TypeError, ValueError):
            raise ValidationError(f"Knob {k!r} is not numeric: {v!r}")

    do_viz = bool(payload.get("do_viz", False))
    return comp_norm, knobs, do_viz
