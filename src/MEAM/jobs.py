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
