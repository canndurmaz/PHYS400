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
