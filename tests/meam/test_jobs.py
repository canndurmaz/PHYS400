"""Unit tests for src/MEAM/jobs.py — pure functions, no LAMMPS."""

import json
import os
import sys

import pytest

# Make src/MEAM importable
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "MEAM"))

from jobs import cache_key


def test_cache_key_is_stable_for_same_inputs():
    k1 = cache_key({"Al": 0.5, "Cu": 0.5}, {"box_size_m": 5e-9}, False, 1234.0)
    k2 = cache_key({"Cu": 0.5, "Al": 0.5}, {"box_size_m": 5e-9}, False, 1234.0)
    assert k1 == k2, "key must be order-independent on composition"


def test_cache_key_changes_when_mtime_changes():
    k1 = cache_key({"Al": 1.0}, {}, False, 1000.0)
    k2 = cache_key({"Al": 1.0}, {}, False, 1001.0)
    assert k1 != k2, "mtime must invalidate"


def test_cache_key_changes_when_do_viz_changes():
    k1 = cache_key({"Al": 1.0}, {}, True, 1000.0)
    k2 = cache_key({"Al": 1.0}, {}, False, 1000.0)
    assert k1 != k2, "do_viz must produce distinct keys"


from jobs import RunCache


def test_runcache_roundtrip(tmp_path):
    path = tmp_path / "runs.json"
    cache = RunCache(str(path))
    assert cache.get("abc") is None

    payload = {"composition": {"Al": 1.0}, "result": {"C11_GPa": 100.0}}
    cache.put("abc", payload)
    assert cache.get("abc") == payload

    # New instance must see the persisted entry
    cache2 = RunCache(str(path))
    assert cache2.get("abc") == payload


def test_runcache_missing_file_starts_empty(tmp_path):
    cache = RunCache(str(tmp_path / "does-not-exist.json"))
    assert cache.get("anything") is None


def test_runcache_corrupt_file_starts_empty(tmp_path, caplog):
    path = tmp_path / "runs.json"
    path.write_text("{ not valid json")
    cache = RunCache(str(path))
    assert cache.get("anything") is None


from jobs import JobStore


def test_jobstore_creates_with_queued_status():
    store = JobStore()
    job_id = store.create(composition={"Al": 1.0}, knobs={}, do_viz=False)
    job = store.get(job_id)
    assert job["status"] == "queued"
    assert job["composition"] == {"Al": 1.0}
    assert list(job["log_lines"]) == []
    assert job["thermo"] == []
    assert job["result"] is None
    assert job["error"] is None


def test_jobstore_state_transitions():
    store = JobStore()
    job_id = store.create({"Al": 1.0}, {}, False)
    store.mark_running(job_id)
    assert store.get(job_id)["status"] == "running"

    store.append_log(job_id, "LAMMPS hello")
    store.append_log(job_id, "step 0")
    assert list(store.get(job_id)["log_lines"])[-2:] == ["LAMMPS hello", "step 0"]

    store.append_thermo(job_id, {"step": 100, "pxx": -1.0})
    assert store.get(job_id)["thermo"] == [{"step": 100, "pxx": -1.0}]

    result = {"C11_GPa": 100.0, "C12_GPa": 50.0, "E_GPa": 75.0, "nu": 0.33}
    store.mark_done(job_id, result, render_path=None)
    job = store.get(job_id)
    assert job["status"] == "done"
    assert job["result"] == result


def test_jobstore_marks_error():
    store = JobStore()
    job_id = store.create({"Al": 1.0}, {}, False)
    store.mark_error(job_id, "boom", "Traceback...")
    job = store.get(job_id)
    assert job["status"] == "error"
    assert job["error"] == {"message": "boom", "traceback": "Traceback..."}


def test_jobstore_recent_returns_newest_first():
    store = JobStore()
    ids = [store.create({"Al": 1.0}, {"i": i}, False) for i in range(5)]
    recent = store.recent(limit=3)
    assert [r["id"] for r in recent] == list(reversed(ids))[:3]


from jobs import validate_payload, ValidationError

ALL = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Mo", "Ni", "Si", "Ti", "Zn"]
DEFAULTS = {
    "box_size_m": 5e-9, "temperature": 300.0,
    "total_steps": 1000, "thermo_interval": 10, "dump_interval": 50,
}


def test_validate_payload_happy_path():
    comp_norm, knobs, do_viz = validate_payload(
        {"composition": {"Al": 0.5, "Cu": 0.5}, "knobs": {}, "do_viz": False},
        ALL, DEFAULTS,
    )
    assert sum(comp_norm.values()) == pytest.approx(1.0)
    assert knobs == DEFAULTS
    assert do_viz is False


def test_validate_payload_normalises_unnormalised_input():
    comp_norm, *_ = validate_payload(
        {"composition": {"Al": 2.0, "Cu": 2.0}},
        ALL, DEFAULTS,
    )
    assert comp_norm == {"Al": 0.5, "Cu": 0.5}


def test_validate_payload_rejects_empty_composition():
    with pytest.raises(ValidationError, match="non-empty"):
        validate_payload({"composition": {}}, ALL, DEFAULTS)


def test_validate_payload_rejects_negative_fraction():
    with pytest.raises(ValidationError, match="negative"):
        validate_payload({"composition": {"Al": -0.1}}, ALL, DEFAULTS)


def test_validate_payload_rejects_unsupported_element():
    with pytest.raises(ValidationError, match="Unsupported element 'Xx'"):
        validate_payload({"composition": {"Al": 0.5, "Xx": 0.5}}, ALL, DEFAULTS)


def test_validate_payload_fills_missing_knobs_from_defaults():
    _, knobs, _ = validate_payload(
        {"composition": {"Al": 1.0}, "knobs": {"temperature": 500.0}},
        ALL, DEFAULTS,
    )
    assert knobs["temperature"] == 500.0
    assert knobs["box_size_m"] == DEFAULTS["box_size_m"]
