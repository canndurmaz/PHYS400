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
