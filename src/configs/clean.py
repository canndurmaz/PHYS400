#!/usr/bin/env python3
"""Remove generated config files, keeping manually created ones.

Generated configs follow the naming pattern: Element_XXXX-Element_XXXX.json
(e.g., Al_8333-Fe_0780-Co_0721.json). Files that don't match this pattern
(e.g., AL7075_simple.json) are preserved.
"""

import argparse
import glob
import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ML", "results.json"))

# Generated filenames match: one or more Element_digits groups joined by dashes
_GENERATED_RE = re.compile(r'^[A-Z][a-z]?_\d{4}(-[A-Z][a-z]?_\d{4})*\.json$')


def clean_results(dry_run=False):
    """Remove generated entries from results.json, keeping manual ones."""
    if not os.path.exists(RESULTS_PATH):
        print("No results.json to clean.")
        return 0

    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    generated_keys = [k for k in results if _GENERATED_RE.match(k)]

    if not generated_keys:
        print("No generated entries in results.json.")
        return 0

    for k in sorted(generated_keys):
        if dry_run:
            print(f"  would remove result: {k}")
        else:
            del results[k]

    if not dry_run:
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=4)

    action = "Would remove" if dry_run else "Removed"
    print(f"{action} {len(generated_keys)} generated result(s) from results.json.")
    return len(generated_keys)


def clean_configs(dry_run=False):
    """Remove generated config files."""
    files = glob.glob(os.path.join(SCRIPT_DIR, "*.json"))
    to_remove = [f for f in files if _GENERATED_RE.match(os.path.basename(f))]

    if not to_remove:
        print("No generated configs to remove.")
        return 0

    for f in sorted(to_remove):
        if dry_run:
            print(f"  would remove: {os.path.basename(f)}")
        else:
            os.remove(f)

    action = "Would remove" if dry_run else "Removed"
    print(f"{action} {len(to_remove)} generated config(s).")
    return len(to_remove)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove generated config files and results")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()
    clean_configs(dry_run=args.dry_run)
    clean_results(dry_run=args.dry_run)
