#!/usr/bin/env python3
"""Remove generated config files, keeping manually created ones.

Generated configs follow the naming pattern: Element_XXXX-Element_XXXX.json
(e.g., Al_8333-Fe_0780-Co_0721.json). Files that don't match this pattern
(e.g., AL7075_simple.json) are preserved.
"""

import argparse
import glob
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Generated filenames match: one or more Element_digits groups joined by dashes
_GENERATED_RE = re.compile(r'^[A-Z][a-z]?_\d{4}(-[A-Z][a-z]?_\d{4})*\.json$')


def clean(dry_run=False):
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
    parser = argparse.ArgumentParser(description="Remove generated config files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()
    clean(dry_run=args.dry_run)
