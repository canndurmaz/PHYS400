#!/usr/bin/env python3
"""Download required pseudopotential files for DFT calculations.

Checks which pseudopotentials are needed for the given elements,
skips files already present, and downloads missing ones from the
QE pseudopotential library.
"""

import os
import sys
import urllib.request
import urllib.error

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.NNIP.logging_config import setup_logger
from src.NNIP.dft_reference import ELEMENT_DATA, PSEUDO_DIR

logger = setup_logger("pseudo_download")

# Base URLs to try, in order of preference
_BASE_URLS = [
    "https://pseudopotentials.quantum-espresso.org/upf_files/",
    "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/",
]


def ensure_pseudopotentials(elements, pseudo_dir=None):
    """Download any missing pseudopotential files for the given elements.

    Args:
        elements: list of element symbols
        pseudo_dir: directory to store files (default: PSEUDO_DIR from dft_reference)

    Returns:
        dict mapping element symbol to full path of its pseudopotential file

    Raises:
        RuntimeError: if a required pseudopotential cannot be found or downloaded
    """
    out_dir = pseudo_dir or PSEUDO_DIR
    os.makedirs(out_dir, exist_ok=True)

    paths = {}
    missing = []

    for sym in elements:
        if sym not in ELEMENT_DATA:
            logger.warning(f"  {sym}: not in ELEMENT_DATA, skipping pseudopotential download")
            continue

        _, _, pseudo_file = ELEMENT_DATA[sym]
        full_path = os.path.join(out_dir, pseudo_file)
        paths[sym] = full_path

        if os.path.isfile(full_path):
            logger.info(f"  {sym}: {pseudo_file} — already present")
        else:
            missing.append((sym, pseudo_file, full_path))

    if not missing:
        logger.info("  All pseudopotentials present.")
        return paths

    logger.info(f"  Need to download {len(missing)} file(s):")
    for sym, fname, _ in missing:
        logger.info(f"    {sym}: {fname}")

    for sym, fname, full_path in missing:
        downloaded = False
        for base_url in _BASE_URLS:
            url = base_url + fname
            logger.info(f"  Downloading {sym}: {url}")
            try:
                urllib.request.urlretrieve(url, full_path)
                # Verify it's not an HTML error page
                with open(full_path, "rb") as f:
                    header = f.read(128)
                if b"<html" in header.lower() or b"<!doctype" in header.lower():
                    os.remove(full_path)
                    logger.info(f"    Got HTML instead of UPF, trying next source...")
                    continue
                size_kb = os.path.getsize(full_path) / 1024
                logger.info(f"    OK ({size_kb:.0f} KB)")
                downloaded = True
                break
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.info(f"    Failed: {e}")
                if os.path.exists(full_path):
                    os.remove(full_path)
                continue

        if not downloaded:
            raise RuntimeError(
                f"Could not download pseudopotential for {sym} ({fname}). "
                f"Please place it manually in {out_dir}"
            )

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download QE pseudopotentials")
    parser.add_argument("elements", nargs="+", help="Element symbols, e.g. Al Cu Fe")
    parser.add_argument("--pseudo-dir", default=None, help=f"Output directory (default: {PSEUDO_DIR})")
    args = parser.parse_args()

    paths = ensure_pseudopotentials(args.elements, args.pseudo_dir)
    for sym, path in sorted(paths.items()):
        print(f"  {sym}: {path}")
