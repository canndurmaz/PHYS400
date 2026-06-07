"""Fetch + cache + parse one ASM-Aerospace (`asm.matweb.com`) datasheet.

Why this host and not ``www.matweb.com``?
-----------------------------------------
``www.matweb.com`` returns **HTTP 403** to every non-browser client (active
anti-bot) and its terms of use forbid automated extraction. The
``asm.matweb.com`` host is the *ASM Aerospace Specification Metals* mirror:
it serves plain static HTML tables and is not bot-walled. Its TLS chain is
missing an intermediate certificate, so we pass ``verify=False`` (the only
wart) and fetch politely with a small inter-request delay.

Each raw page is cached under ``html_cache/<bassnum>.html`` so that every rerun
after the first is fully offline and reproducible. Pass ``refresh=True`` to
re-pull.

The parser extracts the material name, the four elastic/physical properties we
compare against (E, ν, G, ρ), and the weight-percent composition table, then
converts that to a mole-fraction vector over the 12-element basis.
"""

from __future__ import annotations

import html
import os
import re
import time
from typing import Optional

import requests
import urllib3

from materials import ATOMIC_MASS, BASIS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_BASE = "https://asm.matweb.com/search/SpecificMaterial.asp?bassnum="
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "html_cache")
_UA = "Mozilla/5.0 (X11; Linux x86_64) PHYS400-literature-comparison/1.0"

# Element symbols we recognise in composition tables (basis + common
# out-of-basis alloying/impurity elements). Anything matching a two-letter
# pattern that is NOT here is ignored, which keeps stray capitalised words
# (e.g. "Max", "Component") from being mistaken for elements.
_KNOWN_ELEMENTS = set(BASIS) | {
    "C", "H", "N", "O", "P", "S", "B", "V", "W", "Sn", "Pb", "Nb", "Ta",
    "Zr", "Bi", "Ag", "Be", "Cd", "Ca", "Sr", "Li", "Y", "Hf", "Re", "La",
    "Ce", "Ga", "Ge", "As", "Sb", "Se", "Te", "Sc", "Au", "Pt", "Pd",
}

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.verify = False
        s.headers["User-Agent"] = _UA
        _session = s
    return _session


def fetch_html(bassnum: str, *, refresh: bool = False, delay: float = 1.0) -> str:
    """Return raw datasheet HTML for ``bassnum``, using the on-disk cache.

    On a cache miss (or ``refresh``) the page is fetched once and written to
    ``html_cache/<bassnum>.html``; ``delay`` seconds are slept *after* a live
    fetch to stay polite. Decoded as latin-1 because the pages contain raw
    0x80–0xFF bytes (e.g. the ³ in "lb/in³") that are not valid UTF-8.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{bassnum}.html")
    if not refresh and os.path.isfile(path):
        with open(path, encoding="latin-1") as f:
            return f.read()
    resp = _get_session().get(_BASE + bassnum, timeout=20)
    resp.raise_for_status()
    raw = resp.content.decode("latin-1")
    with open(path, "w", encoding="latin-1") as f:
        f.write(raw)
    time.sleep(delay)
    return raw


def _plain_text(raw: str) -> str:
    """Strip tags, unescape entities (``&nbsp;`` → space), collapse runs."""
    txt = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"[ \t\xa0]+", " ", txt)


def _name(raw: str) -> Optional[str]:
    m = re.search(r"<h4[^>]*>(.*?)</h4>", raw, re.S)
    if not m:
        return None
    name = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return name or None


def _property(txt: str, label: str) -> Optional[float]:
    """Pull a numeric property value (single or ``lo - hi`` range → midpoint).

    The datasheet prints e.g. ``Modulus of Elasticity 193 - 200 GPa`` or
    ``Poisson's Ratio 0.33 0.33``. We anchor on the label and take the first
    number, plus an optional ``- hi`` immediately after it.
    """
    m = re.search(re.escape(label) + r"\s+([\d.]+)(?:\s*-\s*([\d.]+))?", txt)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return (lo + hi) / 2.0


def _composition_wt(txt: str) -> dict[str, float]:
    """Parse the ``Wt. %`` table into {element: weight-percent midpoint}.

    Three row shapes appear, in this precedence:
        ``Cr 0.18 - 0.28``   range            → midpoint
        ``Fe Max 0.5``       upper bound only → half (mean of [0, max])
        ``Ti 99.2``          single / balance → as-is
    The word "Component" repeats across the multi-column layout and is stripped
    first so it can't swallow a following number.
    """
    parts = txt.split("Wt. %")
    if len(parts) < 2:
        return {}
    body = " ".join(parts[1:])
    # Cut the table off before the prose / next section.
    for stop in ("Material Notes", "Physical Propert", "References", "Key Words"):
        body = body.split(stop)[0]
    body = body.replace("Component", " ")

    comp: dict[str, float] = {}

    def _set(sym: str, val: float) -> None:
        if sym in _KNOWN_ELEMENTS:
            comp.setdefault(sym, val)

    for sym, lo, hi in re.findall(r"\b([A-Z][a-z]?)\s+([\d.]+)\s*-\s*([\d.]+)", body):
        _set(sym, (float(lo) + float(hi)) / 2.0)
    for sym, mx in re.findall(r"\b([A-Z][a-z]?)\s+Max\s+([\d.]+)", body):
        _set(sym, float(mx) / 2.0)
    for sym, val in re.findall(r"\b([A-Z][a-z]?)\s+([\d.]+)(?!\s*-)", body):
        _set(sym, float(val))
    return comp


def wt_to_mole_fraction(comp_wt: dict[str, float]) -> tuple[dict[str, float], float]:
    """Convert weight-% → basis mole fractions; return (fractions, coverage).

    Only the 12 basis elements contribute; out-of-basis mass (C, V, …) is
    dropped. ``coverage`` is the fraction of total *parsed* weight that stayed
    in-basis — a quality flag (1.0 = nothing dropped). Mole fractions are
    normalised over the in-basis elements so they sum to 1.
    """
    total_wt = sum(comp_wt.values())
    in_basis_wt = sum(v for k, v in comp_wt.items() if k in ATOMIC_MASS)
    coverage = in_basis_wt / total_wt if total_wt > 0 else 0.0

    moles = {k: v / ATOMIC_MASS[k] for k, v in comp_wt.items() if k in ATOMIC_MASS}
    tot = sum(moles.values())
    frac = {k: m / tot for k, m in moles.items()} if tot > 0 else {}
    return frac, coverage


def scrape(bassnum: str, *, refresh: bool = False, delay: float = 1.0) -> Optional[dict]:
    """Scrape one material. Returns ``None`` if the page has no usable data.

    The returned dict carries the literature values, the composition in both
    weight-% and mole-fraction form, the basis coverage, and provenance.
    """
    raw = fetch_html(bassnum, refresh=refresh, delay=delay)
    name = _name(raw)
    if not name:
        return None
    txt = _plain_text(raw)

    E = _property(txt, "Modulus of Elasticity")
    if E is None:
        return None  # no elastic data → not comparable, skip

    comp_wt = _composition_wt(txt)
    comp_frac, coverage = wt_to_mole_fraction(comp_wt)
    if not comp_frac:
        return None  # composition unparseable → can't drive the models

    return {
        "bassnum": bassnum,
        "name": name,
        "url": _BASE + bassnum,
        "source": "ASM Aerospace Specification Metals (asm.matweb.com)",
        "lit": {
            "E_GPa": E,
            "nu": _property(txt, "Poisson's Ratio"),
            "G_GPa": _property(txt, "Shear Modulus"),
            "density_g_cc": _property(txt, "Density"),
        },
        "composition_wt_pct": {k: round(v, 4) for k, v in comp_wt.items()},
        "composition": {k: round(v, 6) for k, v in comp_frac.items()},
        "basis_coverage": round(coverage, 4),
    }
