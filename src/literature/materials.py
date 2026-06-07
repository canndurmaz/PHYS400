"""Seed list of ASM-Aerospace (`asm.matweb.com`) material IDs + basis data.

Every entry here is a ``bassnum`` that resolves on
``https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=<ID>`` and whose
nominal composition is well covered by the 12-element MEAM/NN basis
(Al Co Cr Cu Fe Mg Mn Mo Ni Si Ti Zn). IDs were verified live during
development; the scraper skips any that stop resolving, so the list can be
edited freely.

The set deliberately spans a wide Young's-modulus range so the comparison is
not dominated by near-identical aluminium alloys:

    Al alloys   ~68–73 GPa
    CP titanium ~105 GPa
    Ti-6Al-4V   ~114 GPa   (V is out-of-basis → dropped, ~96% coverage)
    stainless   ~193–200 GPa
    Inco A-286  ~201 GPa
"""

from __future__ import annotations

# Atomic masses (g/mol) for the 12-element basis — used to convert the
# datasheet's weight-percent composition into the mole-fraction vector the
# models expect. Source: IUPAC 2021 standard atomic weights (rounded).
ATOMIC_MASS: dict[str, float] = {
    "Al": 26.982, "Co": 58.933, "Cr": 51.996, "Cu": 63.546,
    "Fe": 55.845, "Mg": 24.305, "Mn": 54.938, "Mo": 95.95,
    "Ni": 58.693, "Si": 28.085, "Ti": 47.867, "Zn": 65.38,
}

BASIS: tuple[str, ...] = tuple(ATOMIC_MASS)

# Candidate bassnum IDs grouped by family. The scraper validates each at
# fetch time, so a few dead IDs here are harmless (they're reported as skipped).
SEED_IDS: dict[str, list[str]] = {
    "aluminium": [
        # 2xxx (Al-Cu)
        "MA2014O", "MA2014T4", "MA2014T6", "MA2017O", "MA2017T4",
        "MA2024O", "MA2024T3", "MA2024T4", "MA2024T6", "MA2219O", "MA2219T87",
        # 5xxx (Al-Mg)
        "MA5052O", "MA5052H32", "MA5052H34", "MA5052H38",
        "MA5083O", "MA5083H32", "MA5083H34", "MA5086O", "MA5086H34",
        # 6xxx (Al-Mg-Si)
        "MA6061O", "MA6061T4", "MA6061T6", "MA6061T8",
        "MA6063O", "MA6063T4", "MA6063T6",
        # 7xxx (Al-Zn-Mg-Cu)
        "MA7075O", "MA7075T6", "MA7075T73",
        "MA7178O", "MA7178T6", "MA7475T651",
    ],
    "titanium": [
        "MTU010", "MTU020", "MTU030", "MTU040",   # CP grades 1–4 (~pure Ti)
        "MTP641", "MTA815",                         # Ti-6Al-4V, Ti-8-1-1
    ],
    "stainless": [
        "MQ304A", "MQ304L", "MQ316A",               # 304 / 304L / 316
    ],
    "nickel_iron": [
        "NINCO01",                                  # Inco A-286 (Fe-Ni-Cr)
    ],
}


def all_ids() -> list[str]:
    """Flattened, de-duplicated list of every candidate bassnum."""
    seen: dict[str, None] = {}
    for group in SEED_IDS.values():
        for bid in group:
            seen.setdefault(bid, None)
    return list(seen)


def family_of(bassnum: str) -> str:
    """Return the material-family key a bassnum belongs to (or 'other')."""
    for fam, group in SEED_IDS.items():
        if bassnum in group:
            return fam
    return "other"
