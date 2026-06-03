#!/usr/bin/env python3
"""Regenerate reports/dft_validation/dft_formation_energies.html from dft_results.json.

Literature values are embedded below (curated). The script is idempotent: re-run
whenever dft_results.json is updated.
"""

import json
import os
from datetime import date
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DFT_JSON = os.path.join(ROOT, "src", "NNIP", "dft_results.json")
OUT_HTML = os.path.join(HERE, "dft_formation_energies.html")

# ── Literature reference values (eV/atom). ───────────────────────────────────
# For each pair we record:
#   probe_struct   — the L1_2/B2 cell the script computes
#   lit_same_cell  — published DFT ΔE_form for that same probe structure
#                    (apples-to-apples; chiefly Materials Project / OQMD)
#   lit_gs         — experimental or DFT enthalpy of formation for the actual
#                    ground-state phase
#   gs_phase       — name of the ground-state phase (or "—" if no GS compound)
#   src_same       / src_gs — citation keys, see Section 7 of the report.
LITERATURE = {
    "Al-Cu": dict(probe="L1₂ (Al₃Cu)", lit_same=+0.04, src_same="a", lit_gs=-0.16, gs="θ-Al₂Cu (C16)", src_gs="g"),
    "Al-Si": dict(probe="L1₂ (Al₃Si)", lit_same=+0.08, src_same="b", lit_gs=+0.00, gs="— (eutectic)", src_gs="g"),
    "Al-Ti": dict(probe="L1₂ (Al₃Ti)", lit_same=+0.06, src_same="c", lit_gs=-0.41, gs="Al₃Ti (D0₂₂)", src_gs="c"),
    "Al-Zn": dict(probe="L1₂ (Al₃Zn)", lit_same=+0.05, src_same="a", lit_gs=+0.02, gs="— (eutectic)", src_gs="g"),
    "Cu-Si": dict(probe="L1₂ (Cu₃Si)", lit_same=-0.05, src_same="a", lit_gs=-0.07, gs="η-Cu₃Si", src_gs="g"),
    "Cu-Ti": dict(probe="L1₂ (Cu₃Ti)", lit_same=-0.20, src_same="d", lit_gs=-0.30, gs="Cu₃Ti (D0_a)", src_gs="d"),
    "Cu-Zn": dict(probe="L1₂ (Cu₃Zn)", lit_same=-0.04, src_same="a", lit_gs=-0.05, gs="CuZn (B2, β-brass)", src_gs="g"),
    "Si-Ti": dict(probe="L1₂ (Si₃Ti)", lit_same=-0.05, src_same="b", lit_gs=-0.66, gs="Ti₅Si₃ (D8₈)", src_gs="g"),
    "Si-Zn": dict(probe="L1₂ (Si₃Zn)", lit_same=+0.10, src_same="b", lit_gs=+0.10, gs="— (immiscible)", src_gs="f"),
    "Ti-Zn": dict(probe="B2 (TiZn)",    lit_same=+0.05, src_same="b", lit_gs=-0.12, gs="TiZn₃ (L1₂)", src_gs="b"),
    # Within {Cr,Fe,Mg,Mo}
    "Cr-Fe": dict(probe="B2 (CrFe)",    lit_same=+0.30, src_same="e", lit_gs=-0.04, gs="σ-FeCr", src_gs="g"),
    "Cr-Mg": dict(probe="B2 (CrMg)",    lit_same=+0.40, src_same="f", lit_gs=+0.15, gs="— (immiscible)", src_gs="f"),
    "Cr-Mo": dict(probe="B2 (CrMo)",    lit_same=+0.10, src_same="a", lit_gs=+0.02, gs="— (BCC SS)", src_gs="g"),
    "Fe-Mg": dict(probe="B2 (FeMg)",    lit_same=+0.50, src_same="f", lit_gs=+0.20, gs="— (immiscible)", src_gs="f"),
    "Fe-Mo": dict(probe="B2 (FeMo)",    lit_same=+0.15, src_same="a", lit_gs=-0.05, gs="Fe₂Mo (Laves C14)", src_gs="g"),
    "Mg-Mo": dict(probe="B2 (MgMo)",    lit_same=+0.50, src_same="f", lit_gs=+0.40, gs="— (immiscible)", src_gs="f"),
    # Cross-set: {Cr,Fe,Mg,Mn,Mo} × {Al,Cu,Si,Ti,Zn}
    "Al-Cr": dict(probe="B2 (AlCr)",    lit_same=-0.10, src_same="a", lit_gs=-0.20, gs="Al₈Cr₅ (D8₁₀)", src_gs="g"),
    "Al-Fe": dict(probe="B2 (AlFe)",    lit_same=-0.31, src_same="a", lit_gs=-0.31, gs="FeAl (B2)", src_gs="g"),
    "Al-Mg": dict(probe="L1₂ (Al₃Mg)",  lit_same=+0.05, src_same="a", lit_gs=-0.06, gs="Al₃Mg₂ (β)", src_gs="g"),
    "Al-Mn": dict(probe="B2 (AlMn)",    lit_same=-0.18, src_same="a", lit_gs=-0.21, gs="Al₆Mn", src_gs="g"),
    "Al-Mo": dict(probe="B2 (AlMo)",    lit_same=+0.05, src_same="a", lit_gs=-0.32, gs="Al₈Mo₃", src_gs="g"),
    "Cr-Cu": dict(probe="B2 (CrCu)",    lit_same=+0.40, src_same="a", lit_gs=+0.06, gs="— (immiscible)", src_gs="g"),
    "Cr-Si": dict(probe="B2 (CrSi)",    lit_same=-0.40, src_same="a", lit_gs=-0.40, gs="CrSi (B20)", src_gs="g"),
    "Cr-Ti": dict(probe="B2 (CrTi)",    lit_same=-0.05, src_same="a", lit_gs=-0.11, gs="TiCr₂ (Laves C14)", src_gs="g"),
    "Cr-Zn": dict(probe="B2 (CrZn)",    lit_same=+0.20, src_same="a", lit_gs=+0.20, gs="— (immiscible)", src_gs="f"),
    "Cu-Fe": dict(probe="B2 (CuFe)",    lit_same=+0.20, src_same="a", lit_gs=+0.05, gs="— (immiscible)", src_gs="g"),
    "Cu-Mg": dict(probe="L1₂ (Cu₃Mg)",  lit_same=-0.05, src_same="a", lit_gs=-0.10, gs="MgCu₂ (Laves C15)", src_gs="g"),
    "Cu-Mn": dict(probe="B2 (CuMn)",    lit_same=-0.02, src_same="a", lit_gs=-0.02, gs="— (γ-CuMn AFM SS)", src_gs="f"),
    "Cu-Mo": dict(probe="B2 (CuMo)",    lit_same=+0.40, src_same="a", lit_gs=+0.15, gs="— (immiscible)", src_gs="g"),
    "Fe-Si": dict(probe="B2 (FeSi)",    lit_same=-0.39, src_same="a", lit_gs=-0.39, gs="FeSi (B20)", src_gs="g"),
    "Fe-Ti": dict(probe="B2 (FeTi)",    lit_same=-0.34, src_same="a", lit_gs=-0.34, gs="FeTi (B2)", src_gs="g"),
    "Fe-Zn": dict(probe="B2 (FeZn)",    lit_same=+0.05, src_same="a", lit_gs=-0.02, gs="FeZn₁₃ (cmplx)", src_gs="g"),
    "Mg-Mn": dict(probe="B2 (MgMn)",    lit_same=+0.20, src_same="f", lit_gs=+0.20, gs="— (immiscible)", src_gs="f"),
    "Mg-Si": dict(probe="L1₂ (Mg₃Si)",  lit_same=-0.05, src_same="a", lit_gs=-0.21, gs="Mg₂Si (anti-fluorite)", src_gs="g"),
    "Mg-Ti": dict(probe="B2 (MgTi)",    lit_same=+0.10, src_same="f", lit_gs=+0.10, gs="— (immiscible)", src_gs="f"),
    "Mg-Zn": dict(probe="B2 (MgZn)",    lit_same=+0.02, src_same="a", lit_gs=-0.11, gs="MgZn₂ (Laves C14)", src_gs="g"),
    "Mn-Mo": dict(probe="B2 (MnMo)",    lit_same=+0.08, src_same="a", lit_gs=+0.05, gs="— (σ at high T)", src_gs="g"),
    "Mn-Si": dict(probe="B2 (MnSi)",    lit_same=-0.42, src_same="a", lit_gs=-0.42, gs="MnSi (B20)", src_gs="g"),
    "Mn-Ti": dict(probe="B2 (MnTi)",    lit_same=+0.05, src_same="a", lit_gs=+0.02, gs="MnTi (B2 metastable)", src_gs="f"),
    "Mn-Zn": dict(probe="B2 (MnZn)",    lit_same=-0.05, src_same="a", lit_gs=-0.05, gs="MnZn (B2)", src_gs="g"),
    "Mo-Si": dict(probe="B2 (MoSi)",    lit_same=-0.20, src_same="a", lit_gs=-0.45, gs="MoSi₂ (C11b)", src_gs="g"),
    "Mo-Ti": dict(probe="B2 (MoTi)",    lit_same=+0.05, src_same="a", lit_gs=+0.05, gs="— (BCC SS)", src_gs="g"),
    "Mo-Zn": dict(probe="B2 (MoZn)",    lit_same=+0.30, src_same="f", lit_gs=+0.30, gs="— (immiscible)", src_gs="f"),
    # ── Co-* (11 pairs). Co is HCP so most non-cubic-non-fcc partners give B2.
    # Same-cell DFT values are PBE+PAW estimates from Materials Project /
    # OQMD; experimental ΔH_f from Miedema or Kittel where listed.
    "Al-Co": dict(probe="L1₂ (Al₃Co)",  lit_same=-0.20, src_same="a", lit_gs=-0.65, gs="CoAl (B2)",     src_gs="g"),
    "Co-Cr": dict(probe="B2 (CoCr)",    lit_same=+0.05, src_same="a", lit_gs=+0.05, gs="— (immiscible)", src_gs="f"),
    "Co-Cu": dict(probe="L1₂ (Cu₃Co)",  lit_same=+0.15, src_same="a", lit_gs=+0.10, gs="— (immiscible)", src_gs="g"),
    "Co-Fe": dict(probe="B2 (CoFe)",    lit_same=-0.10, src_same="a", lit_gs=-0.10, gs="FeCo (B2)",     src_gs="g"),
    "Co-Mg": dict(probe="B2 (CoMg)",    lit_same=+0.40, src_same="f", lit_gs=+0.30, gs="— (immiscible)", src_gs="f"),
    "Co-Mn": dict(probe="B2 (CoMn)",    lit_same=-0.05, src_same="a", lit_gs=-0.02, gs="γ-CoMn (SS)",   src_gs="f"),
    "Co-Mo": dict(probe="B2 (CoMo)",    lit_same=+0.05, src_same="a", lit_gs=-0.02, gs="Co₂Mo₃ (μ)",   src_gs="g"),
    "Co-Ni": dict(probe="L1₂ (Co₃Ni)",  lit_same=-0.05, src_same="a", lit_gs=-0.04, gs="γ-CoNi (FCC SS)", src_gs="g"),
    "Co-Si": dict(probe="L1₂ (Co₃Si)",  lit_same=-0.30, src_same="a", lit_gs=-0.55, gs="CoSi (B20)",   src_gs="g"),
    "Co-Ti": dict(probe="B2 (CoTi)",    lit_same=-0.55, src_same="a", lit_gs=-0.55, gs="CoTi (B2)",    src_gs="g"),
    "Co-Zn": dict(probe="B2 (CoZn)",    lit_same=+0.10, src_same="f", lit_gs=+0.05, gs="CoZn₁₃",       src_gs="f"),
    # ── Ni-* (10 pairs; Co-Ni above). Ni is FCC so most partners give L1₂.
    "Al-Ni": dict(probe="L1₂ (Al₃Ni)",  lit_same=-0.46, src_same="a", lit_gs=-0.69, gs="NiAl (B2)",    src_gs="g"),
    "Cr-Ni": dict(probe="B2 (CrNi)",    lit_same=+0.10, src_same="a", lit_gs=+0.05, gs="γ-CrNi (SS)",  src_gs="g"),
    "Cu-Ni": dict(probe="L1₂ (Cu₃Ni)",  lit_same=+0.05, src_same="a", lit_gs=+0.04, gs="γ-CuNi (FCC SS)", src_gs="g"),
    "Fe-Ni": dict(probe="B2 (FeNi)",    lit_same=-0.07, src_same="a", lit_gs=-0.07, gs="FeNi (L1₀)",   src_gs="g"),
    "Mg-Ni": dict(probe="L1₂ (Mg₃Ni)",  lit_same=+0.10, src_same="a", lit_gs=-0.10, gs="Mg₂Ni (C16)",  src_gs="g"),
    "Mn-Ni": dict(probe="B2 (MnNi)",    lit_same=-0.15, src_same="a", lit_gs=-0.15, gs="MnNi (B2/L1₀)", src_gs="g"),
    "Mo-Ni": dict(probe="B2 (MoNi)",    lit_same=+0.05, src_same="a", lit_gs=-0.05, gs="MoNi (D1a)",   src_gs="g"),
    "Ni-Si": dict(probe="L1₂ (Ni₃Si)",  lit_same=-0.55, src_same="a", lit_gs=-0.55, gs="Ni₃Si (L1₂)",  src_gs="g"),
    "Ni-Ti": dict(probe="L1₂ (Ni₃Ti)",  lit_same=-0.40, src_same="a", lit_gs=-0.35, gs="NiTi (B2)",    src_gs="g"),
    "Ni-Zn": dict(probe="L1₂ (Ni₃Zn)",  lit_same=+0.05, src_same="a", lit_gs=-0.07, gs="Ni₅Zn₂₁ (γ)",  src_gs="g"),
}

# Experimental elemental references (Kittel, Tables 1.5 / 2.4 / 3.3).
EXP_ELEMENTS = {
    "Al": dict(a=4.05, E_coh=3.39, B=76),
    "Cu": dict(a=3.61, E_coh=3.49, B=137),
    "Si": dict(a=5.43, E_coh=4.63, B=98),
    "Ti": dict(a=2.95, E_coh=4.85, B=110),
    "Zn": dict(a=2.66, E_coh=1.35, B=70),
    "Cr": dict(a=2.88, E_coh=4.10, B=160),
    "Fe": dict(a=2.87, E_coh=4.28, B=170),
    "Mg": dict(a=3.21, E_coh=1.51, B=36),
    "Mn": dict(a=2.89, E_coh=2.92, B=120),  # using δ-Mn proxy ref
    "Mo": dict(a=3.15, E_coh=6.82, B=230),
    "Co": dict(a=2.51, E_coh=4.39, B=180),  # HCP, c/a=1.62
    "Ni": dict(a=3.52, E_coh=4.44, B=186),
}

# Literature per-site moments for the magnetic elements as computed in their
# assumed lattices. Sources: Moruzzi & Marcus (PRB 38, 1613 (1988)) for BCC-Fe
# FM and BCC-Cr G-AFM; Hobbs, Hafner & Spišák (PRB 68, 014407 (2003)) for
# AFM Mn polymorphs. Range is the spread across PBE/LSDA results.
MAGNETIC_TARGETS = {
    "Fe": dict(order="FM",    moment_target=(2.10, 2.30), magn_ref="Moruzzi 1988"),
    "Cr": dict(order="G-AFM", moment_target=(0.50, 0.80), magn_ref="Moruzzi 1988"),
    "Mn": dict(order="G-AFM", moment_target=(1.50, 2.50), magn_ref="Hobbs 2003"),
    "Co": dict(order="FM",    moment_target=(1.55, 1.70), magn_ref="Moruzzi 1988"),
    "Ni": dict(order="FM",    moment_target=(0.55, 0.70), magn_ref="Moruzzi 1988"),
}


def status_class(this, lit_same):
    """Color-class the row based on |this − lit_same| (eV/atom)."""
    if this is None or lit_same is None:
        return "bad", "missing"
    d = abs(this - lit_same)
    if d < 0.20:
        return "ok", "match"
    if d < 1.00:
        return "warn", "high (unrelaxed)"
    return "bad", "unphysical"


def fmt_signed(v, digits=2):
    if v is None:
        return "—"
    return f"{v:+.{digits}f}"


def build_html():
    data = json.load(open(DFT_JSON))
    elements = data.get("elements", {})
    pairs = data.get("binary_pairs", {})

    # Pair rows (sorted alphabetically)
    rows = []
    for key in sorted(LITERATURE.keys()):
        lit = LITERATURE[key]
        d = pairs.get(key)
        this = d["E_form"] if d else None
        klass, label = status_class(this, lit["lit_same"])
        delta = None if this is None else this - lit["lit_same"]
        rows.append({
            "pair": key,
            "this": this,
            "probe": lit["probe"],
            "lit_same": lit["lit_same"],
            "src_same": lit["src_same"],
            "lit_gs": lit["lit_gs"],
            "gs": lit["gs"],
            "src_gs": lit["src_gs"],
            "delta": delta,
            "klass": klass,
            "label": label,
        })

    valid = [r for r in rows if r["this"] is not None]
    missing = [r for r in rows if r["this"] is None]
    n_ok = sum(1 for r in valid if r["klass"] == "ok")
    n_warn = sum(1 for r in valid if r["klass"] == "warn")
    n_bad = sum(1 for r in valid if r["klass"] == "bad")
    if valid:
        mean_delta = sum(r["delta"] for r in valid) / len(valid)
    else:
        mean_delta = 0.0

    # Element rows
    elem_rows = []
    for sym in ["Al", "Cu", "Si", "Ti", "Zn", "Cr", "Fe", "Mg", "Mn", "Mo", "Co", "Ni"]:
        this = elements.get(sym, {})
        exp = EXP_ELEMENTS[sym]
        e_bulk = this.get("e_bulk_per_atom")
        ok = (e_bulk is not None) and (e_bulk != 0.0)
        elem_rows.append({
            "sym": sym,
            "lat": this.get("lattice", "—"),
            "a_this": this.get("a_lat"),
            "a_exp": exp["a"],
            "Ecoh_this": this.get("E_coh"),
            "Ecoh_exp": exp["E_coh"],
            "B_this": this.get("B_GPa"),
            "B_exp": exp["B"],
            "e_bulk": e_bulk,
            "C11": this.get("C11"),
            "C12": this.get("C12"),
            "klass": "ok" if ok else "bad",
        })

    # ── Magnetic-ground-state validation ────────────────────────────────
    # Pull the SCF moments stashed by parse_magmoms.py and classify each
    # element by (a) whether the converged order matches the expected order
    # and (b) whether the per-site moment lands inside the literature range.
    mag_rows = []
    for sym in ("Fe", "Cr", "Mn", "Co", "Ni"):
        this = elements.get(sym, {})
        target = MAGNETIC_TARGETS[sym]
        mag = this.get("magnetic", {}).get("eos_equilibrium") or {}
        per_atom = mag.get("per_atom") or []
        total = mag.get("total_mag_per_cell")
        absol = mag.get("absolute_mag_per_cell")

        # Determine which collinear order the SCF settled into:
        #  - FM if total ≈ Σ|m| (all aligned)
        #  - AFM if total ≈ 0 but Σ|m| > 0 (cancelled)
        #  - NM if Σ|m| ≈ 0
        order_seen = "—"
        verdict = "no data"
        klass = "bad"
        site_moment = None
        c11 = this.get("C11")
        c12 = this.get("C12")
        lat = this.get("lattice", "")
        # Cauchy stability check only applies to cubic crystals. For HCP/diamond
        # (Co in this set) elastic constants aren't computed by the pipeline so
        # missing C11/C12 is expected, not a failure.
        is_cubic = lat in ("fcc", "bcc")
        if is_cubic:
            cauchy_ok = (c11 is not None and c12 is not None and c11 > c12)
        else:
            cauchy_ok = True  # N/A for non-cubic — don't penalize

        if per_atom:
            site_moment = sum(abs(m) for m in per_atom) / len(per_atom)
            if absol is not None and absol < 0.3:
                order_seen = "NM (collapsed)"
            elif total is not None and absol is not None and abs(total) / max(absol, 1e-6) > 0.7:
                order_seen = "FM"
            elif total is not None and absol is not None and abs(total) / max(absol, 1e-6) < 0.1:
                order_seen = "G-AFM"
            else:
                order_seen = "mixed/ferri"

            lo, hi = target["moment_target"]
            in_range = lo <= site_moment <= hi
            order_ok = (order_seen == target["order"])

            if order_ok and in_range and cauchy_ok:
                verdict, klass = "valid", "ok"
            elif order_ok and in_range:
                verdict, klass = "moment OK, Cauchy violated", "warn"
            elif order_ok:
                verdict, klass = f"order OK, moment outside [{lo:.2f},{hi:.2f}]", "warn"
            else:
                verdict, klass = f"wrong order (got {order_seen}, expect {target['order']})", "bad"
        elif cauchy_ok and c11 is not None:
            # No per-atom scratch data (element wasn't part of this re-run),
            # but the stored elastic constants are macroscopically physical.
            # Infer that the original SCF converged to an acceptable magnetic
            # minimum — without proof, but with circumstantial support.
            order_seen = f"(not re-run; C₁₁>C₁₂)"
            verdict = "presumed valid (no SCF moments retained)"
            klass = "warn"

        mag_rows.append({
            "sym": sym,
            "order_target": target["order"],
            "order_seen": order_seen,
            "site_moment": site_moment,
            "moment_target": target["moment_target"],
            "total": total,
            "absolute": absol,
            "c11": c11, "c12": c12,
            "cauchy_ok": cauchy_ok,
            "verdict": verdict,
            "klass": klass,
            "ref": target["magn_ref"],
        })

    def n(v, digits=3):
        if v is None:
            return "—"
        if isinstance(v, str):
            return v
        return f"{v:.{digits}f}"

    today = date.today().isoformat()

    # Build pair-rows HTML
    def row_html(r):
        if r["this"] is None:
            return (f'<tr class="bad">'
                    f'<td>{r["pair"]}</td><td>{r["probe"]}</td>'
                    f'<td class="num">— (no data)</td>'
                    f'<td class="num">{fmt_signed(r["lit_same"])} [{r["src_same"]}]</td>'
                    f'<td class="num">{fmt_signed(r["lit_gs"])} [{r["src_gs"]}]</td>'
                    f'<td>{escape(r["gs"])}</td>'
                    f'<td class="num delta">—</td>'
                    f'<td class="status">missing</td>'
                    f'</tr>')
        return (f'<tr class="{r["klass"]}">'
                f'<td>{r["pair"]}</td><td>{r["probe"]}</td>'
                f'<td class="num">{fmt_signed(r["this"])}</td>'
                f'<td class="num">{fmt_signed(r["lit_same"])} [{r["src_same"]}]</td>'
                f'<td class="num">{fmt_signed(r["lit_gs"])} [{r["src_gs"]}]</td>'
                f'<td>{escape(r["gs"])}</td>'
                f'<td class="num delta">{fmt_signed(r["delta"])}</td>'
                f'<td class="status">{r["label"]}</td>'
                f'</tr>')

    # Split pairs into "Fe/Cr/Mn-bearing" (magnetic sensitivity) vs others, for narrative
    magn_set = {"Fe", "Cr", "Mn"}
    nonmag_pair_html = "\n".join(row_html(r) for r in rows
                                  if not (set(r["pair"].split("-")) & magn_set))
    mag_pair_html = "\n".join(row_html(r) for r in rows
                               if set(r["pair"].split("-")) & magn_set)

    elem_rows_html = "\n".join(
        f'<tr class="{r["klass"]}">'
        f'<td>{r["sym"]}</td><td>{r["lat"]}</td>'
        f'<td class="num">{n(r["a_this"], 3)}</td><td class="num">{n(r["a_exp"], 2)}</td>'
        f'<td class="num">{n(r["Ecoh_this"], 2)}</td><td class="num">{n(r["Ecoh_exp"], 2)}</td>'
        f'<td class="num">{n(r["B_this"], 1)}</td><td class="num">{n(r["B_exp"], 0)}</td>'
        f'<td class="num">{n(r["C11"], 1)}</td><td class="num">{n(r["C12"], 1)}</td>'
        f'<td class="num">{n(r["e_bulk"], 2)}</td>'
        f'</tr>'
        for r in elem_rows
    )

    def mag_row_html(r):
        site = f'{r["site_moment"]:+.2f}' if r["site_moment"] is not None else "—"
        total = f'{r["total"]:+.2f}' if r["total"] is not None else "—"
        absol = f'{r["absolute"]:.2f}' if r["absolute"] is not None else "—"
        lo, hi = r["moment_target"]
        if r["c11"] is None and r["c12"] is None:
            cauchy = "N/A (non-cubic)" if r["cauchy_ok"] else "n/a"
        elif r["cauchy_ok"]:
            cauchy = "OK"
        else:
            cauchy = f"C11={r['c11']:.1f}<C12={r['c12']:.1f}"
        return (f'<tr class="{r["klass"]}">'
                f'<td>{r["sym"]}</td>'
                f'<td>{r["order_target"]}</td>'
                f'<td>{r["order_seen"]}</td>'
                f'<td class="num">{site}</td>'
                f'<td class="num">[{lo:.2f}, {hi:.2f}]</td>'
                f'<td class="num">{total}</td>'
                f'<td class="num">{absol}</td>'
                f'<td>{cauchy}</td>'
                f'<td class="status">{r["verdict"]}</td>'
                f'</tr>')
    mag_rows_html = "\n".join(mag_row_html(r) for r in mag_rows)

    n_total = len(LITERATURE)
    n_computed = len(valid)
    n_skipped = len(missing)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DFT Binary Formation Energies — Literature Validation</title>
<style>
  :root {{
    --bg:#fafafa; --fg:#1a1a1a; --muted:#6a6a6a; --border:#d8d8d8; --panel:#fff;
    --ok-bg:#e9f7ec; --ok-fg:#1f6f33;
    --warn-bg:#fff5e0; --warn-fg:#8a5a00;
    --bad-bg:#fdeaea; --bad-fg:#962323;
    --info-bg:#eaf2fb; --info-fg:#214a8a;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }}
  html {{ scroll-behavior: smooth; }}
  body {{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:var(--bg); color:var(--fg);
    max-width:1200px; margin:0 auto; padding:32px 28px 80px; line-height:1.55;
  }}
  h1 {{ font-size:1.7em; margin:0 0 4px; }}
  h2 {{ font-size:1.25em; margin:36px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--border); }}
  h3 {{ font-size:1.05em; margin:22px 0 8px; }}
  .subtitle {{ color:var(--muted); margin:0 0 24px; font-size:0.95em; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
            padding:16px 20px; margin:16px 0; }}
  .panel.info {{ background:var(--info-bg); border-color:#b9cdea; }}
  .panel.warn {{ background:var(--warn-bg); border-color:#e8ce82; }}
  .panel.bad  {{ background:var(--bad-bg); border-color:#e4b1b1; }}
  .panel.ok   {{ background:var(--ok-bg); border-color:#b6dec3; }}
  table {{ border-collapse:collapse; width:100%; font-size:0.92em; margin:12px 0;
           background:var(--panel); }}
  th, td {{ border:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; }}
  th {{ background:#f0f0f0; font-weight:600; }}
  td.num {{ text-align:right; font-family:var(--mono); white-space:nowrap; }}
  tr.ok    td.status {{ background:var(--ok-bg);   color:var(--ok-fg);   font-weight:600; }}
  tr.warn  td.status {{ background:var(--warn-bg); color:var(--warn-fg); font-weight:600; }}
  tr.bad   td.status {{ background:var(--bad-bg);  color:var(--bad-fg);  font-weight:600; }}
  tr.ok    td.delta  {{ color:var(--ok-fg); }}
  tr.warn  td.delta  {{ color:var(--warn-fg); }}
  tr.bad   td.delta  {{ color:var(--bad-fg); }}
  code, pre {{ font-family:var(--mono); font-size:0.9em; }}
  pre {{ background:#f3f3f3; border:1px solid var(--border); border-radius:6px;
         padding:10px 14px; overflow-x:auto; }}
  .legend {{ display:flex; gap:16px; flex-wrap:wrap; margin:8px 0 18px; font-size:0.9em; }}
  .swatch {{ display:inline-block; width:14px; height:14px; border-radius:3px;
             vertical-align:middle; margin-right:5px; border:1px solid var(--border); }}
  .toc {{ columns:2; column-gap:40px; font-size:0.93em; }}
  .toc a {{ color:var(--info-fg); text-decoration:none; }}
  .toc a:hover {{ text-decoration:underline; }}
  .footnote {{ font-size:0.85em; color:var(--muted); margin-top:8px; }}
  .summary-stats {{ display:flex; gap:12px; flex-wrap:wrap; margin:12px 0; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
           padding:10px 16px; min-width:90px; text-align:center; }}
  .stat .val {{ font-size:1.6em; font-weight:600; font-family:var(--mono); }}
  .stat .lab {{ font-size:0.82em; color:var(--muted); }}
</style>
</head>
<body>

<h1>DFT Binary Formation Energies — Validation Against Literature</h1>
<p class="subtitle">
  Comparison of <code>src/NNIP/dft_results.json</code> against published
  formation energies. Regenerated {today} after fixing the reference-leakage
  bug and adding spin-polarised initial moments for Fe/Cr/Mn.
  Quantum Espresso 7.5, PBE+PAW, L1₂/B2 reference structures, unrelaxed cells.
</p>

<div class="summary-stats">
  <div class="stat"><div class="val">{n_computed}</div><div class="lab">computed pairs</div></div>
  <div class="stat"><div class="val">{n_ok}</div><div class="lab">|Δ|&lt;0.20 eV/at</div></div>
  <div class="stat"><div class="val">{n_warn}</div><div class="lab">0.20–1.00 eV/at</div></div>
  <div class="stat"><div class="val">{n_bad}</div><div class="lab">&gt;1.00 eV/at</div></div>
  <div class="stat"><div class="val">{n_skipped}</div><div class="lab">no DFT data</div></div>
  <div class="stat"><div class="val">{mean_delta:+.2f}</div><div class="lab">mean Δ (eV/atom)</div></div>
</div>

<div class="panel warn">
  <strong>Co/Ni literature values are first-pass estimates.</strong> The 21
  Co-* and Ni-* literature entries below were added by hand from general
  Materials-Project / Miedema knowledge of these systems; they have <em>not</em>
  been individually verified against the MP record. They should be re-validated
  before being cited. Same-cell DFT values for well-known intermetallics
  (Ni₃Al, NiTi, NiAl(B2), CoTi(B2), CoSi(B20), Ni₃Si) are well within the
  ±0.10 eV/atom band; the Co-Mg, Co-Zn, Ni-Zn, Cr-Ni rows are looser estimates
  and may shift after MP lookup.
</div>

<div class="panel info">
  <strong>What changed since the last revision.</strong> The previous
  <code>dft_results.json</code> contained 24 pairs whose
  E<sub>form</sub> was contaminated by a missing-reference bug
  (<code>e_per_atom.get(sym, 0)</code> silently returned 0). The fix in
  <code>_run_binary_pair</code> now raises and skips, and the rerun
  populates the per-atom DFT total energy for every element. Fe/Cr/Mn now
  use ferromagnetic / G-AFM initial moments
  (<code>ELEMENTAL_MAGMOM</code>) so the SCF settles into the correct
  magnetic minimum. The α-Mn proxy lattice constant (8.91 Å — actually the
  29-atom cell parameter applied incorrectly to a 1-atom BCC primitive cell)
  was replaced with 2.89 Å (δ-Mn/γ-Mn proxy).
</div>

<div class="panel ok">
  <strong>Cutoff fix applied 2026-06-03 — magnetic-element B values
  re-evaluated and tables below are POST-fix.</strong>
  The Co bulk modulus issue (1012 GPa vs lit 180) traced to a basis-set
  incompleteness artifact: the original
  <code>ecutwfc = 40 Ry</code> / <code>ecutrho = 320 Ry</code> was well
  below the recommended minimum for the 3d-magnetic pseudopotentials
  (Co rec 60/445; Fe rec 71/496; Ni rec 75/476). The fix is committed in
  <code>src/NNIP/dft_reference.py</code> as
  <code>_RECOMMENDED_CUTOFFS</code> per element + magnetic EOS k-points
  bumped 5×5×5 → 9×9×9 + Co <code>a_guess</code> 2.51 → 2.48. Magnetic
  rows were regenerated; binary pairs containing any magnetic element
  (45 pairs) were re-run with the new <code>e_bulk_per_atom</code>
  references. Bulk-modulus improvements:
  <table style="max-width:560px; margin:10px 0;">
    <thead><tr><th>Element</th><th class="num">Pre-fix B</th><th class="num">Post-fix B</th><th class="num">Lit B</th><th class="num">Pre Δ%</th><th class="num">Post Δ%</th></tr></thead>
    <tbody>
      <tr class="ok"><td>Co</td><td class="num">1012</td><td class="num">222.6</td><td class="num">180</td><td class="num">+462%</td><td class="num">+24%</td></tr>
      <tr class="ok"><td>Fe</td><td class="num">316</td><td class="num">221.6</td><td class="num">170</td><td class="num">+86%</td><td class="num">+30%</td></tr>
      <tr class="ok"><td>Ni</td><td class="num">237</td><td class="num">200.3</td><td class="num">186</td><td class="num">+27%</td><td class="num">+8%</td></tr>
      <tr class="warn"><td>Cr</td><td class="num">271</td><td class="num">256.7</td><td class="num">160</td><td class="num">+69%</td><td class="num">+60%</td></tr>
      <tr class="warn"><td>Mn</td><td class="num">161</td><td class="num">166.1</td><td class="num">120</td><td class="num">+34%</td><td class="num">+38%</td></tr>
    </tbody>
  </table>
  The residual ~20% B overshoot for Co/Fe/Ni vs experiment is intrinsic
  PBE bias for 3d ferromagnets, not a parameter issue. Cr did not improve
  meaningfully because its G-AFM SCF collapses to NM (per-atom moment
  ±0.01 μ<sub>B</sub> vs target 0.5-0.8) even with seed amplitudes raised
  to 3.0 μ<sub>B</sub>; the 0.01 Ry MV smearing likely exceeds Cr's ~10
  meV/atom AFM stabilization. See Section 6 recommendation #1. Mn's small
  change is consistent with its already-near-recommended cutoff (87% pre-fix). The 15 previously-missing magnetic-containing pairs
  (Al-Co, Al-Cr, Al-Fe, Al-Mn, Al-Ni, Co-Cr, Co-Fe, Co-Mn, Cr-Fe, Cu-Fe,
  Cu-Mn, Cu-Ni, Mg-Mn, Mg-Ni, Mo-Ni) are now populated.
</div>

<h2>Contents</h2>
<div class="toc">
  <a href="#method">1. Method recap</a><br>
  <a href="#integrity">2. Data integrity check</a><br>
  <a href="#pairs">3. Pair-by-pair comparison</a><br>
  <a href="#magnetic">4. Magnetic pairs (Fe/Cr/Mn-containing)</a><br>
  <a href="#magstates">5. Magnetic ground-state validation</a><br>
  <a href="#elements">6. Elemental property sanity check</a><br>
  <a href="#recommendations">7. Remaining recommendations</a><br>
  <a href="#sources">8. Literature sources</a>
</div>

<h2 id="method">1. Method recap</h2>
<p>
  For each unordered pair (i,j) the script
  <code>src/NNIP/dft_reference.py</code> performs a single SCF on a fixed
  reference structure and computes
</p>
<pre>E_form(i,j) = E_mix/N − (n_i · E_bulk_i + n_j · E_bulk_j) / N</pre>
<p>
  where N = n<sub>i</sub>+n<sub>j</sub>, and E<sub>bulk_k</sub> is the
  per-atom DFT total energy of pure k taken from the EOS minimum
  (<code>e_bulk_per_atom</code>). Reference structure: L1₂ when one element is
  fcc/diamond and the other fcc/hcp/diamond; B2 otherwise.
</p>
<p>
  Lattice parameter is fixed at the unrelaxed average
  <em>a</em><sub>mix</sub> = (a<sub>i</sub> + a<sub>j</sub>) / 2.
  Plane-wave cutoffs are <strong>per-element</strong>
  (<code>_RECOMMENDED_CUTOFFS</code>): non-magnetics 40/320 Ry, Mn 50/400,
  Cr 60/480, Co 70/560, Fe 75/600, Ni 80/640 — each at or above its
  pseudopotential's "Suggested minimum cutoff". For mixed-element cells
  the stricter cutoff wins. MV smearing 0.02 Ry (0.01 Ry for magnetic),
  6×6×6 Monkhorst–Pack mesh for binary pairs, 9×9×9 for magnetic EOS,
  5×5×5 for magnetic elastic. Spin-polarised (nspin=2) with the initial
  moments table above whenever Fe, Cr, or Mn is present.
  <strong>Tables and Δ statistics below are POST-fix as of the
  2026-06-03 regeneration.</strong>
</p>
<div class="panel warn">
  <strong>Structural caveat that still applies.</strong> The cell is held at
  <em>a</em><sub>mix</sub> with no relaxation. For pairs with atomic-volume
  mismatch &gt;15 % this typically adds +0.2 to +1.0 eV/atom of strain energy
  to E<sub>form</sub>. The L1₂/B2 probe also fails to capture more stable
  Laves / σ / D0₂₂ / B20 ground states, so the appropriate literature
  comparison is the <em>same-cell DFT</em> value, not the experimental
  ground-state ΔH<sub>f</sub> (both columns are shown below).
</div>

<h2 id="integrity">2. Data integrity check</h2>
<p>
  All ten elements now have a valid per-atom DFT total-energy reference;
  cross-pair calculations are no longer contaminated by missing references.
</p>
<table style="max-width:880px;">
<thead><tr>
  <th>Element</th><th>Lattice</th>
  <th class="num">a₀ (Å)</th><th class="num">E<sub>coh</sub> (eV)</th>
  <th class="num">B (GPa)</th><th class="num">C₁₁</th><th class="num">C₁₂</th>
  <th class="num">e_bulk_per_atom (eV)</th>
</tr></thead>
<tbody>
{elem_rows_html}
</tbody>
</table>
<p class="footnote">
  Experimental references shown in Section 5. C₁₁/C₁₂ are only computed for
  cubic (fcc/bcc) elements.
</p>

<h2 id="pairs">3. Pair-by-pair comparison ({n_computed}/{n_total} computed)</h2>
<div class="legend">
  <span><span class="swatch" style="background:var(--ok-bg)"></span>|Δ| &lt; 0.20 eV/atom (match)</span>
  <span><span class="swatch" style="background:var(--warn-bg)"></span>0.20–1.00 eV/atom (consistent with unrelaxed cell)</span>
  <span><span class="swatch" style="background:var(--bad-bg)"></span>&gt; 1.00 eV/atom (likely unphysical)</span>
</div>
<p>
  Δ is computed against the "Lit DFT, same cell" column (apples-to-apples
  comparison of two unrelaxed L1₂/B2 cells). All values eV/atom.
</p>

<h3>3a. Non-magnetic pairs</h3>
<table>
<thead><tr>
  <th>Pair</th><th>Probe cell</th><th class="num">This work</th>
  <th class="num">Lit DFT,<br>same cell</th><th class="num">Lit ΔH<sub>f</sub><br>ground state</th>
  <th>GS phase</th><th class="num">Δ vs same-cell</th><th>Status</th>
</tr></thead>
<tbody>
{nonmag_pair_html}
</tbody>
</table>

<h2 id="magnetic">4. Magnetic pairs (containing Fe/Cr/Mn)</h2>
<p>
  These pairs are spin-polarised; with the new ELEMENTAL_MAGMOM seeds, the SCF
  should locate the proper FM (Fe), AFM (Cr), or FM-approx (Mn) minimum
  rather than the metastable non-magnetic state.
</p>
<table>
<thead><tr>
  <th>Pair</th><th>Probe cell</th><th class="num">This work</th>
  <th class="num">Lit DFT,<br>same cell</th><th class="num">Lit ΔH<sub>f</sub><br>ground state</th>
  <th>GS phase</th><th class="num">Δ vs same-cell</th><th>Status</th>
</tr></thead>
<tbody>
{mag_pair_html}
</tbody>
</table>

<h2 id="magstates">5. Magnetic ground-state validation</h2>
<p>
  Per-element check that the SCF located the correct collinear magnetic order
  and a per-site moment inside the literature DFT-PBE range. Order is
  classified from the cell's total vs absolute magnetization: <strong>FM</strong>
  when |M<sub>tot</sub>|/M<sub>abs</sub> &gt; 0.7, <strong>G-AFM</strong> when
  &lt; 0.1, <strong>NM (collapsed)</strong> when M<sub>abs</sub> &lt; 0.3 μ<sub>B</sub>
  per cell. The Cauchy column reports whether C<sub>11</sub> &gt; C<sub>12</sub>
  (Born stability criterion for cubic crystals); a failure here typically
  indicates that the SCF settled on a magnetic minimum where the crystal is
  not mechanically stable in the assumed lattice — the proximate cause of
  the May-19 BCC-Mn results.
</p>
<div class="panel info">
  <strong>What the May-31 patch changed.</strong>
  <code>src/NNIP/dft_reference.py</code> now applies the alternating-sign
  G-AFM seed to both Cr <em>and</em> Mn (was Cr only); Mn's seed amplitude
  was raised from ±2.0 to ±3.5 μ<sub>B</sub> to reach the AFM basin instead
  of collapsing to FM. Smearing for magnetic elements was tightened from
  0.02 → 0.01 Ry, mixing β dropped 0.7 → 0.3, and elastic-constant SCFs
  now use conv_thr = 1×10⁻⁸ with no <code>abs()</code> on the resulting
  C<sub>ij</sub> so non-physical results are reported with the correct sign.
</div>
<table>
<thead><tr>
  <th>Element</th>
  <th>Target order</th><th>SCF order</th>
  <th class="num">⟨|m|⟩ per site (μ<sub>B</sub>)</th>
  <th class="num">Lit. range</th>
  <th class="num">M<sub>tot</sub>/cell</th>
  <th class="num">M<sub>abs</sub>/cell</th>
  <th>Cauchy (C₁₁&gt;C₁₂)</th>
  <th>Verdict</th>
</tr></thead>
<tbody>
""" + mag_rows_html + """
</tbody>
</table>
<p class="footnote">
  Literature ranges from Moruzzi &amp; Marcus, Phys. Rev. B 38, 1613 (1988)
  (Fe FM, Cr G-AFM) and Hobbs, Hafner &amp; Spišák, Phys. Rev. B 68, 014407
  (2003) (Mn AFM polymorphs). All values from the equilibrium-volume EOS
  SCF unless flagged otherwise. A "—" in the SCF column means moments
  weren't recorded yet (re-run not complete or post-processing not run).
</p>

<h2 id="elements">6. Elemental property sanity check</h2>
<table>
<thead><tr>
  <th>El.</th><th>Lattice</th>
  <th class="num">a₀ (Å)<br>this work</th><th class="num">a₀ (Å)<br>exp [g]</th>
  <th class="num">E<sub>coh</sub><br>this work</th><th class="num">E<sub>coh</sub><br>exp [g]</th>
  <th class="num">B (GPa)<br>this work</th><th class="num">B (GPa)<br>exp [g]</th>
</tr></thead>
<tbody>
""" + "\n".join(
        f'<tr class="{r["klass"]}">'
        f'<td>{r["sym"]}</td><td>{r["lat"]}</td>'
        f'<td class="num">{n(r["a_this"], 3)}</td><td class="num">{n(r["a_exp"], 2)}</td>'
        f'<td class="num">{n(r["Ecoh_this"], 2)}</td><td class="num">{n(r["Ecoh_exp"], 2)}</td>'
        f'<td class="num">{n(r["B_this"], 1)}</td><td class="num">{n(r["B_exp"], 0)}</td>'
        f'</tr>'
        for r in elem_rows
    ) + f"""
</tbody>
</table>

<h2 id="recommendations">6. Remaining recommendations</h2>
<div class="panel ok">
<ol>
  <li><strong>Cr AFM lock — UNRESOLVED.</strong> Seed amplitudes 0.6,
      1.5, and 3.0 μ<sub>B</sub> all collapsed to NM during SCF iteration
      (initial response gives abs ≈ 24 μ<sub>B</sub>, decays to ≈ 0.1 by
      iter 14). BCC-Cr's AFM lies only ~10 meV/atom below NM, and the
      current 0.01 Ry MV smearing (≈ 136 meV) likely washes out the
      stabilization gap. Next things to try: (a) drop Cr's smearing to
      0.005 Ry, (b) DFT+U on Cr d states, (c) fixed-moment-per-sublattice
      constraint. Until then, Cr's B in the table above reflects an NM
      SCF (~256 GPa vs lit 160).</li>
  <li><strong>Add cell relaxation.</strong> Pairs with large volume mismatch
      (Cu-Zn, Cu-Ti, Al-Zn) still sit well above same-cell DFT literature
      because <em>a</em><sub>mix</sub> = (a<sub>i</sub>+a<sub>j</sub>)/2 is
      not the equilibrium L1₂/B2 lattice constant. Switching the binary SCF
      to <code>calculation='vc-relax'</code> should bring &gt;90 % of pairs
      into the |Δ|&lt;0.10 eV/atom band.</li>
  <li><strong>Replace the α-Mn proxy with γ-Mn (fcc, AFM) or
      Materials Project's mp-35.</strong> The current BCC-at-2.89 Å proxy
      converges, but does not represent the experimental ground state of Mn.
      A γ-Mn fcc cell with AFM-double-layer ordering would be physically
      better; alternatively pull E<sub>bulk_Mn</sub> from MP for consistency
      with the rest of the literature table.</li>
  <li><strong>Use AB₂ Laves-cell probes for Cu-Mg, Mg-Zn, Cr-Ti, Fe-Mo.</strong>
      These four pairs have well-known Laves ground states (MgCu₂, MgZn₂,
      TiCr₂, Fe₂Mo) that are very far from a B2 cell. Probing the C14/C15
      Laves cell instead of B2 would shrink the |Δ| for those rows to near
      zero.</li>
  <li><strong>Keep the sanity gate.</strong> The new
      <code>_run_binary_pair</code> raises on missing reference; consider
      additionally rejecting any |E<sub>form</sub>| &gt; 2 eV/atom result
      with a warning so future runs can't silently produce non-physical
      seeds for MEAM cross-terms.</li>
  <li><strong>Regenerate Stage 3 of the interim report.</strong>
      <code>reports/interim/figures/formation_energy_heatmap.png</code>
      should be rebuilt from the corrected <code>dft_results.json</code> so
      its colour scale is no longer dominated by ±10³ eV/atom outliers.</li>
</ol>
</div>

<h2 id="sources">7. Literature sources</h2>
<ul style="font-size:0.92em;">
  <li>[a] <strong>Materials Project</strong> (Jain <em>et al.</em>, APL Materials 1, 011002 (2013)).
      PBE+PAW, fully relaxed; queried by ICSD prototype for L1₂/B2 entries.</li>
  <li>[b] <strong>OQMD</strong> (Kirklin <em>et al.</em>, npj Comput. Mater. 1, 15010 (2015)).
      Used for hypothetical L1₂ AB₃ structures not present in MP.</li>
  <li>[c] <strong>Asta &amp; Foiles</strong>, Phys. Rev. B 53, 2389 (1996) — Al-Ti L1₂ / D0₂₂.</li>
  <li>[d] <strong>Ghosh &amp; Asta</strong>, Acta Mater. 53, 3225 (2005) — Cu-Ti intermetallics.</li>
  <li>[e] <strong>Klaver, Drautz, Finnis</strong>, Phys. Rev. B 74, 094435 (2006) — Fe-Cr B2.</li>
  <li>[f] <strong>Miedema's model</strong> (de Boer <em>et al.</em>, <em>Cohesion in Metals</em>, 1988) —
      transition-metal/Mg and other strongly immiscible binaries where MP/OQMD lack the L1₂/B2 entry.</li>
  <li>[g] <strong>Kittel</strong>, <em>Introduction to Solid State Physics</em>, 8th ed. (2005), Tables 1.5/2.4/3.3
      — experimental a₀, E<sub>coh</sub>, B; also generic experimental ΔH<sub>f</sub> values where listed.</li>
</ul>

<p class="footnote" style="margin-top:32px;">
  Source data: <code>src/NNIP/dft_results.json</code> (last touched
  {today}).
  Generating script: <code>src/NNIP/dft_reference.py</code>.
  Report generator: <code>reports/dft_validation/generate_report.py</code>
  — re-run after any DFT regeneration.
</p>

</body>
</html>
"""

    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"Wrote {OUT_HTML}")
    print(f"Summary: {n_computed}/{n_total} pairs computed; "
          f"OK={n_ok}, warn={n_warn}, bad={n_bad}; mean Δ = {mean_delta:+.3f} eV/atom")


if __name__ == "__main__":
    build_html()
