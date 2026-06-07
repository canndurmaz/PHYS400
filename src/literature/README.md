# src/literature — Literature comparison (MEAM & ML vs ASM/MatWeb)

Scrapes experimental elastic data for ~40 basis-covered engineering alloys
(33 aluminium + CP-Ti + stainless + Ni–Fe) from the **ASM Aerospace
Specification Metals** datasheet host (`asm.matweb.com`), feeds the *same*
composition to both project predictors, and writes a single
`literature_comparison.json` with per-material errors and aggregate
(overall + per-family) MAE/MAPE.

The final presentation consumes this JSON: `reports/finalPresentation/
generate_literature_figure.py` renders two booktabs tables
(`sections/_auto_literature_family.tex`, `_auto_literature_al.tex`) shown on
the "External Validation — Quantitative Summary" slide. Regenerate via
`REGEN=1 ./compile.sh` in that directory.

```
ASM/MatWeb datasheet ──scrape──► composition (wt% → mole fraction over basis)
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                             ▼
        ML surrogate (src/ML)            optimized MEAM (src/MEAM/runner.py)
        instant ONNX forward pass        real LAMMPS strain sweep (~6 s each)
                        └─────────────┬─────────────┘
                                      ▼
                       error vs literature E and ν → JSON
```

## Run

```bash
# from the repo root, using the project venv
./phys/bin/python3 src/literature/compare.py             # ML + MEAM (default)
./phys/bin/python3 src/literature/compare.py --no-meam   # ML only (seconds)
./phys/bin/python3 src/literature/compare.py --refresh   # re-pull datasheets
```

Flags: `--no-meam`, `--refresh`, `--limit N`, `--min-coverage 0..1`,
`--box <metres>` (MEAM cubic cell edge, default `2.5e-9`).

## Where the data comes from (and why this host)

The URL the project was pointed at — `asm.matweb.com/search/SpecificMaterial.asp?bassnum=...`
— is the **ASM Aerospace Specification Metals** mirror, *not* the main
`www.matweb.com` site.

| Host | Behaviour |
|------|-----------|
| `www.matweb.com` | **HTTP 403** to any non-browser client (active anti-bot); TOS forbids automated extraction. Not used. |
| `asm.matweb.com` | Plain static HTML tables, not bot-walled. Only wart: a missing TLS *intermediate* cert, so we fetch with `verify=False`. |

Every page is fetched **once**, politely (1 s delay), and cached to
`html_cache/<bassnum>.html`. After the first run the comparison is fully
**offline and reproducible** — re-running never touches the network unless you
pass `--refresh`. This matches the project's offline/self-contained rule.

To add materials, append verified `bassnum` IDs to `SEED_IDS` in
`materials.py` (grouped by family). Dead IDs return HTTP 500 and are skipped
gracefully.

## Files

| File | Role |
|------|------|
| `materials.py` | Seed `bassnum` IDs (grouped by family) + 12-element atomic masses + family lookup. |
| `matweb_scraper.py` | Fetch + on-disk cache + HTML parse; wt% → basis mole fractions with coverage flag. |
| `compare.py` | Driver: scrape → ML + MEAM → per-material errors + aggregates → JSON. |
| `html_cache/` | Cached raw datasheets (enables offline reruns). |
| `literature_comparison.json` | **Output.** Metadata, summary (overall + by-family), per-material records, skips. |

## Output schema (`literature_comparison.json`)

```jsonc
{
  "metadata": { "generated", "source", "basis", "n_materials",
                "n_skipped", "meam_enabled", "meam_unphysical", "meam_knobs", "notes" },
  "summary": {
    "ML":   { "overall": {E_GPa:{n,MAE,MAPE,bias_pct}, nu:{...}}, "by_family": {...} },
    "MEAM": { "overall": {...}, "by_family": {...} }
  },
  "materials": [ {
      "bassnum", "name", "url", "source", "family",
      "lit": {E_GPa, nu, G_GPa, density_g_cc},
      "composition_wt_pct", "composition", "basis_coverage",
      "predictions": { "ML": {...}, "MEAM": {...physical, physical_reason} },
      "comparison": { "ML": {E_GPa:{pred,lit,abs_err,pct_err}, nu:{...}},
                      "MEAM": {...} }
  } ],
  "skipped": [ {bassnum, reason} ]
}
```

## Caveats (read before citing the numbers)

1. **Single-crystal vs polycrystalline.** Both models emit single-crystal
   *cubic* `C11/C12`; datasheet `E`/`ν` are polycrystalline aggregates. The
   comparison is on the derived **E and ν**, the physically appropriate
   common ground.
2. **Out-of-basis elements are dropped.** The 12-element basis has no C, V, N,
   O… so carbon in steels and vanadium in Ti-6Al-4V are removed before
   modelling. `basis_coverage` reports the fraction of mass retained — discount
   any material below ~0.97.
3. **MEAM cubic-strain assumption.** `runner.py` builds the dominant element's
   lattice and extracts *cubic* elastic constants. For **hcp-dominant** alloys
   (CP titanium, Mg-, Zn-rich) this is structurally invalid and the MEAM E is
   not trustworthy (CP-Ti comes out ~2.5× too stiff).
4. **Potential training domain.** The optimized MEAM was fit for Al-rich HEA
   compositions; it transfers well to **aluminium alloys** but produces
   wildly wrong or **mechanically unstable** (negative E, `physical:False`)
   results for **ferrous/stainless** chemistries. Unphysical MEAM runs are
   kept per-material (flagged) but excluded from the MEAM aggregates so they
   don't poison the MAPE.

In short: the aluminium-family numbers are the meaningful headline; the Ti and
steel rows are valuable precisely because they expose where each model's
assumptions break.
