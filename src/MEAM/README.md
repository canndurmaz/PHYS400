# src/MEAM — On-demand MD predictor (optimized MEAM)

Flask app that runs a real LAMMPS strain sweep against the **optimized**
MEAM at `EAM/optimized/optimized_*.meam` for any composition over the
12-element basis. Companion to `src/ML/` (which serves the NN surrogate).

## Run

```bash
./src/MEAM/run.sh
# http://127.0.0.1:5001
```

Requires Stage 4 of the NNIP pipeline to have produced
`EAM/optimized/optimized_library_AlCoCrCuFeMgMnMoNiSiTiZn.meam` and
`EAM/optimized/optimized_AlCoCrCuFeMgMnMoNiSiTiZn.meam`. Boot will fail
loudly if either is missing.

## Endpoints

| Method | Path                       | Purpose                                  |
|--------|----------------------------|------------------------------------------|
| GET    | `/`                        | Composition form                         |
| POST   | `/api/predict`             | Submit a job (returns job_id or cache)   |
| GET    | `/api/jobs/<id>`           | Poll status, log tail, thermo, result    |
| GET    | `/api/renders/<cache_key>` | OVITO PNG (if do_viz was true)           |
| GET    | `/api/jobs`                | Last 20 jobs                             |

## Cache

Completed jobs are persisted to `src/MEAM/runs.json`, keyed by
`sha1({composition, knobs, do_viz, max_meam_mtime})`. Re-running Stage 4
invalidates every cached entry automatically because the MEAM mtime
changes. Delete `runs.json` to clear by hand.

OVITO renders are written to `src/MEAM/renders/<cache_key>.png` and served
via `/api/renders/<cache_key>`. A cache hit with `do_viz` set but no PNG on
disk (earlier render failure, or a deleted file) re-runs the job instead of
returning the render-less cached result.

## Tests

```bash
./phys/bin/python3 -m pytest tests/meam/ -v
# Includes the gated LAMMPS smoke test if you set:
RUN_LAMMPS_TESTS=1 LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH \
  ./phys/bin/python3 -m pytest tests/meam/ -v -s
```

## Manual checklist

- [ ] Submit AL7075 (`Al=0.91, Mg=0.029, Zn=0.061`) → job_id → result within ~30 s. Cards populated; log tail non-empty; thermo trace has ≥ 3 points.
- [ ] Same composition again → `cache hit`, instant result.
- [ ] `touch EAM/optimized/optimized_*.meam` → next submit re-runs (cache key invalidated).
- [ ] Composition with element outside basis (edit DevTools to add `Xx`) → HTTP 400 with helpful message.
- [ ] Tick OVITO render → PNG appears in the result panel within an extra ~10 s.
