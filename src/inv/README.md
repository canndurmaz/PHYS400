# `inv/` — Inverse Alloy Design

Given a target **Young's modulus `E`** and **Poisson's ratio `ν`**, suggest
candidate alloy compositions that achieve it. This is the inverse of the
forward surrogate in [`../ML/`](../ML/): where that model maps
`composition → (C11, C12) → (E, ν)`, this one maps `(E, ν) → composition`.

It deliberately mirrors the ML branch's methodology — same element basis,
same `C_ij`-space convention, same train/val split, deep ensemble, Huber
loss, parity-plot metrics, and a sibling Flask app.

## Why it's not a plain regression

The inverse map is **one-to-many**: many compositions realise the same
`(E, ν)`, and some targets aren't achievable at all. A direct
`(E, ν) → composition` regression would average over all valid answers and
return a blurry, often unphysical alloy.

Instead the network is trained with a **forward-consistency (cycle) loss**:

```
target (C11,C12)  →  inv-net  →  composition  →  [frozen ML forward]  →  recon (C11,C12)
                                                   └── Huber(recon, target) ──┘
```

Any composition that reconstructs the target is rewarded — the correct
objective for a one-to-many map. A small composition-reconstruction term
(MSE against the real alloy) keeps proposals near realistic, few-element
compositions; an optional entropy penalty (`INV_ENTROPY_WEIGHT`) pushes
toward sparser alloys. The 12-d output is a **softmax**, so proposals are
non-negative and sum to 1 automatically.

The forward model used for both the training loss and the inference-time
check is **ML's trained ensemble** (`../ML/alloy_model_*`), so there's a
single source of truth for the composition→property map. **Train the
forward surrogate first** (`../ML/run_nn.sh`) — `inv` loads it frozen.

## Files

| File | Role |
|------|------|
| `inv_design.py` | Trainer. Loads `../ML/results.json`, trains the inverse ensemble against the cycle loss, writes `inv_model_*.keras`, round-trip metrics (`inv_metrics.json`) and parity plots (`plots/`). Mirror of `../ML/nn_alloy.py`. |
| `suggest.py` | Runtime suggester. Inverse-net proposals (one per ensemble member → diverse) + nearest real training alloys (retrieval), constraint projection, optional gradient-free refinement, forward-check + rank. Mirror of `../ML/predict_from_model.py`. |
| `app.py` | Flask UI on **port 5002** (ML=5000, MEAM=5001). |
| `run_inv.sh` | Train wrapper (mirror of `../ML/run_nn.sh`). |
| `templates/`, `static/` | Single-page UI; shares the ML stylesheet for a common look. |

## Usage

```bash
# 1. Train the forward surrogate first (provides the frozen cycle-loss model)
../ML/run_nn.sh

# 2. Train the inverse ensemble (5 members by default)
./run_inv.sh
ENSEMBLE_SIZE=1 ./run_inv.sh            # single model, quick
INV_MAX_EPOCHS=120 INV_PATIENCE=40 ./run_inv.sh   # fast iteration

# 3a. CLI suggestions
../../phys/bin/python3 suggest.py 70 0.33
../../phys/bin/python3 suggest.py 75 0.33 forbid=Mg require=Cu

# 3b. Web app  →  http://127.0.0.1:5002
../../phys/bin/python3 app.py
```

### Tunable env vars (training)

| Var | Default | Meaning |
|-----|---------|---------|
| `ENSEMBLE_SIZE` | 5 (`run_inv.sh`) | independently-trained members → candidate diversity |
| `INV_RECON_WEIGHT` | 1.0 | weight of the forward-consistency Huber term |
| `INV_COMP_WEIGHT` | 0.5 | weight of the composition-reconstruction MSE term |
| `INV_ENTROPY_WEIGHT` | 0.0 | sparsity penalty on the softmax output (off by default) |
| `INV_MAX_EPOCHS` / `INV_PATIENCE` | 4000 / 200 | training length / early-stop patience |

## API

`POST /api/suggest`

```json
{ "E_GPa": 70, "nu": 0.33, "forbid": ["Mg"], "require": ["Cu"], "k": 6, "refine": true }
```

Returns `target`, `constraints`, an `achievability` flag (how close the
target sits to the training `(E, ν)` cloud), and a ranked `candidates` list.
Each candidate carries its forward-predicted `(E, ν, C11, C12)`, ensemble σ,
the `e_err_pct` / `nu_err_pct` versus the target, and its `source`
(`inverse-net` or `retrieval`).
