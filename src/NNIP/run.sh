#!/bin/bash
# Run the full NNIP pipeline: generate configs → DFT → convert → train → validate → MD
#
# Usage:
#   bash src/NNIP/run.sh              # Full pipeline
#   bash src/NNIP/run.sh --clean      # Clear all data and start fresh
#   bash src/NNIP/run.sh --from dft   # Resume from a specific phase
#   bash src/NNIP/run.sh --only md    # Run a single phase
#
# Phases: pseudos, configs, dft, convert, train, validate, md

set -euo pipefail

PROJECT="/home/kenobi/Workspaces/PHYS400"
VENV="$PROJECT/phys"
NNIP="$PROJECT/src/NNIP"

# ── Parse arguments ──────────────────────────────────────────

CLEAN=false
FROM_PHASE=""
ONLY_PHASE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        --from)
            FROM_PHASE="$2"
            shift 2
            ;;
        --only)
            ONLY_PHASE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash src/NNIP/run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean        Remove all generated data before running"
            echo "  --from PHASE   Start from a specific phase (skip earlier ones)"
            echo "  --only PHASE   Run only a single phase"
            echo ""
            echo "Phases (in order):"
            echo "  pseudos   Download & validate pseudopotentials"
            echo "  configs   Generate atomic configurations"
            echo "  dft       Run QE DFT calculations (resumable)"
            echo "  convert   Convert results to DeePMD format"
            echo "  train     Train the neural network"
            echo "  validate  Validate against held-out data"
            echo "  md        Run MD simulation + OVITO visualization"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help)"
            exit 1
            ;;
    esac
done

# ── Phase ordering ───────────────────────────────────────────

PHASES=(pseudos configs dft convert train validate md)

should_run() {
    local phase="$1"

    if [[ -n "$ONLY_PHASE" ]]; then
        [[ "$phase" == "$ONLY_PHASE" ]]
        return
    fi

    if [[ -n "$FROM_PHASE" ]]; then
        local started=false
        for p in "${PHASES[@]}"; do
            if [[ "$p" == "$FROM_PHASE" ]]; then
                started=true
            fi
            if $started && [[ "$p" == "$phase" ]]; then
                return 0
            fi
        done
        return 1
    fi

    return 0
}

# ── Activate venv ────────────────────────────────────────────

source "$VENV/bin/activate"
cd "$PROJECT"

echo "============================================================"
echo "NNIP Pipeline"
echo "============================================================"
echo "Python: $(which python)"
echo "Working dir: $PROJECT"
echo ""

# ── Clean ────────────────────────────────────────────────────

if $CLEAN; then
    echo "[CLEAN] Removing generated data..."
    rm -rf data/training/configs/ data/training/dft_results/ data/training/deepmd/
    rm -rf data/training/qe_scratch/
    rm -rf data/validation/
    rm -rf models/
    rm -f src/NNIP/md_traj.*
    rm -rf src/NNIP/visualization/
    rm -rf src/NNIP/pseudo_test_output/
    echo "[CLEAN] Done."
    echo ""
fi

# ── Phase 1: Pseudopotentials ────────────────────────────────

if should_run pseudos; then
    echo "============================================================"
    echo "[Phase 1] Download & Validate Pseudopotentials"
    echo "============================================================"
    python "$NNIP/download_pseudos.py"
    echo ""
fi

# ── Phase 2a: Generate Configurations ────────────────────────

if should_run configs; then
    echo "============================================================"
    echo "[Phase 2a] Generate Atomic Configurations"
    echo "============================================================"
    python "$NNIP/generate_configs.py"
    echo ""
fi

# ── Phase 2b: Run DFT ───────────────────────────────────────

if should_run dft; then
    echo "============================================================"
    echo "[Phase 2b] Run DFT Calculations"
    echo "============================================================"
    echo "This step is long. Press Ctrl-C to interrupt — re-run to resume."
    echo ""
    python "$NNIP/run_dft.py"
    echo ""
fi

# ── Phase 2c: Convert to DeePMD ─────────────────────────────

if should_run convert; then
    echo "============================================================"
    echo "[Phase 2c] Convert to DeePMD Format"
    echo "============================================================"
    python "$NNIP/convert_to_deepmd.py"
    echo ""
fi

# ── Phase 4: Train ──────────────────────────────────────────

if should_run train; then
    echo "============================================================"
    echo "[Phase 4] Train Neural Network"
    echo "============================================================"
    mkdir -p "$PROJECT/models"
    cd "$PROJECT/models"
    dp train "$NNIP/input.json" 2>&1 | tee train.log
    dp freeze -o model.pb
    cd "$PROJECT"
    echo ""
fi

# ── Phase 5: Validate ───────────────────────────────────────

if should_run validate; then
    echo "============================================================"
    echo "[Phase 5] Validate Model"
    echo "============================================================"
    python "$NNIP/validate.py"
    echo ""
fi

# ── Phase 6: MD + Visualization ──────────────────────────────

if should_run md; then
    echo "============================================================"
    echo "[Phase 6] MD Simulation + Visualization"
    echo "============================================================"
    python "$NNIP/run_md.py"
    echo ""
fi

echo "============================================================"
echo "Pipeline complete."
echo "============================================================"
