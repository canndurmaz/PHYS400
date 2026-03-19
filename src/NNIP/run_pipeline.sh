#!/usr/bin/env bash
# Run the full DFT-driven N-element MEAM potential pipeline.
#
# Usage (from src/NNIP/):
#   ./run_pipeline.sh                        # Full pipeline with GUI
#   ./run_pipeline.sh Al Cu Zn Mg            # Skip GUI, specify elements
#   ./run_pipeline.sh --skip-dft Al Cu Zn Mg # Skip DFT stage
#
# Options (place before element list):
#   --skip-dft       Use existing dft_results.json instead of running QE
#   --skip-optimize  Skip NN optimization stage
#   --skip-verify    Skip verification stage
#   --samples N      Number of NN parameter samples (default: 30)
#   --parallel N     Max parallel DFT workers (default: 4)

set -euo pipefail

PROJECT_ROOT="$(cd ../.. && pwd)"
VENV="$PROJECT_ROOT/phys"
LOG_DIR="./logs"

# ── Activate venv ────────────────────────────────────────────────────────────
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Create it with: python -m venv --system-site-packages $VENV"
    exit 1
fi

# ── Parse flags vs element arguments ─────────────────────────────────────────
FLAGS=()
ELEMENTS=()

for arg in "$@"; do
    case "$arg" in
        --skip-dft|--skip-optimize|--skip-verify)
            FLAGS+=("$arg")
            ;;
        --samples|--parallel)
            FLAGS+=("$arg")
            ;;
        [0-9]*)
            # Number following --samples or --parallel
            if [[ "${FLAGS[-1]:-}" == "--samples" || "${FLAGS[-1]:-}" == "--parallel" ]]; then
                FLAGS+=("$arg")
            else
                ELEMENTS+=("$arg")
            fi
            ;;
        *)
            ELEMENTS+=("$arg")
            ;;
    esac
done

# Build python args
PYARGS=()
PYARGS+=("${FLAGS[@]+"${FLAGS[@]}"}")
if [ ${#ELEMENTS[@]} -gt 0 ]; then
    PYARGS+=("--elements" "${ELEMENTS[@]}")
fi

# ── Setup logging ────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

echo "============================================================"
echo " DFT-Driven MEAM Potential Pipeline"
echo " $(date)"
echo "============================================================"
echo " Project root: $PROJECT_ROOT"
echo " Python:       $(python --version 2>&1)"
echo " Log file:     $LOGFILE"
if [ ${#ELEMENTS[@]} -gt 0 ]; then
    echo " Elements:     ${ELEMENTS[*]}"
else
    echo " Elements:     (GUI selection)"
fi
echo " Flags:        ${FLAGS[*]+"${FLAGS[*]}"}"
echo "============================================================"
echo ""

# ── Ensure merged MEAM potentials exist ──────────────────────────────────────
EAM_DIR="$PROJECT_ROOT/EAM"
MERGE_CONFIG="$PROJECT_ROOT/src/configs/meam_merge_7075.json"
MERGED_LIB="$EAM_DIR/library_AlZnMgCuCrFeMnSiTi.meam"

if [ -f "$MERGE_CONFIG" ] && [ ! -f "$MERGED_LIB" ]; then
    echo "Merged MEAM potentials not found. Running merge_potentials.py..."
    python "$PROJECT_ROOT/src/NNIP/merge_potentials.py" --config "$MERGE_CONFIG" --eam-dir "$EAM_DIR" --output-dir "$EAM_DIR"
    echo ""
fi

# ── Run pipeline ─────────────────────────────────────────────────────────────
python "$PROJECT_ROOT/src/NNIP/pipeline.py" "${PYARGS[@]+"${PYARGS[@]}"}" 2>&1 | tee "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline finished successfully. Log saved to $LOGFILE"
else
    echo "Pipeline failed (exit code $EXIT_CODE). Check $LOGFILE for details."
fi

exit $EXIT_CODE
