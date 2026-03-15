#!/usr/bin/env bash
# Run the full DFT-driven N-element MEAM potential pipeline.
#
# Usage:
#   ./run_pipeline.sh                        # Full pipeline with GUI
#   ./run_pipeline.sh Al Cu Zn Mg            # Skip GUI, specify elements
#   ./run_pipeline.sh --skip-dft Al Cu Zn Mg # Skip DFT stage
#
# Options (place before element list):
#   --skip-dft       Use existing dft_results.json instead of running QE
#   --skip-optimize  Skip NN optimization stage
#   --skip-verify    Skip verification stage
#   --samples N      Number of NN parameter samples (default: 30)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="$PROJECT_ROOT/phys"
LOG_DIR="$SCRIPT_DIR/logs"

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
        --samples)
            FLAGS+=("$arg")
            ;;
        [0-9]*)
            # Number following --samples
            if [[ "${FLAGS[-1]:-}" == "--samples" ]]; then
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

# ── Run pipeline ─────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

python src/NNIP/pipeline.py "${PYARGS[@]+"${PYARGS[@]}"}" 2>&1 | tee "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline finished successfully. Log saved to $LOGFILE"
else
    echo "Pipeline failed (exit code $EXIT_CODE). Check $LOGFILE for details."
fi

exit $EXIT_CODE
