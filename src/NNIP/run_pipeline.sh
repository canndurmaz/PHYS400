#!/usr/bin/env bash
# Run the full DFT-driven N-element MEAM potential pipeline.
#
# Usage (from src/NNIP/):
#   ./run_pipeline.sh                        # Auto-discover elements from EAM/
#   ./run_pipeline.sh Al Cu Zn Mg            # Specify elements explicitly
#   ./run_pipeline.sh --skip-dft Al Cu Zn Mg # Skip DFT stage
#
# Options (place before element list):
#   --skip-dft       Use existing dft_results.json instead of running QE
#   --skip-optimize  Skip NN optimization stage
#   --skip-verify    Skip verification stage
#   --samples N      Number of NN parameter samples (default: 30)
#   --parallel N     Max parallel DFT workers (default: 4)

set -euo pipefail

TIMER_PID=""
TIMER_START=$(date +%s)
HAS_TTY=false
[ -t 1 ] && HAS_TTY=true

_setup_timer() {
    $HAS_TTY || return 0
    local rows
    rows=$(tput lines)
    # Reserve last line: set scroll region to rows 1..(rows-1)
    printf '\e[1;%dr' "$((rows - 1))" > /dev/tty
    # Draw initial timer bar on last row
    printf '\e[%d;1H\e[7m Elapsed: 0m 00s \e[K\e[0m' "$rows" > /dev/tty
    # Position cursor inside scroll region
    printf '\e[1;1H' > /dev/tty
    # Background updater — writes directly to /dev/tty, never through pipes
    (
        while true; do
            sleep 1
            local now elapsed m s rows
            now=$(date +%s)
            elapsed=$((now - TIMER_START))
            m=$((elapsed / 60))
            s=$((elapsed % 60))
            rows=$(tput lines)
            printf '\e[s\e[%d;1H\e[7m Elapsed: %dm %02ds \e[K\e[0m\e[u' \
                "$rows" "$m" "$s" > /dev/tty
        done
    ) &
    TIMER_PID=$!
}

_teardown_timer() {
    # Kill background timer
    [[ -n "${TIMER_PID:-}" ]] && kill "$TIMER_PID" 2>/dev/null || true
    wait "$TIMER_PID" 2>/dev/null || true
    TIMER_PID=""
    $HAS_TTY || return 0
    # Compute final elapsed time
    local now elapsed m s rows
    now=$(date +%s)
    elapsed=$((now - TIMER_START))
    m=$((elapsed / 60))
    s=$((elapsed % 60))
    rows=$(tput lines)
    # Clear timer bar and restore full scroll region
    printf '\e[%d;1H\e[K' "$rows" > /dev/tty
    printf '\e[r' > /dev/tty
    printf '\e[%d;1H' "$rows" > /dev/tty
    echo "Total runtime: ${m}m ${s}s"
}
trap _teardown_timer EXIT

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
    echo " Elements:     (auto-discover from EAM/)"
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
_setup_timer
python "$PROJECT_ROOT/src/NNIP/pipeline.py" "${PYARGS[@]+"${PYARGS[@]}"}" 2>&1 | tee "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline finished successfully. Log saved to $LOGFILE"
else
    echo "Pipeline failed (exit code $EXIT_CODE). Check $LOGFILE for details."
fi

exit $EXIT_CODE
