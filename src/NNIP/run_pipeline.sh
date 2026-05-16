#!/usr/bin/env bash
# Run the full DFT-driven N-element MEAM potential pipeline.
#
# Usage (from src/NNIP/):
#   ./run_pipeline.sh                        # Auto-discover elements from EAM/
#   ./run_pipeline.sh Al Cu Zn Mg            # Specify elements explicitly
#   ./run_pipeline.sh --skip-dft Al Cu Zn Mg # Skip DFT stage
#
# Options (place before element list):
#   --skip-dft               Use existing dft_results.json instead of running QE
#   --skip-optimize          Skip NN optimization stage
#   --skip-verify            Skip verification stage
#   --resume                 Auto-detect completed stages and skip them
#   --no-plots               Skip the visualization stage
#   --perturbations N        Phase-1 parameter perturbations (default: 150)
#   --parallel N             Max parallel DFT/sampling workers (default: 4)
#   --k-representatives N    k-means medoids picked from results.json before split (default: 100)
#   --val-frac F             Fraction of representatives held out for validation (default: 0.3)
#   --split-seed N           Seed for k-means + train/val split (default: 0)
#   --clean                  Remove NNIP-generated content and exit (no pipeline run).
#                            DFT artifacts (dft_results.json, dft_scratch/) are always preserved.
#
# Any --flag (and its value, if any) is forwarded to pipeline.py verbatim — the
# wrapper no longer maintains a hardcoded allowlist.

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
CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN=true
            ;;
        --*)
            # Any --flag (including --foo=value) goes straight to pipeline.py
            FLAGS+=("$arg")
            ;;
        [0-9]*)
            # Bare number: a flag value if it follows a --flag, else an element
            # (real element symbols all start with a capital letter, never a digit)
            if [[ "${FLAGS[-1]:-}" == --* ]]; then
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

# ── --clean: remove NNIP-generated content and exit ──────────────────────────
if $CLEAN; then
    NNIP_DIR="$PROJECT_ROOT/src/NNIP"
    EAM_DIR="$PROJECT_ROOT/EAM"
    ML_DIR="$PROJECT_ROOT/src/ML"
    echo "Cleaning NNIP-generated content..."
    # DFT artifacts (dft_results.json, dft_scratch/) are deliberately excluded
    # — they take hours of QE to regenerate. Delete those manually if needed.
    paths_to_remove=(
        "$NNIP_DIR/nn_checkpoint.json"
        "$NNIP_DIR/nn_checkpoint.json.tmp"
        "$NNIP_DIR/nn_diagnostics.json"
        "$NNIP_DIR/pipeline_summary.json"
        "$NNIP_DIR/tmp_nn"
        "$EAM_DIR/dft_initialized"
        "$EAM_DIR/optimized"
        "$ML_DIR/results_train.json"
        "$ML_DIR/results_val.json"
    )
    removed=0
    for p in "${paths_to_remove[@]}"; do
        if [ -e "$p" ]; then
            rm -rf "$p"
            echo "  removed: $p"
            removed=$((removed + 1))
        fi
    done
    # Pipeline-run logs (preserve the logs/ dir itself in case the user has other things in it)
    if compgen -G "$NNIP_DIR/logs/pipeline_*.log" >/dev/null; then
        rm -f "$NNIP_DIR"/logs/pipeline_*.log
        echo "  removed: $NNIP_DIR/logs/pipeline_*.log"
        removed=$((removed + 1))
    fi
    echo "Cleaned ($removed item(s))."
    trap - EXIT   # skip the timer teardown — no pipeline ran
    exit 0
fi

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
