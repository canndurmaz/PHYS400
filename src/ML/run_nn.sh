#!/bin/bash
TIMER_PID=""
TIMER_START=$(date +%s)
HAS_TTY=false; [ -t 1 ] && HAS_TTY=true

_setup_timer() {
    $HAS_TTY || return 0
    local rows; rows=$(tput lines)
    printf '\e[1;%dr' "$((rows - 1))" > /dev/tty
    printf '\e[%d;1H\e[7m Elapsed: 0m 00s \e[K\e[0m' "$rows" > /dev/tty
    printf '\e[1;1H' > /dev/tty
    ( while true; do sleep 1; local now=$(($(date +%s) - TIMER_START)) rows=$(tput lines)
        printf '\e[s\e[%d;1H\e[7m Elapsed: %dm %02ds \e[K\e[0m\e[u' \
            "$rows" "$((now/60))" "$((now%60))" > /dev/tty; done ) &
    TIMER_PID=$!
}

_teardown_timer() {
    [[ -n "${TIMER_PID:-}" ]] && kill "$TIMER_PID" 2>/dev/null || true
    wait "$TIMER_PID" 2>/dev/null || true; TIMER_PID=""
    $HAS_TTY || return 0
    local elapsed=$(( $(date +%s) - TIMER_START )) rows=$(tput lines)
    printf '\e[%d;1H\e[K\e[r\e[%d;1H' "$rows" "$rows" > /dev/tty
    echo "Total runtime: $((elapsed/60))m $((elapsed%60))s"
}
trap _teardown_timer EXIT

# Path to the virtual environment python
PYTHON_VENV="../../phys/bin/python3"

# Ensure we are in the script's directory so relative paths work
cd "$(dirname "$0")"

echo "Training Alloy Neural Network and predicting for predict.json..."
_setup_timer
$PYTHON_VENV nn_alloy.py
