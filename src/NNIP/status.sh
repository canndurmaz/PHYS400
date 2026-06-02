#!/usr/bin/env bash
# status.sh — show NNIP-pipeline or DFT-generation progress at a glance.
# Usage (from src/NNIP/): ./status.sh
#
# Auto-detects which workload is active. If neither is running, reports the
# last-known state of whichever log was most recently touched.

set -uo pipefail
cd "$(dirname "$0")"

# Prefer the most-recent dft_*.log under logs/ (direct dft_reference runs);
# fall back to the legacy /tmp/dft_rerun.log path if no logs/dft_*.log exists.
DFT_LOG=$(ls -t logs/dft_*.log 2>/dev/null | head -1)
[[ -z "$DFT_LOG" && -f /tmp/dft_rerun.log ]] && DFT_LOG=/tmp/dft_rerun.log
PIPE_LOG=$(ls -t logs/pipeline_*.log 2>/dev/null | head -1)

# ── DFT generation reporter ────────────────────────────────────────────────
dft_status() {
    local log=$1
    local now=$(date +%s)
    local log_mtime=$(stat -c %Y "$log")
    local log_age=$((now - log_mtime))
    local pid=$(pgrep -f 'src\.NNIP\.dft_reference' | head -1 || true)
    local n_pwx=$(pgrep -c 'pw\.x' 2>/dev/null || echo 0)

    echo "============================================================"
    echo " DFT Generation Status — $(date '+%a %b %d %H:%M:%S')"
    echo "============================================================"
    if [[ -n "$pid" ]]; then
        local uptime=$(ps -o etime= -p "$pid" | tr -d ' ')
        echo " Process       : ALIVE   (orchestrator PID $pid, uptime $uptime)"
        echo " pw.x workers  : $n_pwx running"
    else
        echo " Process       : NOT RUNNING   (last log write ${log_age}s ago)"
    fi
    echo " Log file      : $log"

    # Element + stage from the most recent matching log lines.
    local elem=$(grep -E '^  ELEMENT [0-9]+/[0-9]+: [A-Z][a-z]?$' "$log" | tail -1 | sed 's/^  //')
    local stage_line=$(grep -E '^  \[(EOS|ELASTIC|BINARY)\]' "$log" | tail -1 | sed 's/^  //')
    echo ""
    [[ -n "$elem" ]]       && echo " Current elem  : $elem"
    [[ -n "$stage_line" ]] && echo " Stage         : $stage_line"

    # Recent EOS point timings (last 7 — one element's worth).
    local eos_lines=$(grep '\[SCF\] EOS point' "$log" | tail -7)
    if [[ -n "$eos_lines" ]]; then
        echo ""
        echo " Recent EOS points:"
        echo "$eos_lines" | awk '{
            match($0, /point [0-9]+/);   pt  = substr($0, RSTART+6, RLENGTH-6);
            match($0, /a=[0-9.]+/);      a   = substr($0, RSTART+2, RLENGTH-2);
            match($0, /\([0-9.]+s\)/);   dur = substr($0, RSTART+1, RLENGTH-3);
            printf "   pt %s   a=%s   %6.1fs (%4.1f min)\n", pt, a, dur, dur/60;
            sum += dur; n++;
        } END {
            if (n>0) printf "   ────────────  mean %5.1fs (%4.1f min) over %d points\n", sum/n, sum/n/60, n;
        }'
    fi

    # Elastic SCFs (per-element, 2 each)
    local elastic_lines=$(grep -E '\[ELASTIC\] +(Baseline|Strained) done' "$log" | tail -4)
    if [[ -n "$elastic_lines" ]]; then
        echo ""
        echo " Recent elastic SCFs:"
        echo "$elastic_lines" | awk '{
            sub(/^  +\[ELASTIC\] +/, "", $0);
            print "   " $0;
        }'
    fi

    # Binary pair counters. Use grep | wc -l so we always get a single
    # integer; grep -c prints "0" *and* exits 1 on no-match, which combined
    # with `|| echo 0` produces "0\n0" and corrupts the arithmetic below.
    local pair_ok=$(grep '\[BINARY\] .*: E_form=' "$log" 2>/dev/null | wc -l)
    local pair_skip=$(grep '\[BINARY\] .*: SKIPPED' "$log" 2>/dev/null | wc -l)
    local pair_fail=$(grep '\[BINARY\] .*: FAILED' "$log" 2>/dev/null | wc -l)
    if [[ "$pair_ok" -gt 0 || "$pair_skip" -gt 0 || "$pair_fail" -gt 0 ]]; then
        echo ""
        echo " Binary pairs  : $pair_ok completed, $pair_skip skipped, $pair_fail failed"
    fi

    # Magnetic state from the most recently-written pw.x output — the single
    # most important sanity check, since fixing the Mn AFM seed is the whole
    # point of this re-run.
    if [[ -d dft_scratch ]]; then
        local latest_pwo=$(find dft_scratch -name 'espresso.pwo' -printf '%T@ %p\n' 2>/dev/null \
                            | sort -nr | head -1 | awk '{print $2}')
        if [[ -n "$latest_pwo" ]]; then
            local elem_dir=$(echo "$latest_pwo" | awk -F/ '{print $(NF-2)}')
            local sub_dir=$(echo "$latest_pwo" | awk -F/ '{print $(NF-1)}')
            local converged=$(grep 'convergence has been achieved' "$latest_pwo" 2>/dev/null | wc -l)
            local total_mag=$(grep 'total magnetization' "$latest_pwo" 2>/dev/null | tail -1 \
                              | awk '{print $4}')
            local abs_mag=$(grep 'absolute magnetization' "$latest_pwo" 2>/dev/null | tail -1 \
                            | awk '{print $4}')
            local atom_lines=$(grep -E '^\s*atom\s+[0-9]+ \(R=' "$latest_pwo" 2>/dev/null | tail -16)
            echo ""
            echo " Latest SCF    : $elem_dir/$sub_dir   (${converged} converged cycles)"
            [[ -n "$total_mag" ]] && echo " Total mag.    : $total_mag μB/cell"
            [[ -n "$abs_mag"   ]] && echo " Absolute mag. : $abs_mag μB/cell"
            if [[ -n "$atom_lines" ]]; then
                echo " Per-atom moments (latest report):"
                # pw.x prints "atom <n> (R=<r>)  charge= <c>  magn= <m>". With default
                # awk whitespace splitting, "magn=" is its own field and the moment
                # is the next field. The previous version of this awk assigned the
                # *label* field to m, which numeric-formats to 0.000 — masking real
                # AFM moments (Mn was showing 0.000 μB despite |m|=1.8 μB).
                echo "$atom_lines" | awk '{
                    m = 0;
                    for (i=1; i<=NF; i++) {
                        if ($i == "magn=")     { m = $(i+1); break }
                        if ($i ~ /^magn=./)   { sub(/^magn=/, "", $i); m = $i; break }
                    }
                    printf "   atom %2s : %+6.3f μB\n", $2, m;
                }'
            fi
        fi
    fi

    # Rough ETA from mean EOS time. Remaining work assumed = current element's
    # leftover EOS + isolated atom + 2 elastic (~tight) + any later elements'
    # full per-element budget + binary pairs.
    local mean_eos_s=$(grep '\[SCF\] EOS point' "$log" | awk -F'[()]' '{print $2}' \
                       | sed 's/s$//' | awk '{s+=$1; n++} END {if (n) printf "%.0f", s/n; else print 0}')
    if [[ "$mean_eos_s" -gt 0 ]]; then
        echo ""
        printf " Avg EOS SCF   : %ss (~%.1f min)\n" "$mean_eos_s" "$(echo "scale=2; $mean_eos_s/60" | bc -l)"
    fi
    echo "============================================================"
}

# ── Dispatch: prefer the live workload; fall back to the freshest log ──────
DFT_PID=$(pgrep -f 'src\.NNIP\.dft_reference' 2>/dev/null | head -1 || true)
PIPE_PID=$(pgrep -f 'bash.*run_pipeline.sh' 2>/dev/null | head -1 || true)

if [[ -n "$DFT_PID" ]]; then
    dft_status "$DFT_LOG"
    exit 0
elif [[ -n "$PIPE_PID" ]]; then
    : # fall through to pipeline reporter below
elif [[ -f "$DFT_LOG" && -n "$PIPE_LOG" ]]; then
    if [[ $(stat -c %Y "$DFT_LOG") -gt $(stat -c %Y "$PIPE_LOG") ]]; then
        dft_status "$DFT_LOG"
        exit 0
    fi
elif [[ -f "$DFT_LOG" && -z "$PIPE_LOG" ]]; then
    dft_status "$DFT_LOG"
    exit 0
fi

# ── NNIP pipeline reporter ─────────────────────────────────────────────────
LOG="$PIPE_LOG"
[[ -z "$LOG" ]] && { echo "No pipeline log found in logs/"; exit 1; }

PID=$(pgrep -f 'bash.*run_pipeline.sh' | head -1 || true)
NOW=$(date +%s)
LOG_MTIME=$(stat -c %Y "$LOG")
LOG_AGE=$((NOW - LOG_MTIME))

echo "============================================================"
echo " NNIP Pipeline Status — $(date '+%a %b %d %H:%M:%S')"
echo "============================================================"

if [[ -n "$PID" ]]; then
    UPTIME=$(ps -o etime= -p "$PID" | tr -d ' ')
    NWORKERS=$(pgrep -f 'pipeline.py' | wc -l)
    echo " Process       : ALIVE   (wrapper PID $PID, uptime $UPTIME)"
    echo " Workers       : $NWORKERS python processes running"
else
    echo " Process       : NOT RUNNING"
fi
echo " Log file      : $(basename "$LOG")"
echo " Last write    : ${LOG_AGE}s ago"

CUR=$(grep -E '\[active/iter [0-9]+\] [0-9]+/[0-9]+ accepted' "$LOG" | tail -1 || true)
PROG=""
if [[ -n "$CUR" ]]; then
    ITER=$(echo "$CUR" | grep -oE 'iter [0-9]+')
    PROG=$(echo "$CUR" | grep -oE '[0-9]+/[0-9]+' | head -1)
    echo ""
    echo " Stage         : Phase 1 — active learning"
    echo " Progress      : $PROG samples accepted"
    echo " Current iter  : $ITER"
fi

DONE=$(grep -E 'batch done in [0-9.]+s: [0-9]+/[0-9]+ accepted' "$LOG" | tail -5 || true)
if [[ -n "$DONE" ]]; then
    echo ""
    echo " Recent batches (latest 5):"
    echo "$DONE" | awk '{
        match($0, /iter[0-9]+/);                 iter = substr($0, RSTART, RLENGTH);
        match($0, /in [0-9.]+s/);                dur  = substr($0, RSTART+3, RLENGTH-4);
        match($0, /[0-9]+\/[0-9]+ accepted$/);   acc  = substr($0, RSTART, RLENGTH-9);
        printf "   %-8s  %6.1f min   %s accepted\n", iter, dur/60, acc;
    }'

    if [[ -n "$PROG" ]]; then
        echo "$DONE" | tail -3 | awk -v prog="$PROG" '
        {
            match($0, /in [0-9.]+s/);              dur_sum += substr($0, RSTART+3, RLENGTH-4);
            match($0, /: [0-9]+\//); a = substr($0, RSTART+2, RLENGTH-3); acc_sum += a;
            n++;
        }
        END {
            split(prog, p, "/"); cur = p[1]; budget = p[2]; rem = budget - cur;
            avg_dur = dur_sum / n; avg_acc = acc_sum / n;
            print "";
            if (rem <= 0) {
                printf " Budget reached — Phase 1 complete.\n";
            } else if (avg_acc < 1e-9) {
                printf " ETA           : no recent acceptances (last %d batches were 0/6)\n", n;
                printf "                 — tighten PERT_RANGE further or check baseline\n";
            } else {
                batches = rem / avg_acc;
                secs = batches * avg_dur;
                h = int(secs / 3600); m = int((secs - h*3600) / 60);
                printf " ETA           : %.1f more batches × %dm each ≈ %dh %dm to budget\n", batches, avg_dur/60, h, m;
                printf " Acceptance    : %.2f / 6 per batch (avg over last %d batches)\n", avg_acc, n;
            }
        }'
    fi
fi

echo "============================================================"
