#!/usr/bin/env bash
# Wipe every artifact that can be regenerated from the canonical inputs,
# while preserving:
#   - src/ML/results.json        (the LAMMPS-evaluated dataset)
#   - src/NNIP/dft_results.json  (the DFT reference set)
#   - all source code, configuration, and the report .tex sources
#
# Usage:
#   ./clean.sh              # interactive (asks for confirmation)
#   ./clean.sh -y           # skip confirmation
#   ./clean.sh --dry-run    # only print what would be deleted
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -y|--yes)     ASSUME_YES=1 ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set/d; 1d'
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# Items to delete: each entry is "label:glob" (glob is space-separated).
# Globs are interpreted relative to the project root.
ITEMS=(
    "report build artefacts:reports/interim/main.aux reports/interim/main.log reports/interim/main.out reports/interim/main.toc reports/interim/main.bbl reports/interim/main.blg reports/interim/main.pdf"
    "auto-generated report tables:reports/interim/sections/_auto_*.tex"
    "auto-generated report figures:reports/interim/figures/*.png"
    "ML training plots:src/ML/plots/*.png"
    "ML model artefact:src/ML/alloy_model.keras"
    "ML metrics JSON:src/ML/nn_metrics.json"
    "NNIP plots:src/NNIP/plots/*.png src/NNIP/plots/*.pdf"
    "NNIP diagnostics:src/NNIP/nn_diagnostics.json src/NNIP/pipeline_summary.json"
    "NNIP LAMMPS scratch:src/NNIP/tmp_nn"
    "NNIP DFT scratch (kept dft_results.json):src/NNIP/dft_scratch"
    "MD visualizations:src/MD/visualization/*.mp4"
    "MD trajectory dumps:src/MD/traj_*.lammpstrj"
    "Python __pycache__ dirs:src/**/__pycache__ src/__pycache__"
)

# Things we explicitly never touch
PRESERVE=(
    "src/ML/results.json"
    "src/ML/predict.json"
    "src/NNIP/dft_results.json"
)

echo "Project root: $ROOT"
echo "Mode:         $([ $DRY_RUN -eq 1 ] && echo DRY-RUN || echo DELETE)"
echo "Preserved:"
for p in "${PRESERVE[@]}"; do
    if [ -e "$ROOT/$p" ]; then
        echo "  ok    $p"
    else
        echo "  miss  $p (not present, nothing to preserve)"
    fi
done
echo

# Resolve globs and build a deletion list
declare -a TO_DELETE=()
declare -A LABEL_OF=()
for item in "${ITEMS[@]}"; do
    label="${item%%:*}"
    globs="${item#*:}"
    # Split globs string into a literal array (no premature expansion).
    read -ra glob_arr <<< "$globs"
    found_any=0
    shopt -s nullglob globstar
    for pattern in "${glob_arr[@]}"; do
        for match in "$ROOT"/$pattern; do
            [ -e "$match" ] || continue
            rel="${match#$ROOT/}"
            skip=0
            for p in "${PRESERVE[@]}"; do
                [ "$rel" = "$p" ] && skip=1 && break
            done
            [ $skip -eq 1 ] && continue
            TO_DELETE+=("$match")
            LABEL_OF[$match]="$label"
            found_any=1
        done
    done
    shopt -u nullglob globstar
done

if [ ${#TO_DELETE[@]} -eq 0 ]; then
    echo "Nothing to clean."
    exit 0
fi

echo "Will remove ${#TO_DELETE[@]} path(s):"
last_label=""
for m in "${TO_DELETE[@]}"; do
    rel="${m#$ROOT/}"
    lbl="${LABEL_OF[$m]}"
    if [ "$lbl" != "$last_label" ]; then
        echo "[$lbl]"
        last_label="$lbl"
    fi
    if [ -d "$m" ]; then
        n=$(find "$m" -type f 2>/dev/null | wc -l)
        echo "  rm -r  $rel/   (${n} file(s))"
    else
        echo "  rm     $rel"
    fi
done

if [ $DRY_RUN -eq 1 ]; then
    echo
    echo "Dry run: nothing was deleted."
    exit 0
fi

if [ $ASSUME_YES -eq 0 ]; then
    echo
    read -r -p "Proceed with deletion? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

for m in "${TO_DELETE[@]}"; do
    rm -rf -- "$m"
done
echo "Deleted ${#TO_DELETE[@]} path(s)."
