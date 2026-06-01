#!/usr/bin/env bash
#
# PHYS400 environment bootstrap: LAMMPS (patched MEAM), Quantum ESPRESSO 7.5,
# and the `phys/` Python venv. Safe to re-run.
#
# Usage:
#   ./setup.sh                # full bootstrap, skip stages whose artifacts exist
#   ./setup.sh --force        # rebuild every stage (apt step still skipped if satisfied)
#   ./setup.sh --no-apt       # skip the apt step (assumes deps already present)
#   ./setup.sh --help

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACES=$(dirname "$SCRIPT_DIR")
LAMMPS_SRC="$WORKSPACES/lammps"
QE_SRC="$WORKSPACES/qe"
VENV_DIR="$SCRIPT_DIR/phys"
INSTALL_PREFIX="$HOME/.local"

LAMMPS_TAG="patch_7Feb2024"
QE_VERSION="7.5"

FORCE=false
NO_APT=false

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force)  FORCE=true; shift ;;
            --no-apt) NO_APT=true; shift ;;
            --help|-h) usage ;;
            *) die "unknown flag: $1 (try --help)" ;;
        esac
    done
}

APT_PKGS=(
    build-essential cmake gfortran libopenmpi-dev libfftw3-dev
    libblas-dev liblapack-dev git curl python3-venv python3-pip
)

stage_apt_deps() {
    if $NO_APT; then
        log "stage_apt_deps: skipped (--no-apt)"
        return 0
    fi

    local missing=()
    local pkg
    for pkg in "${APT_PKGS[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            missing+=("$pkg")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        log "stage_apt_deps: all packages satisfied"
        return 0
    fi

    log "stage_apt_deps: installing ${missing[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing[@]}"
}

stage_lammps() {
    if [[ -f "$INSTALL_PREFIX/lib/liblammps.so" ]] && ! $FORCE; then
        log "stage_lammps: skipped (liblammps.so already at $INSTALL_PREFIX/lib)"
        return 0
    fi

    # 1. Clone (or reuse existing tree)
    if [[ ! -d "$LAMMPS_SRC" ]]; then
        log "stage_lammps: cloning LAMMPS $LAMMPS_TAG into $LAMMPS_SRC"
        git clone --depth 1 --branch "$LAMMPS_TAG" \
            https://github.com/lammps/lammps.git "$LAMMPS_SRC"
    else
        local actual_tag
        actual_tag=$(git -C "$LAMMPS_SRC" describe --tags --abbrev=0 2>/dev/null || echo "<unknown>")
        if [[ "$actual_tag" != "$LAMMPS_TAG" ]]; then
            warn "stage_lammps: $LAMMPS_SRC is at '$actual_tag', expected '$LAMMPS_TAG' — reusing as-is"
        else
            log "stage_lammps: reusing $LAMMPS_SRC at $actual_tag"
        fi
    fi

    # 2. Patch MEAM/meam.h for maxelt = 20
    local meam_h="$LAMMPS_SRC/src/MEAM/meam.h"
    [[ -f "$meam_h" ]] || die "MEAM header not found at $meam_h"
    if grep -q 'constexpr int maxelt = 20;' "$meam_h"; then
        log "stage_lammps: MEAM patch already applied"
    else
        log "stage_lammps: applying MEAM maxelt=20 patch"
        sed -i 's/constexpr int maxelt = 10;/constexpr int maxelt = 20;/' "$meam_h"
        grep -q 'constexpr int maxelt = 20;' "$meam_h" \
            || die "MEAM patch failed — upstream meam.h may have changed line format"
    fi
}

main() {
    parse_args "$@"
    log "PHYS400 bootstrap starting"
    log "  SCRIPT_DIR=$SCRIPT_DIR"
    log "  WORKSPACES=$WORKSPACES"
    log "  LAMMPS_SRC=$LAMMPS_SRC"
    log "  QE_SRC=$QE_SRC"
    log "  VENV_DIR=$VENV_DIR"
    stage_apt_deps
    stage_lammps
}

main "$@"
