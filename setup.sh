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

main() {
    parse_args "$@"
    log "PHYS400 bootstrap starting"
    log "  SCRIPT_DIR=$SCRIPT_DIR"
    log "  WORKSPACES=$WORKSPACES"
    log "  LAMMPS_SRC=$LAMMPS_SRC"
    log "  QE_SRC=$QE_SRC"
    log "  VENV_DIR=$VENV_DIR"
    stage_apt_deps
}

main "$@"
