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

    # 3. CMake configure
    log "stage_lammps: cmake configure"
    cmake -S "$LAMMPS_SRC/cmake" -B "$LAMMPS_SRC/build" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DBUILD_SHARED_LIBS=yes \
        -DBUILD_MPI=ON \
        -DBUILD_OMP=ON \
        -DPKG_MEAM=yes \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

    # 4. Build + install
    log "stage_lammps: building (this takes several minutes)"
    cmake --build "$LAMMPS_SRC/build" -j"$(nproc)"
    log "stage_lammps: installing to $INSTALL_PREFIX"
    cmake --install "$LAMMPS_SRC/build"

    # 5. Verify
    [[ -f "$INSTALL_PREFIX/lib/liblammps.so" ]] \
        || die "stage_lammps: liblammps.so not produced at $INSTALL_PREFIX/lib"
    log "stage_lammps: complete"
}

stage_qe() {
    if [[ -x "$QE_SRC/bin/pw.x" ]] && ! $FORCE; then
        log "stage_qe: skipped (pw.x already at $QE_SRC/bin)"
        return 0
    fi

    # 1. Download + extract
    if [[ ! -d "$QE_SRC" ]]; then
        local url="https://gitlab.com/QEF/q-e/-/archive/qe-${QE_VERSION}/q-e-qe-${QE_VERSION}.tar.gz"
        log "stage_qe: downloading QE $QE_VERSION from $url"
        local tmp
        tmp=$(mktemp -d)
        curl -fL "$url" | tar -xz -C "$tmp"
        mv "$tmp/q-e-qe-${QE_VERSION}" "$QE_SRC"
        rmdir "$tmp"
    else
        log "stage_qe: reusing existing $QE_SRC"
    fi

    # 2. Configure (auto-detects mpif90, gfortran, FFTW3, BLAS, LAPACK)
    log "stage_qe: configure"
    (cd "$QE_SRC" && ./configure)

    # 3. Build pw.x only (skip CPV, PHonon, EPW, etc.)
    log "stage_qe: building pw.x (this takes several minutes)"
    make -C "$QE_SRC" -j"$(nproc)" pw

    # 4. Verify
    [[ -x "$QE_SRC/bin/pw.x" ]] \
        || die "stage_qe: pw.x not produced at $QE_SRC/bin"
    log "stage_qe: complete"
}

ACTIVATE_SENTINEL="# PHYS400 env wiring"

stage_venv() {
    if [[ -f "$VENV_DIR/pyvenv.cfg" ]] && ! $FORCE; then
        log "stage_venv: skipped (venv already at $VENV_DIR)"
        return 0
    fi

    if $FORCE && [[ -d "$VENV_DIR" ]]; then
        log "stage_venv: --force, removing existing $VENV_DIR"
        rm -rf "$VENV_DIR"
    fi

    # 1. Create venv with system site packages (needed for ase, scipy, etc. via apt)
    log "stage_venv: creating venv at $VENV_DIR"
    python3 -m venv --system-site-packages "$VENV_DIR"

    # 2. Wire env vars into activate (once)
    local activate="$VENV_DIR/bin/activate"
    if ! grep -qF "$ACTIVATE_SENTINEL" "$activate"; then
        log "stage_venv: appending PHYS400 env wiring to activate"
        cat >> "$activate" <<EOF

$ACTIVATE_SENTINEL
export LD_LIBRARY_PATH="\$HOME/.local/lib:\${LD_LIBRARY_PATH:-}"
PATH="\$VIRTUAL_ENV/bin:$WORKSPACES/qe/bin:\$PATH"
EOF
    fi

    # 3. Install requirements
    log "stage_venv: installing pip requirements"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

    log "stage_venv: complete"
}

final_verify() {
    log "final_verify: running smoke checks"
    local all_pass=true
    local label cmd

    _check() {
        local label="$1" cmd="$2"
        if eval "$cmd" >/dev/null 2>&1; then
            printf '  \033[1;32mPASS\033[0m  %s\n' "$label"
        else
            printf '  \033[1;31mFAIL\033[0m  %s\n' "$label"
            all_pass=false
        fi
    }

    _check "liblammps.so present" \
        "test -f '$INSTALL_PREFIX/lib/liblammps.so'"
    _check "pw.x executable" \
        "test -x '$QE_SRC/bin/pw.x'"
    _check "venv python imports lammps, numpy, ase" \
        "'$VENV_DIR/bin/python3' -c 'import lammps, numpy, ase'"
    _check "venv python can load custom liblammps" \
        "LD_LIBRARY_PATH='$INSTALL_PREFIX/lib' '$VENV_DIR/bin/python3' -c 'from lammps import lammps; lammps()'"

    local py_ver lmp_tag qe_ver
    py_ver=$("$VENV_DIR/bin/python3" --version 2>&1 || echo unknown)
    lmp_tag=$(git -C "$LAMMPS_SRC" describe --tags --abbrev=0 2>/dev/null || echo unknown)
    qe_ver=$(grep -oE "version_number = '[^']+'" "$QE_SRC/include/qe_version.h" 2>/dev/null \
             | sed "s/.*'\\(.*\\)'/\\1/" || echo unknown)

    echo
    echo "== PHYS400 setup complete =="
    echo "LAMMPS:  $INSTALL_PREFIX/lib/liblammps.so  (tag $lmp_tag)"
    echo "QE:      $QE_SRC/bin/pw.x  (v$qe_ver)"
    echo "venv:    $VENV_DIR  ($py_ver)"
    echo
    echo "To activate: source phys/bin/activate"

    $all_pass || die "final_verify: one or more smoke checks failed"
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
    stage_qe
    stage_venv
    final_verify
}

main "$@"
