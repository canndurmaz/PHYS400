"""Subprocess entry point for one MD job.

Reads a JSON spec from stdin, runs LAMMPS against the optimized MEAM,
and emits newline-delimited JSON progress objects to stdout:

    {"type":"log",    "line": "..."}
    {"type":"thermo", "step": int, "pxx": float, "pyy": float, "pzz": float}
    {"type":"result", "C11_GPa": ..., "C12_GPa": ..., "E_GPa": ..., "nu": ...,
                      "physical": bool, "physical_reason": str|null,
                      "render_path": str|null}
    {"type":"error",  "message": str, "traceback": str}

LD_LIBRARY_PATH must be set by the caller (run.sh) before this script
imports anything LAMMPS-related.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback

# Make src/MD importable
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "MD"))

# Optimized MEAM (the whole point of this app)
OPTIMIZED_LIBRARY = os.path.join(_REPO, "EAM", "optimized",
                                 "optimized_library_AlCoCrCuFeMgMnMoNiSiTiZn.meam")
OPTIMIZED_PARAMS  = os.path.join(_REPO, "EAM", "optimized",
                                 "optimized_AlCoCrCuFeMgMnMoNiSiTiZn.meam")

NU_FILTER_MAX = 0.48     # mirror src/MD/lmp.py


def _emit(obj: dict) -> None:
    """Write one JSON object as a single line, flush immediately.

    Writes to ``sys.__stdout__`` (the *original* stdout) rather than
    ``sys.stdout`` so it can never recurse through any user-installed
    redirector. The parent process reads this stream as the newline-
    delimited JSON-line protocol.
    """
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


def _parse_thermo_line(line: str) -> dict | None:
    """Pull (step, pxx, pyy, pzz) from a LAMMPS thermo line if present.

    LAMMPS prints thermo as space-separated columns once a header has
    appeared; we don't try to parse the header here, instead we just
    look for lines that start with an integer step and have at least 4
    numeric columns where columns 2-4 look like pressures.
    """
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    try:
        step = int(parts[0])
        pxx, pyy, pzz = float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None
    return {"step": step, "pxx": pxx, "pyy": pyy, "pzz": pzz}


def _render_png(composition, selected, traj_file: str, output_path: str) -> None:
    """Render a single still PNG of the relaxed supercell to ``output_path``.

    Mirrors src/MD/viz.py's atom-type styling, but uses ``render_image``
    (one frame) instead of ``render_anim`` (mp4 video). Imported lazily
    so a missing OVITO doesn't break elastic-only runs.
    """
    import ovito
    from ovito.io import import_file
    from ovito.vis import Viewport, TachyonRenderer
    from viz import get_color

    for p in list(ovito.scene.pipelines):
        p.remove_from_scene()
    pipeline = import_file(traj_file)
    pipeline.add_to_scene()
    pipeline.compute()
    particles = pipeline.source.data.particles_
    types_prop = particles.particle_types_
    for i, elem in enumerate(selected, start=1):
        # OVITO >= 3.15 requires the owning container as second argument
        pt = types_prop.add_type_id(i, particles)
        pt.name = elem.symbol
        pt.color = get_color(elem.symbol)
        pt.radius = 0.8
    vp = Viewport()
    vp.type = Viewport.Type.Perspective
    vp.camera_pos = (40, 40, 40)
    vp.camera_dir = (-1, -1, -1)
    vp.zoom_all()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vp.render_image(filename=output_path, size=(800, 600),
                    renderer=TachyonRenderer())


def _build_optimized_potential():
    """Return a ``pot`` dict shaped like src/MD/config.py:find_potential output,
    but anchored to the optimized MEAM files instead of EAM/*."""
    from element import parse_meam_library

    if not (os.path.isfile(OPTIMIZED_LIBRARY) and os.path.isfile(OPTIMIZED_PARAMS)):
        raise FileNotFoundError(
            f"Optimized MEAM not found at {OPTIMIZED_LIBRARY!r} / "
            f"{OPTIMIZED_PARAMS!r}. Run ./src/NNIP/run_pipeline.sh first."
        )
    elements = parse_meam_library(OPTIMIZED_LIBRARY)
    return {
        "library": OPTIMIZED_LIBRARY,
        "params": OPTIMIZED_PARAMS,
        "elements": elements,
    }


def _run_lammps(spec: dict, tmpdir: str) -> dict:
    """Run LAMMPS for one job. Mirrors src/MD/lmp.py:run_simulation but
    does NOT save to results.json and does NOT delete configs."""
    from lammps import lammps                            # noqa: E402
    from lmp import get_elastic_moduli                   # noqa: E402

    composition = spec["composition"]
    knobs = spec["knobs"]
    do_viz = bool(spec.get("do_viz", False))

    pot = _build_optimized_potential()
    pot_elements_by_symbol = {e.symbol: e for e in pot["elements"]}
    sel = sorted(
        [pot_elements_by_symbol[sym] for sym in composition],
        key=lambda e: e.meam_index,
    )
    dominant = max(sel, key=lambda e: composition[e.symbol])
    a_m = sum(composition[e.symbol] * e.lattice_constant for e in sel)
    box_ang = knobs["box_size_m"] * 1e10
    n_rep = max(1, round(box_ang / a_m))

    library_elements = " ".join(e.meam_label for e in pot["elements"])
    active_elements  = " ".join(e.meam_label for e in sel)

    # Capture LAMMPS's screen output into a file in tmpdir. After the run we
    # read this file and emit each line as a {"type":"log"} progress event
    # (and parse out thermo rows). This avoids fighting Python stdout
    # redirection — LAMMPS writes to libc stdout which doesn't honor it.
    screen_path = os.path.join(tmpdir, "lammps.screen")
    L = lammps(cmdargs=["-log", "none", "-screen", screen_path])
    try:
        L.command("units metal")
        L.command("atom_style atomic")
        L.command("boundary p p p")
        L.command(f"lattice {dominant.lattice_type} {a_m:.4f}")
        L.command(f"region box block 0 {n_rep} 0 {n_rep} 0 {n_rep}")
        L.command(f"create_box {len(sel)} box")
        L.command("create_atoms 1 box")

        remaining = 1.0
        for i, elem in enumerate(sel[1:], start=2):
            frac = composition[elem.symbol] / remaining
            L.command(f"set type 1 type/fraction {i} {frac:.6f} 12345")
            remaining -= composition[elem.symbol]

        L.command("pair_style meam")
        L.command(
            f"pair_coeff * * {pot['library']} {library_elements} "
            f"{pot['params']} {active_elements}"
        )
        for i, elem in enumerate(sel, start=1):
            L.command(f"mass {i} {elem.mass}")

        # Emit thermo as 4 columns "step pxx pyy pzz" so the parent's
        # _parse_thermo_line picks them up cleanly. ``thermo 10`` keeps the
        # event rate moderate (LAMMPS minimize iterations can be very fast).
        L.command("thermo_style custom step pxx pyy pzz")
        L.command("thermo 10")

        _emit({"type": "log", "line": "Relaxing ground state (box + atoms)..."})
        L.command("fix boxrelax all box/relax aniso 0.0 vmax 0.001")
        L.command("minimize 1.0e-6 1.0e-8 1000 10000")
        L.command("unfix boxrelax")
        L.command("minimize 1.0e-6 1.0e-8 200 2000")

        # get_elastic_moduli has print() calls that would corrupt the JSON-line
        # protocol on stdout. Redirect sys.stdout to stderr for that call so
        # its diagnostic prints go to the error stream instead. _emit is safe
        # because it writes to sys.__stdout__ directly.
        _saved_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            E, nu, C11, C12 = get_elastic_moduli(L)
        finally:
            sys.stdout = _saved_stdout

        if not all(math.isfinite(x) for x in (E, nu, C11, C12)):
            raise RuntimeError(
                f"MD did not converge: non-finite elastic constants "
                f"(C11={C11}, C12={C12}, E={E}, nu={nu})"
            )

        physical = True
        physical_reason: str | None = None
        if E < 0:
            physical = False
            physical_reason = f"negative Young's modulus ({E:.2f} GPa)"
        elif nu < 0:
            physical = False
            physical_reason = f"negative Poisson's ratio ({nu:.3f})"
        elif nu >= NU_FILTER_MAX:
            physical = False
            physical_reason = (
                f"Poisson's ratio nu={nu:.3f} >= {NU_FILTER_MAX:.2f} (singularity)"
            )
        elif C11 < C12:
            physical = False
            physical_reason = (
                f"mechanical instability (C11={C11:.2f} < C12={C12:.2f})"
            )

        render_path: str | None = None
        if do_viz and spec.get("render_output_path"):
            traj_file = os.path.join(tmpdir, "traj.lammpstrj")
            # Single dump frame is enough — we render a still, not an animation
            L.command(f"dump one all atom 1 {traj_file}")
            L.command("run 0")
            L.command("undump one")
            try:
                _render_png(composition, sel, traj_file,
                            spec["render_output_path"])
                render_path = spec["render_output_path"]
            except Exception as exc:
                _emit({"type": "log", "line": f"OVITO render failed: {exc}"})
    finally:
        L.close()

    # Now that LAMMPS has flushed its screen buffer, drain the captured log
    # line by line, emitting each as a log/thermo event in chronological
    # order. Reading after the run keeps this simple (no background thread)
    # and the UI receives the full trace on the next poll.
    try:
        with open(screen_path) as f:
            for line in f:
                line = line.rstrip("\n")
                _emit({"type": "log", "line": line})
                thermo = _parse_thermo_line(line)
                if thermo is not None:
                    _emit({"type": "thermo", **thermo})
    except OSError:
        # If for some reason LAMMPS didn't write a screen file, we just
        # don't have a log trace — the elastic constants are still in the
        # result. Don't fail the run for cosmetic output.
        pass

    return {
        "C11_GPa": float(C11),
        "C12_GPa": float(C12),
        "E_GPa": float(E),
        "nu": float(nu),
        "physical": physical,
        "physical_reason": physical_reason,
        "render_path": render_path,
    }


def main() -> int:
    try:
        spec = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        _emit({"type": "error", "message": f"bad stdin JSON: {exc}",
               "traceback": ""})
        return 2

    with tempfile.TemporaryDirectory(prefix="meam-job-") as tmpdir:
        try:
            result = _run_lammps(spec, tmpdir)
            _emit({"type": "result", **result})
            return 0
        except Exception as exc:
            _emit({"type": "error", "message": str(exc),
                   "traceback": traceback.format_exc()})
            return 1


if __name__ == "__main__":
    sys.exit(main())
