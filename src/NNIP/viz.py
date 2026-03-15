#!/usr/bin/env python3
"""OVITO visualization for NNIP MD trajectories.

Renders animated MP4 videos from LAMMPS (.lammpstrj) or ASE (.traj) trajectories
with per-element coloring and a composition legend overlay.
"""

import os

import ovito
from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer, TextLabelOverlay
from PySide6.QtCore import Qt

# Type map — must match DeePMD model order
TYPE_MAP = ["Al", "Zn", "Mg", "Mn", "Cu", "Si", "Cr", "Fe", "Ti"]

# Per-element display colors (RGB 0-1)
ELEMENT_COLORS = {
    "Al": (0.80, 0.80, 0.80),   # Light Gray
    "Zn": (0.50, 0.50, 1.00),   # Slate Blue
    "Mg": (1.00, 1.00, 0.00),   # Yellow
    "Mn": (0.60, 0.00, 0.60),   # Purple
    "Cu": (1.00, 0.50, 0.00),   # Orange
    "Si": (0.00, 0.60, 0.00),   # Green
    "Cr": (0.00, 0.90, 0.90),   # Cyan
    "Fe": (0.55, 0.27, 0.07),   # Brown
    "Ti": (0.40, 0.40, 0.40),   # Dark Gray
}

# Covalent radii (A) for display sizing
ELEMENT_RADII = {
    "Al": 1.21, "Zn": 1.22, "Mg": 1.41, "Mn": 1.39, "Cu": 1.32,
    "Si": 1.11, "Cr": 1.39, "Fe": 1.32, "Ti": 1.60,
}

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")


def get_color(symbol):
    """Get color for a symbol, or generate a deterministic fallback."""
    if symbol in ELEMENT_COLORS:
        return ELEMENT_COLORS[symbol]
    import hashlib
    h = hashlib.md5(symbol.encode()).digest()
    return (h[0] / 255, h[1] / 255, h[2] / 255)


def render_trajectory(traj_path, composition, elements=None, output=None,
                      size=(800, 600), fps=10):
    """Render an animated MP4 from a trajectory file.

    Args:
        traj_path: Path to trajectory file (.lammpstrj or .traj)
        composition: Dict of element symbol -> fraction, e.g. {"Al": 0.95, "Cu": 0.05}
        elements: List of element symbols present (for LAMMPS type ordering).
                  If None, inferred from composition keys.
        output: Output MP4 path. If None, auto-generated from composition.
        size: (width, height) in pixels.
        fps: Frames per second.
    """
    if not os.path.exists(traj_path):
        print(f"Warning: Trajectory file {traj_path} not found. Skipping visualization.")
        return None

    if elements is None:
        elements = list(composition.keys())

    ovito.scene.clear()

    pipeline = import_file(traj_path)

    # Assign element names, colors, and radii per atom type
    def assign_type_visuals(frame, data):
        types = data.particles.particle_types
        for i, elem in enumerate(elements):
            type_id = i + 1  # LAMMPS types are 1-indexed
            try:
                pt = types.type_by_id(type_id)
                pt.name = elem
                pt.color = get_color(elem)
                pt.radius = ELEMENT_RADII.get(elem, 0.8) * 0.6
            except KeyError:
                pass

    pipeline.modifiers.append(assign_type_visuals)
    pipeline.add_to_scene()

    # Camera setup
    vp = Viewport()
    vp.type = Viewport.Type.Perspective
    vp.camera_pos = (40, 40, 40)
    vp.camera_dir = (-1, -1, -1)

    pipeline.compute()
    vp.zoom_all()

    # Composition legend overlay
    legend_elements = sorted(composition.keys(), key=lambda e: -composition[e])
    for i, elem in enumerate(legend_elements):
        frac = composition[elem]
        label = TextLabelOverlay()
        label.text = f"● {elem} ({frac:.0%})"
        label.text_color = get_color(elem)
        label.font_size = 0.04
        label.alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        label.offset_x = -0.01
        label.offset_y = 0.01 + i * 0.05
        label.outline_enabled = True
        vp.overlays.append(label)

    # Title overlay
    title = TextLabelOverlay()
    comp_str = " ".join(f"{e}{composition[e]:.0%}" for e in legend_elements)
    title.text = f"NNIP MD — {comp_str}"
    title.font_size = 0.035
    title.alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
    title.offset_x = 0.01
    title.offset_y = 0.01
    title.outline_enabled = True
    vp.overlays.append(title)

    # Output path
    os.makedirs(VIS_DIR, exist_ok=True)
    if output is None:
        comp_tag = "".join(f"{e}{composition[e]*100:.0f}"
                           for e in legend_elements)
        output = os.path.join(VIS_DIR, f"nnip_{comp_tag}_md.mp4")

    print(f"Rendering animation to {output}...")
    vp.render_anim(filename=output, size=size,
                   renderer=TachyonRenderer(), fps=fps)
    print(f"Video saved as {output}")

    ovito.scene.clear()
    return output


def render_lammps(traj_path, composition, output=None):
    """Render a LAMMPS trajectory with type map ordering."""
    # For LAMMPS dumps, element order follows the type_map
    # Only include types that are actually present
    present = [e for e in TYPE_MAP if e in composition]
    return render_trajectory(traj_path, composition, elements=TYPE_MAP, output=output)


def render_ase(traj_path, composition, output=None):
    """Render an ASE .traj trajectory."""
    return render_trajectory(traj_path, composition, output=output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render NNIP MD trajectory")
    parser.add_argument("trajectory", help="Path to trajectory file")
    parser.add_argument("--composition", "-c", required=True,
                        help="Composition as 'Al:0.95,Cu:0.05'")
    parser.add_argument("--output", "-o", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    args = parser.parse_args()

    comp = {}
    for pair in args.composition.split(","):
        elem, frac = pair.strip().split(":")
        comp[elem.strip()] = float(frac.strip())

    render_trajectory(args.trajectory, comp, output=args.output, fps=args.fps)
