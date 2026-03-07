import os
from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer, TextLabelOverlay
from PySide6.QtCore import Qt


# Per-element display colors (RGB 0-1)
ELEMENT_COLORS = {
    "Fe": (0.6, 0.3, 0.1),   # brown
    "Mn": (0.6, 0.2, 0.6),   # purple
    "Ni": (0.4, 0.7, 0.4),   # green
    "Ti": (0.7, 0.7, 0.7),   # silver
    "Cu": (0.9, 0.5, 0.1),   # copper/orange
    "Cr": (0.3, 0.5, 0.8),   # steel blue
    "Co": (0.2, 0.3, 0.8),   # blue
    "Al": (0.5, 0.8, 1.0),   # light blue
    "Mg": (0.9, 0.9, 0.1),   # yellow
    "Zn": (0.6, 0.6, 0.8),   # bluish-white
}


def get_color(symbol):
    """Get color for a symbol, or generate a deterministic fallback."""
    if symbol in ELEMENT_COLORS:
        return ELEMENT_COLORS[symbol]
    # Simple hash-based fallback (deterministic RGB from symbol)
    import hashlib
    h = hashlib.md5(symbol.encode()).digest()
    return (h[0] / 255, h[1] / 255, h[2] / 255)


def render(composition, selected, traj_file="traj.lammpstrj"):
    """Render animation from a LAMMPS trajectory file."""
    pipeline = import_file(traj_file)

    # Assign colors and radii per atom type via a modifier
    def assign_type_visuals(frame, data):
        types = data.particles_.particle_types_
        for i, elem in enumerate(selected, start=1):
            try:
                pt = types.type_by_id_(i)
            except KeyError:
                continue
            pt.name = elem.symbol
            pt.color = get_color(elem.symbol)
            pt.radius = 0.8  # Reduced size for better visibility

    pipeline.modifiers.append(assign_type_visuals)
    pipeline.add_to_scene()

    # Camera — position scales with box size
    vp = Viewport()
    vp.type = Viewport.Type.Perspective
    vp.camera_pos = (40, 40, 40)
    vp.camera_dir = (-1, -1, -1)
    vp.zoom_all()  # Ensure all atoms are in frame

    # Dynamic legend labels
    for i, elem in enumerate(selected):
        label = TextLabelOverlay()
        frac = composition[elem.symbol]
        label.text = f"● {elem.symbol} ({frac:.0%})"
        label.text_color = get_color(elem.symbol)
        label.font_size = 0.04
        label.alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        label.offset_x = -0.01
        label.offset_y = 0.01 + i * 0.05
        label.outline_enabled = True
        vp.overlays.append(label)

    # Output filename from composition
    comp_str = "".join(f"{e.symbol}{composition[e.symbol]:.0%}"
                       for e in selected).replace("%", "")
    
    # Ensure visualization folder exists
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    filename = os.path.join(vis_dir, f"{comp_str}_vibration.mp4")

    print(f"Rendering animation to {filename}...")
    vp.render_anim(filename=filename,
                   size=(800, 600),
                   renderer=TachyonRenderer(),
                   fps=10)

    print(f"Video saved as {filename}")


if __name__ == "__main__":
    from config import composition, selected
    render(composition, selected)


if __name__ == "__main__":
    from config import composition, selected
    render(composition, selected)
