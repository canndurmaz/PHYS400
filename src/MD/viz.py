import os
import ovito
from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer, TextLabelOverlay
from PySide6.QtCore import Qt


# Per-element display colors (RGB 0-1) - High contrast for 10 elements
ELEMENT_COLORS = {
    "Al": (0.80, 0.80, 0.80),   # Light Gray
    "Co": (0.00, 0.00, 0.70),   # Dark Blue
    "Cr": (0.00, 0.90, 0.90),   # Cyan
    "Cu": (1.00, 0.50, 0.00),   # Orange
    "Fe": (0.55, 0.27, 0.07),   # Brown
    "Mg": (1.00, 1.00, 0.00),   # Yellow
    "Mn": (0.60, 0.00, 0.60),   # Purple
    "Ni": (0.00, 0.80, 0.00),   # Green
    "Ti": (0.40, 0.40, 0.40),   # Dark Gray
    "Zn": (0.50, 0.50, 1.00),   # Slate Blue
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
    if not os.path.exists(traj_file):
        print(f"Warning: Trajectory file {traj_file} not found. Skipping visualization.")
        return

    # Clear any existing pipelines from the global scene
    for p in list(ovito.scene.pipelines):
        p.remove_from_scene()

    pipeline = import_file(traj_file)

    # Assign colors and radii per atom type via a modifier
    def assign_type_visuals(frame, data):
        types_prop = data.particles_.particle_types_
        for i, elem in enumerate(selected, start=1):
            pt = types_prop.type_by_id_(i)
            if pt is not None:
                pt.name = elem.symbol
                pt.color = get_color(elem.symbol)
                pt.radius = 0.8

    pipeline.modifiers.append(assign_type_visuals)
    pipeline.add_to_scene()

    # Camera — position scales with box size
    vp = Viewport()
    vp.type = Viewport.Type.Perspective
    vp.camera_pos = (40, 40, 40)
    vp.camera_dir = (-1, -1, -1)
    
    # Force a refresh to ensure zoom_all sees the data
    pipeline.compute()
    vp.zoom_all()

    # Dynamic legend labels
    for i, elem in enumerate(selected):
        label = TextLabelOverlay()
        frac = composition.get(elem.symbol, 0)
        label.text = f"● {elem.symbol} ({frac:.0%})"
        label.text_color = get_color(elem.symbol)
        label.font_size = 0.04
        label.alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        label.offset_x = -0.01
        label.offset_y = 0.01 + i * 0.05
        label.outline_enabled = True
        vp.overlays.append(label)

    # Output filename from composition
    comp_str = "".join(f"{e.symbol}{composition.get(e.symbol, 0):.0%}"
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
    try:
        from config import composition, selected
        render(composition, selected)
    except ImportError:
        print("Run from project root or ensure src/MD is in PYTHONPATH")
