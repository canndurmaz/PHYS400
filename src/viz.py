from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer, TextLabelOverlay
from PySide6.QtCore import Qt

from config import CONFIG, selected

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
}

pipeline = import_file("traj.lammpstrj")

# Assign colors and radii per atom type via a modifier
def assign_type_visuals(frame, data):
    types = data.particles_.particle_types_
    for i, elem in enumerate(selected, start=1):
        pt = types.type_by_id_(i)
        pt.name = elem.symbol
        pt.color = ELEMENT_COLORS[elem.symbol]
        pt.radius = 0.8

pipeline.modifiers.append(assign_type_visuals)
pipeline.add_to_scene()

# Camera — position scales with box size
vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_pos = (40, 40, 40)
vp.camera_dir = (-1, -1, -1)

# Dynamic legend labels
for i, elem in enumerate(selected):
    label = TextLabelOverlay()
    frac = CONFIG["composition"][elem.symbol]
    label.text = f"● {elem.symbol} ({frac:.0%})"
    label.text_color = ELEMENT_COLORS[elem.symbol]
    label.font_size = 0.04
    label.alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
    label.offset_x = -0.01
    label.offset_y = 0.01 + i * 0.05
    label.outline_enabled = True
    vp.overlays.append(label)

# Output filename from composition
comp_str = "".join(f"{e.symbol}{CONFIG['composition'][e.symbol]:.0%}"
                   for e in selected).replace("%", "")
filename = f"{comp_str}_vibration.mp4"

print(f"Rendering animation to {filename}...")
vp.render_anim(filename=filename,
               size=(800, 600),
               renderer=TachyonRenderer(),
               fps=10)

print(f"Video saved as {filename}")
