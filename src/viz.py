from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer

pipeline = import_file("traj.lammpstrj")
pipeline.add_to_scene()

vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_pos = (40, 40, 40)
vp.camera_dir = (-1, -1, -1)

# Render all available frames into an MP4 file
# size=(width, height)
vp.render_anim(filename="copper_vibration.mp4", 
               size=(800, 600), 
               renderer=TachyonRenderer(),
               fps=60)

print("Video saved as copper_vibration.mp4")