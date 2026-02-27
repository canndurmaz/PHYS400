"""
viz.py — Visualization for Fe-V elastic tensor results.

1. Cij_heatmap.png  — matplotlib 6×6 heatmap with annotated values + moduli box
2. deformation.mp4  — OVITO animation of the 18 strain frames
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]


# ---------------------------------------------------------------------------
# 1. Cij heatmap
# ---------------------------------------------------------------------------

def plot_cij_heatmap(C: np.ndarray, moduli: dict, out_file: str = "Cij_heatmap.png") -> None:
    """
    6×6 colour-mapped heatmap of the elastic tensor, annotated with GPa values.
    A text box below lists the VRH polycrystalline moduli.
    """
    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 10),
        gridspec_kw={"height_ratios": [6, 1.5]},
    )
    ax_hm, ax_txt = axes

    # --- heatmap ---
    vmax = np.abs(C).max()
    im = ax_hm.imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate each cell
    for i in range(6):
        for j in range(6):
            val = C[i, j]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax_hm.text(j, i, f"{val:.1f}", ha="center", va="center",
                       fontsize=10, color=color, fontweight="bold")

    ax_hm.set_xticks(range(6))
    ax_hm.set_yticks(range(6))
    ax_hm.set_xticklabels([f"ε_{l}" for l in VOIGT_LABELS], fontsize=11)
    ax_hm.set_yticklabels([f"σ_{l}" for l in VOIGT_LABELS], fontsize=11)
    ax_hm.set_title("Elastic Tensor $C_{ij}$ (GPa)  —  Fe-V 10 at.%", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.035, pad=0.04)
    cbar.set_label("GPa", fontsize=11)

    # --- moduli text box ---
    ax_txt.axis("off")
    lines = [
        f"Young's modulus  E  = {moduli['E']:.1f} GPa",
        f"Bulk modulus     K  = {moduli['K_H']:.1f} GPa  "
        f"(Voigt {moduli['K_V']:.1f} / Reuss {moduli['K_R']:.1f})",
        f"Shear modulus    G  = {moduli['G_H']:.1f} GPa  "
        f"(Voigt {moduli['G_V']:.1f} / Reuss {moduli['G_R']:.1f})",
        f"Poisson's ratio  ν  = {moduli['nu']:.4f}",
    ]
    text = "\n".join(lines)
    ax_txt.text(
        0.5, 0.5, text,
        transform=ax_txt.transAxes,
        ha="center", va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4f8", edgecolor="#9ab"),
        family="monospace",
    )

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_file}")


# ---------------------------------------------------------------------------
# 2. OVITO deformation animation
# ---------------------------------------------------------------------------

def make_deformation_video(
    frames_file: str = "elastic_frames.lammpstrj",
    out_file: str = "deformation.mp4",
) -> None:
    """
    Render the 18-frame deformation trajectory using OVITO.
    Atoms coloured by particle type (V=type1=blue, Fe=type2=orange).
    """
    try:
        from ovito.io import import_file
        from ovito.modifiers import ColorCodingModifier
        from ovito.vis import Viewport, TachyonRenderer
        import ovito
    except ImportError:
        print("  OVITO not available — skipping video rendering.")
        return

    if not os.path.exists(frames_file):
        print(f"  Frames file '{frames_file}' not found — skipping video.")
        return

    pipeline = import_file(frames_file)

    # Colour by particle type
    pipeline.modifiers.append(
        ColorCodingModifier(
            property="Particle Type",
        )
    )

    vp = Viewport()
    vp.type = Viewport.Type.Perspective
    vp.camera_dir = (1, 1, -1)
    vp.camera_pos = (0, 0, 0)

    # Auto-fit camera once we have data
    data = pipeline.compute(0)
    cell = data.cell
    center = cell.matrix[:, 3] + 0.5 * (cell.matrix[:, 0] + cell.matrix[:, 1] + cell.matrix[:, 2])
    diag = np.linalg.norm(cell.matrix[:, :3].sum(axis=1))
    vp.camera_pos = center.tolist()[:3]
    vp.camera_dir = (-1, -1, 1)
    vp.fov = diag * 1.5

    # Add pipeline to scene before rendering
    pipeline.add_to_scene()

    n_frames = pipeline.source.num_frames
    print(f"  Rendering {n_frames} frames → {out_file}")

    renderer = TachyonRenderer(ambient_occlusion=True, shadows=False)
    vp.render_anim(
        filename=out_file,
        size=(800, 600),
        fps=6,
        range=(0, n_frames - 1),
        renderer=renderer,
    )
    print(f"  Saved: {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cij_file = "Cij.npy"
    moduli_file = "moduli.npy"

    if not os.path.exists(cij_file):
        sys.exit(f"ERROR: '{cij_file}' not found. Run elastic.py first.")
    if not os.path.exists(moduli_file):
        sys.exit(f"ERROR: '{moduli_file}' not found. Run elastic.py first.")

    C = np.load(cij_file)
    moduli = np.load(moduli_file, allow_pickle=True).item()

    print("=== Visualisation ===")
    print(f"  Loaded Cij ({C.shape}), moduli: E={moduli['E']:.1f} GPa")

    plot_cij_heatmap(C, moduli)
    make_deformation_video()


if __name__ == "__main__":
    main()
