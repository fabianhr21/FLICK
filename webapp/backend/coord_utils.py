"""
Coordinate utilities for the FLICK webapp.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from stl import mesh as stl_mesh_module


def get_stl_bbox(stl_path: str):
    """
    Load an STL file and return (min_coords, max_coords, centre) as numpy arrays.
    All arrays have shape (3,) for (x, y, z).
    """
    loaded = stl_mesh_module.Mesh.from_file(stl_path)
    all_verts = np.concatenate([loaded.v0, loaded.v1, loaded.v2])
    min_c = all_verts.min(axis=0)
    max_c = all_verts.max(axis=0)
    centre = (min_c + max_c) / 2.0
    return min_c, max_c, centre


def compute_step_size(min_c: np.ndarray, max_c: np.ndarray) -> float:
    """
    Compute STEP_SIZE = max(x_extent, y_extent) / 2 so the full STL footprint
    fits within the [-STEP_SIZE, +STEP_SIZE]^2 CFD domain.
    Returns a float in the same units as the STL (metres).
    """
    x_extent = float(max_c[0] - min_c[0])
    y_extent = float(max_c[1] - min_c[1])
    return max(x_extent, y_extent) / 2.0


def generate_heatmap_png(umag_csv_path: str, out_png_path: str) -> tuple:
    """
    Read an N×N UMAG CSV (wind speed magnitude matrix output by inference),
    and write a transparent-background jet heatmap PNG.

    Returns (vmin, vmax) of the non-zero wind speed values for the legend.
    """
    data = np.loadtxt(umag_csv_path, delimiter=',')

    # Mask solid-region zeros
    masked = np.ma.masked_where(data <= 0.0, data)

    if masked.count() > 0:
        vmin = float(masked.min())
        vmax = float(masked.max())
    else:
        vmin, vmax = 0.0, 1.0

    cmap = plt.cm.jet.copy()
    cmap.set_bad(alpha=0.0)  # transparent for masked (solid) cells

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0.0)
    ax.set_axis_off()

    ax.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='upper',
        interpolation='nearest',
    )

    plt.tight_layout(pad=0)
    plt.savefig(
        out_png_path,
        dpi=150,
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.close(fig)

    return vmin, vmax
