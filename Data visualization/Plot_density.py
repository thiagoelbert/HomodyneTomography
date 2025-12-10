#!/usr/bin/env python3
"""
Visualize the real and imaginary parts of a reconstructed density matrix.

What it shows
-------------
- Left: 3D bar plot of Re(rho_ij) for the leading ``DIM_PLOT`` levels.
- Right: 3D bar plot of Im(rho_ij) for the leading ``DIM_PLOT`` levels.

How to use
----------
1) Point ``TARGET_FILE`` to the desired ``*.rho.txt`` file (written by ``run_tomography.py``).
2) Adjust ``DIM_PLOT`` to control how many levels are displayed along each axis.
3) Run ``python Plot_density.py`` to view both plots side-by-side.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# Max levels to display on each axis
DIM_PLOT = 5
# File containing both real and imaginary parts (QuTiP text export format)
TARGET_FILE = Path(r"TomoOutput\wigner_CH3_closed_pulse1.rho.txt")


def _parse_matrix(lines) -> np.ndarray:
    rows = []
    for line in lines:
        if not line.strip():
            continue
        rows.append([float(x) for x in line.split()])
    return np.asarray(rows)


def load_density_matrices(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real and imaginary parts from a QuTiP-style text export.

    Expects sections starting with '# real part' and '# imag part'.
    """
    real_lines, imag_lines = [], []
    section = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            lower = stripped.lower()
            if "real" in lower:
                section = "real"
            elif "imag" in lower:
                section = "imag"
            else:
                continue
            continue
        if section == "real":
            real_lines.append(stripped)
        elif section == "imag":
            imag_lines.append(stripped)

    real = _parse_matrix(real_lines)
    imag = _parse_matrix(imag_lines) if imag_lines else np.zeros_like(real)
    return real, imag


def plot_density_part(values: np.ndarray, title: str, ax, dim_plot: int) -> None:
    dim = min(dim_plot, values.shape[0])
    vals = values[:dim, :dim]

    x, y = np.meshgrid(np.arange(dim), np.arange(dim))
    x_flat = x.flatten()
    y_flat = y.flatten()
    z = vals.flatten()

    dx = dy = 0.8
    ax.bar3d(x_flat, y_flat, np.zeros_like(z), dx, dy, z, shade=True)
    ax.set_xlabel("Row (i)")
    ax.set_ylabel("Col (j)")
    ax.set_zlabel(title)
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    ax.set_title(title)
    ax.set_zlim(z.min(), z.max())


def shared_zlims(real: np.ndarray, imag: np.ndarray, dim_plot: int) -> Tuple[float, float]:
    dim = min(dim_plot, real.shape[0], imag.shape[0])
    real_vals = real[:dim, :dim]
    imag_vals = imag[:dim, :dim]
    zmin = min(real_vals.min(), imag_vals.min())
    zmax = max(real_vals.max(), imag_vals.max())
    return zmin, zmax


def main():
    target = TARGET_FILE
    if not target.exists():
        print(f"Target file not found: {target}")
        return

    rho_real, rho_imag = load_density_matrices(target)
    zmin, zmax = shared_zlims(rho_real, rho_imag, DIM_PLOT)

    fig = plt.figure(figsize=(12, 5))
    ax_real = fig.add_subplot(121, projection="3d")
    ax_imag = fig.add_subplot(122, projection="3d")

    plot_density_part(rho_real, "Re(rho_ij)", ax_real, DIM_PLOT)
    plot_density_part(rho_imag, "Im(rho_ij)", ax_imag, DIM_PLOT)
    ax_real.set_zlim(zmin, zmax)
    ax_imag.set_zlim(zmin, zmax)

    fig.suptitle(target.name)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
