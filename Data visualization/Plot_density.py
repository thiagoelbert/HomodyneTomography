#!/usr/bin/env python3
"""
Visualize the real part of a reconstructed density matrix.

What it shows
-------------
- 3D bar plot of Re(rho_ij) for the leading ``DIM_PLOT`` levels.

How to use
----------
1) Point ``TARGET_FILE`` to the desired ``*.rho.txt`` file (written by ``run_tomography.py``).
2) Adjust ``DIM_PLOT`` to control how many levels are displayed along each axis.
3) Run ``python Plot_density.py`` to view the plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Max levels to display on each axis
DIM_PLOT = 5
TARGET_FILE = Path(r"\TomoOutput\wigner_CH3_open_pulse1.rho.txt")


def load_real_density_matrix(path: Path) -> np.ndarray:
    """Load the real-valued density matrix from a QuTiP-style text export."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[2:]:
        if line.startswith("#"):
            break
        rows.append([float(x) for x in line.split()])
    return np.asarray(rows)


def plot_density(rho_real: np.ndarray, fname: str, dim_plot: int) -> None:
    dim = min(dim_plot, rho_real.shape[0])
    values = rho_real[:dim, :dim]

    # Build 3D bar positions
    x, y = np.meshgrid(np.arange(dim), np.arange(dim))
    x_flat = x.flatten()
    y_flat = y.flatten()
    z = values.flatten()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    dx = dy = 0.8

    ax.bar3d(x_flat, y_flat, np.zeros_like(z), dx, dy, z, shade=True)
    ax.set_xlabel("Row (i)")
    ax.set_ylabel("Col (j)")
    ax.set_zlabel("rho_ij (real)")
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    ax.set_title("Density Matrix (real part)")
    fig.suptitle(fname)
    fig.tight_layout()

    plt.show()


def main():
    target = TARGET_FILE
    if not target.exists():
        print(f"Target file not found: {target}")
        return

    rho_real = load_real_density_matrix(target)
    plot_density(rho_real, target.name, DIM_PLOT)


if __name__ == "__main__":
    main()
