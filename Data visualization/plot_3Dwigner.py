#!/usr/bin/env python3
"""
Plot the same 3D Wigner surface used in ``run_tomography.py`` from a saved npz.

How to use
----------
1) Point ``TARGET_FILE`` to a tomography output ``.npz`` file.
2) Run ``python plot_wigner_surface.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


TARGET_FILE = Path(r"TomoOutput\wigner_CH3_closed_pulse1.npz")
BASE_CMAP = "seismic" #"viridis"
FIX_ZERO_COLOR = True


def load_wigner_npz(path: Path):
    data = np.load(path)
    required = ("xvec", "pvec", "W")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"Missing keys in {path}: {', '.join(missing)}")
    return data["xvec"], data["pvec"], data["W"]


def truncated_colormap_for_zero(base_cmap_name: str, zero_position: float, n: int = 256):
    """
    Return a linear subrange of the base colormap such that zero maps to the
    midpoint color while the value scale remains linear.
    """
    base = plt.get_cmap(base_cmap_name)
    zero_position = float(np.clip(zero_position, 0.0, 1.0))

    if zero_position <= 0.0 or zero_position >= 1.0:
        return base

    if zero_position < 0.5:
        cmax = 1.0
        cmin = (0.5 - zero_position) / (1.0 - zero_position)
    elif zero_position > 0.5:
        cmin = 0.0
        cmax = 0.5 / zero_position
    else:
        cmin, cmax = 0.0, 1.0

    samples = np.linspace(cmin, cmax, n)
    colors = base(samples)
    return LinearSegmentedColormap.from_list(f"{base_cmap_name}_trunc", colors, N=n)


def main() -> None:
    if not TARGET_FILE.exists():
        print(f"Target file not found: {TARGET_FILE}")
        return

    xvec, pvec, W = load_wigner_npz(TARGET_FILE)
    X, P = np.meshgrid(xvec, pvec)
    w_min = float(np.min(W))
    w_max = float(np.max(W))

    if FIX_ZERO_COLOR and w_min < 0.0 < w_max:
        norm = Normalize(vmin=w_min, vmax=w_max)
        zero_position = (0.0 - w_min) / (w_max - w_min)
        cmap = truncated_colormap_for_zero(BASE_CMAP, zero_position=zero_position)
    else:
        norm = None
        cmap = plt.get_cmap(BASE_CMAP)

    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(X, P, W, cmap=cmap, norm=norm, linewidth=0, antialiased=False)
    ax.set_title(TARGET_FILE.stem)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_zlabel("Wigner")
    plt.show()


if __name__ == "__main__":
    main()
