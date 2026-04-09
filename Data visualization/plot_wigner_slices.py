#!/usr/bin/env python3
"""
Visualize a saved Wigner reconstruction and extract a 1D quadrature slice.

What it shows
-------------
- Left: 2D heatmap of W(x, p) loaded from the ``npz`` file produced by
  ``run_tomography.py``.
- Right: A single quadrature slice at phase ``SLICE_PHASE`` (0 -> x axis,
  ``np.pi/2`` -> p axis) interpolated via bilinear sampling.

How to use
----------
1) Point ``TARGET_FILE`` to the desired Wigner ``.npz`` file.
2) Adjust ``SLICE_PHASE`` and ``N_SLICE_POINTS`` if needed.
3) Run ``python plot_wigner_slices.py`` to display the plots.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


TARGET_FILE = Path(r"TomoOutput\wigner_CH3_closed_pulse1.npz")
# Target slice phase (radians). 0 -> x axis, np.pi/2 -> p axis.
SLICE_PHASE = 0
# Number of points along the slice
N_SLICE_POINTS = 400
BASE_CMAP = "viridis"# "RdBu_r"
# If True, zero is forced to the midpoint color using a truncated linear
# colormap. If False, the base colormap is applied directly over [min(W), max(W)].
FIX_ZERO_COLOR = True


def load_npz(path: Path):
    data = np.load(path)
    return data["xvec"], data["pvec"], data["W"]


def truncated_colormap_for_zero(base_cmap_name: str, zero_position: float, n: int = 256):
    """
    Return a linear subrange of the base colormap such that:
    - values are still mapped linearly over [vmin, vmax]
    - the normalized data value for zero is mapped to the midpoint color
    - unused colors, if any, are trimmed from one colormap end instead of
      warping the color progression
    """
    base = plt.get_cmap(base_cmap_name)
    zero_position = float(np.clip(zero_position, 0.0, 1.0))

    if zero_position <= 0.0 or zero_position >= 1.0:
        return base

    # Choose the widest linear subrange [cmin, cmax] such that the linear map
    # from data-normalized position t to colormap position c satisfies c(0)=0.5.
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


def bilinear_interpolate(xvec: np.ndarray, pvec: np.ndarray, W: np.ndarray, xq: np.ndarray, pq: np.ndarray):
    """Bilinear interpolation of W at query points (xq, pq)."""
    x_idx = np.searchsorted(xvec, xq) - 1
    p_idx = np.searchsorted(pvec, pq) - 1

    x_idx = np.clip(x_idx, 0, len(xvec) - 2)
    p_idx = np.clip(p_idx, 0, len(pvec) - 2)

    x1 = xvec[x_idx]
    x2 = xvec[x_idx + 1]
    p1 = pvec[p_idx]
    p2 = pvec[p_idx + 1]

    wx = np.where(x2 != x1, (xq - x1) / (x2 - x1), 0.0)
    wp = np.where(p2 != p1, (pq - p1) / (p2 - p1), 0.0)

    w00 = (1 - wx) * (1 - wp)
    w10 = wx * (1 - wp)
    w01 = (1 - wx) * wp
    w11 = wx * wp

    v00 = W[p_idx, x_idx]
    v10 = W[p_idx, x_idx + 1]
    v01 = W[p_idx + 1, x_idx]
    v11 = W[p_idx + 1, x_idx + 1]

    return w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11


def make_slice(
    xvec: np.ndarray, pvec: np.ndarray, W: np.ndarray, phi: float
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Return (q, W_slice) along quadrature axis at angle phi, plus line endpoints
    (x_line, p_line) suitable for overlay on the heatmap.
    """
    c, s = np.cos(phi), np.sin(phi)
    bounds = []
    if abs(c) > 1e-9:
        bounds.extend([abs(xvec.max() / c), abs(xvec.min() / c)])
    if abs(s) > 1e-9:
        bounds.extend([abs(pvec.max() / s), abs(pvec.min() / s)])
    q_lim = min(b for b in bounds if b > 0) if bounds else 0.0
    q = np.linspace(-q_lim, q_lim, N_SLICE_POINTS)
    xq = q * c
    pq = q * s
    Wq = bilinear_interpolate(xvec, pvec, W, xq, pq)
    x_line = (-q_lim * c, q_lim * c)
    p_line = (-q_lim * s, q_lim * s)
    return q, Wq, x_line, p_line


def main():
    target = TARGET_FILE
    if not target.exists():
        print(f"Target file not found: {target}")
        return

    xvec, pvec, W = load_npz(target)
    w_min = float(np.min(W))
    w_max = float(np.max(W))

    q, Wq, x_line, p_line = make_slice(xvec, pvec, W, SLICE_PHASE)
    norm = Normalize(vmin=w_min, vmax=w_max)
    if FIX_ZERO_COLOR and w_min < 0.0 < w_max:
        zero_position = (0.0 - w_min) / (w_max - w_min)
        cmap = truncated_colormap_for_zero(BASE_CMAP, zero_position=zero_position)
    else:
        cmap = plt.get_cmap(BASE_CMAP)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im = ax0.imshow(
        W,
        extent=[xvec.min(), xvec.max(), pvec.min(), pvec.max()],
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="equal",  # keep same scale on x and p
    )
    # Overlay the slice direction on the heatmap for visual reference
    ax0.plot(x_line, p_line, color="red", linestyle="--", linewidth=1.5, label="slice")
    ax0.legend(loc="upper right")
    ax0.set_title("Wigner W(x, p)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("p")
    fig.colorbar(im, ax=ax0, shrink=0.85)

    ax1.plot(q, Wq)
    ax1.set_title(f"Slice at phase phi={SLICE_PHASE:.3f} rad")
    ax1.set_xlabel("quadrature q_phi")
    ax1.set_ylabel("Wigner W")

    fig.suptitle(target.name)
    plt.show()


if __name__ == "__main__":
    main()
