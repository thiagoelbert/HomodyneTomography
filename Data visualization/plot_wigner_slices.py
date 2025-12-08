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
import numpy as np


TARGET_FILE = Path("Wigner3D") / "wigner_CH3_closed_pulse1.npz"
# Target slice phase (radians). 0 -> x axis, np.pi/2 -> p axis.
SLICE_PHASE = 0.0
# Number of points along the slice
N_SLICE_POINTS = 400


def load_npz(path: Path):
    data = np.load(path)
    return data["xvec"], data["pvec"], data["W"]


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


def make_slice(xvec: np.ndarray, pvec: np.ndarray, W: np.ndarray, phi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (q, W_slice) along quadrature axis at angle phi."""
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
    return q, Wq


def main():
    target = TARGET_FILE
    if not target.exists():
        print(f"Target file not found: {target}")
        return

    xvec, pvec, W = load_npz(target)

    q, Wq = make_slice(xvec, pvec, W, SLICE_PHASE)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im = ax0.imshow(
        W,
        extent=[xvec.min(), xvec.max(), pvec.min(), pvec.max()],
        origin="lower",
        cmap="viridis",
        aspect="equal",  # keep same scale on x and p
    )
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
