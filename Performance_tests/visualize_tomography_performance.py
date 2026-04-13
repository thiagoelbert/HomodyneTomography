#!/usr/bin/env python3
"""Visualize tomography benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


SUMMARY_FILE = ROOT / "Performance_tests" / "output" / "benchmark_summary.npz"
STATE_LABELS = {
    "open1": "SPAC",
    "open4": "Coherent",
    "closed1": "Single photon",
    "closed4": "Vacuum",
}


def format_nbins_label(nbins) -> str:
    return "raw" if nbins is None else str(int(nbins))


def barplot3d(
    ax,
    data: np.ndarray,
    nbins_values: np.ndarray,
    eta_values: np.ndarray,
    title: str,
    cmap: str,
    zmin=None,
    zmax=None,
) -> None:
    y_idx, x_idx = np.meshgrid(np.arange(len(nbins_values)), np.arange(len(eta_values)), indexing="ij")
    xpos = x_idx.ravel()
    ypos = y_idx.ravel()
    zpos = np.zeros_like(xpos, dtype=float)
    dx = np.full_like(xpos, 0.6, dtype=float)
    dy = np.full_like(ypos, 0.6, dtype=float)
    dz = np.asarray(data, dtype=float).ravel()

    finite = dz[np.isfinite(dz)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=float)

    zmin = float(np.min(finite)) if zmin is None else float(zmin)
    zmax = float(np.max(finite)) if zmax is None else float(zmax)
    if np.isclose(zmin, zmax):
        zmax = zmin + 1.0

    clipped = np.nan_to_num(dz, nan=zmin)
    norm = Normalize(vmin=zmin, vmax=zmax)
    colors = cm.get_cmap(cmap)(norm(clipped))

    ax.bar3d(xpos, ypos, zpos, dx, dy, clipped, color=colors, shade=True)
    ax.set_xticks(np.arange(len(eta_values)) + 0.3)
    ax.set_yticks(np.arange(len(nbins_values)) + 0.3)
    ax.set_xticklabels([f"{eta:.2f}" for eta in eta_values])
    ax.set_yticklabels([format_nbins_label(nbins) for nbins in nbins_values])
    ax.set_xlabel("Detection efficiency eta")
    ax.set_ylabel("Histogram bins")
    ax.set_zlabel(title)
    ax.set_title(title)
    ax.set_zlim(min(0.0, zmin), zmax)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    plt.colorbar(scalar_mappable, ax=ax, shrink=0.7, pad=0.08)


def main() -> None:
    if not SUMMARY_FILE.exists():
        print(f"Summary file not found: {SUMMARY_FILE}")
        return

    summary = np.load(SUMMARY_FILE, allow_pickle=True)
    nbins_values = summary["nbins_values"]
    eta_values = summary["eta_values"]
    states = [str(state) for state in summary["states"]]
    runtimes = summary["runtime_seconds"]
    average_runtime = np.mean(runtimes, axis=2)
    open4_alpha = summary["open4_alpha"]
    closed1_pop = summary["closed1_single_photon_population"]
    runtime_vmin = float(np.nanmin(runtimes))
    runtime_vmax = float(np.nanmax(runtimes))

    fig_average = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax_average = fig_average.add_subplot(1, 1, 1, projection="3d")
    barplot3d(
        ax_average,
        average_runtime,
        nbins_values,
        eta_values,
        "Average runtime across the four tomographies [s]",
        "magma",
        zmin=float(np.nanmin(average_runtime)),
        zmax=float(np.nanmax(average_runtime)),
    )
    fig_average.suptitle("Average tomography runtime")

    fig_runtime, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, subplot_kw={"projection": "3d"})
    for idx, state_key in enumerate(states):
        row = idx // 2
        col = idx % 2
        barplot3d(
            axes[row, col],
            runtimes[:, :, idx],
            nbins_values,
            eta_values,
            f"Runtime {STATE_LABELS.get(state_key, state_key)} [s]",
            "viridis",
            zmin=runtime_vmin,
            zmax=runtime_vmax,
        )
    fig_runtime.suptitle("Tomography benchmark runtimes")

    alpha_abs = np.abs(open4_alpha)
    alpha_phase = np.mod(np.angle(open4_alpha), 2.0 * np.pi)
    fig_metrics, axes_metrics = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True, subplot_kw={"projection": "3d"})
    barplot3d(
        axes_metrics[0],
        closed1_pop,
        nbins_values,
        eta_values,
        "Single-photon population",
        "cividis",
    )
    barplot3d(
        axes_metrics[1],
        alpha_abs,
        nbins_values,
        eta_values,
        "Coherent-state |alpha|",
        "plasma",
    )
    barplot3d(
        axes_metrics[2],
        alpha_phase,
        nbins_values,
        eta_values,
        "Coherent-state arg(alpha) [rad]",
        "twilight",
        zmin=0.0,
        zmax=2.0 * np.pi,
    )
    fig_metrics.suptitle("State metrics across the benchmark grid")

    plt.show()


if __name__ == "__main__":
    main()
