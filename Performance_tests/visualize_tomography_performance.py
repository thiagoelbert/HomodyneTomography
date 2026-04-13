#!/usr/bin/env python3
"""Visualize tomography benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import qutip as qt
from scipy.optimize import minimize, minimize_scalar

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


def coherent_state_fidelity(rho: np.ndarray, alpha: complex) -> float:
    rho_obj = qt.Qobj(rho)
    target_dm = qt.ket2dm(qt.coherent(rho.shape[0], alpha))
    return float(qt.fidelity(rho_obj, target_dm))


def find_closest_coherent_state(rho: np.ndarray) -> tuple[complex, float]:
    rho_obj = qt.Qobj(rho)
    dim = rho.shape[0]
    alpha0 = qt.expect(qt.destroy(dim), rho_obj)
    alpha_max = 5.0 / np.sqrt(2.0)
    bounds = [(-alpha_max, alpha_max), (-alpha_max, alpha_max)]

    def objective(params: np.ndarray) -> float:
        alpha = complex(params[0], params[1])
        target_dm = qt.ket2dm(qt.coherent(dim, alpha))
        return -float(qt.fidelity(rho_obj, target_dm))

    res = minimize(
        objective,
        x0=np.array([alpha0.real, alpha0.imag], dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )
    alpha_fit = complex(float(res.x[0]), float(res.x[1]))
    fidelity = coherent_state_fidelity(rho, alpha_fit)
    return alpha_fit, fidelity


def single_photon_mixture_density_matrix(dim: int, p1: float) -> qt.Qobj:
    p1 = float(np.clip(p1, 0.0, 1.0))
    diag = np.zeros(dim, dtype=float)
    diag[0] = 1.0 - p1
    if dim > 1:
        diag[1] = p1
    return qt.Qobj(np.diag(diag))


def single_photon_mixture_fidelity(rho: np.ndarray, p1: float) -> float:
    rho_obj = qt.Qobj(rho)
    target_dm = single_photon_mixture_density_matrix(rho.shape[0], p1)
    return float(qt.fidelity(rho_obj, target_dm))


def find_closest_single_photon_mixture(rho: np.ndarray) -> tuple[float, float]:
    rho_obj = qt.Qobj(rho)

    def objective(p1: float) -> float:
        target_dm = single_photon_mixture_density_matrix(rho.shape[0], p1)
        return -float(qt.fidelity(rho_obj, target_dm))

    res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    p1_best = float(np.clip(res.x, 0.0, 1.0))
    fidelity = single_photon_mixture_fidelity(rho, p1_best)
    return p1_best, fidelity


def compute_fidelity_metrics(summary, states: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nbins_values = summary["nbins_values"]
    eta_values = summary["eta_values"]
    result_files = summary["result_files"]
    state_index = {state: idx for idx, state in enumerate(states)}

    open4_alpha = np.full((len(nbins_values), len(eta_values)), np.nan + 1j * np.nan, dtype=np.complex128)
    open4_fidelity = np.full((len(nbins_values), len(eta_values)), np.nan, dtype=float)
    closed1_pop = np.full((len(nbins_values), len(eta_values)), np.nan, dtype=float)
    closed1_fidelity = np.full((len(nbins_values), len(eta_values)), np.nan, dtype=float)

    open4_idx = state_index.get("open4")
    closed1_idx = state_index.get("closed1")

    for i in range(len(nbins_values)):
        for j in range(len(eta_values)):
            if open4_idx is not None:
                open4_file = Path(str(result_files[i, j, open4_idx]))
                if open4_file.exists():
                    with np.load(open4_file, allow_pickle=False) as data:
                        rho = np.array(data["rho"], copy=False)
                    alpha_fit, fidelity = find_closest_coherent_state(rho)
                    open4_alpha[i, j] = alpha_fit
                    open4_fidelity[i, j] = fidelity

            if closed1_idx is not None:
                closed1_file = Path(str(result_files[i, j, closed1_idx]))
                if closed1_file.exists():
                    with np.load(closed1_file, allow_pickle=False) as data:
                        rho = np.array(data["rho"], copy=False)
                    p1_best, fidelity = find_closest_single_photon_mixture(rho)
                    closed1_pop[i, j] = p1_best
                    closed1_fidelity[i, j] = fidelity

    return open4_alpha, open4_fidelity, closed1_pop, closed1_fidelity


def barplot3d(
    ax,
    data: np.ndarray,
    nbins_values: np.ndarray,
    eta_values: np.ndarray,
    title: str,
    cmap: str,
    zmin=None,
    zmax=None,
    base_at_min: bool = False,
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
    if base_at_min:
        zpos = np.full_like(xpos, zmin, dtype=float)
        dz_plot = clipped - zmin
    else:
        dz_plot = clipped

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz_plot, color=colors, shade=True)
    ax.set_xticks(np.arange(len(eta_values)) + 0.3)
    ax.set_yticks(np.arange(len(nbins_values)) + 0.3)
    ax.set_xticklabels([f"{eta:.2f}" for eta in eta_values])
    ax.set_yticklabels([format_nbins_label(nbins) for nbins in nbins_values])
    ax.set_xlabel("Detection efficiency eta")
    ax.set_ylabel("Histogram bins")
    ax.set_zlabel(title)
    ax.set_title(title)
    ax.set_zlim(zmin if base_at_min else min(0.0, zmin), zmax)
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
    open4_alpha, open4_fidelity, closed1_pop, closed1_fidelity = compute_fidelity_metrics(summary, states)
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
        "viridis",
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

    fig_fidelity, axes_fidelity = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, subplot_kw={"projection": "3d"})
    barplot3d(
        axes_fidelity[0],
        open4_fidelity,
        nbins_values,
        eta_values,
        "Coherent-state fidelity",
        "viridis",
        base_at_min=True,
    )
    barplot3d(
        axes_fidelity[1],
        closed1_fidelity,
        nbins_values,
        eta_values,
        "Single-photon-mixture fidelity",
        "viridis",
        base_at_min=True,
    )
    fig_fidelity.suptitle("Closest-state fidelities")

    plt.show()


if __name__ == "__main__":
    main()
