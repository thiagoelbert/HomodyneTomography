"""
Lvovsky iterative maximum-likelihood reconstruction for homodyne tomography.

This module reconstructs a density matrix from homodyne quadrature samples
collected at multiple phases. It implements the operator-update scheme from
Lvovsky (2004), which iteratively refines the state until the forward model
matches the measured quadrature statistics.

Inputs expected by this module
------------------------------
- ``quadratures``: mapping phase (radians) -> 1D numpy array of samples.
- ``cutoff``: Fock-space cutoff dimension (matrix will be ``cutoff x cutoff``).
- Optional histogram binning (``nbins``) for speed on large datasets.

Outputs
-------
- ``rho_hat``: estimated density matrix.
- ``info``: convergence metadata (iterations, deltas, probability extrema).
"""

from __future__ import annotations

from typing import Dict, Tuple, cast, Optional

import numpy as np
from scipy.special import eval_hermite, gammaln


def quadrature_psi(q: np.ndarray, n: int) -> np.ndarray:
    """
    Harmonic-oscillator wavefunction psi_n(q) for the X quadrature.

    The quadrature is scaled so the vacuum has std = 0.5 (x = (a + a†)/2),
    hence we evaluate the standard wavefunction at q_std = sqrt(2) * q and
    include the Jacobian factor sqrt(dq_std/dq) = 2**0.25.
    """
    q_std = np.sqrt(2.0) * q
    norm = np.exp(-0.5 * q_std * q_std) / (np.pi ** 0.25 * np.sqrt(2.0 ** n * np.exp(gammaln(n + 1))))
    return (2.0 ** 0.25) * norm * eval_hermite(n, q_std)

def _build_wavefunction_matrix(
    quadratures: Dict[float, np.ndarray],
    cutoff: int,
    nbins: Optional[int] = None,
    bin_pad_frac: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the quadrature wavefunction matrix ``W`` for all measurements.

    Each row of ``W`` corresponds to |x, phi> expressed in the Fock basis up to
    ``cutoff``. If ``nbins`` is provided, raw samples are histogrammed per phase
    and each bin center becomes a row weighted by its counts; this reduces
    iterations for large datasets.

    Returns
    -------
    W:
        Complex matrix of shape (N, cutoff), one row per sample/bin.
    weights:
        1D array of counts per row (all ones when unbinned).
    """
    n = np.arange(cutoff)

    if nbins is None:
        rows = []
        for phi in sorted(quadratures.keys()):
            x = np.asarray(quadratures[phi], dtype=float)
            psi_vals = np.stack([quadrature_psi(x, k) for k in range(cutoff)], axis=1)  # (len(x), cutoff)
            phase = np.exp(-1j * n * phi)
            rows.append(psi_vals * phase)
        W = np.vstack(rows)
        weights = np.ones(W.shape[0], dtype=float)
        return W, weights

    # Binned path: histogram per phase and only keep non-empty bins.
    all_vals = np.concatenate(list(quadratures.values()))
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = bin_pad_frac * (vmax - vmin + 1e-12)
    edges = np.linspace(vmin - pad, vmax + pad, nbins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []
    weights_list = []
    for phi in sorted(quadratures.keys()):
        hist, _ = np.histogram(quadratures[phi], bins=edges)
        mask = hist > 0
        if not np.any(mask):
            continue
        x = bin_centers[mask]
        counts = hist[mask].astype(float)
        psi_vals = np.stack([quadrature_psi(x, k) for k in range(cutoff)], axis=1)
        phase = np.exp(-1j * n * phi)
        rows.append(psi_vals * phase)
        weights_list.append(counts)

    if not rows:
        return np.empty((0, cutoff), dtype=np.complex128), np.array([], dtype=float)

    W = np.vstack(rows)
    weights = np.concatenate(weights_list)
    return W, weights


def _lvovsky_step(
    rho: np.ndarray,
    W: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_prob: float = 1e-12,
) -> Tuple[np.ndarray, float, float, float]:
    """
    One Lvovsky iteration: ``rho_{k+1} = R rho R / Tr(R rho R)``.

    Args
    ----
    rho:
        Current density matrix (``cutoff x cutoff``).
    W:
        Wavefunction matrix from ``_build_wavefunction_matrix``.
    weights:
        Optional per-row counts (used when histogram binning is enabled).
    min_prob:
        Numerical floor to avoid division by zero.

    Returns
    -------
    rho_next:
        Updated density matrix.
    delta:
        Frobenius-norm difference to previous ``rho`` (convergence metric).
    p_min, p_max:
        Extremal probabilities encountered in this step.
    """
    probs = np.real(np.sum((W @ rho) * np.conj(W), axis=1))
    probs = np.clip(probs, min_prob, None)
    if weights is None:
        weights_arr = np.ones_like(probs)
    else:
        weights_arr = weights
    total_counts = float(np.sum(weights_arr))

    # R = (1/N) sum_i (weights_i / p_i) |psi_i><psi_i|
    scaled = weights_arr / probs
    weighted_W = W * scaled[:, None]
    R = (W.conj().T @ weighted_W) / total_counts

    rho_next = R @ rho @ R
    rho_next = 0.5 * (rho_next + rho_next.conj().T)  # enforce Hermiticity
    rho_next /= np.trace(rho_next).real

    delta: float = cast(float, np.linalg.norm(rho_next - rho, ord="fro").item())
    p_min: float = float(probs.min())
    p_max: float = float(probs.max())
    return rho_next, delta, p_min, p_max


def run_lvovsky_mle(
    quadratures: Dict[float, np.ndarray],
    cutoff: int,
    max_iter: int = 200,
    tol: float = 1e-7,
    min_prob: float = 1e-12,
    nbins: Optional[int] = None,
):
    """
    Run Lvovsky iterative MLE on homodyne quadrature samples.

    Args
    ----
    quadratures:
        Mapping phase (radians) -> 1D array of quadrature samples.
    cutoff:
        Fock cutoff dimension for the reconstruction (rho is ``cutoff x cutoff``).
    max_iter:
        Maximum number of iterations before giving up.
    tol:
        Convergence tolerance on Frobenius norm between successive states.
    min_prob:
        Probability floor to avoid singular updates.
    nbins:
        Histogram bins per phase. Set to None to use raw samples.

    Returns
    -------
    rho_hat:
        Estimated density matrix (``cutoff x cutoff``).
    info:
        Dict with convergence metadata (iterations, converged, deltas, p_min/max).
    """
    W, weights = _build_wavefunction_matrix(quadratures, cutoff, nbins=nbins)
    rho = np.eye(cutoff, dtype=np.complex128) / float(cutoff)

    deltas = []
    pmins = []
    pmaxs = []
    converged = False

    for it in range(1, max_iter + 1):
        rho, delta, p_min, p_max = _lvovsky_step(rho, W, weights=weights, min_prob=min_prob)
        deltas.append(delta)
        pmins.append(p_min)
        pmaxs.append(p_max)
        if delta < tol:
            converged = True
            break

    info = {
        "iterations": it,
        "converged": converged,
        "delta": deltas[-1],
        "deltas": deltas,
        "p_min": min(pmins),
        "p_max": max(pmaxs),
        "nbins": nbins,
    }
    return rho, info
