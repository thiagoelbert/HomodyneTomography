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

from typing import Dict, List, Tuple, cast, Optional

import numpy as np
from scipy.special import eval_hermite, gammaln


def quadrature_psi(q: np.ndarray, n: int) -> np.ndarray:
    """
    Harmonic-oscillator wavefunction psi_n(q) for the X quadrature.

    Optics convention: x = (a + a^dagger)/sqrt(2), vacuum variance = 1/2.
    """
    norm = np.exp(-0.5 * q * q) / (np.pi ** 0.25 * np.sqrt(2.0 ** n * np.exp(gammaln(n + 1))))
    return norm * eval_hermite(n, q)


def _validate_eta(eta: float) -> float:
    """Validate the homodyne efficiency parameter."""
    eta = float(eta)
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"`eta` must satisfy 0 < eta <= 1, got {eta}.")
    return eta


def _loss_diagonal_coefficients(cutoff: int, eta: float) -> List[np.ndarray]:
    """
    Return Bernoulli coefficients for each loss sector ``k`` within the cutoff.

    For a fixed ``k``, the returned vector has length ``cutoff - k`` and stores
    ``B_{n+k,n}(eta)`` for ``n = 0, ..., cutoff-k-1``.
    """
    eta = _validate_eta(eta)
    if cutoff <= 0:
        return []

    if eta == 1.0:
        return [np.ones(cutoff, dtype=float)] + [np.zeros(cutoff - k, dtype=float) for k in range(1, cutoff)]

    log_eta = np.log(eta)
    log_one_minus_eta = np.log1p(-eta)
    coeffs: List[np.ndarray] = []

    for k in range(cutoff):
        n = np.arange(cutoff - k, dtype=float)
        log_binom = gammaln(n + k + 1.0) - gammaln(n + 1.0) - gammaln(k + 1.0)
        log_coeff = 0.5 * (log_binom + n * log_eta + k * log_one_minus_eta)
        coeffs.append(np.exp(log_coeff))
    return coeffs


def apply_loss_map(rho: np.ndarray, eta: float) -> np.ndarray:
    """
    Apply the Bernoulli loss channel to ``rho`` in the truncated Fock basis.

    The output has the same truncation dimension as ``rho`` and is given by
    Lvovsky's generalized Bernoulli transformation restricted consistently to
    the available matrix elements.
    """
    eta = _validate_eta(eta)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("`rho` must be a square matrix.")
    if eta == 1.0:
        return np.array(rho, dtype=np.complex128, copy=True)

    cutoff = rho.shape[0]
    coeffs = _loss_diagonal_coefficients(cutoff, eta)
    rho_eta = np.zeros_like(rho, dtype=np.complex128)

    for k, coeff in enumerate(coeffs):
        if coeff.size == 0:
            continue
        block = rho[k:, k:]
        rho_eta[: cutoff - k, : cutoff - k] += coeff[:, None] * block * coeff[None, :]

    rho_eta = 0.5 * (rho_eta + rho_eta.conj().T)
    trace = np.trace(rho_eta).real
    if trace > 0.0:
        rho_eta /= trace
    return rho_eta


def quadrature_probability(
    rho: np.ndarray,
    x: np.ndarray,
    phi: float,
    eta: float = 1.0,
) -> np.ndarray:
    """
    Evaluate the predicted quadrature distribution at phase ``phi``.

    When ``eta < 1``, the function first applies the Bernoulli loss channel and
    then evaluates the ideal quadrature probability of the resulting state.
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("`rho` must be a square matrix.")

    x = np.asarray(x, dtype=float)
    cutoff = rho.shape[0]
    n = np.arange(cutoff)
    psi_vals = np.stack([quadrature_psi(x, k) for k in range(cutoff)], axis=1)
    W = psi_vals * np.exp(-1j * n * phi)
    rho_eff = apply_loss_map(rho, eta)
    probs = np.real(np.sum((W @ rho_eff) * np.conj(W), axis=1))
    return np.clip(probs, 0.0, None)


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
    eta: float = 1.0,
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
    eta = _validate_eta(eta)
    coeffs = _loss_diagonal_coefficients(rho.shape[0], eta)
    probs = np.zeros(W.shape[0], dtype=float)

    for k, coeff in enumerate(coeffs):
        if coeff.size == 0 or not np.any(coeff):
            continue
        phi_k = W[:, : rho.shape[0] - k] * coeff[None, :]
        rho_block = rho[k:, k:]
        probs += np.real(np.sum((phi_k @ rho_block) * np.conj(phi_k), axis=1))

    probs = np.clip(probs, min_prob, None)
    if weights is None:
        weights_arr = np.ones_like(probs)
    else:
        weights_arr = weights
    total_counts = float(np.sum(weights_arr))

    # R = (1/N) sum_i (weights_i / p_i) |psi_i><psi_i|
    scaled = weights_arr / probs
    R = np.zeros_like(rho, dtype=np.complex128)
    for k, coeff in enumerate(coeffs):
        if coeff.size == 0 or not np.any(coeff):
            continue
        phi_k = W[:, : rho.shape[0] - k] * coeff[None, :]
        weighted_phi_k = phi_k * scaled[:, None]
        R[k:, k:] += phi_k.conj().T @ weighted_phi_k
    R /= total_counts

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
    eta: float = 1.0,
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
    eta:
        Homodyne detection efficiency. ``eta=1`` recovers the ideal detector
        model, while ``eta<1`` reconstructs the pre-loss state.
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
    eta = _validate_eta(eta)
    W, weights = _build_wavefunction_matrix(quadratures, cutoff, nbins=nbins)
    rho = np.eye(cutoff, dtype=np.complex128) / float(cutoff)

    deltas = []
    pmins = []
    pmaxs = []
    converged = False

    for it in range(1, max_iter + 1):
        rho, delta, p_min, p_max = _lvovsky_step(rho, W, eta=eta, weights=weights, min_prob=min_prob)
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
        "eta": eta,
    }
    return rho, info
