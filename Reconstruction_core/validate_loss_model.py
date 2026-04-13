"""
Minimal numerical validation for the loss-aware homodyne reconstruction.

Checks performed
----------------
1. The Bernoulli loss map preserves Hermiticity, positivity, and trace.
2. ``eta=1`` reproduces the input state exactly.
3. A lossy single-photon dataset reconstructed with the matching ``eta`` yields
   a pre-loss state close to ``|1><1|``.
4. The quadrature prediction changes when losses are enabled.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Reconstruction_core.mle_lvovsky import apply_loss_map, quadrature_probability, run_lvovsky_mle


RNG = np.random.default_rng(1234)
VAC_STD = 1.0 / np.sqrt(2.0)


def sample_vacuum(size: int) -> np.ndarray:
    return RNG.normal(loc=0.0, scale=VAC_STD, size=size)


def sample_single_photon(size: int) -> np.ndarray:
    u = RNG.gamma(shape=1.5, scale=1.0, size=size)
    signs = RNG.choice((-1.0, 1.0), size=size)
    return signs * np.sqrt(u)


def sample_lossy_single_photon(size: int, eta: float) -> np.ndarray:
    single_count = int(np.round(size * eta))
    vacuum_count = size - single_count
    samples = np.concatenate((sample_single_photon(single_count), sample_vacuum(vacuum_count)))
    RNG.shuffle(samples)
    return samples


def build_single_photon_dataset(phases: np.ndarray, samples_per_phase: int, eta: float) -> Dict[float, np.ndarray]:
    return {float(phi): sample_lossy_single_photon(samples_per_phase, eta) for phi in phases}


def single_photon_density(cutoff: int) -> np.ndarray:
    rho = np.zeros((cutoff, cutoff), dtype=np.complex128)
    rho[1, 1] = 1.0
    return rho


def coherent_density(cutoff: int, alpha: complex) -> np.ndarray:
    coeffs = np.array(
        [alpha ** n / np.sqrt(np.exp(math.lgamma(n + 1))) for n in range(cutoff)],
        dtype=np.complex128,
    )
    coeffs *= np.exp(-0.5 * np.abs(alpha) ** 2)
    coeffs /= np.linalg.norm(coeffs)
    return np.outer(coeffs, coeffs.conj())


def main() -> None:
    cutoff = 8
    eta = 0.8
    tol = 1e-10

    rho0 = 0.65 * coherent_density(cutoff, alpha=0.6 + 0.2j) + 0.35 * single_photon_density(cutoff)
    rho0 = 0.5 * (rho0 + rho0.conj().T)
    rho0 /= np.trace(rho0).real

    rho_eta = apply_loss_map(rho0, eta)
    evals = np.linalg.eigvalsh(rho_eta)

    print("Loss-map validation")
    print(f"trace(rho_eta) = {np.trace(rho_eta).real:.12f}")
    print(f"hermiticity error = {np.linalg.norm(rho_eta - rho_eta.conj().T):.3e}")
    print(f"min eigenvalue = {evals.min():.3e}")
    print(f"eta=1 identity error = {np.linalg.norm(apply_loss_map(rho0, 1.0) - rho0):.3e}")

    if not np.allclose(rho_eta, rho_eta.conj().T, atol=tol):
        raise AssertionError("Loss map did not preserve Hermiticity.")
    if not np.isclose(np.trace(rho_eta).real, 1.0, atol=tol):
        raise AssertionError("Loss map did not preserve trace.")
    if evals.min() < -1e-10:
        raise AssertionError("Loss map produced a non-positive matrix.")
    if not np.allclose(apply_loss_map(rho0, 1.0), rho0, atol=tol):
        raise AssertionError("eta=1 did not reproduce the original state.")

    x = np.linspace(-3.5, 3.5, 401)
    p_ideal = quadrature_probability(single_photon_density(cutoff), x, phi=0.0, eta=1.0)
    p_lossy = quadrature_probability(single_photon_density(cutoff), x, phi=0.0, eta=eta)
    distribution_gap = np.trapezoid(np.abs(p_ideal - p_lossy), x)
    print(f"quadrature L1 difference (eta=1 vs eta={eta:.1f}) = {distribution_gap:.6f}")
    if distribution_gap < 1e-3:
        raise AssertionError("Loss-aware quadrature distribution did not change measurably.")

    phases = np.linspace(0.0, np.pi, 12, endpoint=True)
    quadratures = build_single_photon_dataset(phases, samples_per_phase=3000, eta=eta)
    rho_hat, info = run_lvovsky_mle(
        quadratures,
        cutoff=6,
        eta=eta,
        max_iter=1200,
        tol=1e-7,
        min_prob=1e-9,
        nbins=80,
    )
    fidelity_proxy = float(np.real(rho_hat[1, 1]))
    print(
        "reconstruction status = "
        f"converged={info['converged']} iterations={info['iterations']} "
        f"rho_hat[1,1]={fidelity_proxy:.4f}"
    )
    if fidelity_proxy < 0.8:
        raise AssertionError("Loss-aware reconstruction did not recover the pre-loss single-photon population.")

    print("Validation completed successfully.")


if __name__ == "__main__":
    main()
