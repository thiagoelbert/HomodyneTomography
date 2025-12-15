#!/usr/bin/env python3
"""
Generate a synthetic, calibrated homodyne dataset with phase-dependent quadratures.

Output layout (written to ``Simulation/single_photon_calib``):
- ``Acq_list.dat`` with phase/shutter metadata that ``collect_processed`` expects.
- Pulse 1: closed → single-photon quadrature state; open → photon-added coherent state (|alpha>, then a^†).
- Pulse 4: closed → vacuum; open → coherent state |alpha>.

The vacuum quadrature variance matches calibrated data (std = 1/sqrt(2)) and the
quadrature histogram responds to the local-oscillator phase.
"""

from pathlib import Path
from typing import Iterable
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "single_photon_calib"
CHANNEL = "CH3"  # default channel used in the tomography script
TARGET_PULSE = 1
CALIBRATION_PULSE = 4
PHASES = np.linspace(0.0, np.pi, 12, endpoint=True)  # equally spaced LO phases
SAMPLES_PER_PHASE = 8000
SHUTTERS: Iterable[str] = ("open", "closed")
VAC_STD = 1 / np.sqrt(2)  # calibrated vacuum std matching reconstruction convention
# Coherent amplitude used for the open shots (can be complex)
ALPHA = .0 + 0.0j
EFFICIENCY = 1
RNG = np.random.default_rng(42)


def sample_vacuum(size: int) -> np.ndarray:
    """Return calibrated vacuum quadratures (mean 0, var 0.5)."""
    return RNG.normal(loc=0.0, scale=VAC_STD, size=size)


def sample_single_photon(size: int) -> np.ndarray:
    """
    Sample quadratures from the |1> Fock state distribution.

    We draw x^2 from Gamma(k=3/2, theta=1) then assign a random sign so that
    the resulting pdf is P_1(x) = (2/sqrt(pi)) x^2 exp(-x^2), consistent with
    vacuum variance 1/2 used in the reconstruction.
    """
    u = RNG.gamma(shape=1.5, scale=1.0, size=size)
    signs = RNG.choice((-1.0, 1.0), size=size)
    return signs * np.sqrt(u)

def sample_noisy_single_photon(size: int, eta: float) -> np.ndarray:
    """
    Sample quadratures from a lossy single photon state with efficiency eta.
    """
    single = sample_single_photon(np.round(size*eta).astype(int))
    vacuum = sample_vacuum(np.round(size*(1-eta)).astype(int))
    return np.concatenate((single, vacuum))

def sample_coherent(alpha: complex, phase: float, size: int) -> np.ndarray:
    """Sample a coherent state; quadrature mean follows the LO phase."""
    alpha_rot = alpha * np.exp(-1j * phase)  # rotate into the measurement frame
    mean = np.sqrt(2) * alpha_rot.real
    return RNG.normal(loc=mean, scale=VAC_STD, size=size)


def spac_quadratures_pdf(x: np.ndarray, alpha: float, theta: float, eta: float) -> np.ndarray:
    """Single-photon-added coherent state quadrature PDF with efficiency eta."""
    c = np.sqrt(1 / np.pi) / (1 + np.abs(alpha) ** 2)
    m = (
        1
        - eta
        + 2 * eta * x ** 2
        + np.abs(alpha) ** 2 * (1 + 2 * eta * (eta - 1))
        - 2 * np.sqrt(2) * np.abs(alpha) * x * np.sqrt(eta) * (2 * eta - 1) * np.cos(theta)
        + 2 * np.abs(alpha) ** 2 * eta * (eta - 1) * np.cos(2 * theta)
    )
    e = np.exp(-1 * (x - np.sqrt(2 * eta) * np.abs(alpha) * np.cos(theta)) ** 2)
    return c * m * e



def draw_from_pdf(pdf: np.ndarray, grid: np.ndarray, size: int) -> np.ndarray:
    """Inverse-transform sampling on a fixed grid."""
    dx = grid[1] - grid[0]
    cdf = np.cumsum(pdf) * dx
    cdf /= cdf[-1]
    # Prepend boundaries so np.interp covers [0, 1].
    xp = np.concatenate(([0.0], cdf))
    fp = np.concatenate(([grid[0]], grid))
    return np.interp(RNG.random(size), xp, fp)


def sample_photon_added_coherent(alpha: complex, phase: float, size: int) -> np.ndarray:
    """
    Sample a photon-added coherent state a^†|alpha>.

    Measurement at phase ``phase`` is equivalent to rotating the state by -phase
    before measuring the X quadrature, which rotates the displacement and the |1> phase.
    """
    alpha_rot = alpha * np.exp(-1j * phase)
    x0 = alpha_rot.real

    # Grid wide enough to cover displaced tails.
    span = 5
    grid = np.linspace(x0-span, x0+span, 2001)

    pdf = spac_quadratures_pdf(grid, alpha, phase, eta=EFFICIENCY)
    pdf /= np.trapezoid(pdf, grid)
    return draw_from_pdf(pdf, grid, size)


def write_numeric_file(path: Path, values: np.ndarray) -> None:
    """Write one value per line in scientific notation (ASCII)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{v:.6e}" for v in values) + "\n"
    path.write_text(text, encoding="ascii")


def acq_header() -> str:
    return (
        "# idx SigMean SigVar SigAll DarkM DarkV FitEff3 Delay PhaseHD PhaseMic "
        "FitEff1 IntDC IntDCErr TrigCount TrigStd ThirdPulseM ThirdPulseVar FileRoot AcqTime"
    )


def format_acq_line(idx: int, phase: float, shutter: str, base_prefix: str) -> str:
    """
    Build one line for Acq_list.dat; only PhaseHD (col 8) and FileRoot (col -2)
    are used by the loader, the rest are placeholders to match the expected width.
    """
    return (
        f"{idx} 0 0 0 0 0 0 0 "
        f"{phase:.6f} 0 0 0 0 0 0 0 0 "
        f"{base_prefix}-{shutter} 0"
    )


def generate_dataset() -> None:
    lines = [acq_header()]
    for idx, phase in enumerate(PHASES, start=1):
        base_prefix = f"SimExp_{idx:03d}"
        for shutter in SHUTTERS:
            lines.append(format_acq_line(idx, phase, shutter, base_prefix))
            if shutter == "open":
                main_samples = sample_photon_added_coherent(ALPHA, phase, SAMPLES_PER_PHASE)
            else:
                main_samples = sample_noisy_single_photon(SAMPLES_PER_PHASE, EFFICIENCY)

            pulse_path = OUTPUT_DIR / f"{base_prefix}{CHANNEL}-{shutter}_{TARGET_PULSE:02d}.dat"
            write_numeric_file(pulse_path, main_samples)

            calib_path = OUTPUT_DIR / f"{base_prefix}{CHANNEL}-{shutter}_{CALIBRATION_PULSE:02d}.dat"
            if shutter == "open":
                calib_samples = sample_coherent(ALPHA, phase, SAMPLES_PER_PHASE)
            else:
                calib_samples = sample_vacuum(SAMPLES_PER_PHASE)
            write_numeric_file(calib_path, calib_samples)

    acq_list_path = OUTPUT_DIR / "Acq_list.dat"
    acq_list_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {len(lines) - 1} acquisitions to {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_dataset()
