#!/usr/bin/env python3
"""
End-to-end real-data pipeline.

What you need on disk
---------------------
- A raw homodyne dataset folder that contains an ``Acq_list.dat`` file and one
  processed file per pulse with the pattern ``<base>CH{1|3}-<shutter>_<NN>.dat``.
  Pulse 4 must represent vacuum and is used for calibration.
- This script will create a sibling ``*_calib`` folder with the calibrated data
  (mean shifted to zero, variance scaled) if it does not already exist.

What this script does
---------------------
1. Calibrate the raw data (or reuse an existing ``*_calib`` folder).
2. Gather the calibrated quadrature samples per phase and channel.
3. Run Lvovsky's iterative MLE to reconstruct the density matrix.
4. Save the Wigner grid plus reconstruction metadata to ``TomoOutput`` and
   show a quick 3D surface plot for visual inspection.

Set the constants below to point at your dataset, then run ``main()``.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import minimize
import time

from Reconstruction_core.calibrate_dataset import calibrate_folder
from Reconstruction_core.calibrate_phase import calibrate_phase

from Reconstruction_core.collect_processed import collect
from Reconstruction_core.mle_lvovsky import run_lvovsky_mle

# Reconstruction defaults (tune here)
DATA_FOLDER = Path(r"I:\230226\3")
# Channel, pulses and shutters to reconstruct
CHANNEL = "CH3"
PULSES = (1, 4)
SHUTTERS = ["open", "closed"]
# Fock cutoff dimension for density matrix (larger is slower)
CUTOFF = 20
# Histogram bins per phase for the Lvovsky MLE (None to use raw samples)
NBINS_LVOVSKY = 120
# Convergence variables for Lvovsky reconstruction (tol)
TOL = 1e-7
MAX_ITER = 5000
MIN_PROB = 1e-9
# Wigner grid resolution (points per axis) and half-width range
WIGNER_POINTS = 500
WIGNER_XMAX = 5.0
# Coherent alpha fit bounds and optimizer settings
ALPHA_FIT_MAX_ITER = 200
#Output directory
OUTPUT_DIR = Path("TomoOutput")
# Optional phase calibration after scale calibration
ENABLE_PHASE_CALIBRATION = True

STATE_LABELS = {
    ("open", 1): "SPAC",
    ("open", 4): "Coherent",
    ("closed", 1): "Single photon",
    ("closed", 4): "Vacuum",
}


def build_quadrature_dict(subset) -> Dict[float, np.ndarray]:
    """Collect concatenated quadrature samples per phase from a filtered DataFrame."""
    quadratures: Dict[float, np.ndarray] = {}
    if subset.empty:
        return quadratures
    phases = np.array(sorted(subset["phase_hd"].unique()), dtype=float)
    for ph in phases:
        vals = np.concatenate(subset.loc[subset["phase_hd"] == ph, "values"].to_numpy())
        quadratures[ph] = vals
    return quadratures


def estimate_single_photon_efficiency(rho: np.ndarray) -> Optional[float]:
    """Estimate |1><1| population as the single-photon efficiency."""
    if rho.shape[0] < 2:
        return None
    eta = float(np.real(rho[1, 1]))
    return max(0.0, min(1.0, eta))


def fit_coherent_alpha(
    rho: np.ndarray,
    alpha_max: float,
    max_iter: int = ALPHA_FIT_MAX_ITER,
) -> complex:
    """Fit alpha by maximizing overlap with coherent states (L-BFGS-B)."""
    rho_obj = qt.Qobj(rho)
    dim = rho.shape[0]
    alpha0 = qt.expect(qt.destroy(dim), rho_obj)
    bounds = [(-alpha_max, alpha_max), (-alpha_max, alpha_max)]

    def objective(params: np.ndarray) -> float:
        alpha = complex(params[0], params[1])
        ket = qt.coherent(dim, alpha)
        return -float((ket.dag() * rho_obj * ket).real)

    res = minimize(
        objective,
        x0=np.array([alpha0.real, alpha0.imag], dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    return complex(float(res.x[0]), float(res.x[1]))


def _format_complex(val: complex) -> str:
    return f"{val.real:.3f}{val.imag:+.3f}j"


def reconstruct_wigner(
    quadratures: Dict[float, np.ndarray],
    title: str,
    save_path: Optional[Path] = None,
    state_label: str = "",
):
    """
    Run Lvovsky MLE on the provided quadrature samples and optionally persist
    the reconstructed Wigner grid plus density matrix to ``save_path`` (npz and
    human-readable ``.rho.txt``).
    """
    if not quadratures:
        print("No quadrature data available for reconstruction.")
        return None
    if qt is None:
        print("QuTiP is required for Wigner reconstruction; install with `pip install qutip`.")
        return None

    rho_hat, info = run_lvovsky_mle(
        quadratures,
        cutoff=CUTOFF,
        max_iter=MAX_ITER,
        tol=TOL,
        min_prob=MIN_PROB,
        nbins=NBINS_LVOVSKY,
    )
    mle_status = (
        f"Lvovsky converged={info['converged']} iterations={info['iterations']} "
        f"delta={info['delta']:.2e} nbins={info['nbins']}"
    )
    print(f"MLE status: {mle_status}")

    xvec = np.linspace(-WIGNER_XMAX, WIGNER_XMAX, WIGNER_POINTS)
    pvec = np.linspace(-WIGNER_XMAX, WIGNER_XMAX, WIGNER_POINTS)
    W = qt.wigner(qt.Qobj(rho_hat), xvec, pvec)

    efficiency = None
    alpha_fit = None
    label_lower = state_label.lower()
    if "single photon" in label_lower:
        efficiency = estimate_single_photon_efficiency(rho_hat)
    if label_lower == "coherent" or "coherent" in label_lower:
        alpha_fit = fit_coherent_alpha(
            rho_hat,
            alpha_max=WIGNER_XMAX / np.sqrt(2.0),
        )

    if efficiency is not None:
        print(f"Single-photon efficiency (eta): {efficiency:.3f}")
    if alpha_fit is not None:
        print(f"Coherent alpha fit: {_format_complex(alpha_fit)}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        alpha_payload = alpha_fit if alpha_fit is not None else np.nan + 1j * np.nan
        efficiency_payload = efficiency if efficiency is not None else np.nan
        np.savez(
            save_path,
            xvec=xvec,
            pvec=pvec,
            W=W,  # type: ignore
            rho=rho_hat,
            mle_status=mle_status,
            nbins_lvovsky=NBINS_LVOVSKY, # type: ignore
            cutoff=CUTOFF,
            tol=TOL,
            max_iter=MAX_ITER,
            fit_single_photon_efficiency=efficiency_payload,
            fit_coherent_alpha=alpha_payload,
        )
        # Also write density matrix as a text file
        rho_txt = save_path.with_suffix(".rho.txt")
        rho_real = np.real(rho_hat)
        rho_imag = np.imag(rho_hat)
        with open(rho_txt, "w", encoding="utf-8") as f:
            f.write("# rho complex matrix\n# real part\n")
            np.savetxt(f, rho_real, fmt="%.6e")
            f.write("# imag part\n")
            np.savetxt(f, rho_imag, fmt="%.6e")

    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    X, P = np.meshgrid(xvec, pvec)
    ax.plot_surface(X, P, W, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_zlabel("Wigner")
    return fig


def main():
    t0 = time.perf_counter()
    if DATA_FOLDER.name.endswith("_phasecalib"):
        calib_folder = DATA_FOLDER
        print(f"Using existing phase-calibrated folder: {calib_folder}")
    elif DATA_FOLDER.name.endswith("_calib"):
        calib_folder = DATA_FOLDER
        print(f"Using existing calibrated folder: {calib_folder}")
    else:
        existing_calib = DATA_FOLDER.with_name(DATA_FOLDER.name + "_calib")
        if existing_calib.exists() and (existing_calib / "Acq_list.dat").exists():
            calib_folder = existing_calib
            print(f"Using existing calibrated folder: {calib_folder}")
        else:
            calib_folder = calibrate_folder(DATA_FOLDER)

    if ENABLE_PHASE_CALIBRATION:
        if calib_folder.name.endswith("_phasecalib"):
            print(f"Phase calibration already present: {calib_folder}")
        else:
            existing_phasecalib = calib_folder.with_name(calib_folder.name + "_phasecalib")
            if existing_phasecalib.exists() and (existing_phasecalib / "Acq_list.dat").exists():
                calib_folder = existing_phasecalib
                print(f"Using existing phase-calibrated folder: {calib_folder}")
            else:
                calib_folder = calibrate_phase(calib_folder)
    t_calib = time.perf_counter() - t0

    t1 = time.perf_counter()
    df = collect(calib_folder, channels=[CHANNEL], pulses=list(PULSES), shutters=SHUTTERS)
    t_collect = time.perf_counter() - t1

    t2 = time.perf_counter()
    for pulse in PULSES:
        for shutter in SHUTTERS:
            subset = df[(df["channel"] == CHANNEL) & (df["shutter"] == shutter) & (df["pulse"] == pulse)]
            if subset.empty:
                print(f"No data for {CHANNEL} {shutter} pulse {pulse}")
                continue
            quadratures = build_quadrature_dict(subset)
            outfile = OUTPUT_DIR / f"wigner_{CHANNEL}_{shutter}_pulse{pulse}.npz"
            state_label = STATE_LABELS.get((shutter.lower(), pulse), "Unknown state")
            title = f"Wigner {CHANNEL} {shutter} pulse {pulse} ({state_label})"
            reconstruct_wigner(quadratures, title, save_path=outfile, state_label=state_label)
    t_reconstruct = time.perf_counter() - t2

    print(
        f"Timing: calibration={t_calib:.2f}s, collect={t_collect:.2f}s, "
        f"reconstruction+plot={t_reconstruct:.2f}s"
    )

    plt.show()


if __name__ == "__main__":
    main()
