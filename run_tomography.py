#!/usr/bin/env python3
"""
End-to-end real-data pipeline (IDE-friendly).

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
import time

from Reconstruction_core.calibrate_dataset import calibrate_folder
from Reconstruction_core.collect_processed import collect
from Reconstruction_core.mle_lvovsky import run_lvovsky_mle

# Reconstruction defaults (tune here)
DATA_FOLDER = Path(r"C:\Users\Thiago Guimarães\Documents\Repositorios\Marco-Setup\Data03121")  # raw data; calibrated folder is created automatically
CHANNEL = "CH3"
PULSE = 1
SHUTTERS = ("open", "closed")
CUTOFF = 20
NBINS_LVOVSKY = 120
TOL = 1e-7
MAX_ITER = 2000
MIN_PROB = 1e-9
WIGNER_POINTS = 60
WIGNER_XMAX = 5.0
OUTPUT_DIR = Path("TomoOutput")


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


def reconstruct_wigner(quadratures: Dict[float, np.ndarray], title: str, save_path: Optional[Path] = None):
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

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            xvec=xvec,
            pvec=pvec,
            W=W,  # type: ignore
            rho=rho_hat,
            mle_status=mle_status,
            nbins_lvovsky=NBINS_LVOVSKY,
            cutoff=CUTOFF,
            tol=TOL,
            max_iter=MAX_ITER,
        )
        # Also write density matrix as a text file for quick inspection (separate real/imag blocks)
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
    if DATA_FOLDER.name.endswith("_calib"):
        calib_folder = DATA_FOLDER
        print(f"Using existing calibrated folder: {calib_folder}")
    else:
        calib_folder = calibrate_folder(DATA_FOLDER)
    t_calib = time.perf_counter() - t0

    t1 = time.perf_counter()
    df = collect(calib_folder, channels=[CHANNEL], pulses=[PULSE], shutters=SHUTTERS)
    t_collect = time.perf_counter() - t1

    t2 = time.perf_counter()
    for shutter in SHUTTERS:
        subset = df[(df["channel"] == CHANNEL) & (df["shutter"] == shutter) & (df["pulse"] == PULSE)]
        if subset.empty:
            print(f"No data for {CHANNEL} {shutter} pulse {PULSE}")
            continue
        quadratures = build_quadrature_dict(subset)
        outfile = OUTPUT_DIR / f"wigner_{CHANNEL}_{shutter}_pulse{PULSE}.npz"
        reconstruct_wigner(quadratures, f"Wigner {CHANNEL} {shutter} pulse {PULSE}", save_path=outfile)
    t_reconstruct = time.perf_counter() - t2

    print(
        f"Timing: calibration={t_calib:.2f}s, collect={t_collect:.2f}s, "
        f"reconstruction+plot={t_reconstruct:.2f}s"
    )

    plt.show()


if __name__ == "__main__":
    main()
