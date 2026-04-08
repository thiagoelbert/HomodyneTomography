#!/usr/bin/env python3
"""
Calibrate homodyne phases by fitting a sinusoid to open-shutter signal means.

This module writes a phase-calibrated copy of the dataset:
- Copies all files from input folder to a sibling output folder.
- Uses open-shutter rows and their ``SigMean`` values (column 2) as the fit data.
- Fits ``y = offset + amp * sin(omega * idx + phi0)`` over acquisition index.
- Writes fitted phase into ``PhaseHD`` of output ``Acq_list.dat``.
"""
from pathlib import Path
from typing import List, Optional
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PULSE_FOR_PHASE = 4
PHASE_CHANNEL = "CH3"
DATA_FOLDER = Path(r"I:\230226\3 - Copia")
SHOW_PLOT = True


def _parse_acq_line(line: str):
    parts = line.split()
    if len(parts) < 18:
        return None
    file_root = Path(parts[-2]).name
    shutter = "open" if file_root.endswith("-open") else "closed" if file_root.endswith("-closed") else "unknown"
    if shutter == "open":
        base_prefix = file_root[: -len("-open")]
    elif shutter == "closed":
        base_prefix = file_root[: -len("-closed")]
    else:
        base_prefix = file_root
    return {"parts": parts, "shutter": shutter, "base_prefix": base_prefix}


def _read_numeric_file(path: Path) -> np.ndarray:
    try:
        arr = np.fromfile(path, dtype=float, sep=" ")
    except ValueError:
        arr = np.loadtxt(path, dtype=float, ndmin=1)
    if arr.size == 0:
        raise RuntimeError(f"No numeric data found in {path}")
    return arr


def _pulse_mean_for_row(folder: Path, row: dict, channel: str, pulse: int) -> Optional[float]:
    if row["shutter"] != "open":
        return None
    path = folder / f"{row['base_prefix']}{channel}-open_{pulse:02d}.dat"
    if not path.exists():
        return None
    vals = _read_numeric_file(path)
    return float(np.mean(vals))


def _cos_model(x: np.ndarray, offset: float, amp: float, omega: float, phi0: float) -> np.ndarray:
    return offset + amp * np.cos(omega * x + phi0)


def _initial_guess_from_frequency_scan(open_means: np.ndarray):
    """
    Coarse global search on omega. For each candidate omega, solve
    y ~ c0 + a*cos(omega*k) + b*sin(omega*k) by least squares and pick the
    minimum SSE candidate. Returns (offset, amp, omega, phi0).
    """
    n = open_means.size
    k = np.arange(n, dtype=float)
    if n < 4:
        raise RuntimeError("Need at least 4 open-shutter points to fit a sinusoid.")

    # Reasonable omega range for sampled acquisitions.
    omega_min = 2.0 * np.pi / max(4.0 * n, 8.0)
    omega_max = np.pi
    omegas = np.linspace(omega_min, omega_max, 2000)

    best = None
    y = open_means.astype(float)
    for omega in omegas:
        c = np.cos(omega * k)
        s = np.sin(omega * k)
        X = np.column_stack([np.ones_like(k), c, s])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coeffs
        sse = float(np.dot(resid, resid))
        if (best is None) or (sse < best[0]):
            best = (sse, float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(omega))

    _, offset, a_cos, b_sin, omega0 = best  # type: ignore[misc]
    amp0 = float(np.hypot(a_cos, b_sin))
    phi0 = float(np.arctan2(-b_sin, a_cos))
    return offset, amp0, omega0, phi0


def _fit_phase_from_open_rows(open_means: np.ndarray):
    if open_means.size < 4:
        raise RuntimeError("Need at least 4 open-shutter points to fit a sinusoid.")

    offset0, amp0, omega0, phi00 = _initial_guess_from_frequency_scan(open_means)
    p0 = np.array([offset0, amp0, omega0, phi00], dtype=float)
    lower = np.array([-np.inf, -np.inf, 1e-8, -20.0 * np.pi], dtype=float)
    upper = np.array([np.inf, np.inf, np.pi, 20.0 * np.pi], dtype=float)

    popt, _ = curve_fit(
        _cos_model,
        np.arange(open_means.size, dtype=float),
        open_means,
        p0=p0,
        bounds=(lower, upper),
        maxfev=100000,
    )
    offset, amp, omega, phi0 = [float(v) for v in popt]
    return offset, amp, omega, phi0


def _plot_open_mean_vs_phase(
    acq_path: Path,
    open_phase: np.ndarray,
    open_means: np.ndarray,
    offset: float,
    amp: float,
    omega: float,
    phi0: float,
) -> None:
    order = np.argsort(open_phase)
    phase_sorted = open_phase[order]
    means_sorted = open_means[order]

    phase_grid = np.linspace(0.0, 2.0 * np.pi, 400)
    fit_curve = offset + amp * np.cos(phase_grid)

    plt.figure(figsize=(7, 4), constrained_layout=True)
    plt.scatter(phase_sorted, means_sorted, s=20, label="Open-shutter means")
    plt.plot(phase_grid, fit_curve, "r-", lw=2, label="Fitted sinusoid")
    plt.xlabel("Phase (rad)")
    plt.ylabel("SigMean")
    plt.title(f"Phase calibration fit: {acq_path.parent.name}")
    plt.xlim(0.0, 2.0 * np.pi)
    plt.legend()
    plt.grid(alpha=0.25)


def calibrate_phase(folder: Path) -> Path:
    """
    Fit phase from per-acquisition open-shutter pulse means (pulse 4 by default)
    and write a sibling output folder with calibrated ``PhaseHD`` (column index 8).
    """
    input_acq_path = folder / "Acq_list.dat"
    if not input_acq_path.exists():
        raise FileNotFoundError(f"Acq_list.dat not found in {folder}")

    lines = input_acq_path.read_text().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"Acq_list.dat has no data rows in {folder}")

    header = lines[0]
    raw_rows = lines[1:]
    row_items: List[Optional[dict]] = []
    for line in raw_rows:
        if not line.strip():
            row_items.append(None)
            continue
        item = _parse_acq_line(line)
        row_items.append(item)

    parsed_rows = [row for row in row_items if row is not None]
    if not parsed_rows:
        raise RuntimeError("No valid acquisition rows found in Acq_list.dat.")

    open_idx = []
    open_means = []
    for idx, row in enumerate(row_items):
        if row is None:
            continue
        pulse_mean = _pulse_mean_for_row(folder, row, channel=PHASE_CHANNEL, pulse=PULSE_FOR_PHASE)
        if pulse_mean is not None and np.isfinite(pulse_mean):
            open_idx.append(idx)
            open_means.append(pulse_mean)

    if len(open_idx) < 4:
        raise RuntimeError(
            f"Not enough open-shutter rows with valid {PHASE_CHANNEL} pulse {PULSE_FOR_PHASE} means."
        )

    open_means_fit = np.array(open_means, dtype=float)
    offset, amp, omega, phi0 = _fit_phase_from_open_rows(open_means_fit)

    # Fit phase over open-shot order (0, 1, 2, ...), not raw Acq_list row index.
    open_order = np.arange(len(open_idx), dtype=float)
    open_phase = np.mod(omega * open_order + phi0, 2.0 * np.pi)
    _plot_open_mean_vs_phase(input_acq_path, open_phase, open_means_fit, offset, amp, omega, phi0)

    open_idx_arr = np.array(open_idx, dtype=int)
    open_phase_by_idx = {
        int(idx): float(np.mod(omega * float(order) + phi0, 2.0 * np.pi))
        for order, idx in enumerate(open_idx_arr)
    }

    def _phase_for_row(idx: int, row: dict) -> float:
        if row["shutter"] == "open":
            return open_phase_by_idx[idx]
        if row["shutter"] == "closed":
            pos = int(np.searchsorted(open_idx_arr, idx + 1, side="left"))
            if pos >= open_idx_arr.size:
                raise RuntimeError(
                    f"Closed-shutter acquisition at row {idx + 2} has no next open-shutter row."
                )
            next_open_idx = int(open_idx_arr[pos])
            return open_phase_by_idx[next_open_idx]
        return float(np.mod(omega * float(idx) + phi0, 2.0 * np.pi))

    new_lines = [header]
    for idx, line in enumerate(raw_rows):
        row = row_items[idx]
        if row is None:
            new_lines.append(line)
            continue
        phase = _phase_for_row(idx, row)
        parts = row["parts"]
        parts[8] = f"{phase:.8f}"
        new_lines.append("\t".join(parts))

    output_folder = folder.with_name(folder.name + "_phasecalib")
    shutil.copytree(folder, output_folder, dirs_exist_ok=True)
    output_acq_path = output_folder / "Acq_list.dat"
    output_acq_path.write_text("\n".join(new_lines) + "\n")

    print(
        f"Phase calibration complete: {output_acq_path}. "
        f"Fitted omega={omega:.6f} rad/sample, phi0={phi0:.6f} rad."
    )
    return output_folder


def main() -> None:
    calibrate_phase(DATA_FOLDER)
    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
