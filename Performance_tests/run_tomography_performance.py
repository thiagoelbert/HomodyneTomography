#!/usr/bin/env python3
"""
Benchmark homodyne tomography over a grid of histogram bin counts and
detection efficiencies.

For each ``(nbins, eta)`` pair this script reconstructs four states:
- open pulse 1
- open pulse 4
- closed pulse 1
- closed pulse 4

It stores:
- runtime per reconstruction
- the reconstructed density matrix for each state
- MLE convergence metadata
- fitted coherent-state alpha for open pulse 4
- single-photon population for closed pulse 1

Outputs are written under ``Performance_tests/output``.
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import qutip as qt
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Reconstruction_core.calibrate_dataset import calibrate_folder
from Reconstruction_core.calibrate_phase import calibrate_phase
from Reconstruction_core.collect_processed import collect
from Reconstruction_core.mle_lvovsky import run_lvovsky_mle


DATA_FOLDER = Path(r"I:\230226\3")
CHANNEL = "CH3"
PULSES = (1, 4)
SHUTTERS = ("open", "closed")
STATE_ORDER = ("open1", "open4", "closed1", "closed4")
STATE_MAP = {
    "open1": ("open", 1, "SPAC"),
    "open4": ("open", 4, "Coherent"),
    "closed1": ("closed", 1, "Single photon"),
    "closed4": ("closed", 4, "Vacuum"),
}

CUTOFF = 20
NBINS_VALUES = (120, 140, 160, None)
DETECTION_EFFICIENCIES = (1.00, 0.80, 0.70, 0.60)
TOL = 1e-7
MAX_ITER = 5000
MIN_PROB = 1e-9
ENABLE_PHASE_CALIBRATION = True

ALPHA_FIT_MAX_ITER = 200
ALPHA_SEARCH_MAX = 5.0 / np.sqrt(2.0)

OUTPUT_DIR = ROOT / "Performance_tests" / "output"
HEARTBEAT_INTERVAL_SECONDS = 30.0
RESUME_PREVIOUS_RUN = True


def build_benchmark_config() -> dict:
    return {
        "data_folder": str(DATA_FOLDER),
        "channel": CHANNEL,
        "pulses": list(PULSES),
        "shutters": list(SHUTTERS),
        "state_order": list(STATE_ORDER),
        "state_map": {key: {"shutter": value[0], "pulse": value[1], "label": value[2]} for key, value in STATE_MAP.items()},
        "cutoff": CUTOFF,
        "nbins_values": [None if value is None else int(value) for value in NBINS_VALUES],
        "detection_efficiencies": [float(value) for value in DETECTION_EFFICIENCIES],
        "tol": TOL,
        "max_iter": MAX_ITER,
        "min_prob": MIN_PROB,
        "enable_phase_calibration": ENABLE_PHASE_CALIBRATION,
        "alpha_fit_max_iter": ALPHA_FIT_MAX_ITER,
        "alpha_search_max": float(ALPHA_SEARCH_MAX),
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "resume_previous_run": RESUME_PREVIOUS_RUN,
        "output_dir": str(OUTPUT_DIR),
    }


def build_quadrature_dict(subset) -> Dict[float, np.ndarray]:
    quadratures: Dict[float, np.ndarray] = {}
    if subset.empty:
        return quadratures

    phases = np.array(sorted(subset["phase_hd"].unique()), dtype=float)
    for ph in phases:
        vals = np.concatenate(subset.loc[subset["phase_hd"] == ph, "values"].to_numpy())
        quadratures[float(ph)] = vals
    return quadratures


def estimate_single_photon_population(rho: np.ndarray) -> float:
    if rho.shape[0] < 2:
        return float("nan")
    pop1 = float(np.real(rho[1, 1]))
    return max(0.0, min(1.0, pop1))


def fit_coherent_alpha(
    rho: np.ndarray,
    alpha_max: float = ALPHA_SEARCH_MAX,
    max_iter: int = ALPHA_FIT_MAX_ITER,
) -> complex:
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


def ensure_calibrated_folder(data_folder: Path) -> Path:
    if data_folder.name.endswith("_phasecalib"):
        return data_folder
    if data_folder.name.endswith("_calib"):
        calib_folder = data_folder
    else:
        existing_calib = data_folder.with_name(data_folder.name + "_calib")
        if existing_calib.exists() and (existing_calib / "Acq_list.dat").exists():
            calib_folder = existing_calib
        else:
            calib_folder = calibrate_folder(data_folder)

    if not ENABLE_PHASE_CALIBRATION:
        return calib_folder

    if calib_folder.name.endswith("_phasecalib"):
        return calib_folder

    existing_phasecalib = calib_folder.with_name(calib_folder.name + "_phasecalib")
    if existing_phasecalib.exists() and (existing_phasecalib / "Acq_list.dat").exists():
        return existing_phasecalib
    return calibrate_phase(calib_folder)


def collect_quadratures(calib_folder: Path) -> Dict[str, Dict[float, np.ndarray]]:
    df = collect(calib_folder, channels=[CHANNEL], pulses=list(PULSES), shutters=list(SHUTTERS))
    quadratures_by_state: Dict[str, Dict[float, np.ndarray]] = {}

    for state_key, (shutter, pulse, _) in STATE_MAP.items():
        subset = df[(df["channel"] == CHANNEL) & (df["shutter"] == shutter) & (df["pulse"] == pulse)]
        quadratures = build_quadrature_dict(subset)
        if not quadratures:
            raise ValueError(f"No data found for {CHANNEL} {shutter} pulse {pulse}.")
        quadratures_by_state[state_key] = quadratures

    return quadratures_by_state


def format_eta(eta: float) -> str:
    return f"{eta:.3f}".replace(".", "p")


def format_nbins_label(nbins: Optional[int]) -> str:
    return "raw" if nbins is None else str(int(nbins))


def format_nbins_dirname(nbins: Optional[int]) -> str:
    return "raw" if nbins is None else f"{int(nbins):04d}"


class Heartbeat:
    """Emit periodic status lines while a long reconstruction is running."""

    def __init__(self, label: str, interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self._start = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Heartbeat":
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        elapsed = time.perf_counter() - self._start
        status = "failed" if exc_type is not None else "finished"
        print(f"[heartbeat] {self.label}: {status} after {elapsed:.1f}s")

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            elapsed = time.perf_counter() - self._start
            print(f"[heartbeat] {self.label}: running for {elapsed:.1f}s")


def run_single_reconstruction(
    quadratures: Dict[float, np.ndarray],
    nbins: Optional[int],
    eta: float,
    state_key: str,
) -> Tuple[np.ndarray, dict, float]:
    label = f"state={state_key} nbins={format_nbins_label(nbins)} eta={eta:.3f}"
    with Heartbeat(label):
        t0 = time.perf_counter()
        rho_hat, info = run_lvovsky_mle(
            quadratures,
            cutoff=CUTOFF,
            eta=eta,
            max_iter=MAX_ITER,
            tol=TOL,
            min_prob=MIN_PROB,
            nbins=nbins,
        )
        elapsed = time.perf_counter() - t0
    return rho_hat, info, elapsed


def save_density_matrix_text(path: Path, rho: np.ndarray) -> None:
    rho_real = np.real(rho)
    rho_imag = np.imag(rho)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# rho complex matrix\n# real part\n")
        np.savetxt(handle, rho_real, fmt="%.6e")
        handle.write("# imag part\n")
        np.savetxt(handle, rho_imag, fmt="%.6e")


def save_run_result(
    run_dir: Path,
    state_key: str,
    rho_hat: np.ndarray,
    info: dict,
    runtime_seconds: float,
    alpha_fit: complex,
    single_photon_population: float,
) -> None:
    np.savez(
        run_dir / f"{state_key}.npz",
        rho=rho_hat,
        runtime_seconds=runtime_seconds,
        iterations=info["iterations"],
        converged=info["converged"],
        delta=info["delta"],
        p_min=info["p_min"],
        p_max=info["p_max"],
        deltas=np.asarray(info["deltas"], dtype=float),
        nbins=info["nbins"],
        eta=info["eta"],
        alpha_fit=np.complex128(alpha_fit),
        single_photon_population=single_photon_population,
        state_key=state_key,
    )
    save_density_matrix_text(run_dir / f"{state_key}.rho.txt", rho_hat)


def write_summary_csv(rows: Iterable[dict], csv_path: Path) -> None:
    fieldnames = [
        "nbins",
        "eta",
        "state",
        "runtime_seconds",
        "iterations",
        "converged",
        "delta",
        "alpha_real",
        "alpha_imag",
        "single_photon_population",
        "result_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_summary_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _same_nbins_grid(saved_values: np.ndarray, current_values: np.ndarray) -> bool:
    saved_labels = [format_nbins_label(None if value is None else int(value)) for value in saved_values]
    current_labels = [format_nbins_label(None if value is None else int(value)) for value in current_values]
    return saved_labels == current_labels


def load_previous_snapshot(
    *,
    output_dir: Path,
    nbins_values: np.ndarray,
    eta_values: np.ndarray,
    states: np.ndarray,
):
    summary_path = output_dir / "benchmark_summary.npz"
    csv_path = output_dir / "benchmark_summary.csv"
    if not RESUME_PREVIOUS_RUN or not summary_path.exists():
        return None

    summary = np.load(summary_path, allow_pickle=True)
    if not _same_nbins_grid(summary["nbins_values"], nbins_values):
        return None
    if not np.allclose(summary["eta_values"].astype(float), eta_values):
        return None
    if [str(state) for state in summary["states"]] != [str(state) for state in states]:
        return None

    return {
        "runtime_seconds": np.array(summary["runtime_seconds"], copy=True),
        "iterations": np.array(summary["iterations"], copy=True),
        "converged": np.array(summary["converged"], copy=True),
        "deltas": np.array(summary["deltas"], copy=True),
        "open4_alpha": np.array(summary["open4_alpha"], copy=True),
        "closed1_pop": np.array(summary["closed1_single_photon_population"], copy=True),
        "result_files": np.array(summary["result_files"], dtype=object, copy=True),
        "csv_rows": load_summary_csv_rows(csv_path),
        "sweep_complete": bool(summary["sweep_complete"]) if "sweep_complete" in summary.files else False,
    }


def has_csv_row(csv_rows: list[dict], nbins_label: str, eta: float, state_key: str) -> bool:
    eta_label = f"{float(eta):.12g}"
    for row in csv_rows:
        if (
            str(row.get("nbins", "")) == nbins_label
            and str(row.get("state", "")) == state_key
            and str(row.get("eta", "")) == eta_label
        ):
            return True
    return False


def build_csv_row_from_saved_result(result_file: Path, nbins_label: str, eta: float, state_key: str) -> dict:
    with np.load(result_file, allow_pickle=False) as data:
        alpha_fit = complex(data["alpha_fit"])
        single_photon_population = float(data["single_photon_population"])
        return {
            "nbins": nbins_label,
            "eta": float(eta),
            "state": state_key,
            "runtime_seconds": float(data["runtime_seconds"]),
            "iterations": int(data["iterations"]),
            "converged": bool(data["converged"]),
            "delta": float(data["delta"]),
            "alpha_real": float(np.real(alpha_fit)) if state_key == "open4" else np.nan,
            "alpha_imag": float(np.imag(alpha_fit)) if state_key == "open4" else np.nan,
            "single_photon_population": single_photon_population if state_key == "closed1" else np.nan,
            "result_file": str(result_file),
        }


def save_summary_snapshot(
    *,
    output_dir: Path,
    nbins_values: np.ndarray,
    eta_values: np.ndarray,
    states: np.ndarray,
    runtime_seconds: np.ndarray,
    iterations: np.ndarray,
    converged: np.ndarray,
    deltas: np.ndarray,
    open4_alpha: np.ndarray,
    closed1_pop: np.ndarray,
    t_calib: float,
    t_collect: float,
    reconstruction_elapsed: float,
    data_folder: Path,
    calib_folder: Path,
    result_files: np.ndarray,
    csv_rows: list[dict],
    sweep_complete: bool,
) -> None:
    benchmark_config = build_benchmark_config()
    tmp_npz = output_dir / "benchmark_summary.tmp.npz"
    final_npz = output_dir / "benchmark_summary.npz"
    np.savez(
        tmp_npz,
        nbins_values=nbins_values,
        eta_values=eta_values,
        states=states,
        runtime_seconds=runtime_seconds,
        iterations=iterations,
        converged=converged,
        deltas=deltas,
        open4_alpha=open4_alpha,
        closed1_single_photon_population=closed1_pop,
        calibration_time_seconds=t_calib,
        collect_time_seconds=t_collect,
        reconstruction_time_seconds=reconstruction_elapsed,
        data_folder=str(data_folder),
        calibrated_folder=str(calib_folder),
        benchmark_config_json=json.dumps(benchmark_config),
        result_files=result_files,
        sweep_complete=sweep_complete,
    )
    tmp_npz.replace(final_npz)

    tmp_csv = output_dir / "benchmark_summary.tmp.csv"
    final_csv = output_dir / "benchmark_summary.csv"
    write_summary_csv(csv_rows, tmp_csv)
    tmp_csv.replace(final_csv)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    calib_folder = ensure_calibrated_folder(DATA_FOLDER)
    t_calib = time.perf_counter() - t0

    t1 = time.perf_counter()
    quadratures_by_state = collect_quadratures(calib_folder)
    t_collect = time.perf_counter() - t1

    nbins_values = np.asarray(NBINS_VALUES, dtype=object)
    eta_values = np.asarray(DETECTION_EFFICIENCIES, dtype=float)
    states = np.asarray(STATE_ORDER)

    shape = (len(nbins_values), len(eta_values), len(states))
    runtime_seconds = np.full(shape, np.nan, dtype=float)
    iterations = np.zeros(shape, dtype=int)
    converged = np.zeros(shape, dtype=bool)
    deltas = np.full(shape, np.nan, dtype=float)
    open4_alpha = np.full((len(nbins_values), len(eta_values)), np.nan + 1j * np.nan, dtype=np.complex128)
    closed1_pop = np.full((len(nbins_values), len(eta_values)), np.nan, dtype=float)
    result_files = np.full(shape, "", dtype=object)
    csv_rows = []

    previous = load_previous_snapshot(
        output_dir=OUTPUT_DIR,
        nbins_values=nbins_values,
        eta_values=eta_values,
        states=states,
    )
    if previous is not None:
        runtime_seconds = previous["runtime_seconds"]
        iterations = previous["iterations"]
        converged = previous["converged"]
        deltas = previous["deltas"]
        open4_alpha = previous["open4_alpha"]
        closed1_pop = previous["closed1_pop"]
        result_files = previous["result_files"]
        csv_rows = previous["csv_rows"]
        print(
            "Resuming from existing snapshot."
            if not previous["sweep_complete"]
            else "Existing snapshot is already complete; finished states will be skipped."
        )

    t2 = time.perf_counter()
    for i, nbins in enumerate(nbins_values):
        for j, eta in enumerate(eta_values):
            nbins_value = None if nbins is None else int(nbins)
            run_dir = OUTPUT_DIR / f"nbins_{format_nbins_dirname(nbins_value)}_eta_{format_eta(float(eta))}"
            run_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                **build_benchmark_config(),
                "data_folder": str(DATA_FOLDER),
                "calibrated_folder": str(calib_folder),
                "nbins": nbins_value,
                "eta": float(eta),
            }
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            print(f"Running benchmark for nbins={format_nbins_label(nbins_value)} eta={float(eta):.3f}")
            for k, state_key in enumerate(states):
                expected_result_file = run_dir / f"{state_key}.npz"
                nbins_label = format_nbins_label(nbins_value)
                if (
                    str(result_files[i, j, k])
                    and Path(str(result_files[i, j, k])).exists()
                    and np.isfinite(runtime_seconds[i, j, k])
                ) or expected_result_file.exists():
                    if not str(result_files[i, j, k]):
                        result_files[i, j, k] = str(expected_result_file)
                    if not has_csv_row(csv_rows, nbins_label, float(eta), str(state_key)):
                        csv_rows.append(
                            build_csv_row_from_saved_result(
                                expected_result_file,
                                nbins_label,
                                float(eta),
                                str(state_key),
                            )
                        )
                    print(
                        f"Skipping completed state={state_key} "
                        f"nbins={nbins_label} eta={float(eta):.3f}"
                    )
                    continue

                rho_hat, info, elapsed = run_single_reconstruction(
                    quadratures_by_state[str(state_key)],
                    nbins=nbins_value,
                    eta=float(eta),
                    state_key=str(state_key),
                )

                alpha_fit = np.nan + 1j * np.nan
                single_photon_population = float("nan")
                if state_key == "open4":
                    alpha_fit = fit_coherent_alpha(rho_hat)
                    open4_alpha[i, j] = alpha_fit
                if state_key == "closed1":
                    single_photon_population = estimate_single_photon_population(rho_hat)
                    closed1_pop[i, j] = single_photon_population

                save_run_result(
                    run_dir=run_dir,
                    state_key=str(state_key),
                    rho_hat=rho_hat,
                    info=info,
                    runtime_seconds=elapsed,
                    alpha_fit=alpha_fit,
                    single_photon_population=single_photon_population,
                )

                result_file = expected_result_file
                runtime_seconds[i, j, k] = elapsed
                iterations[i, j, k] = int(info["iterations"])
                converged[i, j, k] = bool(info["converged"])
                deltas[i, j, k] = float(info["delta"])
                result_files[i, j, k] = str(result_file)

                csv_rows.append(
                    {
                        "nbins": format_nbins_label(nbins_value),
                        "eta": float(eta),
                        "state": str(state_key),
                        "runtime_seconds": elapsed,
                        "iterations": int(info["iterations"]),
                        "converged": bool(info["converged"]),
                        "delta": float(info["delta"]),
                        "alpha_real": float(np.real(alpha_fit)) if state_key == "open4" else np.nan,
                        "alpha_imag": float(np.imag(alpha_fit)) if state_key == "open4" else np.nan,
                        "single_photon_population": single_photon_population if state_key == "closed1" else np.nan,
                        "result_file": str(result_file),
                    }
                )

                save_summary_snapshot(
                    output_dir=OUTPUT_DIR,
                    nbins_values=nbins_values,
                    eta_values=eta_values,
                    states=states,
                    runtime_seconds=runtime_seconds,
                    iterations=iterations,
                    converged=converged,
                    deltas=deltas,
                    open4_alpha=open4_alpha,
                    closed1_pop=closed1_pop,
                    t_calib=t_calib,
                    t_collect=t_collect,
                    reconstruction_elapsed=time.perf_counter() - t2,
                    data_folder=DATA_FOLDER,
                    calib_folder=calib_folder,
                    result_files=result_files,
                    csv_rows=csv_rows,
                    sweep_complete=False,
                )

    t_reconstruct = time.perf_counter() - t2

    save_summary_snapshot(
        output_dir=OUTPUT_DIR,
        nbins_values=nbins_values,
        eta_values=eta_values,
        states=states,
        runtime_seconds=runtime_seconds,
        iterations=iterations,
        converged=converged,
        deltas=deltas,
        open4_alpha=open4_alpha,
        closed1_pop=closed1_pop,
        t_calib=t_calib,
        t_collect=t_collect,
        reconstruction_elapsed=t_reconstruct,
        data_folder=DATA_FOLDER,
        calib_folder=calib_folder,
        result_files=result_files,
        csv_rows=csv_rows,
        sweep_complete=True,
    )

    print(
        f"Saved benchmark results to {OUTPUT_DIR}\n"
        f"Timing: calibration={t_calib:.2f}s collect={t_collect:.2f}s grid_reconstruction={t_reconstruct:.2f}s"
    )


if __name__ == "__main__":
    main()
