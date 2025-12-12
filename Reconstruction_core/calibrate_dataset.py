#!/usr/bin/env python3
"""
Calibrate processed homodyne pulse files by using pulse 4 as vacuum.

Pipeline in plain words
-----------------------
1) For closed shutters, reuse the pulse-4 mean/variance already stored in
   ``Acq_list.dat`` (DarkM/DarkV columns treated as vacuum); for open shutters,
   reuse the most recent closed entry (open pulse 4 is not vacuum).
2) Shift every pulse so the vacuum mean is 0 and scale it so the vacuum std is
   ``1/sqrt(2)`` (vacuum quadrature variance).
3) Write all calibrated pulses to a sibling folder with suffix ``_calib`` and
   copy ``Acq_list.dat`` unchanged for reproducibility.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Reconstruction_core.collect_processed import load_acq_list, read_numeric_file

TARGET_STD = 1 / np.sqrt(2)  # vacuum quadrature std



def write_numeric_file(path: Path, values: np.ndarray):
    """Write one value per line using scientific notation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{v:.6e}" for v in values) + "\n"
    path.write_text(text)


def find_calibration(meta_df: pd.DataFrame) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
    """
    Return mapping ``(base_prefix, channel, shutter) -> (mean, std)``.

    Calibration rules:
    - Closed shutter entries use DarkM/DarkV from ``Acq_list.dat`` as vacuum stats.
    - Open shutter entries reuse the most recent closed pulse-4 statistics for
      the same channel encountered earlier in ``Acq_list`` (open pulse 4 is not
      vacuum).
    """
    calib: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    last_closed: Dict[str, Tuple[float, float]] = {}
    for _, meta in meta_df.iterrows():
        base_prefix = meta.base_prefix
        shutter = meta.shutter
        dark_mean = getattr(meta, "dark_mean", np.nan)
        dark_var = getattr(meta, "dark_var", np.nan)
        if shutter == "closed":
            if np.isnan(dark_mean) or np.isnan(dark_var) or dark_var <= 0:
                continue
            cal = (float(dark_mean), float(np.sqrt(dark_var)))
            for channel in ("CH1", "CH3"):
                last_closed[channel] = cal
                calib[(base_prefix, channel, shutter)] = cal
        elif shutter == "open":
            for channel in ("CH1", "CH3"):
                if channel in last_closed:
                    calib[(base_prefix, channel, shutter)] = last_closed[channel]
    return calib


def calibrate_folder(input_folder: Path) -> Path:
    meta_df = load_acq_list(input_folder / "Acq_list.dat")
    if meta_df.empty:
        raise RuntimeError(f"No entries found in Acq_list.dat under {input_folder}")
    print(meta_df)

    calib_map = find_calibration(meta_df)
    if not calib_map:
        raise RuntimeError("No calibration stats found in Acq_list.dat (DarkM/DarkV missing or zero).")
    missing_open = []
    for _, meta in meta_df.iterrows():
        if meta.shutter != "open":
            continue
        for channel in ("CH1", "CH3"):
            key = (meta.base_prefix, channel, meta.shutter)
            if key not in calib_map:
                missing_open.append(f"{meta.base_prefix}{channel}-{meta.shutter}")
    if missing_open:
        missing_str = ", ".join(sorted(set(missing_open)))
        raise RuntimeError(
            f"No closed pulse-4 reference found before open shutter entries: {missing_str}"
        )

    output_folder = input_folder.with_name(input_folder.name + "_calib")
    output_folder.mkdir(parents=True, exist_ok=True)
    # Copy Acq_list unchanged
    (output_folder / "Acq_list.dat").write_text((input_folder / "Acq_list.dat").read_text())

    for _, meta in meta_df.iterrows():
        base_prefix = meta.base_prefix
        shutter = meta.shutter
        for channel in ("CH1", "CH3"):
            key = (base_prefix, channel, shutter)
            if key not in calib_map:
                continue
            mean, std = calib_map[key]
            scale = TARGET_STD / std
            pattern = f"{base_prefix}{channel}-{shutter}_*.dat"
            for path in sorted(input_folder.glob(pattern)):
                vals = np.array(read_numeric_file(path))
                calibrated = (vals - mean) * scale
                out_path = output_folder / path.name
                write_numeric_file(out_path, calibrated)
    print(f"Calibrated data written to {output_folder}")
    return output_folder
