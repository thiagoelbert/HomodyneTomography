#!/usr/bin/env python3
"""
Calibrate processed homodyne pulse files by using pulse 4 as vacuum.

Pipeline in plain words
-----------------------
1) For each acquisition group (``base_prefix`` + channel + shutter), read the
   processed vacuum pulse (``*_04.dat``) and measure its mean and std.
2) Shift every pulse so the vacuum mean is 0 and scale it so the vacuum std is
   ``1/sqrt(2)`` (vacuum quadrature variance).
3) Write all calibrated pulses to a sibling folder with suffix ``_calib`` and
   copy ``Acq_list.dat`` unchanged for reproducibility.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from Reconstruction_core.collect_processed import load_acq_list, read_numeric_file

TARGET_STD = 1 / np.sqrt(2)  # vacuum quadrature std
CALIBRATION_PULSE = 4


def write_numeric_file(path: Path, values: np.ndarray):
    """Write one value per line using scientific notation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{v:.6e}" for v in values) + "\n"
    path.write_text(text)


def find_calibration(meta_df, folder: Path) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
    """
    Return mapping ``(base_prefix, channel, shutter) -> (mean, std)`` for pulse 4.
    If a channel lacks a valid vacuum file or has zero variance, it is skipped.
    """
    calib = {}
    for _, meta in meta_df.iterrows():
        base_prefix = meta.base_prefix
        shutter = meta.shutter
        phase = meta.phase_hd
        # Only need one entry per base_prefix; pulse handled via file pattern
        for channel in ("CH1", "CH3"):
            pattern = f"{base_prefix}{channel}-{shutter}_{CALIBRATION_PULSE:02d}.dat"
            path = folder / pattern
            if not path.exists():
                continue
            vals = np.array(read_numeric_file(path))
            std = float(np.std(vals))
            if std == 0.0:
                continue
            mean = float(np.mean(vals))
            key = (base_prefix, channel, shutter)
            calib[key] = (mean, std)
    return calib


def calibrate_folder(input_folder: Path) -> Path:
    meta_df = load_acq_list(input_folder / "Acq_list.dat")
    if meta_df.empty:
        raise RuntimeError(f"No entries found in Acq_list.dat under {input_folder}")

    calib_map = find_calibration(meta_df, input_folder)
    if not calib_map:
        raise RuntimeError("No calibration pulse found (pulse 4 missing or zero std).")

    output_folder = input_folder.with_name(input_folder.name + "_calib")
    output_folder.mkdir(parents=True, exist_ok=True)
    # Copy Acq_list unchanged
    (output_folder / "Acq_list.dat").write_text((input_folder / "Acq_list.dat").read_text())

    for _, meta in meta_df.iterrows():
        base_prefix = meta.base_prefix
        shutter = meta.shutter
        phase = meta.phase_hd
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
