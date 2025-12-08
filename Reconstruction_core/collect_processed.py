#!/usr/bin/env python3
"""
Collect processed pulse-area files (CH1/CH3) into a pandas DataFrame.
Metadata (phase, shutter, base name) comes from Acq_list.dat.
Values are the processed pulse areas stored in the *_NN.dat files (one column).
"""

from pathlib import Path
from typing import List, Iterable, Optional
import pandas as pd
import numpy as np

DEFAULT_FOLDER = "thiagoTest"  # set your default data folder here
DEFAULT_SAVE_JSON = None       # e.g., "processed.json"
DEFAULT_SAVE_CSV = None        # e.g., "processed.csv"


def read_numeric_file(path: Path) -> np.ndarray:
    """
    Fast numeric reader using numpy; returns 1D float array.
    """
    try:
        arr = np.fromfile(path, dtype=float, sep=" ")
    except ValueError:
        arr = np.loadtxt(path, dtype=float, ndmin=1)
    if arr.size == 0:
        raise RuntimeError(f"No numeric data found in {path}")
    return arr


def load_acq_list(acq_list_path: Path) -> pd.DataFrame:
    """
    Parse Acq_list.dat into a DataFrame with columns:
    base_prefix, shutter, phase_hd, file_root.
    """
    rows = []
    lines = [ln for ln in acq_list_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        return pd.DataFrame(columns=["base_prefix", "shutter", "phase_hd", "file_root"])
    for line in lines[1:]:  # skip header
        parts = line.split()
        # Expected columns (20): N, SigMean, SigVar, SigAll, DarkM, DarkV, FitEff3, Delay,
        # PhaseHD (idx 8), PhaseMic (idx 9), FitEff1, IntDC, IntDCErr, TrigCount, TrigStd,
        # 3rdPulseM, 3rdPulseVar, FileRoot (idx -2), AcqTime
        if len(parts) < 18:
            continue
        phase_hd = float(parts[8])
        file_root = parts[-2]
        file_root_name = Path(file_root).name  # e.g., Acq_251202_155054-open
        if file_root_name.endswith("-open"):
            shutter = "open"
            base_prefix = file_root_name[:-len("-open")]
        elif file_root_name.endswith("-closed"):
            shutter = "closed"
            base_prefix = file_root_name[:-len("-closed")]
        else:
            shutter = "unknown"
            base_prefix = file_root_name
        rows.append(
            {
                "base_prefix": base_prefix,
                "shutter": shutter,
                "phase_hd": phase_hd,
                "file_root": file_root_name,
            }
        )
    return pd.DataFrame(rows)


def collect(
    folder: Path,
    channels: Optional[Iterable[str]] = None,
    pulses: Optional[Iterable[int]] = None,
    shutters: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per processed file:
    columns: file, channel, shutter, pulse, phase_hd, values (list), mean, var, count.
    """
    acq_list_path = folder / "Acq_list.dat"
    if not acq_list_path.exists():
        raise FileNotFoundError(f"Acq_list.dat not found in {folder}")

    meta_df = load_acq_list(acq_list_path)
    if shutters is not None:
        shutters_set = set(shutters)
        meta_df = meta_df[meta_df["shutter"].isin(shutters_set)]
    if meta_df.empty:
        return pd.DataFrame(
            columns=["file", "channel", "shutter", "pulse", "phase_hd", "values", "mean", "var", "count"]
        )

    channels_set = set(channels) if channels is not None else None
    pulses_set = set(pulses) if pulses is not None else None

    records = []
    for _, meta in meta_df.iterrows():
        for channel in ("CH1", "CH3"):
            if channels_set is not None and channel not in channels_set:
                continue
            pattern = f"{meta.base_prefix}{channel}-{meta.shutter}_*.dat"
            for path in sorted(folder.glob(pattern)):
                pulse_str = path.stem.rsplit("_", 1)[-1]
                try:
                    pulse = int(pulse_str)
                except ValueError:
                    continue
                if pulses_set is not None and pulse not in pulses_set:
                    continue
                vals = read_numeric_file(path)
                records.append(
                    {
                        "file": str(path),
                        "channel": channel,
                        "shutter": meta.shutter,
                        "pulse": pulse,
                        "phase_hd": meta.phase_hd,
                        "values": vals,
                        "mean": float(np.mean(vals)),
                        "var": float(np.var(vals)),
                        "count": int(vals.size),
                    }
                )
    return pd.DataFrame(records)


def main():
    folder = Path(DEFAULT_FOLDER)
    df = collect(folder)

    print(f"Collected {len(df)} files from {folder}")
    print(df[["channel", "shutter", "pulse", "phase_hd"]].value_counts().rename("files").reset_index().head())

    if DEFAULT_SAVE_JSON:
        Path(DEFAULT_SAVE_JSON).write_text(df.to_json(orient="records", indent=2))
        print(f"Saved JSON to {DEFAULT_SAVE_JSON}")
    if DEFAULT_SAVE_CSV:
        df.to_csv(DEFAULT_SAVE_CSV, index=False)
        print(f"Saved CSV to {DEFAULT_SAVE_CSV}")


if __name__ == "__main__":
    main()
