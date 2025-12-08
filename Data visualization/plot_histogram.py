#!/usr/bin/env python3
"""
Quick-look histogram for a single processed pulse file (e.g., CH3-open_01.dat).

Edit ``file_path`` below to point at any calibrated/processed ``*.dat`` file.
The script reads all numeric values (robust to commas/spaces) and displays a
simple histogram so you can spot outliers or check calibration visually.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def read_numeric_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    vals = []
    for token in text.replace(",", " ").split():
        try:
            vals.append(float(token))
        except ValueError:
            continue
    if not vals:
        raise RuntimeError(f"No numeric data found in {path}")
    return vals


def main():
    # Set your target file here
    file_path = Path(r"Data03121_calib\Acq_251203_152035CH3-closed_01.dat")

    vals = read_numeric_file(file_path)
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=50, alpha=0.85)
    plt.title(file_path.name)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    main()
