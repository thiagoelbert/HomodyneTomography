#!/usr/bin/env python3
"""
Quick-look histograms for processed pulse files in a folder.

Edit the constants below to point at a dataset folder (must contain
``Acq_list.dat`` and processed ``*.dat`` files). The script filters files using
the same criteria as ``run_tomography.py`` (channel, pulses, shutter states),
then plots a histogram for each matching file and annotates it with the phase
from ``Acq_list.dat``.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Reconstruction_core.collect_processed import collect, load_acq_list  # noqa: E402

# Folder containing Acq_list.dat and processed files
DATA_FOLDER = Path(r"I:\290126\1")
# Same filter knobs as run_tomography.py
CHANNEL = "CH3"
PULSES = [4]
SHUTTERS = ["closed"]
# Histogram bins
BINS = 50


def infer_base_prefix(file_path: str, channel: str, shutter: str) -> str:
    stem = Path(file_path).stem
    marker = f"{channel}-{shutter}_"
    if marker in stem:
        return stem.split(marker)[0]
    if channel in stem:
        return stem.split(channel)[0]
    return stem


def build_sequence_mapping(acq_list_path: Path, shutter: str):
    meta_df = load_acq_list(acq_list_path)
    meta_df = meta_df[meta_df["shutter"] == shutter]
    if meta_df.empty:
        return [], {}, 0

    phases_order = []
    for phase in meta_df["phase_hd"]:
        if phase not in phases_order:
            phases_order.append(phase)
    n_phases = len(phases_order)

    base_prefixes = meta_df["base_prefix"].tolist()
    mapping = {}
    for idx, base in enumerate(base_prefixes):
        phase_index = idx % n_phases
        seq_index = idx // n_phases
        mapping[base] = {
            "phase_hd": phases_order[phase_index],
            "phase_index": phase_index,
            "sequence_index": seq_index,
        }

    n_sequences = (len(base_prefixes) + n_phases - 1) // n_phases
    return phases_order, mapping, n_sequences


def main():
    shutters = [SHUTTERS] if isinstance(SHUTTERS, str) else list(SHUTTERS)
    pulses = list(PULSES)
    df = collect(DATA_FOLDER, channels=[CHANNEL], pulses=pulses, shutters=shutters)
    if df.empty:
        print(
            f"No matching files in {DATA_FOLDER} "
            f"for {CHANNEL}, pulses={pulses}, shutters={shutters}"
        )
        return

    df["base_prefix"] = df.apply(
        lambda row: infer_base_prefix(row["file"], row["channel"], row["shutter"]),
        axis=1,
    )

    acq_list_path = DATA_FOLDER / "Acq_list.dat"
    for pulse in PULSES:
        for shutter in SHUTTERS:
            subset = df[(df["pulse"] == pulse) & (df["shutter"] == shutter)]
            if subset.empty:
                print(f"No data for {CHANNEL} {shutter} pulse {pulse}")
                continue

            phases_order, mapping, n_sequences = build_sequence_mapping(acq_list_path, shutter)
            if not phases_order or n_sequences == 0:
                print(f"No Acq_list entries for shutter={shutter}")
                continue

            n_phases = len(phases_order)
            fig, axes = plt.subplots(
                nrows=n_phases,
                ncols=n_sequences,
                figsize=(3.2 * n_sequences, 2.6 * n_phases),
                constrained_layout=False,
                sharex=True,
                sharey=True,
            )
            axes = np.array(axes).reshape(n_phases, n_sequences)

            all_vals = [row["values"] for _, row in subset.iterrows()]
            x_min = min(float(vals.min()) for vals in all_vals)
            x_max = max(float(vals.max()) for vals in all_vals)

            used = set()
            for _, row in subset.iterrows():
                base_prefix = row["base_prefix"]
                info = mapping.get(base_prefix)
                if info is None:
                    continue
                phase_index = info["phase_index"]
                seq_index = info["sequence_index"]
                ax = axes[phase_index, seq_index]

                ax.hist(row["values"], bins=BINS, alpha=0.85)
                ax.set_xlim(x_min, x_max)
                if phase_index == 0:
                    ax.set_title(f"Seq {seq_index + 1}")
                if seq_index == 0:
                    ax.set_ylabel(f"phase={info['phase_hd']:.3f}")
                if phase_index == n_phases - 1:
                    ax.set_xlabel("Value")
                used.add((phase_index, seq_index))

            for r in range(n_phases):
                for c in range(n_sequences):
                    if (r, c) not in used:
                        axes[r, c].axis("off")

            fig.suptitle(f"{CHANNEL} {shutter} pulse {pulse}", y=0.995, fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))

    plt.show()


if __name__ == "__main__":
    main()
