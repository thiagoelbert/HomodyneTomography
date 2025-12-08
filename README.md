# Homodyne Tomography Pipeline 

Source code for quantum state tomography from homodyne measurements.

## 1) Install Python (Windows)

1. Download Python 3.11 or newer from https://www.python.org/downloads/
2. Run the installer **with "Add Python to PATH" checked**.
3. Confirm install: open PowerShell and run:
   ```pwsh
   python --version
   ```
   You should see `Python 3.x.x`.

## 2) Get this repository

- If you already have the folder, skip.
- Otherwise, download the source from GitHub: https://github.com/thiagoelbert/HomodyneTomography (Code → Download ZIP) and unzip it.
  
You should see files like `run_tomography.py`, the `Reconstruction_core` folder, and `Data visualization`.

## 3) Create a virtual environment (recommended)

In the repo folder:

```pwsh
python -m venv .venv
.\.venv\Scripts\activate
```

You should see `(.venv)` in your prompt. To exit later: `deactivate`.

## 4) Install required packages

With the virtual environment activated:

```pwsh
pip install numpy scipy matplotlib pandas qutip
```

These cover calibration, data handling, plotting, and Wigner reconstruction.

## 5) Prepare your data

You need a raw processed data folder that contains:
- `Acq_list.dat`
- Processed pulse files named like `<base>CH1-open_01.dat`, `<base>CH3-closed_04.dat`, etc.
- Pulse 4 must be vacuum (used for calibration).

If you already have a calibrated folder ending with `_calib`, you can point to it directly.

## 6) Configure and run the pipeline

Open `run_tomography.py` and adjust the constants near the top:

- `DATA_FOLDER`: full path to your raw data folder (or `_calib` folder).
- `CHANNEL`: `CH1` or `CH3`.
- `PULSE`: pulse number to reconstruct (e.g., `1`).
- `SHUTTERS`: which shutter states to process, e.g., `(\"open\", \"closed\")`.
- `CUTOFF`, `NBINS_LVOVSKY`, `TOL`, `MAX_ITER`, `MIN_PROB`: reconstruction settings (defaults work to start).
- `WIGNER_POINTS`, `WIGNER_XMAX`: grid resolution/range for the Wigner plot.
- `OUTPUT_DIR`: where results are saved (default `TomoOutput`).

Run the script (inside the venv):

```pwsh
python run_tomography.py
```

What happens:
1. If `DATA_FOLDER` is raw, it creates a sibling `_calib` folder with mean-shifted, variance-scaled pulses.
2. It collects quadratures per phase and channel.
3. It runs Lvovsky MLE to estimate the density matrix.
4. It saves `wigner_<channel>_<shutter>_pulse<pulse>.npz` and a human-readable `<...>.rho.txt` into `OUTPUT_DIR`.
5. It shows a 3D Wigner surface plot.

## 7) Visualize results

- **1D histogram of a pulse**: edit `file_path` inside `Data visualization/plot_histogram.py`, then run:
  ```pwsh
  python "Data visualization/plot_histogram.py"
  ```
- **Wigner slice viewer**: set `TARGET_FILE` and `SLICE_PHASE` in `Data visualization/plot_wigner_slices.py`, then run:
  ```pwsh
  python "Data visualization/plot_wigner_slices.py"
  ```

## 8) Common issues

- **Module not found (qutip, numpy, etc.)**: ensure the venv is activated and `pip install ...` completed without errors.
- **File not found**: double-check `DATA_FOLDER` points to the folder containing `Acq_list.dat`.
- **No calibration pulse**: the calibrator needs pulse 4 with nonzero variance; ensure those files exist.

## 9) Updating dependencies later

Inside the venv:

```pwsh
pip install --upgrade numpy scipy matplotlib pandas qutip
```

## 10) Deactivate the virtual environment

```pwsh
deactivate
```
