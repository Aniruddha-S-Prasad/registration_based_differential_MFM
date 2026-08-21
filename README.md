# differential_MFM

Differential MFM (Magnetic Force Microscopy) processing utilities for Gwyddion (.gwy) files.

This small Python package provides tools to register two MFM scans (taken with the same tip in two magnetization states), compute differential and summed phase images, and export results as a new Gwyddion file. It includes two registration approaches: a greedy hill-descent search over integer pixel translations and a high-precision phase cross-correlation (PCC) based registration.

Files of interest
- [differential_MFM.py](/home/aniruddha/Projects/differential_MFM.worktrees/add-readme-documentation/differential_MFM.py) — main processing classes: MFMData and DifferentialMFMData.
- [registration.py](/home/aniruddha/Projects/differential_MFM.worktrees/add-readme-documentation/registration.py) — helper functions and registration algorithms (hill descent and cost function utilities).
- [demo.ipynb](/home/aniruddha/Projects/differential_MFM.worktrees/add-readme-documentation/demo.ipynb) — example notebook demonstrating typical usage.
- [data/](/home/aniruddha/Projects/differential_MFM.worktrees/add-readme-documentation/data/) — example Gwyddion files included for quick testing (e.g. `1um_25nm.gwy`).

Requirements
- Python 3.9 or newer
- The following Python packages (installable via pip):
  - gwyfile
  - numpy
  - scipy
  - scikit-image
  - matplotlib
  - matplotlib-scalebar

Install (recommended)

1. Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install gwyfile numpy scipy scikit-image matplotlib matplotlib-scalebar

Quick usage

1. Load two Gwyddion files (initial and after reversing tip magnetization) as MFMData objects, create a DifferentialMFMData object, then inspect or save results.

```python
from differential_MFM import MFMData, DifferentialMFMData

# Paths to two .gwy scans
init_gwy = "data/1um_25nm.gwy"
aftr_gwy = "data/1um_25nm_after.gwy"

# Create MFMData wrappers
init = MFMData(init_gwy)
aftr = MFMData(aftr_gwy)

# Register and compute differential MFM using hill descent (default)
diff = DifferentialMFMData(init, aftr, displacement_guess=(0, 0), method="hill descent")

# Inspect registration result
print(diff.registration_result)
print("RMS residual:", diff.registration_result.get('rms_residual'))

# Save differential results to a new .gwy file
diff.save_gwy("diff_result.gwy")
```

Notes on registration methods
- "hill descent" (default)
  - Greedy integer-pixel search using a custom cost function (mean squared error on overlapping area).
  - Returns a discrete displacement and provides a cost map generator to inspect the cost landscape.
- "phase cross correlation" / "pcc"
  - Uses skimage.registration.phase_cross_correlation for sub-pixel registration, resamples the second image via scipy.ndimage.map_coordinates, and computes differential images on the registered result.

Plotting
- Both MFMData and DifferentialMFMData expose a plot(channel, ax, color_range=None, color_map=None) method. Channel names supported include:
  - For MFMData: "topography" / "topo" / "t", "phase" / "p"
  - For DifferentialMFMData: "topography" / "topo", "phase", "diff_topo" / "dtopo"

Example (plot using matplotlib):

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

diff.plot('topography', axs[0])
diff.plot('phase', axs[1], color_range='equal')
diff.plot('diff_topo', axs[2])

plt.show()
```

Cost map
- DifferentialMFMData.gen_cost_map(del_px=50) computes a cost map of the registration cost function around zero displacement (useful to inspect local minima).
- plot_cost_map(ax, del_px=50, ...) visualizes the cost map and overlays contour lines.

API summary

MFMData(file_path: str)
- Loads a .gwy file and exposes:
  - .topography: np.ndarray — topography channel
  - .phase: np.ndarray — phase channel
  - .x_px_size, .y_px_size — pixel size in meters (from file metadata)
  - .plot(channel, ax, color_range=None, color_map=None)
  - .reset() — reloads from original file path

DifferentialMFMData(init_data: MFMData, aftr_data: MFMData, displacement_guess=(0,0), method='hill descent')
- Performs registration and differential calculations. Exposes:
  - .topography, .diff_topo, .phase, .sum_phase, .phi_1, .phi_2
  - .registration_result — dict with keys depending on method (displacement, path, cost, pcc_error, ...)
  - .rms_residual (available as registration_result['rms_residual'])
  - .save_gwy(filename: str) — writes results to a .gwy file
  - .plot(channel, ax, color_range=None, color_map=None)
  - .gen_cost_map(del_px=50) and .plot_cost_map(ax, del_px=50, ...)

Examples and demos
- The included [demo.ipynb](/home/aniruddha/Projects/differential_MFM.worktrees/add-readme-documentation/demo.ipynb) contains runnable examples showing how to load the sample data in [data/](/home/aniruddha/Projects/differential_MFM.worktrees/add-readme-documentation/data/), run registration, plot results, and save a .gwy output.
