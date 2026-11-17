# Task 4: Dynamic Cooling Effectiveness

This package contains the Task 4 model for Barahat Msheireb: OpenFOAM case generation, device-field surrogate fitting, seasonal exposure, greedy dispatch, economics, multi-objective optimisation and robustness analysis. `data/` contains the saved OSM geometry and June–September 2025 Doha weather; `inputs.csv` records input provenance and assumptions. Model parameters are coded in the modules: editing the register alone does not change them.

## Environment

Use Python 3.12 and install `pip install -r requirements.txt` from this directory. Versions record the original environment; platform-specific dependencies, OpenFOAM and a working graphics backend must also be installed. For publication plots, set `LATIN_MODERN_FONT_DIR` to the directory containing the Latin Modern Roman OTF files; otherwise Matplotlib falls back to an available serif font.

CFD generation uses OpenFOAM 12 tutorials. Set `FOAM_TUTORIALS` to its tutorials directory, or `BLUECFD_ROOT` to the directory containing `OpenFOAM-12/tutorials`. The Windows execution wrapper defaults to `../tools/of.cmd` and may be overridden with `HIMCM_OF`. Configure the wrapper as described in the repository README. Commands below are PowerShell, from this directory. Running case generation or a forced pipeline stage overwrites its corresponding generated outputs.

## Workflow

```powershell
python src/venue.py                         # geometry and occupancy diagnostic
python src/run_venue.py 5                   # 16 venue wind/roof CFD cases
python src/run_doe.py '.' 8 2000            # device CFD cases
python -c "import sys; sys.path.insert(0,'src'); import post; [post.latest_vtk('cfd/devices/off_'+b+'_1.0') for b in 'AED']"
python src/surrogate.py AEDM                # fit device-field surrogates
python src/pipeline.py all --safety-target 0
```

The VTK command exports the three reference meshes required by the surrogate fitter. Examine mesh, residual and stability checks before fitting; `src/check_venue.py` and `src/analyse_checks.py` provide additional checks when their inputs are available.

The pipeline uses 60 representative hours, a population of 80 and 200 generations by default. The reported run used `--safety-target 0` after the 95% target proved unattainable in the investigated layouts; it is not evidence of satisfying a 95% safety requirement. Use `python src/pipeline.py --help` for stages and settings. The full workflow is computationally intensive. Venue and device cases require separate convergence and mesh checks: completion markers alone do not establish either.

## Synthetic software check

This test uses synthetic data, runs all eight pipeline stages and checks six generated figure pairs. It writes only under `work/pipeline_test/`; those outputs are illustrative and are not venue results. Set the font directory to an installed Latin Modern Roman OTF directory before running: the fallback font may cause the figure overlap check to fail.

```powershell
$env:LATIN_MODERN_FONT_DIR = "<path-to-latin-modern-otf-directory>"
python src/pipeline.py test
```

## Included and omitted results

Small summaries in `results/` preserve selected design-day, seasonal and robustness outputs. Large CFD cases, meshes, solution fields, fitted surrogates and intermediate optimisation caches are omitted. Therefore the full saved analysis and plots cannot be reproduced from this package alone: regenerate and independently verify all required CFD fields and downstream caches, or supply the corresponding original artifacts. `figures.py` also requires additional result JSON files, including Pareto and selection outputs. This package does not contain a validation dataset for the venue fields. The installed-device optimisation and scenario robustness results remain conditional on the modelling assumptions and approximations in the code.
