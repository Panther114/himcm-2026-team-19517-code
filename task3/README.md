# Task 3: venue applications

The analysis compares chilled-air cooling at Bayfront and misting at the Medina courtyard. `inputs/` contains scenario and geometry inputs. `src/` contains the matched on/off device-case generation, extraction, and venue analysis. The scripts reuse `../task4/src`; set `HIMCM_ENGINE` if it is located elsewhere. OpenFOAM 12 and the launcher described in the parent instructions are required to run CFD. Python dependencies include numpy, scipy, pandas, shapely 2, pvlib, pyvista, and matplotlib.

Run the physical-invariant checks with `python -m unittest discover -s tests -v`.

`results/venue_maps.npz` and `venue_results.json` are saved outputs of the venue analysis. Run the sensitivity calculation using:

```
python venue_scale/scripts/make_sensitivity.py
```

Set `HIMCM_TASK3_OUTPUT` to choose its output directory. The calculation changes each input while holding the saved CFD cooling field fixed. The script reports the calculation, including slopes and reference efficiencies.

`venue_scale/scripts/make_figures.py` creates the maps and kernel figures from the saved results, scenario inputs and matched CFD kernel files in `results/kernels/`. Generate the cases with `src/run_cases.py`, sample them with `src/extract.py` and combine the kernels with `src/analyse.py`. `src/check_results.py` produces the `results/checks.json` input used by `src/figures.py`. Case and kernel files are stored separately from this source package. Generator hashes and input fingerprints identify matching simulation products.

`venue_scale/scripts/make_startup_scaled.py` calculates start-up curves by scaling steady efficiencies with transient device cooling fractions. `venue_scale/figures/efficiency_startup.json` contains the saved plotted points. Set `HIMCM_TASK3_CASES` to the directory containing the required steady off cases and transient cases. The script reads `D_fast` and `M_coarse`, prepared by `make_fast_transient.py` and `make_m_coarse.py`. The builders replace their named destination cases when run; `make_fast_transient.py` requires the corresponding steady cases.
