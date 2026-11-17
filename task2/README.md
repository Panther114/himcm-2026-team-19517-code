# Task 2: thermal-comfort elasticity regression

This package estimates the apparent-temperature ideal and thermal-discomfort elasticity used in the Task 2 model. It reads thermal-comfort votes, groups them by building type and metabolic rate, fits a utility-shortfall elasticity for each stratum, and regresses those elasticities on activity and estimated occupant density. The included density profile, standards tables, and citation registry supply and document the density assumptions.

## Input and execution

Python 3.9 or later is required; all imports are from the standard library. Place the cleaned ASHRAE Global Thermal Comfort Database II file at `data/ashrae_clean.csv`. The exact cleaned file is excluded from this package because of its size; no public download for that exact cleaned version is provided. Its SHA-256 is `0a12d133cef92f25ac753454ea92e6e26a190553294356a0d6b7291cb61b1ec6`. The raw database alone is not a drop-in replacement: the cleaning procedure is not included. The cleaned file must contain `building_id`, `building_type`, `ta`, `rh`, `vel`, `met`, `dissat_ta`, and `dissat_tsv`. Respect the source data's terms when obtaining or redistributing it.

From this directory run:

```sh
python main.py --out out
python main.py --cov cluster --out out_cluster
python main.py --help
```

The first command runs the original defaults and writes generated tables, diagnostics, citations, and `summary.json` to `out/`; existing output files of the same names are overwritten. `--out` accepts an absolute path or a path relative to this directory. The second command requests stratum-clustered covariance estimates. The help text lists alternative binning, search, and shortfall settings.

## Scope and limitations

This is the elasticity regression, not executable implementations of the bus-shelter or rail-platform cooling models. Their code is not included in the source archive. Results require the exact cleaned survey input; the supplied small input files alone cannot reproduce the fit. The code estimates one elasticity per stratum and repeats it across survey rows, so the row count overstates independent information. Density is a building-type proxy, not a measured density for each vote. The regression is descriptive and sensitive to its shortfall transformation; use cluster covariance and interpret extrapolation cautiously. The encoded standards values and the California density unit convention are documented in `src/standards.py` and the generated provenance table.
