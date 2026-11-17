# HiMCM 2026 Team 19517: Cooling Efficiency and Effectiveness Code

Essential computational code for Problem B, organised by task. Each task README describes its inputs, execution order and limitations.

| Directory | Scope |
| --- | --- |
| [task2](task2/README.md) | Thermal-comfort utility and elasticity regression. |
| [task3](task3/README.md) | Matched cooling-device CFD, venue integration and the associated sensitivity and startup calculations. |
| [task4](task4/README.md) | Seasonal cooling effectiveness, device dispatch, multi-objective optimisation and scenario robustness. |
| `tools` | Configurable Windows launcher for OpenFOAM 12. |

## Environment

Use Python 3.12. Task 2 uses only the Python standard library. For Tasks 3 and 4, create a virtual environment and install the numerical dependencies below. Run each task's commands from its own directory. Task 3 also uses the Task 4 numerical engine included in this repository.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r task4/requirements.txt
```

CFD generation and execution require a separate installation of OpenFOAM 12 through blueCFD-Core on Windows. Set `BLUECFD_ROOT` to the installation directory containing `setvars_OF12.bat`. Set `FOAM_TUTORIALS` to the OpenFOAM 12 tutorials directory if it cannot be located beneath that installation. `HIMCM_OF` can select an alternative compatible Windows command wrapper.

```powershell
$env:BLUECFD_ROOT = 'C:\path\to\blueCFD-Core'
$env:FOAM_TUTORIALS = "$env:BLUECFD_ROOT\OpenFOAM-12\tutorials"
```

## Included material and reproducibility

The package retains executable source, small model inputs and selected compact saved results. Full CFD fields, solver installations, fitted binary models, large datasets and generated figures are excluded. The Task 2 regression dataset must be obtained separately; Task 3 and Task 4 describe the simulation products that must be regenerated for a complete calculation. Bundled results are saved model outputs, not independent field measurements.

The scientific equations and numerical coefficients are retained from the source implementations. Packaging changes configure paths, remove machine-specific assumptions and omit obsolete workflows. The task documentation distinguishes calculated evidence from modelling assumptions and identifies limits relevant to interpreting the paper.

Validation of the packaged code included six Task 3 physical-invariant tests, reproduction of the archived Task 3 sensitivity calculation, and the Task 4 synthetic pipeline test across all eight stages. Python syntax, imports, input geometry and weather records were also checked. These checks do not replace full CFD regeneration or a rerun of the Task 2 regression with its excluded survey data.
