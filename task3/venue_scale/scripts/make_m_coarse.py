"""Coarse-mesh mist transient: fewer, more diffusive cells, stable to 20 s."""
import os
import sys
import shutil
import re

SRC = os.environ.get("HIMCM_ENGINE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "task4", "src")))
sys.path.insert(0, SRC)
import cfd_gen
import devices

BASE = os.environ.get("HIMCM_TASK3_CASES", os.path.join(os.path.dirname(__file__), "..", "..", "cfd", "devices"))
case = os.path.join(BASE, "M_coarse")

spec = devices.spec("M", w_b=2.70, Ta=42.0, RH=10.0, setting=1.0, on=True, endTime=20, fine=0.6)
info = cfd_gen.make(case, spec)
print("cells spec fine=0.6, blocks", info["blocks"])

for name in os.listdir(case):
    p = os.path.join(case, name)
    if os.path.isdir(p) and name not in ("0", "constant", "system"):
        shutil.rmtree(p)
cp = os.path.join(case, "constant", "cloudProperties")
if os.path.exists(cp):
    os.remove(cp)
open(os.path.join(case, "constant", "fvModels"), "w", newline="\n").write(
    "FoamFile\n{\n    format ascii;\n    class dictionary;\n    object fvModels;\n}\n")


def set_internal(path, value):
    t = open(path, encoding="utf-8").read()
    t = re.sub(r"internalField\s+uniform\s+[^;]+;", f"internalField   uniform {value};", t, count=1)
    open(path, "w", encoding="utf-8", newline="\n").write(t)


set_internal(os.path.join(case, "0", "U"), "(0 0 0)")
set_internal(os.path.join(case, "0", "T"), "315.15")
set_internal(os.path.join(case, "0", "H2O"), "0.0050535177")
set_internal(os.path.join(case, "0", "k"), "0.001")
set_internal(os.path.join(case, "0", "epsilon"), "0.001")
set_internal(os.path.join(case, "0", "nut"), "0.0001")

t = open(os.path.join(case, "0", "T"), encoding="utf-8").read()
t = t.replace("type fixedValue; value uniform 315.15;", "type fixedValue; value uniform 292.85;")
open(os.path.join(case, "0", "T"), "w", encoding="utf-8", newline="\n").write(t)
h = open(os.path.join(case, "0", "H2O"), encoding="utf-8").read()
h = h.replace("type fixedValue; value uniform 0.0050535177;", "type fixedValue; value uniform 0.01526;")
open(os.path.join(case, "0", "H2O"), "w", encoding="utf-8", newline="\n").write(h)

ctl = '''FoamFile
{
    format ascii; class dictionary; location "system"; object controlDict;
}
application foamRun;
solver multicomponentFluid;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 20;
deltaT 0.1;
adjustTimeStep yes;
maxCo 5;
maxDeltaT 0.25;
writeControl runTime;
writeInterval 1;
purgeWrite 0;
writeFormat ascii;
writePrecision 7;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
libs ("libatmosphericModels.so");
'''
open(os.path.join(case, "system", "controlDict"), "w", newline="\n").write(ctl)

f = os.path.join(case, "system", "fvSchemes")
t = open(f, encoding="utf-8").read().replace("default         steadyState;", "default         Euler;")
open(f, "w", encoding="utf-8", newline="\n").write(t)

FIN = ["p_rgh", "U", "h", "T", "k", "epsilon", "omega", "air", "H2O", "Yi"]
block = "    rho\n    {\n        solver diagonal;\n    }\n    rhoFinal\n    {\n        solver diagonal;\n    }\n"
for n in FIN:
    block += f"    {n}Final\n    {{\n        solver smoothSolver;\n        smoother GaussSeidel;\n        tolerance 1e-6;\n        relTol 0;\n        maxIter 20;\n    }}\n"
block += "    pFinal\n    {\n        solver GAMG;\n        smoother DICGaussSeidel;\n        tolerance 1e-6;\n        relTol 0;\n        maxIter 50;\n    }\n"
f = os.path.join(case, "system", "fvSolution")
t = open(f, encoding="utf-8").read()
t = t.replace("PIMPLE\n{\n    nCorrectors     1;\n    nNonOrthogonalCorrectors 0;\n}",
              "PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors     2;\n    nNonOrthogonalCorrectors 0;\n}")
anchor = '''    "(U|Yi|h|k|epsilon|omega).*"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       0;
        relTol          0.1;
        maxIter         20;
    }
}'''
t = t.replace(anchor, anchor[:-1] + "\n" + block + "}")
open(f, "w", encoding="utf-8", newline="\n").write(t)

open(os.path.join(case, "constant", "fvConstraints"), "w", newline="\n").write('''FoamFile
{
    format ascii; class dictionary; object fvConstraints;
}
limitT { type limitTemperature; libs ("libfvConstraints.so"); min 250; max 350; selectionMode all; }
limitK { type limit; libs ("libfvConstraints.so"); field k; min 1e-10; max 50; selectionMode all; }
limitEps { type limit; libs ("libfvConstraints.so"); field epsilon; min 1e-10; max 500; selectionMode all; }
''')
print("made", case)
