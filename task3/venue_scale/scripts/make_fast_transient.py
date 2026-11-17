"""Fast, low-accuracy transient start-up cases (single phase, coarse time step)."""
import os
import shutil

BASE = os.environ.get("HIMCM_TASK3_CASES", os.path.join(os.path.dirname(__file__), "..", "..", "cfd", "devices"))
JOBS = [("D_bf_on", "D_fast", False), ("M_md_on", "M_fast", True)]
END = 12.0
DT = 0.1
WRITE = 1.0

FINALS = ["p_rgh", "U", "h", "T", "k", "epsilon", "omega", "air", "H2O", "Yi"]


def final_block():
    out = ["    rho\n    {\n        solver          diagonal;\n    }\n",
           "    rhoFinal\n    {\n        solver          diagonal;\n    }\n"]
    for name in FINALS:
        out.append(f"    {name}Final\n    {{\n        solver smoothSolver;\n        smoother GaussSeidel;\n        tolerance 1e-6;\n        relTol 0;\n        maxIter 20;\n    }}\n")
    out.append("    pFinal\n    {\n        solver GAMG;\n        smoother DICGaussSeidel;\n        tolerance 1e-6;\n        relTol 0;\n        maxIter 50;\n    }\n")
    return "\n".join(out)


for src, dst, mist in JOBS:
    s = os.path.join(BASE, src); d = os.path.join(BASE, dst)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(s, d)
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if os.path.isdir(p) and name not in ("0", "constant", "system"):
            shutil.rmtree(p)
    if os.path.exists(os.path.join(d, "postProcessing")):
        shutil.rmtree(os.path.join(d, "postProcessing"))

    # single phase everywhere: no droplets, no cloud model
    cp = os.path.join(d, "constant", "cloudProperties")
    if os.path.exists(cp):
        os.remove(cp)
    open(os.path.join(d, "constant", "fvModels"), "w", newline="\n").write(
        'FoamFile\n{\n    format ascii;\n    class dictionary;\n    object fvModels;\n}\n')

    if mist:
        # evaporative outlet: saturated air at the wet-bulb temperature (19.7 C)
        tfile = os.path.join(d, "0", "T"); txt = open(tfile, encoding="utf-8").read()
        txt = txt.replace("type fixedValue; value uniform 315.15;", "type fixedValue; value uniform 292.85;")
        open(tfile, "w", encoding="utf-8", newline="\n").write(txt)
        yfile = os.path.join(d, "0", "H2O"); txt = open(yfile, encoding="utf-8").read()
        txt = txt.replace("type fixedValue; value uniform 0.0050535177;", "type fixedValue; value uniform 0.01526;")
        open(yfile, "w", encoding="utf-8", newline="\n").write(txt)

    ctl = f"""FoamFile
{{
    format ascii; class dictionary; location "system"; object controlDict;
}}
application foamRun;
solver multicomponentFluid;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {END};
deltaT {DT};
adjustTimeStep yes;
maxCo 5;
maxDeltaT 0.25;
writeControl runTime;
writeInterval {WRITE};
purgeWrite 0;
writeFormat ascii;
writePrecision 7;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
libs ("libatmosphericModels.so");
"""
    open(os.path.join(d, "system", "controlDict"), "w", newline="\n").write(ctl)

    f = os.path.join(d, "system", "fvSchemes"); t = open(f, encoding="utf-8").read()
    t = t.replace("default         steadyState;", "default         Euler;")
    open(f, "w", encoding="utf-8", newline="\n").write(t)

    f = os.path.join(d, "system", "fvSolution"); t = open(f, encoding="utf-8").read()
    t = t.replace("PIMPLE\n{\n    nCorrectors     1;\n    nNonOrthogonalCorrectors 0;\n}",
                  "PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors     2;\n    nNonOrthogonalCorrectors 0;\n}")
    anchor = """    "(U|Yi|h|k|epsilon|omega).*"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       0;
        relTol          0.1;
        maxIter         20;
    }
}"""
    if anchor not in t:
        raise SystemExit("anchor missing")
    t = t.replace(anchor, anchor[:-1] + "\n" + final_block() + "}")
    open(f, "w", encoding="utf-8", newline="\n").write(t)

    cons = os.path.join(d, "constant", "fvConstraints")
    open(cons, "w", newline="\n").write("""FoamFile
{
    format ascii; class dictionary; object fvConstraints;
}
limitT
{
    type            limitTemperature;
    libs            ("libfvConstraints.so");
    min             250;
    max             350;
    selectionMode   all;
}
""")
    print("made", d)
