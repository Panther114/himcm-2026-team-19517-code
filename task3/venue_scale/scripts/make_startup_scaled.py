"""Start-up efficiency: scale the steady venue efficiency by the transient cooling fraction."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "viz")))
import of_viz

D = os.environ.get("HIMCM_TASK3_CASES", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cfd", "devices")))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)

# steady venue efficiencies from the venue analysis
ETA_INF = {"bayfront": 0.579, "medina": 2.484}
M = {"bayfront": 1.2, "medina": 2.0}
P_HPA = {"bayfront": 1008.0, "medina": 942.0}
CASES = {"bayfront": dict(off="D_bf_off", trans="D_fast", z=1.1, th=0.5),
         "medina": dict(off="M_md_off", trans="M_coarse", z=3.5, th=1.0)}


def times(case):
    out = []
    for d in os.listdir(case):
        try:
            v = float(d)
        except ValueError:
            continue
        if v >= 0 and os.path.isdir(os.path.join(case, d)):
            out.append((v, d))
    return sorted(out)


def at_fields(case, tdir, n, m, p):
    T = np.asarray(of_viz.load_display_field(case, tdir, "T", n), dtype=float) - 273.15
    Y = np.asarray(of_viz.load_display_field(case, tdir, "H2O", n), dtype=float)
    nc, U = of_viz.read_field(case, tdir, "U")
    U = np.asarray(U, dtype=float)
    if U.shape[0] == 1:
        U = np.repeat(U, n, axis=0)
    sp = np.linalg.norm(U, axis=1)
    om = Y / (1 - Y)
    e = p * om / (0.622 + om)
    return T + 0.33 * e - 0.70 * sp + 58.2 * m / (sp + 10.0) - 12.15



def fine_to_coarse_AT(case, tdir, m, p, target_centres):
    fc, _ = of_viz.cell_centres(case)
    fc = np.asarray(fc)
    AT = at_fields(case, tdir, len(fc), m, p)
    tree = cKDTree(fc)
    _, idx = tree.query(target_centres)
    return AT[idx]

rows = {}
for venue, cfg in CASES.items():
    off = os.path.join(D, cfg["off"]); tr = os.path.join(D, cfg["trans"])
    centres, _ = of_viz.cell_centres(tr)
    centres = np.asarray(centres)
    slab = np.abs(centres[:, 2] - cfg["z"]) <= 0.25
    n = len(centres)
    if len(of_viz.cell_centres(off)[0]) != n:
        AT_off = fine_to_coarse_AT(off, times(off)[-1][1], M[venue], P_HPA[venue], centres)[slab]
    else:
        AT_off = at_fields(off, times(off)[-1][1], len(of_viz.cell_centres(off)[0]), M[venue], P_HPA[venue])[slab]
    AT_inf = at_fields(tr, times(tr)[-1][1], n, M[venue], P_HPA[venue])[slab]
    dAT_inf = AT_off - AT_inf
    foot = dAT_inf > cfg["th"]
    ts, eta = [], []
    for t, d in times(tr):
        AT = at_fields(tr, d, n, M[venue], P_HPA[venue])[slab]
        d = AT_off - AT
        r = float(np.mean(d[foot]) / np.mean(dAT_inf[foot])) if foot.any() else 0.0
        ts.append(t); eta.append(ETA_INF[venue] * max(r, 0.0))
    rows[venue] = dict(t=ts, eta=eta, eta_inf=ETA_INF[venue])
    print(venue, "points", len(ts), "start %.3f" % eta[0], "last %.3f" % eta[-1], "asym %.3f" % ETA_INF[venue])

from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for ax, venue in zip(axs, ("bayfront", "medina")):
    r = rows[venue]
    ax.plot(r["t"], r["eta"], "o-", color="tab:blue", label="efficiency during start-up")
    t = np.array(r["t"]); y = np.array(r["eta"]); e0 = r["eta_inf"]
    best = None
    for tau in np.linspace(0.1, 30.0, 6000):
        pred = e0 * (1 - np.exp(-t / tau))
        err = float(np.sum((pred - y) ** 2))
        if best is None or err < best[1]:
            best = (tau, err)
    tau = best[0]
    tt = np.linspace(0, 20, 200)
    ax.plot(tt, e0 * (1 - np.exp(-tt / tau)), "--", color="tab:orange", lw=1.2,
            label="exponential fit, tau = %.1f s (95%% at %.1f s)" % (tau, 3 * tau))
    ax.axhline(e0, color="tab:red", ls="--", lw=1.2, label="steady value %.2f per kWh" % e0)
    ax.set_ylabel("$\\eta$ (share of gap per kWh)", fontsize=8)
    ax.set_title(("Bayfront Event Space" if venue == "bayfront" else "Al-Masjid an-Nabawi"), fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.tick_params(labelsize=7)
axs[1].set_xlabel("time after switch-on (s)", fontsize=8)
fig.suptitle("Efficiency after switch-on: steady venue efficiency scaled by the transient cooling fraction", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "efficiency_startup.png"), dpi=170)
json.dump(rows, open(os.path.join(OUT, "efficiency_startup.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "efficiency_startup.png"))
