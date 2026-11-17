"""One-at-a-time sensitivity of equilibrium efficiency for both venues.

Factors: ambient temperature, relative humidity, wind, device power, occupancy
density, metabolic rate, and solar load. Efficiency is the normalised share per
kWh at the reference hour (steady fields, so "equilibrium").
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.environ.get("HIMCM_TASK3_OUTPUT", os.path.join(ROOT, "figures"))
os.makedirs(OUT, exist_ok=True)
VM = np.load(os.path.join(ROOT, "results", "venue_maps.npz"))
RES = json.load(open(os.path.join(ROOT, "results", "venue_results.json")))
SC = json.load(open(os.path.join(ROOT, "inputs", "scenarios.json")))

ATI = 23.17
STEPS = [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10]


def eps_of(m, rho):
    return float(np.exp(1.27825 - 1.79203 * m / (m + 1.0) - 0.01036 * np.log(rho)))


def ref(venue):
    rows = [r for r in RES if r["venue"] == venue]
    r = max(rows, key=lambda q: q["devices"])
    sc = SC[venue]
    return dict(r=r, sc=sc, m=sc["met"], rho=sc["density_persons_m2"],
                RH=sc["RH_percent"], T=sc["T_C"], w=sc["wind10_m_s"],
                P=r["resources"]["power_kW"], Q=r["solar"]["Q_applied_W_m2"])


def es(T):
    return 6.105 * np.exp(17.27 * T / (237.7 + T))


def eta(venue, base, m, rho, T, RH, w, P, Q):
    AT_off = VM[venue + "_AT_off"].copy()
    dAT = VM[venue + "_dAT"].copy()
    area = VM[venue + "_area"]
    e0 = base["RH"] / 100 * es(base["T"])
    e = RH / 100 * es(T)
    AT_off = (AT_off + (T - base["T"]) + 0.348 * (e - e0)
              - 0.70 * (w - base["w"])
              + (Q + 58.2 * m) / (w + 10.0) - (base["Q"] + 58.2 * base["m"]) / (base["w"] + 10.0))
    AT_on = AT_off - dAT
    epsi = eps_of(m, rho)
    share = 1 - (np.abs(ATI - AT_on) / np.maximum(np.abs(ATI - AT_off), 1e-9)) ** epsi
    return float(rho * (area * share).sum() / P)


if __name__ == "__main__":
    factors = [
        ("$T_a$", "T", "ambient temperature"),
        ("RH", "RH", "relative humidity"),
        ("$w_s$", "w", "wind at 10 m"),
        ("$P$", "P", "electrical power"),
        ("$\\rho$", "rho", "occupancy density"),
        ("$m$", "m", "metabolic rate"),
        ("$Q$", "Q", "solar load"),
    ]
    curves = {}
    rows = []
    for venue in ("bayfront", "medina"):
        base = ref(venue)
        eta0 = eta(venue, base, base["m"], base["rho"], base["T"], base["RH"], base["w"], base["P"], base["Q"])
        curves[venue] = {}
        for label, key, name in factors:
            ys = []
            for s in STEPS:
                kw = dict(m=base["m"], rho=base["rho"], T=base["T"], RH=base["RH"],
                          w=base["w"], P=base["P"], Q=base["Q"])
                kw[key] = base[key] * (1 + s)
                ys.append(100 * (eta(venue, base, **kw) / eta0 - 1))
            curves[venue][label] = (np.array(STEPS) * 100, np.array(ys), name)
            slope = (ys[-1] - ys[0]) / 20.0      # percent efficiency per percent factor
            rows.append(dict(venue=venue, factor=label, name=name, eta0=eta0, slope=round(slope, 3),
                             at_m10=round(ys[0], 2), at_p10=round(ys[-1], 2)))

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, venue in zip(axs, ("bayfront", "medina")):
        for label, (x, y, name) in curves[venue].items():
            ax.plot(x, y, "o-", lw=1.4, ms=3.5, label=label + "  (" + name + ")")
        ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
        ax.set_title(("Bayfront Event Space" if venue == "bayfront" else "Al-Masjid an-Nabawi"), fontsize=9)
        ax.set_xlabel("change in factor (%)", fontsize=8)
        ax.set_ylabel("change in efficiency (%)", fontsize=8)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.2, loc="best")
    fig.suptitle("One-at-a-time sensitivity of equilibrium efficiency (normalised share per kWh)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sensitivity_efficiency.png"), dpi=170)
    json.dump(rows, open(os.path.join(OUT, "sensitivity_efficiency.json"), "w"), indent=1)
    for v in ("bayfront", "medina"):
        print(v)
        for r in sorted([q for q in rows if q["venue"] == v], key=lambda q: -abs(q["slope"])):
            print("  %-8s %-22s slope %+0.3f  eta0 %.3f" % (r["factor"], r["name"], r["slope"], r["eta0"]))
    print("wrote", os.path.join(OUT, "sensitivity_efficiency.png"))
