"""Layer 4 core: combine device fields, dispatch devices hour by hour, and evaluate the effectiveness measures of a design.

Representation for one hour type (a representative hour; weight = number of season hours it stands for):
  T0 [cells], w_hum0 [cells]   baseline air temperature and humidity ratio
  wnd0 [cells]                 baseline 10 m-equivalent wind (device off)
  Qc [cells], sun [cells]      absorbed radiation and sun flag
  n  [cells]                   persons per cell (cell area 4 m2)
Device contribution (precomputed from the surrogate): dT (K, drop), dW (humidity-ratio increase, kg/kg), v2 (added squared
10 m-equivalent speed). Superposition:
  - temperature drops and humidity-ratio changes add (delivery fractions add, Task 2a balance), then both are scaled per cell by
    min(1, dT_cap / sum dT), where dT_cap is the largest supply drop among the devices switched on: mixed air cannot be colder than
    the coldest supply;
  - the humidity ratio is capped at saturation at the mixed temperature;
  - speeds add in quadrature.
Elasticity is evaluated per cell from the local density (capped at 0.16 persons/m2, Task 2a) and m = 1.2.
"""
import numpy as np
from climate import AT_shade, AT_sun, utility, AT_SAFE, eps_fn

P_HPA = 1008.0
CELL_AREA = 4.0


def e_of_w(w):
    return P_HPA * w / (0.622 + w)


def w_of_e(e):
    return 0.622 * e / (P_HPA - e)


def w_sat(T):
    return w_of_e(6.105 * np.exp(17.27 * T / (237.7 + T)))


def eps_cells(n, m=1.2):
    rho = np.clip(np.asarray(n, float) / CELL_AREA, 0.02, 0.16)   # fitted density range; ln(rho) needs rho > 0
    return np.exp(1.27825 - 1.79203 * m / (1 + m) - 0.01036 * np.log(rho))   # main.tex, checked 2026-11-16


class HourType:
    def __init__(self, T0, w0, wnd0, Qc, sun, n, W=1.0, m=1.2, rad_coef=1.0, eps=None, Ta=None, at_safe=AT_SAFE):
        self.T0, self.w0, self.wnd0, self.Qc, self.sun, self.n, self.W = T0, w0, wnd0, Qc, sun, n, W
        self.m, self.rad_coef, self.Ta = m, rad_coef, Ta
        self.at_safe = at_safe
        self.eps = eps if eps is not None else eps_cells(n, m)
        self.AT0 = self.at(0.0, 0.0, 0.0, None)

    def at(self, dT, dW, v2, cap):
        dT = np.asarray(dT, float); dW = np.asarray(dW, float)
        if cap is not None:
            s = np.minimum(1.0, cap / np.maximum(dT, 1e-12)) if np.ndim(dT) else min(1.0, cap / max(dT, 1e-12))
            s = np.where(dT > cap, s, 1.0)
            dT = dT * s; dW = dW * s
        T = self.T0 - dT
        w = np.minimum(self.w0 + dW, w_sat(T))
        e = e_of_w(w); wnd = np.sqrt(self.wnd0 ** 2 + v2)
        return np.where(self.sun, AT_sun(T, e, wnd, self.Qc, self.m, self.rad_coef), AT_shade(T, e, wnd, self.m))

    def measures(self, AT):
        n = self.n
        return dict(P=n.sum(axis=-1) if np.ndim(AT) == 1 else n.sum(), safe=(n * (AT <= self.at_safe + 1e-6)).sum(-1),
                    H=(n * np.maximum(AT - self.at_safe, 0.0) ** 2).sum(-1), U=(n * utility(self.AT0, AT, self.eps)).sum(-1))

    def evaluate(self, dT=0.0, dW=0.0, v2=0.0, cap=None):
        AT = self.at(dT, dW, v2, cap)
        r = self.measures(AT); r['AT'] = AT
        return r


def dispatch(ht, fields, run_cost, caps, mu_comfort=0.0, cutoffs=None, target=1.0):
    """Greedy dispatch for one hour type.
    fields: list of (dT, dW, v2) per installed device; run_cost: USD per hour when on; caps: supply drop of each device (K).
    Stage 1 (safety first): while the protected share is below `target`, switch on the device that removes the most heat burden H
    per USD of running cost, as long as H decreases.
    Stage 2 (comfort): switch on further devices while mu_comfort x (utility-person gain) exceeds the running cost, provided that
    neither the protected count falls nor H rises.
    cutoffs: True = device not allowed this hour (fan above 35 C air temperature). Returns (on mask, evaluation)."""
    nd = len(fields)
    on = np.zeros(nd, bool)
    if ht.n.sum() <= 0 or nd == 0:
        return on, ht.evaluate()
    cut = np.zeros(nd, bool) if cutoffs is None else np.asarray(cutoffs, bool)
    FT = np.stack([f[0] for f in fields]); FW = np.stack([f[1] for f in fields]); FV = np.stack([f[2] for f in fields])
    rc = np.maximum(np.asarray(run_cost, float), 1e-6); cp = np.asarray(caps, float)
    sT = np.zeros_like(ht.T0); sW = np.zeros_like(ht.T0); sV = np.zeros_like(ht.T0); cap = 0.0
    cur = ht.evaluate(sT, sW, sV, None)
    P = ht.n.sum()
    for stage in (1, 2):
        if stage == 2 and mu_comfort <= 0:
            break
        while True:
            if stage == 1 and cur['safe'] >= target * P - 1e-9:
                break
            cand = np.where(~on & ~cut)[0]
            if len(cand) == 0:
                break
            capc = np.maximum(cap, cp[cand])[:, None]
            dTc = sT + FT[cand]; scale = np.where(dTc > capc, capc / np.maximum(dTc, 1e-12), 1.0)
            T = ht.T0 - dTc * scale
            w = np.minimum(ht.w0 + (sW + FW[cand]) * scale, w_sat(T))
            wnd = np.sqrt(ht.wnd0 ** 2 + sV + FV[cand])
            AT = np.where(ht.sun, AT_sun(T, e_of_w(w), wnd, ht.Qc, ht.m, ht.rad_coef), AT_shade(T, e_of_w(w), wnd, ht.m))
            M = ht.measures(AT)
            if stage == 1:
                gain = cur['H'] - M['H']
                val = np.where((gain > 1e-6 * max(cur['H'], 1.0)) & (M['safe'] >= cur['safe'] - 1e-9), gain / rc[cand], 0.0)
            else:
                g = mu_comfort * (M['U'] - cur['U']) - rc[cand]
                ok = (g > 0) & (M['H'] <= cur['H'] + 1e-9) & (M['safe'] >= cur['safe'] - 1e-9)
                val = np.where(ok, g, 0.0)
            j = int(np.argmax(val))
            if val[j] <= 0:
                break
            b = cand[j]; on[b] = True
            sT = sT + FT[b]; sW = sW + FW[b]; sV = sV + FV[b]; cap = max(cap, cp[b])
            cur = ht.evaluate(sT, sW, sV, cap)
    return on, cur
