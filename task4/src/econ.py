"""Economics and resources of the device classes (values from inputs.csv; sourced or assumed as labelled there)."""
import math
import numpy as np

DEV = {
    'A': dict(price=300.0, life=10, maint=0.05, power=0.26, water_Lph=0.0, noise=65.0),
    'M': dict(price=499.0, life=7, maint=0.10, power=0.26, water_Lph=5.0, noise=68.0),
    'E': dict(price=385.0, life=8, maint=0.10, power=1.10, water_Lph=None, noise=68.0, V=5.0),
    'D': dict(price=755.0 * 38.0, life=15, maint=0.04, power=0.50, water_Lph=0.0, noise=68.0, V=1.57, T_sup=18.0),
}
ECON = dict(r=0.031, install=1.3, tariff=0.036, water=1.21, carbon=0.636, cop_dc=3.7, noise_limit=65.0)


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def annual_capex(cls, econ=ECON):
    d = DEV[cls]; c = d['price'] * econ['install']
    return c * crf(econ['r'], d['life']) + c * d['maint']


def hourly_running(cls, Ta, RH, econ=ECON, p_hpa=1008.0):
    """Electricity (kWh per hour on), water (L per hour on) for one device in ambient Ta, RH."""
    d = DEV[cls]
    if cls == 'E':
        from devices import twb
        dT = max(0.0, min(8.0, 0.8 * (Ta - twb(Ta, RH))))
        dw = 1.006 * dT / 2450.0
        return d['power'], 1.18 * d['V'] * dw * 3600.0          # kg/s -> L/h (rho_air about 1.18)
    if cls == 'D':
        e = RH / 100 * 6.105 * math.exp(17.27 * Ta / (237.7 + Ta)); w_a = 0.622 * e / (p_hpa - e)
        es = 0.9 * 6.105 * math.exp(17.27 * d['T_sup'] / (237.7 + d['T_sup'])); w_s = min(0.622 * es / (p_hpa - es), w_a)
        h = lambda T, w: 1.006 * T + w * (2501 + 1.86 * T)        # kJ/kg dry air
        q_th = 1.18 * d['V'] * max(h(Ta, w_a) - h(d['T_sup'], w_s), 0.0)   # kW of cooling
        return d['power'] + q_th / econ['cop_dc'], 0.0
    return d['power'], d['water_Lph']


def noise_field(slots_xy, classes, cells_xy):
    """Sound pressure level at cells from point sources (L_1m - 20 log10 r), energetic sum."""
    tot = np.zeros(len(cells_xy))
    for s, c in zip(slots_xy, classes):
        r = np.maximum(np.linalg.norm(cells_xy - s, axis=1), 2.0)   # 2 m clearance: visitors do not stand inside a unit
        tot += 10 ** ((DEV[c]['noise'] - 20 * np.log10(r)) / 10)
    return 10 * np.log10(np.maximum(tot, 1e-12))
