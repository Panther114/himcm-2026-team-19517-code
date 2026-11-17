"""Device library for Task 4: one CFD case template per device class, and the design of experiments (DOE).

Subdomain: x downwind (-10 to 30 m), y span 24 m (cyclic, i.e. one device per 24 m of crosswind row: effectively isolated),
z up to 16 m. Ground at ambient temperature (shaded plaza; sun and roof are handled in the dynamic layer).
Ambient wind is a log profile matched to the body-level wind w_b at 1.1 m (z0 = 0.5 m, sheltered urban plaza).

Classes (physics parameters; costs are in inputs.csv):
  A  outdoor fan (650 mm)      0.8 x 0.8 m outlet at 1.1-1.9 m, airflow 2.3 m3/s (8,280 m3/h, 0.26 kW class), ambient air
  M  misting fan                fan A + 5 L/h of 5-40 um droplets at 25 C (PIN IMF66 class)
  E  industrial evaporative cooler 1.0 x 1.0 m outlet at 0.6-1.6 m, 5.0 m3/s (18,000 m3/h, 1.1 kW class), supply wet-bulb limited, drop <= 8 K
  D  chilled-air unit           0.6 x 0.6 m outlet at 0.7-1.3 m, 1.57 m3/s (analog of the Bayfront unit), supply 16-24 C, RH 90 %
"""
import math
import numpy as np
from cfd_gen import Y_h2o

P_PA = 100800.0   # Doha summer surface pressure (ERA5 mean, about 1,008 hPa)
OFF_REF = (0.3, 1.0, 2.5)   # m/s, body-level wind of the device-off reference cases


def twb(T, RH):   # Stull (2011), as in Appendix of main.tex; bounded by the dry-bulb temperature
    return min(T, T * math.atan(0.151977 * (RH + 8.313659) ** 0.5) + math.atan(T + RH) - math.atan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * math.atan(0.023101 * RH) - 4.686035)


def _mesh_base(fine=1.0):
    f = lambda n: max(1, int(round(n * fine)))
    return dict(xb=[-10, -0.4, 0, 0.6, 3, 10, 30], xn=[f(12), f(2), f(3), f(8), f(18), f(25)],
                yb=[0, 10.5, 11.5, 12.5, 13.5, 24], yn=[f(14), f(4), f(4), f(4), f(14)],
                zb=[0, 0.7, 1.1, 1.6, 1.9, 4, 16], zn=[f(4), f(3), f(4), f(2), f(8), f(12)])


def _amb(Ta, RH, w_b):
    # log profile u(z) = (u*/k) ln((z+z0)/z0); choose Uref at Zref = 1.1 m equal to w_b
    return dict(T=Ta + 273.15, RH=RH, p=P_PA, Uref=max(w_b, 0.05), Zref=1.1, z0=0.5)


def spec(cls, w_b=1.0, Ta=38.0, RH=40.0, setting=1.0, T_sup=20.0, on=True, endTime=1500, fine=1.0):
    """Return a cfd_gen spec. setting = airflow fraction (0.4-1); T_sup only for class D."""
    m = _mesh_base(fine)
    Ya = Y_h2o(Ta, RH, P_PA / 100)
    cloud = None
    if cls in ('A', 'M'):
        V = 2.3 * setting; area = 0.8 * 0.8; box = (-0.4, 0, 11.6, 12.4, 1.1, 1.9)
        jet = dict(patch='dev_out', face='xp', U=round(V / area, 4) if on else 0.0, T=Ta + 273.15, Y=Ya)
        if cls == 'M' and on:
            cloud = dict(mdot=5 / 3600.0, T_water=25 + 273.15, parcels_per_s=2000, pos=(0.05, 12.0, 1.5), Umag=V / area,
                         half_angle=20, dmin=5e-6, dmax=40e-6, rr_lambda=1.5e-5, n=2.5)
    elif cls == 'E':
        V = 5.0 * setting; area = 1.0 * 1.0; box = (-0.4, 0, 11.5, 12.5, 0.6, 1.6)
        dT = max(0.0, min(8.0, 0.8 * (Ta - twb(Ta, RH))))
        Ts = Ta - dT
        # adiabatic humidification: c_pa dT = h_fg d(omega)
        om_a = Ya / (1 - Ya); om_s = om_a + 1.006 * dT / 2450.0
        jet = dict(patch='dev_out', face='xp', U=round(V / area, 4) if on else 0.0, T=Ts + 273.15, Y=om_s / (1 + om_s))
        m['yb'] = [0, 10.5, 11.5, 12.5, 13.5, 24]
        m['zb'] = [0, 0.6, 1.1, 1.6, 1.9, 4, 16]
    elif cls == 'D':
        V = 1.57 * setting; area = 0.6 * 0.6; box = (-0.4, 0, 11.7, 12.3, 0.7, 1.3)
        Ys = min(Y_h2o(T_sup, 90.0, P_PA / 100), Ya)     # cooling coil dries the air or leaves it unchanged
        jet = dict(patch='dev_out', face='xp', U=round(V / area, 4) if on else 0.0, T=T_sup + 273.15, Y=Ys)
        m['yb'] = [0, 10.5, 11.7, 12.3, 13.5, 24]
        m['zb'] = [0, 0.7, 1.0, 1.3, 1.9, 4, 16]
    else:
        raise ValueError(cls)
    if not on:
        jet = None
    solids = [dict(name='dev_body', box=box, **({'jet': jet} if jet else {}))]
    return dict(**m, solids=solids, ambient=_amb(Ta, RH, w_b), T_ground=Ta + 273.15, endTime=endTime, writeInterval=max(endTime // 2, 1), cloud=cloud)


def lhs(n, dims, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(dims)))
    for j, (lo, hi) in enumerate(dims):
        perm = rng.permutation(n)
        out[:, j] = lo + (hi - lo) * (perm + rng.random(n)) / n
    return out


def doe():
    """Design of experiments. Returns list of (name, spec_kwargs)."""
    jobs = []
    # A: isothermal momentum jet -> wind and setting only (Ta, RH fixed; the speed field is independent of them)
    for i, (wb, s) in enumerate(lhs(16, [(0.2, 3.0), (0.4, 1.0)], 11)):
        jobs.append((f'A_{i:02d}', dict(cls='A', w_b=round(wb, 3), setting=round(s, 3), Ta=38.0, RH=40.0)))
    # E: wind, setting, and climate (sets the wet-bulb-limited supply drop and buoyancy)
    for i, (wb, s, Ta, RH) in enumerate(lhs(24, [(0.2, 3.0), (0.4, 1.0), (30, 46), (15, 80)], 12)):
        jobs.append((f'E_{i:02d}', dict(cls='E', w_b=round(wb, 3), setting=round(s, 3), Ta=round(Ta, 2), RH=round(RH, 1))))
    # D: wind, setting, supply temperature, ambient temperature
    for i, (wb, s, Ts, Ta) in enumerate(lhs(24, [(0.2, 3.0), (0.4, 1.0), (16, 24), (30, 46)], 13)):
        jobs.append((f'D_{i:02d}', dict(cls='D', w_b=round(wb, 3), setting=round(s, 3), T_sup=round(Ts, 2), Ta=round(Ta, 2), RH=45.0)))
    # M: wind and climate (evaporation depends on Ta, RH); setting fixed at full
    for i, (wb, Ta, RH) in enumerate(lhs(24, [(0.2, 3.0), (30, 46), (15, 80)], 14)):
        jobs.append((f'M_{i:02d}', dict(cls='M', w_b=round(wb, 3), Ta=round(Ta, 2), RH=round(RH, 1))))
    return jobs
