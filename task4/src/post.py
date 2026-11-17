"""Post-processing of device CFD cases: body-level (1.1 +- 0.25 m) fields of T, vapour pressure and speed on a regular 0.5 m grid."""
import os, sys, subprocess, json
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from runner import OF

GX = np.arange(-9.75, 30.0, 0.5)      # cell centres, x downwind
GY = np.arange(0.25, 24.0, 0.5)       # span


def latest_time(case):
    ts = []
    for d in os.listdir(case):
        try:
            v = float(d)
        except ValueError:
            continue
        if v > 0 and os.path.isdir(os.path.join(case, d)): ts.append((v, d))
    if not ts: raise FileNotFoundError(f'no solution time in {case}')
    return max(ts)[1]


def latest_vtk(case, time=None):
    """VTK export of the latest (or given) solution time; exported if missing."""
    t = time or latest_time(case)
    vt = os.path.join(case, 'VTK')
    want = [f for f in (os.listdir(vt) if os.path.isdir(vt) else []) if f.endswith(f'_{t}.vtk')]
    if not want:
        r = subprocess.run(['cmd', '/c', OF, 'foamToVTK', '-time', t, '-noFaceZones', '-excludePatches', '(".*")'], cwd=case, capture_output=True, text=True)
        want = [f for f in (os.listdir(vt) if os.path.isdir(vt) else []) if f.endswith(f'_{t}.vtk')]
        if not want: raise RuntimeError(f'foamToVTK failed for {case} at {t}: {r.stderr[-300:]}')
    return os.path.join(vt, want[0])


def body_level(case, p_hpa=1008.0, z=1.1, dz=0.25, time=None):
    """Return dict of 2D arrays (len(GY), len(GX)): T (C), e (hPa), speed (m/s), averaged over the body slab."""
    cd = pv.read(latest_vtk(case, time))
    c = cd.cell_centers().points
    sel = np.abs(c[:, 2] - z) <= dz
    T = cd.cell_data['T'][sel] - 273.15
    Y = cd.cell_data['H2O'][sel]
    om = Y / (1 - Y); e = p_hpa * om / (0.622 + om)
    U = np.linalg.norm(cd.cell_data['U'][sel], axis=1)
    xy = c[sel][:, :2]
    out = {}
    ix = np.clip(np.searchsorted(GX - 0.25, xy[:, 0]) - 1, 0, len(GX) - 1)
    iy = np.clip(np.searchsorted(GY - 0.25, xy[:, 1]) - 1, 0, len(GY) - 1)
    for k, v in (('T', T), ('e', e), ('speed', U)):
        s = np.zeros((len(GY), len(GX))); n = np.zeros_like(s)
        np.add.at(s, (iy, ix), v); np.add.at(n, (iy, ix), 1)
        g = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        # fill empty grid cells (coarse CFD cells larger than the grid) by nearest filled neighbour along x then y
        for _ in range(6):
            m = np.isnan(g)
            if not m.any(): break
            gp = np.pad(g, 1, mode='edge')
            nb = np.nanmean(np.stack([gp[:-2, 1:-1], gp[2:, 1:-1], gp[1:-1, :-2], gp[1:-1, 2:]]), axis=0)
            g[m] = nb[m]
        out[k] = g
    return out


def residual_summary(case):
    log = open(os.path.join(case, 'log.foamRun')).read().splitlines()
    last = [l for l in log if 'Solving for Ux' in l][-1:]
    return last[0].strip() if last else ''
