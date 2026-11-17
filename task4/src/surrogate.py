"""Layer 2: surrogate of the device fields (POD + Gaussian-process regression) and placement of device fields in the plaza.

Targets on the body-level grid (post.GX x post.GY, device frame: outlet face at x = 0, device axis y = 12 m, x downwind):
  A (fan):            v2  = max(speed_on^2 - speed_off^2, 0)                       added squared body-level speed (m2/s2)
  E, D (cooled air):  phiT = (T_off - T_on) / dT_supply ; phiW = (w_on - w_off) / dw_supply (humidity ratio); v2
  M (mist):           dT = T_off - T_on (K) ; de = e_on - e_off (hPa) ; v2
Inputs per class (columns of X): see INPUTS.
"""
import os, sys, json, math, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import devices, post
from cfd_gen import Y_h2o

ROOT = os.path.abspath(os.path.join(HERE, '..'))
INPUTS = {'A': ['w_b', 'setting'], 'E': ['w_b', 'setting', 'Ta', 'RH'], 'D': ['w_b', 'setting', 'T_sup', 'Ta'], 'M': ['w_b', 'Ta', 'RH']}
TARGETS = {'A': ['v2'], 'E': ['phiT', 'phiW', 'v2'], 'D': ['phiT', 'v2'], 'M': ['dT', 'de', 'v2']}
P_HPA = devices.P_PA / 100


def omega(e):
    return 0.622 * e / (P_HPA - e)


def supply_state(cls, kw):
    Ta, RH = kw.get('Ta', 38.0), kw.get('RH', 40.0)
    Ya = Y_h2o(Ta, RH, P_HPA); oma = Ya / (1 - Ya)
    if cls == 'E':
        dT = max(0.0, min(8.0, 0.8 * (Ta - devices.twb(Ta, RH))))
        return dT, 1.006 * dT / 2450.0
    if cls == 'D':
        Ys = min(Y_h2o(kw['T_sup'], 90.0, P_HPA), Ya); oms = Ys / (1 - Ys)
        return Ta - kw['T_sup'], oms - oma
    return None, None


_OFF = {}
_CENTRES = {}


def _body_level(case, p_hpa=P_HPA, time=None):
    """Read saved fields without exporting VTK or writing into a CFD case.

    Reproduce post.body_level's slab averaging and empty-cell filling, using
    existing reference-mesh VTK cell centres and saved ASCII solution fields.
    Cache only in work/, keyed by the source field modification times.
    """
    import hashlib
    import warnings
    import pyvista as pv
    time = time or post.latest_time(case)
    stamp = [(f, os.stat(os.path.join(case, time, f)).st_mtime_ns)
             for f in ('U', 'T', 'H2O')]
    key = hashlib.sha256(repr((case, time, p_hpa, stamp)).encode()).hexdigest()[:20]
    cache = os.path.join(ROOT, 'work', 'surrogate_fields')
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, key + '.npz')
    if os.path.exists(path):
        with np.load(path) as a:
            return {k: a[k] for k in a.files}
    # The on/off DOE cases of each body use the identical block mesh and
    # OpenFOAM cell order. Reuse an existing off VTK for its cell centres;
    # parse saved ASCII fields directly (no OpenFOAM executable).
    body = os.path.basename(case).split('_')[0]
    if body == 'off': body = os.path.basename(case).split('_')[1]
    if body == 'M': body = 'A'
    if body not in _CENTRES:
        ref = os.path.join(ROOT, 'cfd', 'devices', f'off_{body}_1.0')
        vt = os.path.join(ref, 'VTK')
        files = [os.path.join(vt, f) for f in os.listdir(vt)
                 if f.endswith('_' + post.latest_time(ref) + '.vtk')]
        if not files: raise FileNotFoundError(f'Existing reference VTK required: {ref}')
        _CENTRES[body] = pv.read(files[0]).cell_centers().points
    c = _CENTRES[body]
    def read_field(name, components=1):
        import re
        with open(os.path.join(case, time, name)) as f:
            raw = f.read()
        match = re.search(r'internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\(', raw)
        if match:
            count = int(match[1])
            end = raw.index(';', match.end())
            values = np.fromstring(raw[match.end():end].replace('(', ' ').replace(')', ' '), sep=' ')
            if count != len(c) or len(values) != count * components:
                raise ValueError(f'Mesh/field size mismatch: {case}, {name}')
            return values.reshape(count, components) if components > 1 else values
        match = re.search(r'internalField\s+uniform\s+([^;]+)', raw)
        if not match: raise ValueError(f'Expected ASCII field: {case}, {name}')
        values = np.fromstring(match[1].replace('(', ' ').replace(')', ' '), sep=' ')
        return np.tile(values, (len(c), 1)) if components > 1 else np.full(len(c), values[0])
    sel = np.abs(c[:, 2] - 1.1) <= 0.25
    T = read_field('T')[sel] - 273.15
    Y = read_field('H2O')[sel]
    om = Y / (1 - Y)
    e = p_hpa * om / (0.622 + om)
    U = np.linalg.norm(read_field('U', 3)[sel], axis=1)
    xy = c[sel, :2]
    ix = np.clip(np.searchsorted(post.GX - 0.25, xy[:, 0]) - 1, 0, len(post.GX) - 1)
    iy = np.clip(np.searchsorted(post.GY - 0.25, xy[:, 1]) - 1, 0, len(post.GY) - 1)
    out = {}
    for k, v in (('T', T), ('e', e), ('speed', U)):
        s = np.zeros((len(post.GY), len(post.GX))); n = np.zeros_like(s)
        np.add.at(s, (iy, ix), v); np.add.at(n, (iy, ix), 1)
        g = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        for _ in range(6):
            m = np.isnan(g)
            if not m.any(): break
            gp = np.pad(g, 1, mode='edge')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                nb = np.nanmean(np.stack([gp[:-2, 1:-1], gp[2:, 1:-1],
                                         gp[1:-1, :-2], gp[1:-1, 2:]]), axis=0)
            g[m] = nb[m]
        if not np.isfinite(g).all():
            raise ValueError(f'Unfilled/nonfinite {k} cells: {case} at {time}')
        out[k] = g
    np.savez_compressed(path, **out)
    return out


def off_reference(cls, w_b):
    """Device-off state for gravity-off cases: ambient T and e everywhere; speed field scaled linearly from the reference
    case with the nearest body-level wind (A and M share the fan body)."""
    body = 'A' if cls in ('A', 'M') else cls
    ref = min(devices.OFF_REF, key=lambda r: abs(r - w_b))
    key = (body, ref)
    if key not in _OFF:
        _OFF[key] = _body_level(os.path.join(ROOT, 'cfd', 'devices', f'off_{body}_{ref:.1f}'), P_HPA)['speed']
    return _OFF[key] * (w_b / ref)


def targets_for(cls, name, kw):
    case = os.path.join(ROOT, 'cfd', 'devices')
    on = _body_level(os.path.join(case, f'{name}_on'), P_HPA)
    Ta, RH = kw.get('Ta', 38.0), kw.get('RH', 40.0)
    e_a = RH / 100 * 6.105 * math.exp(17.27 * Ta / (237.7 + Ta))
    off = dict(T=np.full_like(on['T'], Ta), e=np.full_like(on['e'], e_a), speed=off_reference(cls, kw['w_b']))
    v2 = np.maximum(on['speed'] ** 2 - off['speed'] ** 2, 0.0)
    out = {'v2': v2}
    if cls in ('E', 'D'):
        dTs, dws = supply_state(cls, kw)
        out['phiT'] = (off['T'] - on['T']) / max(dTs, 1e-3)
        if cls == 'E':
            out['phiW'] = (omega(on['e']) - omega(off['e'])) / max(dws, 1e-6)
    if cls == 'M':
        out['dT'] = off['T'] - on['T']; out['de'] = on['e'] - off['e']
    return out


def collect(cls):
    X, Y = [], {t: [] for t in TARGETS[cls]}
    names = []
    for name, kw in devices.doe():
        if kw['cls'] != cls: continue
        d = os.path.join(ROOT, 'cfd', 'devices')
        if not os.path.exists(os.path.join(d, f'{name}_on', 'done.json')):
            continue
        tg = targets_for(cls, name, kw)
        X.append([kw[k] if k in kw else 1.0 for k in INPUTS[cls]])
        for t in TARGETS[cls]: Y[t].append(np.nan_to_num(tg[t]).ravel())
        names.append(name)
    return np.array(X, float), {t: np.array(v) for t, v in Y.items()}, names


class FieldGP:
    """Fold-local POD and a regularised GP with a shared spatial-mode kernel.

    Keep all numerically nonzero modes by default. A common kernel avoids
    unstable independent hyperparameter estimates for tiny POD coefficients.
    Targets are always decoded to their original units before scoring/return.
    """
    def __init__(self, energy=1.0, max_modes=32, cls='A', target='v2',
                 input_transform='raw', transform='none', kernel='15',
                 speed_reference=False, restarts=1):
        self.energy, self.max_modes = energy, max_modes
        self.cls, self.target = cls, target
        self.input_transform, self.transform = input_transform, transform
        self.kernel, self.speed_reference, self.restarts = kernel, speed_reference, restarts

    def _setting(self, X):
        return np.maximum(X[:, 1], 1e-6) if self.cls != 'M' else np.ones(len(X))

    def _inputs(self, X):
        ratio = X[:, 0] / self._setting(X)
        if self.input_transform == 'ratio': return ratio[:, None]
        if self.input_transform == 'logratio': return np.log(np.maximum(ratio, 1e-6))[:, None]
        if self.input_transform == 'ratio_w': return np.column_stack([ratio, X[:, 0]])
        if self.input_transform == 'logratio_w':
            return np.column_stack([np.log(np.maximum(ratio, 1e-6)), X[:, 0]])
        if self.input_transform == 'flow_density':
            # Ideal-gas supply/ambient density ratio at approximately equal p.
            if self.cls == 'D': density = (X[:,3]+273.15)/(X[:,2]+273.15)
            elif self.cls == 'E':
                drop=np.array([supply_state('E',dict(Ta=r[2],RH=r[3]))[0] for r in X])
                density=(X[:,2]+273.15)/(X[:,2]+273.15-drop)
            else: raise ValueError('flow_density requires E or D')
            return np.column_stack([np.log(np.maximum(ratio,1e-6)),X[:,0],density])
        # Delivery fractions divide out supply amplitudes. Remaining density
        # effects are small for those targets. M retains ambient temperature.
        return X[:, :2]

    def _off2(self, X):
        winds = np.asarray(devices.OFF_REF)
        nearest = np.argmin(abs(X[:, 0, None] - winds), axis=1)
        return self.off_unit2[nearest] * X[:, 0, None] ** 2

    def fit(self, X, F, on_speed2=None):
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C
        from sklearn.exceptions import ConvergenceWarning
        X = np.atleast_2d(X).astype(float)
        self.training_bounds = np.array([X.min(0), X.max(0)])
        scale = self._setting(X)[:, None] ** 2 if self.target == 'v2' else 1.0
        if self.speed_reference:
            if on_speed2 is None: raise ValueError('Signed speed fit requires training on-state speed squared')
            self.off_unit2 = np.array([off_reference(self.cls, w).ravel() ** 2 / w ** 2
                                       for w in devices.OFF_REF])
            self.base2 = self.off_unit2[list(devices.OFF_REF).index(1.0)]
            G = (on_speed2 - X[:, 0, None] ** 2 * self.base2) / scale
        else:
            G = F / scale
        if self.transform == 'sqrt': H = np.sqrt(np.maximum(G, 0))
        elif self.transform == 'log': H = np.log1p(np.maximum(G, 0))
        elif self.transform == 'asinh': H = np.arcsinh(G)
        else: H = G
        Z = self._inputs(X)
        self.xm, self.xs = Z.mean(0), Z.std(0) + 1e-9
        Z = (Z - self.xm) / self.xs
        self.mean = H.mean(0); centered = H - self.mean
        # Snapshot Gram matrix is small (n <= 24), unlike the spatial matrix.
        val, U = np.linalg.eigh(centered @ centered.T)
        order = np.argsort(val)[::-1]; val, U = val[order], U[:, order]
        rank = int(np.count_nonzero(val > max(val[0] * 1e-12, 1e-12)))
        if rank == 0:
            self.modes = np.empty((0, F.shape[1])); self.gp = None
            return self
        k = min(rank, self.max_modes, int(np.searchsorted(np.cumsum(val[:rank]) / val[:rank].sum(), self.energy) + 1))
        self.modes = (U[:, :k].T @ centered) / np.sqrt(val[:k, None])
        coef = centered @ self.modes.T
        self.coef_scale = max(float(np.sqrt(np.mean(centered ** 2) * F.shape[1])), 1e-12)
        spatial = RBF(np.ones(Z.shape[1]), (.05, 30)) if self.kernel == 'rbf' else Matern(
            np.ones(Z.shape[1]), (.05, 30), nu=1.5 if self.kernel == '15' else 2.5)
        kern = C(1., (.01, 100)) * spatial + WhiteKernel(1e-5, (1e-8, .1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            self.gp = GaussianProcessRegressor(kern, normalize_y=False,
                n_restarts_optimizer=self.restarts, random_state=0).fit(Z, coef / self.coef_scale)
        return self

    def predict(self, X):
        X = np.atleast_2d(X).astype(float)
        Z = (self._inputs(X) - self.xm) / self.xs
        H = np.broadcast_to(self.mean, (len(X), len(self.mean))).copy()
        if self.gp is not None:
            coef = np.asarray(self.gp.predict(Z)).reshape(len(X), len(self.modes))
            H += (coef * self.coef_scale) @ self.modes
        if self.transform == 'sqrt': G = np.maximum(H, 0) ** 2
        elif self.transform == 'log': G = np.expm1(np.clip(H, -30, 30))
        elif self.transform == 'asinh': G = np.sinh(np.clip(H, -20, 20))
        else: G = H
        if self.target == 'v2':
            G = G * self._setting(X)[:, None] ** 2
            if self.speed_reference:
                G += X[:, 0, None] ** 2 * self.base2 - self._off2(X)
        G = np.maximum(G, 0)
        if self.target in ('phiT', 'phiW'):
            # Thermal mixing cannot exceed the prescribed supply drop.
            G = np.minimum(G, 1.0 if self.target == 'phiT' else 1.2)
        if self.target == 'dT' and self.cls == 'M':
            # No single supply-air temperature for mist. Conservative bound
            # combines adiabatic wet bulb with the 25 C injected water.
            cap = np.array([max(0., row[1] - min(25., devices.twb(row[1], row[2]))) for row in X])
            G = np.minimum(G, cap[:, None])
        return G


def loo_r2(X, F, on_speed2=None, **kw):
    """Leave-one-out R2 of the reconstructed field (pooled over cells and cases)."""
    pred = np.zeros_like(F)
    for i in range(len(X)):
        m = np.ones(len(X), bool); m[i] = False
        extra = None if on_speed2 is None else on_speed2[m]
        pred[i] = FieldGP(**kw).fit(X[m], F[m], extra).predict(X[i:i + 1])[0]
    ss = ((F - pred) ** 2).sum(); st = ((F - F.mean()) ** 2).sum()
    return 1 - ss / st, pred


def fit_all(classes='AEDM', cv=True):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _fit_all(classes, cv)


def _fit_all(classes, cv):
    models, report, diagnostic = {}, {}, []
    configs = {
        ('A', 'v2'): dict(input_transform='logratio_w', transform='asinh', kernel='25', speed_reference=True),
        ('E', 'phiT'): dict(input_transform='logratio'),
        ('E', 'phiW'): dict(input_transform='logratio'),
        ('E', 'v2'): dict(input_transform='ratio_w', speed_reference=True),
        ('D', 'phiT'): dict(input_transform='logratio', kernel='25'),
        ('D', 'v2'): dict(input_transform='flow_density', transform='asinh', kernel='25', speed_reference=True),
        ('M', 'dT'): dict(transform='sqrt', kernel='rbf'),
        ('M', 'de'): dict(transform='sqrt'),
        ('M', 'v2'): dict(),
    }
    for cls in classes:
        X, Y, names = collect(cls)
        if len(X) < 4:
            report[cls] = dict(n=len(X), note='too few cases'); continue
        omitted = [name for name, kw in devices.doe() if kw['cls']==cls and name not in names]
        models[cls] = {}; report[cls] = dict(n=len(X), cases=names,
            omitted_without_done=omitted, targets={})
        speed2 = np.array([_body_level(os.path.join(ROOT, 'cfd', 'devices', name + '_on'))['speed'].ravel() ** 2
                           for name in names])
        worst = (-1, None)
        for t, F in Y.items():
            config = dict(cls=cls, target=t, **configs[(cls, t)])
            extra = speed2 if config.get('speed_reference') else None
            model = models[cls][t] = FieldGP(**config).fit(X, F, extra)
            detail = dict(n=len(X), modes=len(model.modes), transformation=model.transform,
                inputs=model.input_transform, kernel=str(model.gp.kernel_) if model.gp else 'constant',
                setting_squared_scaling=t == 'v2', signed_speed_reference=model.speed_reference,
                cfd_min=float(F.min()), cfd_max=float(F.max()), nonfinite_cells=int((~np.isfinite(F)).sum()))
            if cv:
                r2, pred = loo_r2(X, F, on_speed2=extra, **config)
                error = ((F - pred) ** 2).sum(1)
                variance = ((F - F.mean(1, keepdims=True)) ** 2).sum(1)
                case_r2 = 1 - error / np.maximum(variance, 1e-20)
                detail.update(loo_r2=float(r2), loo_rmse=float(np.sqrt(np.mean((F-pred)**2))),
                    case_loo_r2=dict(zip(names, case_r2.tolist())),
                    worst_case=names[int(np.argmin(case_r2))],
                    loo_min=float(pred.min()), loo_max=float(pred.max()),
                    passes_r2_0_9=bool(r2 >= .9))
                i = int(np.argmin(case_r2))
                if 1-case_r2[i] > worst[0]: worst = (1-case_r2[i], (cls, t, names[i], F[i], pred[i], case_r2[i]))
                report[cls][f'loo_r2_{t}'] = round(float(r2), 6)
                np.savez_compressed(os.path.join(ROOT, 'work', f'surrogate_loo_{cls}_{t}.npz'),
                                    X=X, truth=F, prediction=pred, cases=np.array(names))
            report[cls][f'modes_{t}'] = len(model.modes)
            report[cls]['targets'][t] = detail
            print(cls, t, detail.get('loo_r2'), 'modes', len(model.modes), flush=True)
        if worst[1] is not None: diagnostic.append(worst[1])
        report[cls]['data_checks'] = _data_checks(cls, X, Y, names)
        report[cls]['prediction_checks'] = _prediction_checks(cls, models[cls], X)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    pickle.dump(models, open(os.path.join(ROOT, 'results', 'surrogate.pkl'), 'wb'))
    json.dump(report, open(os.path.join(ROOT, 'results', 'surrogate_report.json'), 'w'), indent=1)
    if diagnostic: _diagnostic_plot(diagnostic)
    _write_report(report)
    return models, report


def _prediction_checks(cls, models, X):
    """Check the fitted interface at DOE points and 256 domain samples."""
    bounds = {'A':[(.2,3),(.4,1)], 'E':[(.2,3),(.4,1),(30,46),(15,80)],
              'D':[(.2,3),(.4,1),(16,24),(30,46)], 'M':[(.2,3),(30,46),(15,80)]}
    sample = np.vstack([X, devices.lhs(256,bounds[cls],43)])
    out = {}
    for target, model in models.items():
        pred = model.predict(sample)
        upper = 1. if target=='phiT' else 1.2 if target=='phiW' else np.inf
        finite = bool(np.isfinite(pred).all())
        good = finite and bool((pred>=0).all()) and bool((pred<=upper).all())
        if target=='dT':
            cap=np.array([max(0.,r[1]-min(25.,devices.twb(r[1],r[2]))) for r in sample])
            good=good and bool((pred<=cap[:,None]+1e-12).all())
        out[target]=dict(n_inputs=len(sample), shape=list(pred.shape), finite=finite,
                         min=float(pred.min()), max=float(pred.max()), physical_bounds_pass=good)
        if not good: raise ValueError(f'Prediction sanity failed: {cls}/{target}')
    return out


def _data_checks(cls, X, Y, names):
    """Audit raw truths, without deleting cases or clipping validation data."""
    checks = dict(bound_violations={}, snapshot_changes={}, residuals={}, background_v2={})
    yy, xx = np.meshgrid(post.GY, post.GX, indexing='ij')
    far = ((abs(yy-12) > 4) & (xx > 3)).ravel()
    for target, F in Y.items():
        upper = 1.2 if target in ('phiT', 'phiW') else np.inf
        checks['bound_violations'][target] = dict(
            negative_cells=int((F < 0).sum()), below_minus_0_001=int((F < -1e-3).sum()),
            above_1_2=int((F > upper).sum()) if np.isfinite(upper) else 0,
            affected_cases=[names[i] for i in range(len(X)) if (F[i] < -1e-3).any() or (F[i] > upper).any()])
    for i, name in enumerate(names):
        case = os.path.join(ROOT, 'cfd', 'devices', name + '_on')
        times = sorted(float(d) for d in os.listdir(case)
                       if d.replace('.', '', 1).isdigit() and float(d) > 0 and os.path.isdir(os.path.join(case, d)))
        now = _body_level(case)
        if len(times) >= 2:
            old = _body_level(case, time=f'{times[-2]:g}')
            dv = np.sqrt(np.mean((now['speed']-old['speed'])**2))
            checks['snapshot_changes'][name] = dict(times=times[-2:],
                speed_relative_rms=float(dv / max(np.sqrt(np.mean(now['speed']**2)), 1e-12)),
                temperature_rms_K=float(np.sqrt(np.mean((now['T']-old['T'])**2))),
                vapour_pressure_rms_hPa=float(np.sqrt(np.mean((now['e']-old['e'])**2))))
        log = os.path.join(case, 'log.foamRun')
        if os.path.exists(log):
            import re
            with open(log) as f: text = f.read()
            matches = re.findall(r'Solving for Ux, Initial residual = ([\deE.+-]+)', text)
            checks['residuals'][name] = float(matches[-1]) if matches else None
        else: checks['residuals'][name] = None
        checks['background_v2'][name] = dict(far_mean=float(Y['v2'][i, far].mean()),
            far_max=float(Y['v2'][i, far].max()), far_sum_share=float(Y['v2'][i, far].sum()/max(Y['v2'][i].sum(),1e-12)))
    checks['interpretation'] = (
        'Small negative temperature/delivery/humidity increments are numerical or output-rounding errors; '
        'raw truths remain in all LOO scores. Broad positive far-field v2 is sensitive to subtracting '
        'the nearest wind-scaled off reference; it is not by itself evidence of unconverged CFD. '
        'Snapshot changes and available final Ux initial residuals are listed separately. '
        'Null residual means the solver log was unavailable or lacked Ux records.')
    return checks


def _diagnostic_plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figstyle
    fig, axes = plt.subplots(len(rows), 3, figsize=(12.5, 3.1*len(rows)), squeeze=False,
                             layout='constrained')
    extent = (post.GX[0]-.25, post.GX[-1]+.25, post.GY[0]-.25, post.GY[-1]+.25)
    for ax, (cls, target, name, truth, pred, score) in zip(axes, rows):
        vmax = max(float(truth.max()), float(pred.max()), 1e-9)
        difference = pred-truth; lim = max(float(abs(difference).max()), 1e-9)
        for j, (values, title) in enumerate([(truth, 'CFD'), (pred, 'LOO prediction'), (difference, 'Prediction − CFD')]):
            im = ax[j].imshow(values.reshape(len(post.GY),len(post.GX)), origin='lower', extent=extent,
                vmin=-lim if j==2 else 0, vmax=lim if j==2 else vmax,
                cmap=figstyle.DIVERGING_CMAP if j==2 else figstyle.BLUE_CMAP, aspect='equal')
            ax[j].set_title(f'{title}: {target}', fontsize=11)
            ax[j].set_xlabel('Downwind distance (m)')
            fig.colorbar(im, ax=ax[j], shrink=.8, pad=.02)
        ax[0].set_ylabel(f'{cls}: {name}\nCase R² = {score:.3f}\nCrosswind distance (m)')
    fig.savefig(os.path.join(ROOT,'work','surrogate_check.png'),dpi=160)
    plt.close(fig)


def _write_report(report):
    lines = ['# Device-Field Surrogate Validation', '',
        'LOO R² is 1 − SSE/SST pooled over all cases and all grid cells in original physical units. '
        'Every fold refits input scaling, field mean, POD basis, and GP hyperparameters using only training cases. '
        'All completed cases are retained, and validation truths are never clipped. '
        'Configurations were selected through exploratory LOO comparisons, so these are model-selection scores, '
        'not an independent test-set estimate.', '',
        '| Class | Target | n | POD modes | LOO R² | Target transform | Inputs |',
        '|---|---|---:|---:|---:|---|---|']
    for cls, result in report.items():
        for target, detail in result.get('targets',{}).items():
            transform = detail['transformation']
            if detail['setting_squared_scaling']: transform += '; divide by setting²'
            if detail['signed_speed_reference']: transform += '; signed on-speed correction, restore off subtraction'
            lines.append(f"| {cls} | {target} | {detail['n']} | {detail['modes']} | {detail.get('loo_r2',float('nan')):.6f} | {transform} | {detail['inputs']} |")
    lines += ['', '## Model choices and physical constraints', '',
        '- Wind/setting represents the ambient-to-jet velocity ratio. Squared speed scales with setting². '
        'Absolute wind is also retained for the A/E/D speed corrections to capture Reynolds-number and off-reference effects.',
        '- D speed also uses (Ta+273.15)/(T_sup+273.15), the ideal-gas supply/ambient density ratio at equal pressure. '
        'Even without gravity, variable density affects jet momentum and background velocity. '
        'Including this ratio substantially improves the worst individual D speed case as well as its pooled score.',
        '- E and D delivery fractions remove passive supply amplitudes. Their temperature and humidity inputs '
        'are omitted from the delivery-shape model: gravity is disabled and the residual density effects are small in LOO. '
        'This is a validated approximation, not exact independence from supply conditions. '
        'Log(wind/setting) resolves the long low-wind footprint.',
        '- A/E/D speed fits (speed_on² − wind² × speed_off_at_1²)/setting² without positive clipping. '
        'Prediction restores the fixed-reference baseline, subtracts the nearest scaled off reference exactly, '
        'and applies max(.,0). Independent off-library fields are stored in the pickle. '
        'This removes a reference-dependent kink without changing targets_for or excluding data.',
        '- Square root for M cooling/humidification and asinh for the signed A/D speed corrections reduce peak dominance. '
        'Asinh preserves the correction sign and is linear near zero. '
        'M uses wind and ambient temperature; its weak RH dependence is assessed by the full LOO error. '
        'Other targets use the identity transform because transformed variants did not offer a necessary benefit.',
        '- All nonzero snapshot POD modes are kept (relative eigenvalue cutoff 1e-12, at most 32). '
        'This avoids conflating spatial truncation with interpolation error. A shared regularised GP kernel '
        'estimates a common input smoothness across modes, reducing the hyperparameter count relative to independent-mode fits. '
        'Kernel details are recorded per target in the JSON report. No parametric plume fit was needed.',
        '- Predictions are nonnegative. Thermal delivery is capped at 1, humidity delivery at 1.2, '
        'so E/D cooling cannot exceed the supply drop. M has no prescribed supply-air temperature; '
        'its conservative cooling cap is Ta − min(wet-bulb temperature, 25 °C injection-water temperature).',
        '- Adding density to E speed worsened validation and was rejected. '
        'Large extrapolation errors rejected unconstrained cubic/PCHIP interpolation. '
        'Full fold predictions and original truths are saved in work/surrogate_loo_<class>_<target>.npz.', '',
        '## Individual-case limitations', '',
        'Pooled R² does not guarantee the same accuracy in every case. The minimum individual-case spatial R² per target is:', '',
        '| Class | Target | Worst case | Case R² |', '|---|---|---|---:|']
    for cls,result in report.items():
        for target,detail in result.get('targets',{}).items():
            scores=detail.get('case_loo_r2',{})
            if scores:
                name=min(scores,key=scores.get)
                lines.append(f'| {cls} | {target} | {name} | {scores[name]:.6f} |')
    lines += ['', 'The small-footprint speed cases can have substantial relative errors even when the pooled field score is high. '
              'These limitations are visible in the diagnostic and must be retained when interpreting optimization results.', '',
              '## CFD checks', '']
    for cls, result in report.items():
        ck=result.get('data_checks',{}); changes=ck.get('snapshot_changes',{})
        if not changes: continue
        worst=max(changes,key=lambda n:changes[n]['speed_relative_rms'])
        temp=max(changes,key=lambda n:changes[n]['temperature_rms_K'])
        lines.append(f"- {cls}: maximum saved-snapshot speed RMS change = {100*changes[worst]['speed_relative_rms']:.5g}% ({worst}); "
                     f"maximum temperature RMS change = {changes[temp]['temperature_rms_K']:.5g} K ({temp}).")
        residuals=[v for v in ck['residuals'].values() if v is not None]
        if residuals:
            lines.append(f"  - Available final Ux initial residuals: {len(residuals)}/{result['n']} logs; maximum {max(residuals):.6g}.")
        for target, b in ck['bound_violations'].items():
            if b['negative_cells'] or b['above_1_2']:
                lines.append(f"  - {target}: {b['negative_cells']} negative cells; {b['below_minus_0_001']} below −0.001; "
                             f"{b['above_1_2']} above 1.2. Raw min/max: {result['targets'][target]['cfd_min']:.6g}, "
                             f"{result['targets'][target]['cfd_max']:.6g}.")
        backgrounds=ck['background_v2']; broad=[n for n,b in backgrounds.items() if b['far_sum_share']>.1]
        if broad: lines.append('  - Broad far-field positive v2 (>10% of integrated v2 outside |y−12|≤4 m, x>3 m): '+', '.join(broad)+'.')
        if result.get('omitted_without_done'):
            lines.append('  - Omitted because done.json was absent at collection: '+', '.join(result['omitted_without_done'])+'.')
    lines += ['', 'Broad positive speed increments require caution in superposition: subtraction of two nearly equal '
        'wind fields can magnify small background differences. They are preserved and modeled here. '
        'Two saved snapshots cannot exclude intervening oscillations, and missing solver logs limit residual verification. '
        'The JSON report records per-case changes/residuals and background contributions; no case was silently labeled converged or discarded.', '',
        'The diagnostic selects the lowest per-case spatial R² across all targets in each class. '
        'This is stricter than selecting the largest absolute-error field and can select a small-footprint endpoint case. '
        'The pooled acceptance threshold remains 0.9 per target.', '',
        'Physical bounds passed at all DOE inputs and 256 additional Latin-hypercube inputs per class. '
        'The JSON records the finite-value checks, output shapes, and sampled minima/maxima. '
        'Interpret these sampled checks within the evaluated input domain.', '']
    with open(os.path.join(ROOT,'results','surrogate_report.md'),'w',encoding='utf-8') as f: f.write('\n'.join(lines))


# ------------------------------------------------------------------ placement in the plaza
def place(field2d, slot_xy, flow_deg, cells_xy):
    """Sample a device-frame field at plaza cells for a device at slot_xy aimed along flow_deg (plaza-frame math degrees)."""
    from scipy.interpolate import RegularGridInterpolator
    a = math.radians(flow_deg)
    d = cells_xy - slot_xy
    xr = d[:, 0] * math.cos(a) + d[:, 1] * math.sin(a)
    yr = -d[:, 0] * math.sin(a) + d[:, 1] * math.cos(a) + 12.0
    f = RegularGridInterpolator((post.GY, post.GX), field2d, bounds_error=False, fill_value=0.0)
    return f(np.stack([yr, xr], 1))


if __name__ == '__main__':
    # Pickles must refer to surrogate.FieldGP even when launched as a script.
    import surrogate
    m, r = surrogate.fit_all(sys.argv[1] if len(sys.argv) > 1 else 'AEDM')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'cases'} for k, v in r.items()}, indent=1))
