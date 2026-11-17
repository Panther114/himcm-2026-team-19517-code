"""Verification runs: mesh-independence study (one DOE point per class, cells x1.6 in each direction) and superposition test
(two devices in one domain, compared with the sum of single-device fields from the surrogate inputs).
python src/run_checks.py <mesh|super|all> <workers>"""
import sys, os, copy, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import devices, runner

MESH_POINTS = ['A_03', 'E_05', 'D_07', 'M_04']


def two_devices(cls, layout, **kw):
    """Two units of class cls: 'side' = 6 m apart across the wind, 'line' = second unit 6 m downwind of the first."""
    s = devices.spec(cls, **kw)
    s = copy.deepcopy(s)
    a = s['solids'][0]; x0, x1, y0, y1, z0, z1 = a['box']
    b = copy.deepcopy(a); b['name'] = 'dev_body2'
    if 'jet' in b: b['jet']['patch'] = 'dev_out2'
    if layout == 'side':
        b['box'] = (x0, x1, y0 + 6, y1 + 6, z0, z1)
        s['yb'] = sorted(set(s['yb'] + [y0 + 6, y1 + 6]))
    else:
        b['box'] = (x0 + 6, x1 + 6, y0, y1, z0, z1)
        s['xb'] = sorted(set(s['xb'] + [x0 + 6, x1 + 6]))
    # rebuild cell counts for the new breakpoints at about the original spacing
    def counts(bp, h):
        return [max(1, int(round((bp[i + 1] - bp[i]) / h(bp[i], bp[i + 1])))) for i in range(len(bp) - 1)]
    s['yn'] = counts(s['yb'], lambda a_, b_: 0.25 if b_ - a_ <= 2 else 0.75)
    s['xn'] = counts(s['xb'], lambda a_, b_: 0.2 if (a_ >= -0.5 and b_ <= 7) else (0.4 if b_ <= 10 else 0.8))
    s['solids'].append(b)
    if s.get('cloud'):
        c2 = copy.deepcopy(s['cloud'])
        s['cloud2'] = c2  # second injector is not supported by the generator: mist is not used in the superposition test
    return s


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    doe = dict(devices.doe())
    jobs = []
    if what in ('mesh', 'all'):
        for n in MESH_POINTS:
            k = dict(doe[n], on=True, endTime=1000, fine=1.6)
            jobs.append((f'mesh_{n}_on', devices.spec(**k), k))
        for body in ('A', 'E', 'D'):
            k = dict(cls=body, w_b=1.0, Ta=38.0, RH=40.0, on=False, endTime=1000, fine=1.6)
            jobs.append((f'mesh_off_{body}_1.0', devices.spec(**k), k))
    if what in ('super', 'all'):
        for cls in ('E', 'D'):
            for lay in ('side', 'line'):
                k = dict(w_b=1.0, Ta=40.0, RH=30.0, setting=1.0, T_sup=18.0, on=True, endTime=1000)
                jobs.append((f'super_{cls}_{lay}', two_devices(cls, lay, **k), dict(k, cls=cls, layout=lay)))
                k1 = dict(cls=cls, **k)
                jobs.append((f'super_{cls}_single', devices.spec(**k1), k1))
    seen = set(); uniq = []
    for j in jobs:
        if j[0] not in seen: seen.add(j[0]); uniq.append(j)
    t = time.time()
    res = runner.run_many(uniq, workers=workers, sub='checks')
    print('done', len(res), 'ok', sum(r.get('ok', False) for r in res), f'{time.time() - t:.0f}s')
