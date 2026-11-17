"""Solve already-meshed venue cases (potentialFoam initialisation + foamRun).  python src/run_venue_solve.py <workers>"""
import sys, os, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import venue_cfd as v
from runner import sh
order = [f'w{d:03d}_{r}' for d in (0, 315, 45, 90, 270, 135, 225, 180) for r in ('closed', 'open')]


def go(name):
    case = os.path.join(v.ROOT, 'cfd', 'venue', name); t = time.time(); out = {}
    for s in (('potentialFoam',), ('foamRun',)):
        rc, dt = sh(case, *s, log=f'log.{s[0]}'); out[s[0]] = [rc, round(dt)]
        if rc: break
    out['ok'] = out.get('foamRun', [1])[0] == 0
    json.dump(out, open(os.path.join(case, 'run_status.json'), 'w'))
    return name, out, time.time() - t


todo = [n for n in order if not os.path.exists(os.path.join(v.ROOT, 'cfd', 'venue', n, 'run_status.json'))]
with ThreadPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 5) as ex:
    for f in as_completed([ex.submit(go, n) for n in todo]):
        n, r, t = f.result(); print(time.strftime('%H:%M:%S'), n, r, f'{t / 60:.0f} min', flush=True)
