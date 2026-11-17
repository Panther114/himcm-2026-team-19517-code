"""Run all venue wind cases: 8 directions x roof open/closed.  python src/run_venue.py <workers>"""
import sys, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import venue_cfd as v
sc = v.load_scene()
jobs = []
for d in (0, 315, 45, 90, 270, 135, 225, 180):       # worst-diverging first (check), then by frequency
    for roof in (True, False):
        name = f'w{d:03d}_{"closed" if roof else "open"}'
        case = os.path.join(v.ROOT, 'cfd', 'venue', name)
        if os.path.exists(os.path.join(case, 'run_status.json')):
            continue
        v.make_case(case, sc, wind_from_deg=d, roof=roof, endTime=800)
        jobs.append((name, case))
def go(nc):
    n, c = nc; t = time.time(); r = v.run_case(c); return n, r, time.time() - t
with ThreadPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 5) as ex:
    for f in as_completed([ex.submit(go, j) for j in jobs]):
        n, r, t = f.result(); print(time.strftime('%H:%M:%S'), n, r, f'{t/60:.0f} min', flush=True)
