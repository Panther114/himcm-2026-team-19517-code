"""Continue every finished venue case from its last write to iteration 1,400 while averaging U over the continuation (fieldAverage),
because most cases oscillate (10-21 % local change between iterations 400 and 800; results/venue_convergence.json).
Cases still running are processed once their run_status.json reports ok.  python src/run_venue_average.py <workers>"""
import sys, os, re, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import venue_cfd as v
from runner import sh

END = 1400
FUNC = """
functions
{
    UAverage
    {
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        restartOnRestart true;
        fields          ( U { mean on; prime2Mean off; base iteration; } );
    }
}
"""
names = [f'w{d:03d}_{r}' for d in (0, 315, 45, 90, 270, 135, 225, 180) for r in ('closed', 'open')]


def ready(n):
    p = os.path.join(v.ROOT, 'cfd', 'venue', n, 'run_status.json')
    if not os.path.exists(p): return False
    s = json.load(open(p))
    return bool(s.get('ok')) and not s.get('averaged')


def go(n):
    case = os.path.join(v.ROOT, 'cfd', 'venue', n)
    cd = os.path.join(case, 'system', 'controlDict'); s = open(cd).read()
    s = re.sub(r'endTime \d+;', f'endTime {END};', s)
    s = re.sub(r'writeInterval \d+;', 'writeInterval 200;', s)
    s = s.replace('startFrom startTime;', 'startFrom latestTime;')
    if 'fieldAverage' not in s: s += FUNC
    open(cd, 'w', newline='\n').write(s)
    t = time.time(); rc, dt = sh(case, 'foamRun', log='log.foamRun.avg')
    st = json.load(open(os.path.join(case, 'run_status.json')))
    st.update(averaged=rc == 0, average_window=f'from the restart time to {END}', avg_rc=rc, avg_s=round(dt))
    json.dump(st, open(os.path.join(case, 'run_status.json'), 'w'))
    return n, rc, (time.time() - t) / 60


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    pending = list(names); running = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while pending or running:
            for n in list(pending):
                if len(running) < workers and ready(n):
                    running[ex.submit(go, n)] = n; pending.remove(n)
            done = [f for f in running if f.done()]
            for f in done:
                n, rc, m = f.result(); print(time.strftime('%H:%M:%S'), n, 'rc', rc, f'{m:.0f} min', flush=True); running.pop(f)
            time.sleep(30)
