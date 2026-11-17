"""Run the device design of experiments (gravity off).
python src/run_doe.py <regex> <workers> [endTime]
'On' cases for every DOE point; 'off' reference cases (device body, no jet) at w_b = 0.3, 1.0, 2.5 m/s per body type
(A and M share a body). With g = 0 and the ground at ambient temperature, the off-state temperature and humidity equal ambient,
and the off-state speed field is scaled linearly with w_b from the nearest reference."""
import sys, os, re, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import devices, runner

pat = re.compile(sys.argv[1] if len(sys.argv) > 1 else '.')
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 7
endTime = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
jobs = []
for body in ('A', 'E', 'D'):
    for wb in devices.OFF_REF:
        name = f'off_{body}_{wb:.1f}'
        if pat.match(name):
            k = dict(cls=body, w_b=wb, Ta=38.0, RH=40.0, on=False, endTime=endTime)
            jobs.append((name, devices.spec(**k), k))
for name, kw in devices.doe():
    if pat.match(name):
        k = dict(kw, on=True, endTime=endTime)
        jobs.append((f'{name}_on', devices.spec(**k), k))
t = time.time()
res = runner.run_many(jobs, workers=workers, sub='devices')
print('done', len(res), 'ok', sum(r.get('ok', False) for r in res), f'{time.time() - t:.0f}s')
