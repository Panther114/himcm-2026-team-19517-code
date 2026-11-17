"""Re-optimise the cost-heat-burden front with all device fields scaled by a common effectiveness factor (validation suggests the
device CFD overestimates jet reach in calm air) and compare the p = 0.5 selection with the base case.
python src/run_eff_sensitivity.py <factor> [pop] [gens] [workers]   ->  results/pareto_KH_eff<factor>.json, results/eff_sensitivity_<factor>.json
Workers default to all CPU cores; each worker limits BLAS to one thread."""
import sys, os, json
import multiprocessing as mp
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import optimise, pipeline

if __name__ == '__main__':
    from pymoo.parallelization.starmap import StarmapParallelization
    from threadpoolctl import threadpool_limits
    _threads = threadpool_limits(limits=1)
    f = float(sys.argv[1]); pop = int(sys.argv[2]) if len(sys.argv) > 2 else 80; gens = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    ev = optimise.Evaluator()
    for k in ('dT', 'dW', 'v2'):
        ev.pre[k] = ev.pre[k] * f
    tag = f'{f:g}'.replace('.', 'p')
    workers = int(sys.argv[4]) if len(sys.argv)>4 else os.cpu_count() or 1
    with mp.get_context('spawn').Pool(workers,initializer=optimise.init_worker,initargs=(ev,)) as pool:
        out = optimise.nsga(ev, objectives=('K_usd', 'H'), pop=pop, gens=gens, seed=4204, safety_target=None,
                            runner=StarmapParallelization(pool.starmap),
                            out=os.path.join(optimise.ROOT, 'results', f'pareto_KH_eff{tag}.json'))
    D = out['designs']
    vals = np.array([[d['K_usd'], d['H']] for d in D])
    ideal, nadir = vals.min(0), vals.max(0)
    res = {}
    for p in (0.25, 0.5, 0.75):
        s = pipeline.scalarise(vals, p, ideal, nadir); d = D[int(np.argmin(s))]
        res[str(p)] = dict(K_usd=d['K_usd'], H=d['H'], P_safe=d['P_safe'], n_dev=d['n_dev'], mix={int(k): v for k, v in Counter(d['x']).items()})
    base = json.load(open(os.path.join(optimise.ROOT, 'results', 'selection.json')))['marked']['recommended']
    res['base_recommended'] = dict(K_usd=base['K_usd'], H=base['H'], P_safe=base['P_safe'], n_dev=base['n_dev'],
                                   mix={int(k): v for k, v in Counter(base['x']).items()})
    res['factor'] = f; res['front_size'] = len(D)
    json.dump(res, open(os.path.join(optimise.ROOT, 'results', f'eff_sensitivity_{tag}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
