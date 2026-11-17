"""Layer 4: design evaluation over the season (hour types) and multi-objective optimisation.

Design vector: x[s] in {0: none, 1: A, 2: M, 3: E, 4: D} for each candidate slot s, plus log10 of the comfort price mu (USD per
utility-person-hour) that sets how far the hourly dispatch goes beyond safety.
Measures (season, per person-hour where stated): P_safe (share of person-hours with AT <= AT_safe), H (mean squared exceedance, K^2 per
person-hour), U (mean utility per person-hour), E (kWh per season), W (m3 per season), K (USD per year: annualised capital + energy + water),
noise (max dBA at occupied cells), eta (utility-person-hours per kWh, Task 2a measure).
"""
import os, sys, json, math, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import climate, econ, effect
from precompute import CLASSES

ROOT = climate.ROOT
MU_LEVELS = 9            # comfort price levels: 0 (safety only) and 10^(-3 + 0.5 (L - 1)) USD per utility-person-hour for L = 1..8


def mu_of(level):
    L = int(round(float(level)))
    return 0.0 if L <= 0 else 10 ** (-3 + 0.5 * (L - 1))


class Evaluator:
    def __init__(self, base=None, pre=None, econ_par=None, rad_coef=1.0, fan_cutoff=35.0, eps=None, at_safe=climate.AT_SAFE, elasticity_scale=1.0):
        if base is None:
            import season
            base = season.build()
        self.base = base
        self.pre = pre if pre is not None else dict(np.load(os.path.join(ROOT, 'results', 'precomputed.npz')))
        self.E = dict(econ.ECON, **(econ_par or {}))
        self.reps, self.W = self.pre['reps'], self.pre['W']
        self.eps = eps            # None: per-cell elasticity from local density (effect.eps_cells)
        self.hts = []
        for h in self.reps:
            b = self.base
            nC = len(b['xy'])
            ht = effect.HourType(np.full(nC, b['T'][h]), np.full(nC, effect.w_of_e(b['e'][h])), b['w_eq'][h].astype(float),
                                 b['Q'][h].astype(float), b['sun'][h], b['n'][h].astype(float), 1.0, rad_coef=rad_coef,
                                 eps=(self.eps if self.eps is not None else effect.eps_cells(b['n'][h])) * elasticity_scale,
                                 Ta=b['T'][h], at_safe=at_safe)
            self.hts.append(ht)
        self.fan_cutoff = fan_cutoff
        self.cells = self.base['xy']; self.slots = self.base['slots']

    def evaluate(self, x, mu=0.0, mode='dispatch', detail=False, target=1.0):
        """mode: 'dispatch' (hourly optimal operation), 'always' (every installed device on whenever people are present)."""
        inst = [(s, int(c) - 1) for s, c in enumerate(x) if int(c) > 0]
        capex = sum(econ.annual_capex(CLASSES[ci], self.E) for _, ci in inst)
        tot = dict(P=0.0, safe=0.0, H=0.0, U=0.0, E=0.0, Wat=0.0, P0safe=0.0)
        on_hours = np.zeros(len(inst)); per_type = []
        for k, ht in enumerate(self.hts):
            Wk = self.W[k]
            fields = [(self.pre['dT'][k, s, ci].astype(float), self.pre['dW'][k, s, ci].astype(float), self.pre['v2'][k, s, ci].astype(float)) for s, ci in inst]
            caps = [float(self.pre['caps'][k, s, ci]) for s, ci in inst]
            rc = [self.pre['run_e'][k, ci] * self.E['tariff'] + self.pre['run_w'][k, ci] / 1000 * self.E['water'] for _, ci in inst]
            cut = [CLASSES[ci] == 'A' and ht.Ta >= self.fan_cutoff for _, ci in inst]
            if mode == 'dispatch':
                on, ev = effect.dispatch(ht, fields, rc, caps, mu, cut, target)
            else:
                on = np.array([(ht.n.sum() > 0) and not c for c in cut], bool)
                if len(inst):
                    sdT = sum(f[0] for f, o in zip(fields, on) if o) if on.any() else 0.0
                    sde = sum(f[1] for f, o in zip(fields, on) if o) if on.any() else 0.0
                    sv2 = sum(f[2] for f, o in zip(fields, on) if o) if on.any() else 0.0
                    ev = ht.evaluate(sdT, sde, sv2, max([c for c, o in zip(caps, on) if o], default=0.0))
                else:
                    ev = ht.evaluate()
            ev0 = ht.evaluate()
            tot['P'] += Wk * ev['P']; tot['safe'] += Wk * ev['safe']; tot['H'] += Wk * ev['H']; tot['U'] += Wk * ev['U']
            tot['P0safe'] += Wk * ev0['safe']
            for j, (s, ci) in enumerate(inst):
                if on[j]:
                    on_hours[j] += Wk
                    tot['E'] += Wk * self.pre['run_e'][k, ci]; tot['Wat'] += Wk * self.pre['run_w'][k, ci] / 1000
            if detail: per_type.append(dict(k=k, on=on.tolist(), safe=ev['safe'], P=ev['P'], H=ev['H'], U=ev['U']))
        K = capex + tot['E'] * self.E['tariff'] + tot['Wat'] * self.E['water']
        occupied = self.base['n'][self.reps].sum(0) > 0
        noise = float(econ.noise_field(self.slots[[s for s, _ in inst]], [CLASSES[ci] for _, ci in inst], self.cells)[occupied].max()) if inst and occupied.any() else 0.0
        P = max(tot['P'], 1e-9)
        res = dict(P_safe=tot['safe'] / P, P_safe0=tot['P0safe'] / P, H=tot['H'] / P, U=tot['U'] / P, E_kWh=tot['E'], W_m3=tot['Wat'],
                   K_usd=K, capex_usd=capex, CO2_t=tot['E'] * self.E['carbon'] / 1000, noise_dBA=noise,
                   eta=(tot['U'] / max(tot['E'], 1e-9)) if tot['E'] > 0 else 0.0, n_dev=len(inst), person_hours=P,
                   cost_per_protected_ph=K / max(tot['safe'] - tot['P0safe'], 1e-9))
        if detail: res['per_type'] = per_type; res['on_hours'] = on_hours.tolist()
        return res


_WORKER_EV = None


def init_worker(ev):
    """Install the large read-only evaluator once per Windows-spawn worker."""
    global _WORKER_EV
    _WORKER_EV = ev
    # Avoid one BLAS thread pool per process.
    from threadpoolctl import threadpool_limits
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


from pymoo.core.problem import ElementwiseProblem


class DesignProblem(ElementwiseProblem):
    """Picklable problem; parallel tasks carry no evaluator arrays."""
    def __init__(self, ev, objectives, noise_limit, safety_target, runner=None):
        S = len(ev.slots)
        kw = dict(elementwise_runner=runner) if runner is not None else {}
        super().__init__(n_var=S+1, n_obj=len(objectives), n_ieq_constr=1+(safety_target is not None),
                         xl=[0]*(S+1), xu=[4]*S+[MU_LEVELS-1], vtype=int, **kw)
        self.ev = ev if runner is None else None
        self.objectives, self.noise_limit, self.safety_target = objectives, noise_limit, safety_target

    def _evaluate(self, x, out, *args, **kwargs):
        ev = self.ev if self.ev is not None else _WORKER_EV
        r = ev.evaluate(np.rint(x[:-1]).astype(int), mu_of(x[-1]))
        sign = {'K_usd': 1, 'H': 1, 'P_safe': -1, 'U': -1, 'E_kWh': 1, 'W_m3': 1}
        out['F'] = [sign[o]*r[o] for o in self.objectives]
        out['G'] = [r['noise_dBA']-self.noise_limit]
        if self.safety_target is not None:
            out['G'].append(self.safety_target-r['P_safe'])
        out['metrics'] = r


def nondominated(designs, objectives=('K_usd', 'H')):
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    if not designs: return []
    F = np.array([[(-1 if o in ('P_safe', 'U') else 1)*r[o] for o in objectives] for r in designs])
    idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return [designs[i] for i in idx]


def nsga(ev, objectives=('K_usd', 'H'), pop=80, gens=200, seed=1, mu_range=(-3, 1), noise_limit=None,
         log_every=10, out=None, runner=None, safety_target=None):
    """NSGA-II with every evaluated design archived and fixed-reference HV.

    runner may be pymoo StarmapParallelization(pool.starmap), with pool
    initialized using init_worker(ev). mu_range is retained for compatibility;
    the actual discrete comfort prices are defined by mu_of.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize
    from pymoo.indicators.hv import HV
    nl = noise_limit if noise_limit is not None else ev.E['noise_limit']
    hist, archive = [], {}
    front_F = np.zeros((0,len(objectives)))
    def cb(alg):
        nonlocal front_F
        for ind in alg.off:
            xx = np.rint(ind.X).astype(int)
            archive[tuple(xx)] = dict(x=xx[:-1].tolist(), mu=mu_of(xx[-1]), **ind.get('metrics'))
        # Track the best feasible front found so far, matching the final
        # archived front. Crowding selection can discard objective duplicates
        # and temporarily lose front points from a small current population.
        feasible = alg.off.get('FEAS').ravel()
        F = np.unique(np.vstack([front_F,alg.off.get('F')[feasible]]),axis=0)
        if len(F):
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
        front_F = F
        hist.append(dict(gen=alg.n_gen, n_front=len(F), F=F.tolist()))
        if alg.n_gen % log_every == 0 or alg.n_gen == 1:
            print(time.strftime('%H:%M:%S'), 'gen', alg.n_gen, 'front', len(F), flush=True)
    S = len(ev.slots)
    rng = np.random.default_rng(seed)
    initial = rng.integers(0,5,size=(pop,S+1))
    initial[:,-1] = rng.integers(0,MU_LEVELS,size=pop)
    initial[0] = 0  # Always retain the no-device cost/noise reference.
    for i in range(1,min(pop,5)):
        initial[i] = 0
        initial[i,(i-1)%S] = i
    minimize(DesignProblem(ev, objectives, nl, safety_target, runner), NSGA2(pop_size=pop, sampling=initial, crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                                 mutation=PM(eta=20, vtype=float, repair=RoundingRepair()), eliminate_duplicates=True),
                   ('n_gen', gens), seed=seed, callback=cb, verbose=False, save_history=False)
    # hypervolume history against a common reference point
    available = [np.asarray(h['F']) for h in hist if h['F']]
    # A positive range-based margin also works for negative -P_safe values.
    allF = np.vstack(available) if available else np.array([[0.0]*len(objectives)])
    ref = allF.max(0) + 0.05*np.maximum(np.ptp(allF, axis=0), np.maximum(abs(allF.max(0)), 1.0))
    hv = HV(ref_point=ref)
    for h in hist: h['hv'] = float(hv(np.array(h['F']))) if h['F'] else 0.0
    change = None
    if len(hist) >= 21 and hist[-21]['hv'] > 0:
        change = float((max(h['hv'] for h in hist[-21:])-min(h['hv'] for h in hist[-21:]))/hist[-21]['hv'])
    evaluated = list(archive.values())
    noise_ok = [r for r in evaluated if r['noise_dBA'] <= nl+1e-9]
    feasible = [r for r in noise_ok if safety_target is None or r['P_safe'] >= safety_target-1e-9]
    # Preserve a reviewable attainable front when the safety target is infeasible.
    designs = nondominated(feasible or noise_ok, objectives)
    convergence = dict(converged=change is not None and change < .01, window_generations=20,
                       relative_change=change, criterion='HV range / HV at start < 0.01',
                       feasible_designs=len(feasible), safety_target=safety_target,
                       safety_target_feasible=bool(feasible), exploratory_fallback=not bool(feasible))
    out_d = dict(objectives=objectives, pop=pop, gens=gens, seed=seed, hv_reference=ref.tolist(), convergence=convergence,
                 history=[{k: v for k, v in h.items() if k != 'F'} for h in hist], designs=designs, evaluated=evaluated)
    print('Hypervolume:', [round(h['hv'], 6) for h in hist], 'convergence:', convergence, flush=True)
    if out:
        json.dump(out_d, open(out, 'w'), indent=1)
    return out_d
