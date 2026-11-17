"""Task 4 cached results pipeline (local clock, USD/year, person-hours).

Run from the project with .venv/Scripts/python src/pipeline.py STAGE.
STAGE is base, precompute, optimise, select, designday, season, robustness, candidates,
all, or test. Defaults: K=60, population=80, generations=200, Sobol N=256,
all available CPU cores, fixed independent NSGA seeds 4204 and 4205.
--force reruns the requested stage; otherwise manifests compare its parameters
and input-file signatures. Individual stages require their upstream caches;
all visits stages in dependency order. Changed dependencies invalidate caches.

Artifacts in results/:
  base: season_base.npz, directly from season.build (all 16 venue CFD cases).
  precompute: precomputed.npz, from precompute.run(K); surrogate.pkl required.
  optimise: pareto_KH.json, pareto_KP.json, including ALL evaluated designs,
    generation-by-generation fixed-reference hypervolume and convergence.
    KH enforces --safety-target (default .95) and noise; KP enforces noise.
    If no safe design is found, KH explicitly reports an exploratory noise-
    feasible front; no feasible noise design is a hard error at selection.
  select: pareto.json (exact figures.py interface), selection.json (dial,
    scalarisation, normalisation, knee, marginal protection costs and flags).
  designday: designday.json, maps.npz (figures.py interfaces).
  season: season.npz, season_summary.json; selected device fields in float32
    fields_season_<fingerprint>_*.npy, produced in --chunk-hours batches.
  robustness: robustness.json (figures.py interface), robustness_detail.json
    (Saltelli samples/responses, both Sobol orders and confidence intervals,
    scenario metrics, scalarised values, regret and approximation statement).

Selection uses augmented Chebyshev max((1-p)*cost_norm,p*H_norm) plus
1e-4 times the weighted sum. Normalisation uses the KH front's ideal/nadir.
The efficiency optimum searches all evaluations of BOTH runs, including
dominated/infeasible designs, as requested, and reports its feasibility.
Marginal cost divides annual cost change by additional protected person-hours;
zero/negative protection changes are labelled and have no finite unit cost.

Real-hour verification keeps the original baseline, occupancy and dispatch
physics. The design day is the hottest daily Tmax on a Qatar weekday (Sun--
Thu); the map is its hottest occupied evening (18--23) air-temperature hour.
Daily safety shares and seasonal totals use person-hour weights. Savings
are (constant-hourly)/constant, including safety differences; they do NOT
assert equal safety. Constant operation respects occupancy and fan cutoff.

Robustness is a conditional, fixed-delivery-field approximation, NOT new CFD:
T/RH/vapour, wind and occupancy are perturbed in memory on representative
hours; supply drops/humidity delivery are rescaled for the new weather,
resource rates/COP, roof radiation, safety threshold and elasticity update.
Wind does not reshape/reorient device plumes, medoids and roof schedules stay
fixed; device_eff scales dT, dW and v2; physical supply caps still apply. Solar/shadow geometry is
fixed, roof_tau scales closed-roof Q relative to baseline .15. The +2 C
scenario uses this same approximation on the 60 whole-season medoids.
The humid fortnight and extreme day instead use their actual hours, each
with weight one, with fields from the same precompute functions and Evaluator.
Regret benchmarks every typical-season KH-front design re-evaluated in each
scenario, under p=.5 with the fixed typical-season ideal/nadir. Regret is the
score minus the scenario front minimum, as a percentage of that front's score
range (zero for a degenerate range). The six dial/efficiency rows are retained;
the full-front minimax design is appended if not already among them.
Annualised capital is retained in short scenarios (annual investment choice).
Operating-only scores and regret are also reported for the short scenarios.

test creates a small analytic synthetic venue, 16 days of hourly weather,
surrogate predictors and caches ONLY under work/pipeline_test/. It executes
the production stages with tiny settings, exercises Windows spawn with two
workers, verifies numeric invariants and all figures.py interfaces, calls
figures.py's main entry point, and redirects its output to that test folder.
Synthetic results are illustrative software checks, never venue evidence.
No stage launches CFD or OpenFOAM. base/maps read existing CFD; test does not.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, timedelta
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from unittest.mock import patch

import numpy as np
import climate
import econ
import effect
import optimise
import precompute

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('base', 'precompute', 'optimise', 'select', 'designday', 'season', 'robustness', 'candidates')
PRIORITIES = (0., .25, .5, .75, 1.)
INPUTS = ['wind_scale', 'T_off', 'RH_off', 'occ_scale', 'tariff', 'cop_dc',
          'device_eff', 'roof_tau', 'AT_safe', 'elasticity_scale']
BOUNDS = [[.7,1.3],[-1.5,1.5],[-10,10],[.5,2.],[.036,.15],[3.,4.5],
          [.7,1.3],[.05,.30],[30.,34.],[.8,1.2]]
APPROXIMATION = __doc__.split('Robustness is ')[1].split('\ntest creates')[0]


def json_default(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, default=json_default, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k:z[k] for k in z.files}


def signature(path):
    path = Path(path)
    if not path.exists(): return None
    s = path.stat()
    return [str(path.resolve()), s.st_size, s.st_mtime_ns]


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=json_default).encode()).hexdigest()[:16]


def design_key(d):
    return tuple(d['x']), float(d['mu'])


def unique(designs):
    return list({design_key(d):d for d in designs}.values())


def scalarise(values, p=.5, ideal=None, nadir=None):
    v = np.asarray(values, float)
    lo = v.min(0) if ideal is None else np.asarray(ideal)
    hi = v.max(0) if nadir is None else np.asarray(nadir)
    z = (v-lo)/np.maximum(hi-lo, 1e-12)
    weighted = z*np.array([1-p,p])
    return weighted.max(1)+1e-4*weighted.sum(1)


def date_axis(base):
    if 'dates' in base: return np.asarray(base['dates']).astype(str)
    # The baseline interface has hour/dow but no timestamp: verify against
    # the same weather file, never infer dates from an arbitrary year.
    df = climate.load_weather()
    if len(df) != len(base['T']) or not np.array_equal(df['hour'], base['hour']) or not np.array_equal(df['dow'], base['dow']):
        raise ValueError('Cached baseline and source weather calendar differ')
    return df['time'].dt.strftime('%Y-%m-%d').to_numpy()


def make_ht(base, h):
    N = len(base['xy'])
    return effect.HourType(np.full(N,base['T'][h]), np.full(N,effect.w_of_e(base['e'][h])),
                           base['w_eq'][h], base['Q'][h], base['sun'][h], base['n'][h], Ta=base['T'][h])


def running(base, hours, E=None):
    E = econ.ECON if E is None else E
    rates = np.array([[econ.hourly_running(c,float(base['T'][h]),float(base['RH'][h]),E)
                       for c in precompute.CLASSES] for h in hours])
    return rates[:,:,0], rates[:,:,1]


class Pipeline:
    def __init__(self, args, root=ROOT, synthetic=False):
        self.a, self.root, self.synthetic = args, Path(root), synthetic
        self.folder = self.root/'results'
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, name): return self.folder/name
    def base(self): return load_npz(self.path('season_base.npz'))
    def pre(self): return load_npz(self.path('precomputed.npz'))
    def selected(self): return read_json(self.path('selection.json'))['marked']['recommended']

    def execute(self, stage):
        outputs = dict(base=['season_base.npz'], precompute=['precomputed.npz'],
                       optimise=['pareto_KH.json','pareto_KP.json'], select=['pareto.json','selection.json'],
                       designday=['designday.json','maps.npz'], season=['season.npz','season_summary.json'],
                       robustness=['robustness.json','robustness_detail.json'], candidates=['real_hour_candidates.json'])
        deps = dict(base=[],precompute=['season_base.npz','surrogate.pkl'],
                    optimise=['season_base.npz','precomputed.npz'], select=['pareto_KH.json','pareto_KP.json'],
                    designday=['selection.json','season_base.npz','surrogate.pkl'],
                    season=['selection.json','season_base.npz','surrogate.pkl'],
                    robustness=['selection.json','pareto_KH.json','season_base.npz','precomputed.npz','surrogate.pkl'],
                    candidates=['selection.json','robustness_detail.json','season_base.npz','surrogate.pkl'])
        missing = [n for n in deps[stage] if not self.path(n).exists()]
        if missing: raise FileNotFoundError(f'{stage}: missing upstream caches {missing}; run their stages first')
        cfg_keys = dict(base=[],precompute=['K'],optimise=['pop','gens','workers','safety_target'],select=['safety_target'],
                        designday=['chunk_hours'],season=['chunk_hours'],robustness=['N','workers'],candidates=['chunk_hours','workers'])
        cfg = {k:getattr(self.a,k) for k in cfg_keys[stage]}
        cfg['synthetic'] = self.synthetic
        stamp = dict(parameters=cfg, inputs={n:signature(self.path(n)) for n in deps[stage]},
                     code={n:signature(ROOT/'src'/n) for n in ['pipeline.py','effect.py','optimise.py','precompute.py','econ.py','season.py','climate.py','surrogate.py','venue.py']})
        if stage == 'base' and not self.synthetic:
            # Stat existing case outputs without calling any CFD utility.
            files = list((ROOT/'cfd'/'venue').glob('w*/[0-9]*/U*'))
            stamp['venue_fields'] = [signature(p) for p in sorted(files)]
            stamp['weather'] = signature(ROOT/'data'/'weather_doha_2025_JJAS.json')
            stamp['geometry'] = signature(ROOT/'data'/'osm_msheireb.json')
        manifest = self.path(stage+'.manifest.json')
        if not self.a.force and manifest.exists() and read_json(manifest).get('stamp') == stamp and all(self.path(n).exists() for n in outputs[stage]):
            print(stage, 'cached', flush=True); return
        print(stage, 'starting', flush=True); start = time.perf_counter()
        getattr(self, 'stage_'+stage)()
        write_json(manifest, dict(stamp=stamp, seconds=time.perf_counter()-start, outputs=outputs[stage]))
        print(stage, f'completed in {time.perf_counter()-start:.1f}s', flush=True)

    def stage_base(self):
        if self.synthetic:
            np.savez_compressed(self.path('season_base.npz'), **synthetic_base())
        else:
            import season as season_model
            # cache=False is intentional when the manifest says to rebuild.
            b = season_model.build(cache=False)
            np.savez_compressed(self.path('season_base.npz'), **b)

    def stage_precompute(self):
        import season as season_model
        if self.a.K > len(self.base()['T']): raise ValueError('K exceeds season hours')
        with patch.object(precompute,'ROOT',str(self.root)), patch.object(season_model,'build',self.base):
            precompute.run(self.a.K)

    def stage_optimise(self):
        from pymoo.parallelization.starmap import StarmapParallelization
        ev = optimise.Evaluator(self.base(), self.pre())
        @contextmanager
        def runner():
            if self.a.workers == 1:
                yield None
            else:
                with mp.get_context('spawn').Pool(self.a.workers, initializer=optimise.init_worker, initargs=(ev,)) as pool:
                    yield StarmapParallelization(pool.starmap)
        # The same worker pool serves both runs and initializes each evaluator once.
        with runner() as r:
            for objectives,name,seed,target in [(('K_usd','H'),'KH',4204,self.a.safety_target),
                                                 (('K_usd','P_safe'),'KP',4205,None)]:
                optimise.nsga(ev,objectives,pop=self.a.pop,gens=self.a.gens,seed=seed,runner=r,
                              safety_target=target,out=str(self.path('pareto_'+name+'.json')))

    def stage_select(self):
        kh, kp = [read_json(self.path('pareto_'+s+'.json')) for s in ('KH','KP')]
        front = sorted(unique(kh['designs']),key=lambda d:(d['K_usd'],d['H']))
        if not front: raise RuntimeError('No noise-feasible KH design; inspect pareto_KH.json')
        all_d = unique(kh['evaluated']+kp['evaluated'])
        efficiency = max((d for d in all_d if d['n_dev'] >= 1),key=lambda d:d['eta'])
        values = np.array([[d['K_usd'],d['H']] for d in front])
        ideal,nadir = values.min(0),values.max(0)
        z = (values-ideal)/np.maximum(nadir-ideal,1e-12)
        line = z[-1]-z[0]
        distances = np.abs(line[0]*(z[:,1]-z[0,1])-line[1]*(z[:,0]-z[0,0]))/max(np.linalg.norm(line),1e-12)
        knee = int(np.argmax(distances))
        dial = []
        for p in PRIORITIES:
            scores = scalarise(values,p,ideal,nadir)
            idx = int(np.argmin(scores))
            dial.append(dict(p=p,index=idx,score=scores[idx],design=front[idx]))
        marked = dict(efficiency_optimum=efficiency,knee=front[knee],recommended=dial[2]['design'])
        keys = {design_key(d) for d in front}
        other = [d for d in all_d if design_key(d) not in keys]
        # Evenly spaced deterministic sample of other evaluations, at most 400.
        other = [other[i] for i in np.linspace(0,len(other)-1,min(400,len(other)),dtype=int)] if other else []
        marginal = []
        for i,(a,b) in enumerate(zip(front,front[1:])):
            gain = b['P_safe']*b['person_hours']-a['P_safe']*a['person_hours']
            marginal.append(dict(from_index=i,to_index=i+1,delta_cost_usd=b['K_usd']-a['K_usd'],
                                 delta_protected_ph=gain,usd_per_additional_protected_ph=(b['K_usd']-a['K_usd'])/gain if gain>1e-9 else None,
                                 status='additional protection' if gain>1e-9 else 'no additional protection'))
        write_json(self.path('pareto.json'),dict(designs=front,dominated=other,marked=marked,history=kh['history']))
        write_json(self.path('selection.json'),dict(marked=marked,priority_dial=dial,
                   ideal=dict(zip(('K_usd','H'),ideal)),nadir=dict(zip(('K_usd','H'),nadir)),
                   scalarisation='max((1-p)*K_norm,p*H_norm)+1e-4*((1-p)*K_norm+p*H_norm)',
                   knee_distances=distances,marginal_protection=marginal,convergence=kh['convergence'],
                   efficiency_to_recommendation=dict(delta_cost_usd=marked['recommended']['K_usd']-efficiency['K_usd'],
                       delta_protected_person_hours=marked['recommended']['P_safe']*marked['recommended']['person_hours']-efficiency['P_safe']*efficiency['person_hours']),
                   efficiency_optimum_feasible=efficiency['noise_dBA']<=econ.ECON['noise_limit'] and efficiency['P_safe']>=self.a.safety_target,
                   evaluated_count=len(all_d),efficiency_scope='all evaluated designs from both NSGA runs, n_dev>=1'))

    def fields(self, base, hours, design):
        """Return only installed contributions (H,J,N), using actual-hour GPs."""
        inst = np.flatnonzero(np.asarray(design['x'])>0)
        codes = np.asarray(design['x'],int)[inst]
        slim = dict(base,slots=base['slots'][inst])
        if not hasattr(self, '_models'):
            self._models = pickle.loads(self.path('surrogate.pkl').read_bytes())
        models = self._models
        if not len(inst):
            shape = (len(hours),0,len(base['xy']))
            return [np.zeros(shape,np.float32) for _ in range(3)]+[np.zeros((len(hours),0),np.float32)]
        fields = precompute.device_fields(models,slim,hours,design=codes)
        j = np.arange(len(inst))
        return [a[:,j,codes-1] for a in fields]

    def field_cache(self, base, design):
        """Memory-map only installed fields; publish metadata after full write."""
        key = fingerprint([signature(self.path('season_base.npz')),signature(self.path('surrogate.pkl')),design['x'],
                           *[signature(ROOT/'src'/n) for n in ('precompute.py','surrogate.py','climate.py')]])
        prefix = 'fields_season_'+key
        names = ['dT','dW','v2','caps']
        files = [self.path(prefix+'_'+n+'.npy') for n in names]
        done = self.path(prefix+'.json')
        if not done.exists() or not all(f.exists() for f in files):
            H,N = len(base['T']),len(base['xy']); J = int(np.count_nonzero(design['x']))
            arrays = [np.lib.format.open_memmap(f,mode='w+',dtype='float32',shape=(H,J,N) if n!='caps' else (H,J)) for f,n in zip(files,names)]
            for h in range(0,H,self.a.chunk_hours):
                stop = min(h+self.a.chunk_hours,H)
                fields = self.fields(base,np.arange(h,stop),design)
                for a,v in zip(arrays,fields): a[h:stop] = v
                if h % (self.a.chunk_hours*10) == 0: print('real-hour fields',stop,'/',H,flush=True)
            for a in arrays: a.flush()
            del arrays
            write_json(done,dict(hours=H,installed=J,fields=names,complete=True))
        return [np.load(f,mmap_mode='r') for f in files]

    def simulate(self, base, hours, design, fields, mode='dispatch', capture=False):
        """Shared real-hour dispatch/constant simulation and resource accounting."""
        x = np.asarray(design['x'],int); inst = np.flatnonzero(x>0); codes = x[inst]-1
        re,rw = running(base,hours)
        capex = sum(econ.annual_capex(precompute.CLASSES[c]) for c in codes)
        rows, pictures = [], []
        for k,h in enumerate(hours):
            ht = make_ht(base,h)
            ff = [(fields[0][k,j],fields[1][k,j],fields[2][k,j]) for j in range(len(inst))]
            caps = fields[3][k]
            cut = (codes==0)&(base['T'][h]>=35.)
            costs = re[k,codes]*econ.ECON['tariff']+rw[k,codes]/1000*econ.ECON['water']
            if mode == 'dispatch': on,ev = effect.dispatch(ht,ff,costs,caps,design['mu'],cut)
            else:
                on = (~cut)&(ht.n.sum()>0)
                sums = [a[k,on].sum(0) for a in fields[:3]]
                ev = ht.evaluate(*sums, float(max(caps[on],default=0.)))
            e = float(re[k,codes][on].sum()); w = float(rw[k,codes][on].sum()/1000)
            ev0 = ht.evaluate()
            row = dict(P=float(ev['P']),safe=float(ev['safe']),H=float(ev['H']),U=float(ev['U']),
                       E=e,W=w,safe0=float(ev0['safe']),on_counts=[int(np.sum(on&(codes==c))) for c in range(4)])
            if capture:
                for name,AT in [('base',ht.AT0),('cooled',ev['AT'])]:
                    row['AT_'+name+'_mean'] = float(np.average(AT,weights=ht.n)) if ht.n.sum() else float(np.mean(AT))
                    order = np.argsort(AT); weights = ht.n[order]
                    row['AT_'+name+'_p90'] = float(AT[order][min(np.searchsorted(np.cumsum(weights),.9*weights.sum()),len(order)-1)]) if weights.sum() else float(np.percentile(AT,90))
                pictures.append((ht.AT0,ev['AT']))
            rows.append(row)
        totals = {n:sum(r[n] for r in rows) for n in ('P','safe','H','U','E','W','safe0')}
        P = max(totals['P'],1e-9); E = totals['E']
        summary = dict(P_safe=totals['safe']/P,H=totals['H']/P,U=totals['U']/P,E_kWh=E,W_m3=totals['W'],
                       K_usd=capex+E*econ.ECON['tariff']+totals['W']*econ.ECON['water'],capex_usd=capex,
                       CO2=E*econ.ECON['carbon']/1000,CO2_units='tonnes',eta=totals['U']/E if E>0 else 0.,
                       person_hours=totals['P'],P_safe0=totals['safe0']/P)
        return summary,rows,pictures

    def stage_designday(self):
        b,d = self.base(),self.selected(); dates = date_axis(b)
        days = np.unique(dates)
        weekdays = [day for day in days if b['dow'][np.flatnonzero(dates==day)[0]] not in (4,5)]
        day = max(weekdays,key=lambda day:np.max(b['T'][dates==day]))
        hours = np.flatnonzero(dates==day)
        if len(hours)!=24 or not np.array_equal(b['hour'][hours],np.arange(24)): raise ValueError('Design day must have 24 real hours')
        fields = self.fields(b,hours,d)
        summary,rows,pics = self.simulate(b,hours,d,fields,capture=True)
        out = dict(date=day,hours=b['hour'][hours],T=b['T'][hours],RH=b['RH'][hours],w10=b['w10'][hours],
                   roof_closed=b['roof'][hours],occupancy=b['n'][hours].sum(1),
                   on_counts={c:[r['on_counts'][i] for r in rows] for i,c in enumerate(precompute.CLASSES)},energy_kWh=[r['E'] for r in rows])
        for name in ('AT_base_mean','AT_base_p90','AT_cooled_mean','AT_cooled_p90'): out[name] = [r[name] for r in rows]
        write_json(self.path('designday.json'),out)
        evening = [i for i,h in enumerate(hours) if b['hour'][h]>=18 and b['n'][h].sum()>0]
        k = max(evening,key=lambda i:b['T'][hours[i]])
        if self.synthetic:
            poly,buildings,angle,ratio = synthetic_scene(b)
        else:
            import venue
            sc = venue.build()
            poly = np.asarray(sc['plaza'].exterior.coords)
            buildings = [np.asarray(bb['poly'].exterior.coords) for bb in sc['buildings']]
            angle = sc['angle_deg']
            import venue_cfd
            ratio = venue_cfd.sample_case(str(ROOT/'cfd'/'venue'/'w315_closed'),sc)['ratio']
        obj = np.empty(len(buildings),object)
        for i,p in enumerate(buildings): obj[i] = p
        np.savez_compressed(self.path('maps.npz'),xy=b['xy'],plaza_poly=poly,buildings=obj,slots=b['slots'],
                            design=d['x'],wind_ratio=ratio,AT_base=pics[k][0],AT_cooled=pics[k][1],
                            plaza_angle_deg=angle,hour_index=hours[k],date=day,hour=b['hour'][hours[k]])

    def stage_season(self):
        b,d = self.base(),self.selected(); hours = np.arange(len(b['T'])); dates = date_axis(b)
        days = np.unique(dates)
        if len(hours)!=24*len(days) or not np.array_equal(b['hour'],np.tile(np.arange(24),len(days))):
            raise ValueError('Season must contain complete consecutive local days')
        fields = self.field_cache(b,d)
        dispatch,dr,_ = self.simulate(b,hours,d,fields)
        constant,cr,_ = self.simulate(b,hours,d,fields,mode='always')
        saved = dict(days=days)
        for name,rows in [('dispatch',dr),('constant',cr)]:
            saved[name+'_kWh'] = np.array([r['E'] for r in rows]).reshape(-1,24)
            safe = np.array([r['safe'] for r in rows]).reshape(-1,24).sum(1)
            people = np.array([r['P'] for r in rows]).reshape(-1,24).sum(1)
            saved['P_safe_'+name] = np.divide(safe,people,out=np.zeros_like(safe),where=people>0)
        np.savez_compressed(self.path('season.npz'),**saved)
        metrics = ('P_safe','H','U','E_kWh','W_m3','K_usd','CO2','eta')
        savings = {k:100*(constant[k]-dispatch[k])/constant[k] if abs(constant[k])>1e-12 else None for k in metrics}
        write_json(self.path('season_summary.json'),dict(dispatch=dispatch,constant=constant,savings_percent=savings,
                   savings_definition='100*(constant-dispatch)/constant; null if denominator zero; negative means hourly is larger',
                   protection_difference_pp=100*(dispatch['P_safe']-constant['P_safe']),hours=len(hours),
                   note='Different safety outcomes are reported; savings do not imply equal safety. CO2 in tonnes.'))

    def stage_robustness(self):
        from SALib.sample import sobol as sample_sobol
        from SALib.analyze import sobol as analyze_sobol
        b,pre,d = self.base(),self.pre(),self.selected()
        problem = dict(num_vars=len(INPUTS),names=INPUTS,bounds=BOUNDS)
        samples = sample_sobol.sample(problem,self.a.N,calc_second_order=False,seed=4206)
        # Each worker retains only the recommended installed device slots.
        active = np.flatnonzero(np.asarray(d['x'])>0)
        slim_b = dict(b,slots=b['slots'][active])
        slim_pre = dict(pre,**{n:pre[n][:,active] for n in ('dT','dW','v2','caps')})
        slim_d = dict(d,x=np.asarray(d['x'])[active].tolist())
        if self.a.workers>1:
            with mp.get_context('spawn').Pool(self.a.workers,initializer=init_sensitivity,initargs=(slim_b,slim_pre,slim_d)) as pool:
                response = list(pool.imap(sensitivity_one,samples,chunksize=8))
        else:
            init_sensitivity(slim_b,slim_pre,slim_d)
            response = [sensitivity_one(s) for s in samples]
        Y = np.asarray(response); outputs = ['K_usd','P_safe','H']
        sobol = dict(outputs=outputs,inputs=INPUTS,S1=[],ST=[],ST_conf=[],S1_conf=[])
        constant_outputs = []
        for i,name in enumerate(outputs):
            if np.ptp(Y[:,i]) <= 1e-12:
                result = {n:np.zeros(len(INPUTS)) for n in ('S1','ST','ST_conf','S1_conf')}; constant_outputs.append(name)
            else:
                # SALib 1.5.1 passes names to pandas.unique; pandas 3 requires
                # an ndarray rather than a plain list. Keep sampling/JSON plain.
                analysis_problem = dict(problem,names=np.asarray(INPUTS,dtype=object))
                result = analyze_sobol.analyze(analysis_problem,Y[:,i],calc_second_order=False,seed=4207,print_to_console=False)
            for n in ('S1','ST','ST_conf','S1_conf'): sobol[n].append(result[n])
        selection = read_json(self.path('selection.json'))
        designs = [v['design'] for v in selection['priority_dial']]+[selection['marked']['efficiency_optimum']]
        names = ['p=0','p=.25','p=.5','p=.75','p=1','Efficiency']
        front = sorted(unique(read_json(self.path('pareto_KH.json'))['designs']),key=lambda v:(v['K_usd'],v['H']))
        if not front: raise RuntimeError('Regret requires a nonempty typical-season KH front')
        ideal = [selection['ideal'][k] for k in ('K_usd','H')]
        nadir = [selection['nadir'][k] for k in ('K_usd','H')]
        # Evaluate duplicate dial/front designs only once per scenario.
        benchmark = unique(front+designs)
        indices = {design_key(v):i for i,v in enumerate(benchmark)}
        front_idx = [indices[design_key(v)] for v in front]
        candidate_idx = [indices[design_key(v)] for v in designs]
        dates = date_axis(b); days = np.unique(dates)
        daily_rh = np.array([b['RH'][dates==day].mean() for day in days])
        width = min(14,len(days)); start = int(np.argmax(np.convolve(daily_rh,np.ones(width)/width,mode='valid')))
        humid_days = days[start:start+width]
        hottest = max(days,key=lambda day:b['T'][dates==day].max())
        masks = [np.ones(len(dates),bool),np.isin(dates,humid_days),dates==hottest,np.ones(len(dates),bool)]
        scenario_names = ['Typical','Humid 14 d','Extreme heat','Season +2 C']
        benchmark_metrics = []; benchmark_values = np.zeros((len(benchmark),4))
        for j,mask in enumerate(masks):
            if j in (1,2):
                folder = self.actual_pre(b,np.flatnonzero(mask),j)
                if self.a.workers>1:
                    with mp.get_context('spawn').Pool(self.a.workers,initializer=init_scenario,initargs=(b,folder)) as pool:
                        mm = list(pool.imap(scenario_one,benchmark,chunksize=2))
                else:
                    init_scenario(b,folder)
                    mm = [scenario_one(v) for v in benchmark]
            else:
                ev = perturbed_evaluator(b,pre,{'T_off':2.} if j==3 else {})
                mm = [ev.evaluate(v['x'],v['mu']) for v in benchmark]
            for r in mm: r['operating_usd'] = r['K_usd']-r['capex_usd']
            benchmark_metrics.append(mm)
            val = np.array([[r['K_usd'],r['H']] for r in mm])
            benchmark_values[:,j] = scalarise(val,.5,ideal,nadir)
        front_values = benchmark_values[front_idx]
        minima = front_values.min(0); ranges = np.ptp(front_values,axis=0)
        benchmark_regret = np.divide(100*(benchmark_values-minima),ranges,
                                    out=np.zeros_like(benchmark_values),where=ranges>1e-12)
        front_regret = benchmark_regret[front_idx]
        robust_front = int(np.argmin(front_regret.max(1)))
        robust_key = design_key(front[robust_front])
        matches = [i for i,v in enumerate(designs) if design_key(v)==robust_key]
        if matches:
            robust = matches[0]
        else:
            robust = len(designs)
            designs.append(front[robust_front]); names.append(f'Minimax (KH {robust_front})')
            candidate_idx.append(front_idx[robust_front])
        values = benchmark_values[candidate_idx]; regret = benchmark_regret[candidate_idx]
        metrics = [[mm[i] for i in candidate_idx] for mm in benchmark_metrics]
        operating = []
        for j in (1,2):
            mm = benchmark_metrics[j]
            scores = scalarise([[r['operating_usd'],r['H']] for r in mm],.5,ideal,nadir)
            fs = scores[front_idx]; span = np.ptp(fs)
            regrets = 100*(scores-fs.min())/span if span>1e-12 else np.zeros_like(scores)
            operating.append(dict(scenario=scenario_names[j],normalisation='same fixed typical-season ideal/nadir; cost excludes capital and maintenance',
                                  front_scalarised_values=fs,front_regret_percent=regrets[front_idx],
                                  scalarised_values=scores[candidate_idx],regret_percent=regrets[candidate_idx],
                                  optimum_front_index=int(np.argmin(fs)),scenario_minimum=fs.min(),scenario_range=span))
        write_json(self.path('robustness.json'),dict(sobol={k:v for k,v in sobol.items() if k!='S1_conf'},scenarios=scenario_names,
                   designs=names,regret=regret,robust_choice=names[robust],
                   regret_units='percent of scenario KH-front scalarised range',robust_front_index=robust_front))
        write_json(self.path('robustness_detail.json'),dict(problem=problem,N=self.a.N,sampling='scrambled Sobol Saltelli, no second order',
                   sample_count=len(samples),samples=samples,responses=Y,sobol=sobol,constant_outputs=constant_outputs,
                   approximation=APPROXIMATION,scenario_names=scenario_names,scenario_metrics=metrics,designs=designs,
                   scenario_definitions=['Typical full season','Fourteen consecutive days with highest mean relative humidity',
                                         'Day with highest daily maximum air temperature','Full season with air temperature increased by 2 K'],
                   design_names=names,scalarised_values=values,regret_percent=regret,robust_choice=names[robust],
                   maximum_regret_percent=regret.max(1),scalarisation_priority=.5,
                   typical_ideal=ideal,typical_nadir=nadir,scenario_minimum=minima,scenario_range=ranges,
                   front_designs=front,front_scalarised_values=front_values,front_regret_percent=front_regret,
                   front_scenario_metrics=[[mm[i] for i in front_idx] for mm in benchmark_metrics],
                   front_maximum_regret_percent=front_regret.max(1),robust_front_index=robust_front,
                   robust_design=front[robust_front],
                   regret_definition='100*(score - min(front scores))/ptp(front scores); zero when range <= 1e-12',
                   humid_dates=humid_days,extreme_heat_date=hottest,
                   scenario_hours=[int(m.sum()) for m in masks],normalisation='fixed typical-season KH-front ideal/nadir',
                   scenario_representation=['60 whole-season medoids','actual scenario hours, unit weights',
                                            'actual scenario hours, unit weights','60 whole-season medoids with +2 K offset'],
                   scenario_representative_hours=[pre['reps'],np.flatnonzero(masks[1]),np.flatnonzero(masks[2]),pre['reps']],
                   scenario_weights=[pre['W'],np.ones(masks[1].sum()),np.ones(masks[2].sum()),pre['W']],
                   short_scenario_cost='Full annualised capital plus resource cost for scenario hours; operating-only secondary comparison also saved',
                   short_scenario_operating_only=operating))

    def actual_pre(self, base, hours, scenario):
        """Unit-weight actual hours; memory maps share delivery arrays across workers."""
        folder = self.root/'work'/'scenarios'/f'scenario_{scenario}'
        folder.mkdir(parents=True,exist_ok=True)
        if not hasattr(self,'_models'): self._models = pickle.loads(self.path('surrogate.pkl').read_bytes())
        shape = (len(hours),len(base['slots']),4,len(base['xy']))
        arrays = {n:np.lib.format.open_memmap(folder/(n+'.npy'),mode='w+',dtype='float32',
                  shape=shape[:-1] if n=='caps' else shape) for n in ('dT','dW','v2','caps')}
        for k in range(0,len(hours),self.a.chunk_hours):
            stop = min(k+self.a.chunk_hours,len(hours))
            fields = precompute.device_fields(self._models,base,hours[k:stop])
            for n,a in zip(arrays,fields): arrays[n][k:stop] = a
        for a in arrays.values(): a.flush()
        re,rw = running(base,hours)
        for n,a in dict(reps=hours,W=np.ones(len(hours)),run_e=re,run_w=rw).items(): np.save(folder/(n+'.npy'),a)
        return folder

    def stage_candidates(self):
        selection = read_json(self.path('selection.json'))
        marked = selection['marked']
        names = ['Balanced p=0.5','Geometric knee','Efficiency optimum','Minimax regret','p=0.25','p=0.75','p=1.0','Previous eight-E']
        previous = np.zeros(len(self.base()['slots']),int)
        previous[[1,2,4,6,7,13,15,19] if not self.synthetic else list(range(len(previous)))] = 3
        designs = [marked['recommended'],marked['knee'],marked['efficiency_optimum'],
                   read_json(self.path('robustness_detail.json'))['robust_design'],
                   *[selection['priority_dial'][i]['design'] for i in (1,3,4)],dict(x=previous.tolist(),mu=0.)]
        distinct = unique(designs)
        tasks = [(self.a,self.root,self.synthetic,d) for d in distinct]
        if self.a.workers>1:
            with mp.get_context('spawn').Pool(self.a.workers) as pool: results = pool.map(real_candidate_one,tasks)
        else: results = [real_candidate_one(t) for t in tasks]
        lookup = {design_key(d):r for d,r in zip(distinct,results)}
        rows = [dict(name=n,x=d['x'],mu=d['mu'],**lookup[design_key(d)]) for n,d in zip(names,designs)]
        values = np.array([[r['K_usd'],r['H']] for r in rows])
        ideal,nadir = values.min(0),values.max(0)
        scores = scalarise(values,.5,ideal,nadir)
        for r,s in zip(rows,scores): r['score_p05'] = s
        write_json(self.path('real_hour_candidates.json'),dict(hours=len(self.base()['T']),candidates=rows,
                   priority=.5,ideal=dict(zip(('K_usd','H'),ideal)),nadir=dict(zip(('K_usd','H'),nadir)),
                   normalisation='common componentwise minimum and maximum of (annual K, H) over the eight named real-hour candidates; no clipping',
                   scalarisation=selection['scalarisation'],preferred=names[int(np.argmin(scores))],
                   note='Finite candidate comparison, not a real-hour global optimisation; annual capital plus season operating cost.'))


def perturbed_evaluator(base, pre, params):
    """Fast conditional sensitivity model; never calls season.build or CFD."""
    import surrogate
    p = dict(wind_scale=1.,T_off=0.,RH_off=0.,occ_scale=1.,device_eff=1.,roof_tau=.15,
             AT_safe=32.,elasticity_scale=1.,tariff=econ.ECON['tariff'],cop_dc=econ.ECON['cop_dc'])
    p.update(params)
    # Compress the baseline to medoids before copying spatial arrays.
    reps = pre['reps'].astype(int)
    b = {n:(a[reps].copy() if n not in ('xy','slots') and hasattr(a,'shape') and a.ndim>0 and len(a)==len(base['T']) else a)
         for n,a in base.items()}
    b['T'] = b['T']+p['T_off']; b['RH'] = np.clip(b['RH']+p['RH_off'],1,100)
    b['e'] = climate.vapour(b['T'],b['RH'])
    for n in ('w_eq','wb','w10'): b[n] = b[n]*p['wind_scale']
    b['n'] = b['n']*p['occ_scale']; b['Q'][b['roof']] *= p['roof_tau']/.15
    pp = dict(pre,reps=np.arange(len(reps)))
    # Rescale delivery using supply-state changes, without refitting GPs.
    factors_t = np.ones((len(reps),4)); factors_w = np.ones_like(factors_t)
    for k,h in enumerate(reps):
        for ci,c in enumerate(precompute.CLASSES):
            if c in ('E','D'):
                old = surrogate.supply_state(c,dict(Ta=float(base['T'][h]),RH=float(base['RH'][h]),T_sup=18.))
                new = surrogate.supply_state(c,dict(Ta=float(b['T'][k]),RH=float(b['RH'][k]),T_sup=18.))
                factors_t[k,ci] = new[0]/old[0] if abs(old[0])>1e-9 else 0.
                factors_w[k,ci] = new[1]/old[1] if abs(old[1])>1e-9 else 0.
            elif c=='M':
                old = max(float(base['T'][h])-min(25.,surrogate.devices.twb(float(base['T'][h]),float(base['RH'][h]))),0.)
                new = max(float(b['T'][k])-min(25.,surrogate.devices.twb(float(b['T'][k]),float(b['RH'][k]))),0.)
                factors_t[k,ci] = factors_w[k,ci] = new/old if old>1e-9 else 0.
    eff = p['device_eff']
    pp['dT'] = pre['dT']*factors_t[:,None,:,None]*eff
    pp['dW'] = pre['dW']*factors_w[:,None,:,None]*eff
    pp['v2'] = pre['v2']*eff
    pp['caps'] = pre['caps']*factors_t[:,None,:]
    E = dict(econ.ECON,tariff=p['tariff'],cop_dc=p['cop_dc'])
    pp['run_e'],pp['run_w'] = running(b,np.arange(len(reps)),E)
    return optimise.Evaluator(b,pp,econ_par=E,at_safe=p['AT_safe'],elasticity_scale=p['elasticity_scale'])


_SENSITIVITY = None


def init_scenario(base, folder):
    pre = {p.stem:np.load(p,mmap_mode='r') for p in Path(folder).glob('*.npy')}
    optimise.init_worker(optimise.Evaluator(base,pre))


def scenario_one(design):
    return optimise._WORKER_EV.evaluate(design['x'],design['mu'])


def real_candidate_one(task):
    from threadpoolctl import threadpool_limits
    args,root,synthetic,design = task
    with threadpool_limits(limits=1):
        pipe = Pipeline(args,root,synthetic)
        base = pipe.base()
        result,_,_ = pipe.simulate(base,np.arange(len(base['T'])),design,pipe.field_cache(base,design))
    return result


def init_sensitivity(base, pre, design):
    global _SENSITIVITY, _SENSITIVITY_THREADS
    from threadpoolctl import threadpool_limits
    _SENSITIVITY_THREADS = threadpool_limits(limits=1)
    _SENSITIVITY = base,pre,design


def sensitivity_one(sample):
    b,pre,d = _SENSITIVITY
    ev = perturbed_evaluator(b,pre,dict(zip(INPUTS,sample)))
    r = ev.evaluate(d['x'],d['mu'])
    return [r[n] for n in ('K_usd','P_safe','H')]


class SyntheticGP:
    """Analytic fake field predictor exercising production batching/placement."""
    def __init__(self, cls, target): self.cls,self.target = cls,target
    def predict(self, X):
        import surrogate
        gx,gy = np.meshgrid(surrogate.post.GX,surrogate.post.GY)
        plume = np.exp(-((gx-3)/10)**2-((gy-12)/7)**2).ravel()
        amplitude = dict(v2=5.,phiT=.6,phiW=.6,dT=3.,de=1.)[self.target]
        return np.broadcast_to(plume*amplitude,(len(X),len(plume))).copy()


def synthetic_base():
    import venue
    H = 16*24; t = np.arange(H); hour = t%24; day = t//24
    gx,gy = np.meshgrid(np.arange(-39,40,2),np.arange(-5,6,2))
    xy = np.column_stack([gx.ravel(),gy.ravel()]); N = len(xy)
    dates = np.array([(date(2025,6,1)+timedelta(days=int(d))).isoformat() for d in day])
    dow = np.array([date.fromisoformat(d).weekday() for d in dates])
    T = 31.+5*np.cos(2*np.pi*(hour-15)/24)+.12*day
    RH = 47.-13*np.cos(2*np.pi*(hour-15)/24)+3*np.sin(day/4)
    w10 = 3.+.5*np.sin(hour/24*2*np.pi)
    ratio = .3+.08*np.cos(xy[:,0]/12)
    wb = w10[:,None]*ratio[None,:]; w_eq = wb/climate.PROFILE_BODY
    roof = (hour>=7)&(hour<19)
    Q = np.broadcast_to((roof*24.)[:,None],(H,N)).copy()
    n = np.array([venue.occupancy_total(int(h),int(d)) for h,d in zip(hour,dow)])[:,None]*np.ones((1,N))/N
    # Include a genuinely empty hour to exercise constant-operation occupancy.
    n[hour==4] = 0.
    return dict(T=T,RH=RH,e=climate.vapour(T,RH),w10=w10,wb=wb,w_eq=w_eq,
                wdir_loc=np.zeros((H,N)),Q=Q,sun=np.broadcast_to(roof[:,None],(H,N)).copy(),
                n=n,roof=roof,hour=hour,dow=dow,dates=dates,xy=xy,slots=np.array([[-25.,-3.],[0.,3.],[25.,-3.]]))


def synthetic_scene(base):
    poly = np.array([[-40.,-6.],[40.,-6.],[40.,6.],[-40.,6.]])
    buildings = [np.array([[-43.,y],[43.,y],[43.,y+3],[-43.,y+3]]) for y in (-10.,7.)]
    return poly,buildings,16.3,.3+.08*np.cos(base['xy'][:,0]/12)


def run_test(args):
    import surrogate
    from threadpoolctl import threadpool_limits
    root = ROOT/'work'/'pipeline_test'; folder = root/'results'; folder.mkdir(parents=True,exist_ok=True)
    # Import by canonical name so the synthetic pickle also reloads in a new CLI.
    import pipeline as canonical
    models = {c:{t:canonical.SyntheticGP(c,t) for t in surrogate.TARGETS[c]} for c in precompute.CLASSES}
    (folder/'surrogate.pkl').write_bytes(pickle.dumps(models))
    write_json(folder/'surrogate_report.json',{c:dict(n=4,targets={t:dict(loo_r2=.99) for t in surrogate.TARGETS[c]}) for c in precompute.CLASSES})
    write_json(folder/'venue_convergence.json',dict(cases={f'synthetic_{i}':{} for i in range(16)}))
    a = argparse.Namespace(**vars(args)); a.K=4; a.pop=12; a.gens=4; a.N=8; a.workers=2; a.chunk_hours=24; a.force=True
    pipe = Pipeline(a,root,synthetic=True)
    with threadpool_limits(limits=1):
        for stage in STAGES: pipe.execute(stage)
        verify_test(pipe)
        import figures
        figures.fs.FIGURES_DIR = root/'figures'
        heading = figures.heading
        with patch.object(sys,'argv',['figures.py',str(folder)]), patch.object(figures,'heading',lambda fig,title,result,synthetic:heading(fig,title,result,True)):
            figures.main()
    files = sorted((root/'figures').glob('fig4_*.*'))
    assert len(files)==12, files
    write_json(root/'test_report.json',dict(passed=True,synthetic=True,stages=list(STAGES),
               workers=2,hours=len(pipe.base()['T']),figure_files=files,
               checks='Stage caches, spawn evaluation, conservation, limits, scalarisation, constant occupancy/cutoff, actual-hour scenarios, candidates, sensitivity perturbations, F1-F6 rendering and overlap checks'))
    print('PASS: all eight stages, numeric checks and six PDF/PNG figure pairs:',root,flush=True)


def verify_test(pipe):
    b,pre = pipe.base(),pipe.pre(); H = len(b['T'])
    assert pre['W'].sum()==H and len(pre['labels'])==H
    for name in ('KH','KP'):
        r = read_json(pipe.path('pareto_'+name+'.json'))
        assert r['evaluated'] and len(r['history'])==pipe.a.gens
        assert r['convergence']['converged'] is False  # insufficient 20-generation window
        assert all(np.isfinite(h['hv']) and h['hv']>=0 for h in r['history'])
        # Archive metrics from spawned workers must reproduce serial evaluation.
        ev = optimise.Evaluator(b,pre)
        for d in r['evaluated'][:3]:
            check = ev.evaluate(d['x'],d['mu'])
            for k in ('K_usd','P_safe','H','eta'): assert np.isclose(check[k],d[k]), (k,check[k],d[k])
    selection = read_json(pipe.path('selection.json'))
    all_d = unique(sum([read_json(pipe.path('pareto_'+n+'.json'))['evaluated'] for n in ('KH','KP')],[]))
    assert selection['marked']['efficiency_optimum']['eta']==max(d['eta'] for d in all_d if d['n_dev']>=1)
    dial = selection['priority_dial']; front = read_json(pipe.path('pareto.json'))['designs']
    assert dial[0]['design']['K_usd']==min(d['K_usd'] for d in front)
    assert dial[-1]['design']['H']==min(d['H'] for d in front)
    day = read_json(pipe.path('designday.json'))
    assert all(len(day[n])==24 for n in ('hours','T','RH','w10','roof_closed','occupancy','energy_kWh','AT_cooled_p90'))
    assert date.fromisoformat(day['date']).weekday() not in (4,5)
    season = load_npz(pipe.path('season.npz')); summary = read_json(pipe.path('season_summary.json'))
    for mode in ('dispatch','constant'):
        assert season[mode+'_kWh'].shape==(H//24,24)
        assert np.isclose(season[mode+'_kWh'].sum(),summary[mode]['E_kWh'])
        assert 0<=summary[mode]['P_safe']<=1 and summary[mode]['H']>=0
        assert np.all(season[mode+'_kWh'][:,4]==0)
    assert summary['dispatch']['E_kWh']<=summary['constant']['E_kWh']+1e-8
    fan_b = dict(b,T=b['T'].copy()); fan_b['T'][1:3] = [34.,36.]
    fan = dict(x=[1]+[0]*(len(b['slots'])-1),mu=0.)
    ff = [np.zeros((2,1,len(b['xy']))) for _ in range(3)]+[np.zeros((2,1))]
    _,fan_rows,_ = pipe.simulate(fan_b,np.array([1,2]),fan,ff,mode='always')
    assert np.isclose(fan_rows[0]['E'],econ.DEV['A']['power']) and fan_rows[1]['E']==0
    maps = load_npz(pipe.path('maps.npz'))
    assert maps['AT_base'].shape==(len(b['xy']),) and maps['wind_ratio'].shape==maps['AT_base'].shape
    robust = read_json(pipe.path('robustness.json')); detail = read_json(pipe.path('robustness_detail.json'))
    assert np.asarray(robust['sobol']['ST']).shape==(3,10)
    regret = np.asarray(robust['regret']); assert regret.shape in ((6,4),(7,4))
    fr = np.asarray(detail['front_regret_percent'])
    assert np.allclose(fr.min(0),0.) and np.all(fr>=-1e-9)
    assert detail['robust_front_index']==int(np.argmin(fr.max(1)))
    ri = robust['designs'].index(robust['robust_choice'])
    assert np.allclose(regret[ri],fr[detail['robust_front_index']])
    assert detail['sample_count']==pipe.a.N*12 and detail['scenario_hours'][1]==14*24
    for j in (1,2):
        assert np.all(np.asarray(detail['scenario_weights'][j])==1)
        assert len(detail['scenario_representative_hours'][j])==detail['scenario_hours'][j]
    candidates = read_json(pipe.path('real_hour_candidates.json'))
    assert len(candidates['candidates'])==8 and candidates['hours']==H
    # Physical perturbations must reach HourType, resource and threshold paths.
    ev = perturbed_evaluator(b,pre,{'T_off':1.,'occ_scale':2.,'AT_safe':30.,'cop_dc':3.,'elasticity_scale':1.2})
    assert ev.hts[0].at_safe==30 and np.isclose(ev.hts[0].Ta,b['T'][pre['reps'][0]]+1)
    assert np.isclose(ev.hts[0].n.sum(),2*b['n'][pre['reps'][0]].sum())
    assert not np.allclose(ev.pre['run_e'][:,3],pre['run_e'][:,3])
    ht = make_ht(b,int(pre['reps'][0]))
    ht.at_safe = 100.; assert np.isclose(ht.evaluate()['safe'],ht.n.sum())
    ht.at_safe = -100.; assert ht.evaluate()['safe']==0
    # A known fixed trade-off front must have zero HV change, including -P_safe.
    class FixedFront:
        slots = np.zeros((1,2))
        E = econ.ECON
        def evaluate(self,x,mu=0.):
            c = int(x[0])
            return dict(K_usd=float(c),H=float(4-c),P_safe=.1+.2*c,noise_dBA=0.)
    for obj in (('K_usd','H'),('K_usd','P_safe')):
        check = optimise.nsga(FixedFront(),obj,pop=8,gens=22,seed=4210,log_every=100)
        assert check['convergence']['converged'] and np.isclose(check['convergence']['relative_change'],0.)
        assert all(h['hv']>0 for h in check['history'])
    # Identical stage parameters and dependencies must load, not rewrite caches.
    pipe.a.force = False
    for stage in STAGES:
        manifest = pipe.path(stage+'.manifest.json'); stamp = manifest.stat().st_mtime_ns
        pipe.execute(stage)
        assert manifest.stat().st_mtime_ns==stamp, ('cache unexpectedly rebuilt',stage)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage',choices=(*STAGES,'all','test'))
    p.add_argument('--K','--hour-types',type=int,default=60)
    p.add_argument('--pop',type=int,default=80)
    p.add_argument('--gens',type=int,default=200)
    p.add_argument('--N','--sobol-n',type=int,default=256)
    p.add_argument('--workers',type=int,default=os.cpu_count() or 1)
    p.add_argument('--chunk-hours',type=int,default=24)
    p.add_argument('--safety-target',type=float,default=.95)
    p.add_argument('--force',action='store_true')
    a = p.parse_args(argv)
    if min(a.K,a.N,a.workers,a.chunk_hours,a.gens)<1 or a.pop<4: p.error('Positive counts and pop >= 4 required')
    if not 0<=a.safety_target<=1: p.error('Safety target must be in [0,1]')
    if a.N & (a.N-1): p.error('Sobol N must be a power of two')
    if a.stage=='test': return run_test(a)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        pipeline = Pipeline(a)
        for stage in STAGES if a.stage=='all' else [a.stage]: pipeline.execute(stage)


if __name__=='__main__':
    mp.freeze_support()
    main()
