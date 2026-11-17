"""Immutable, fingerprinted Task 3 CFD cases using the reviewed Task 4 engine.

Run with python src/run_cases.py --workers 4.
Case data are research outputs, not scratch build files. Existing cases are
never regenerated or removed. Only a successful identical case is reusable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from common import ROOT, ENGINE, OF
import devices
import cfd_gen
import climate
from scipy.optimize import brentq

CONFIG=json.loads((ROOT/'inputs'/'scenarios.json').read_text())
CLIMATES = {cid:dict(T=CONFIG[name]['T_C'],RH=CONFIG[name]['RH_percent'],
                     w10=CONFIG[name]['wind10_m_s'],p=CONFIG[name]['pressure_Pa'])
            for cid,name in [('bf','bayfront'),('md','medina')]}

def enthalpy(T, w):
    return 1.006 * T + w * (2501. + 1.86 * T)

def supply(T, RH, p, efficiency=.8, water_Lph=100., flow=2.5):
    """Enthalpy-conserving evaporated-air approximation, bounded by saturation.

    Un-evaporated liquid is not credited as air cooling or body wetting.
    Saturation is solved at local pressure rather than the Stull fit limit.
    """
    Ya = cfd_gen.Y_h2o(T, RH, p / 100.)
    wa = Ya / (1 - Ya)
    ha = enthalpy(T, wa)
    def ws(t):
        y = cfd_gen.Y_h2o(t, 100., p / 100.)
        return y / (1-y)
    tw = brentq(lambda t: enthalpy(t, ws(t)) - ha, -10., T)
    Ts = T - efficiency * (T - tw)
    wnew = (ha - 1.006 * Ts) / (2501 + 1.86 * Ts)
    # Volume flow is specified at the outlet, so its dry-air density must be
    # used in the water balance as well as in the chilled-air enthalpy load.
    def evaporated(w):
        ts=(ha-2501*w)/(1.006+1.86*w)
        es=p*w/(.622+w)
        rho=(p-es)/(287.05*(ts+273.15))
        return (w-wa)*rho*flow*3600
    if evaporated(wnew)>water_Lph:
        wnew=brentq(lambda w:evaporated(w)-water_Lph,wa,wnew)
    Ts = (ha - 2501*wnew)/(1.006 + 1.86*wnew)
    return Ts, wnew/(1+wnew), dict(Twb=tw, evaporated_Lph=evaporated(wnew),
                                water_available_Lph=water_Lph, h_ambient=ha,
                                h_supply=enthalpy(Ts,wnew))

def case_spec(climate_id, cls, on, fine=1., height=None, gravity=False, angle=0., ground_delta=0., wind_factor=1., flow_factor=1., hydrostatic=False):
    c = CLIMATES[climate_id]
    # Same power-law convention as current Task 4; 10 m input retained in metadata.
    wb = c['w10'] * climate.PROFILE_BODY * wind_factor
    s = devices.spec(cls='D' if cls=='D' else 'A', w_b=wb, Ta=c['T'], RH=c['RH'],
                     T_sup=24., on=on, fine=fine, endTime=1000)
    s['ambient']['p'] = c['p']
    s['gravity'] = gravity
    s['T_ground'] += ground_delta
    s['writeInterval'] = 250
    info = dict(climate=climate_id, cls=cls, on=on, fine=fine, height=height,
                gravity=gravity, angle=angle, ground_delta=ground_delta,
                wind_factor=wind_factor, flow_factor=flow_factor, hydrostatic=hydrostatic)
    body = s['solids'][0]
    if cls=='M':
        z = 3.5 if height is None else height
        # Explicit 0.8 m outlet, fine levels fixed around body height and outlet.
        breaks = sorted(set(round(v,6) for v in [0., .7, .85, 1.1, 1.35, 1.9, z-.4, z, z+.4, 5., 16.]))
        s['zb'] = breaks
        s['zn'] = [max(1, round((b-a)/(.22 if b<=5 else 1.)*fine)) for a,b in zip(breaks,breaks[1:])]
        s['yb'] = [0., 10.5, 11.6, 12., 12.4, 13.5, 24.]
        s['yn'] = [max(1,round(n*fine)) for n in [14,4,2,2,4,14]]
        body['box'] = (-.4,0.,11.6,12.4,z-.4,z+.4)
        Ts,Ys,evap = supply(c['T'],c['RH'],c['p'],flow=2.5*flow_factor)
        info['evaporation'] = evap
        if on:
            body['jet'].update(U=2.5*flow_factor/.64,T=Ts+273.15,Y=Ys)
    elif on:
        body['jet'].update(U=1.57*flow_factor/.36,
                           Y=min(cfd_gen.Y_h2o(24.,90.,c['p']/100),cfd_gen.Y_h2o(c['T'],c['RH'],c['p']/100)))
    # Both initial atmosphere and jet use the case pressure.
    info['supply'] = body.get('jet')
    return s, info

def run(job):
    label,spec,meta = job
    engine_sha = hashlib.sha256((ENGINE/'cfd_gen.py').read_bytes()).hexdigest()
    fingerprint=dict(spec=spec,engine=engine_sha)
    if meta['angle']:fingerprint['background_angle']=meta['angle']
    if meta.get('hydrostatic'):fingerprint['hydrostatic_outlet']='uniform-density-pressure-v1'
    signature = hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    folder = ROOT/'cfd'/(label+'_'+signature[:10])
    accepted = folder/'run.json'
    if folder.exists():
        if accepted.exists() and json.loads(accepted.read_text()).get('ok'):
            print('reusing',folder.name,flush=True)
            return json.loads(accepted.read_text())
        raise RuntimeError(f'Unaccepted case already exists; inspect without overwriting: {folder}')
    folder.parent.mkdir(exist_ok=True)
    cfd_gen.make(str(folder),spec)
    if meta.get('hydrostatic'):
        a=spec['ambient'];Ta=a['T']-273.15
        e=a['RH']/100*cfd_gen.es_hpa(Ta)*100
        rho=(a['p']-e)/(287.05*a['T'])+e/(461.5*a['T'])
        p=folder/'0'/'p_rgh'
        t=p.read_text().replace('type prghPressure; p uniform 0; value uniform 0;',
            f'type prghUniformDensityHydrostaticPressure; rhoRef {rho}; pRef 0; pRefPoint (0 0 0); value uniform 0;')
        p.write_text(t,newline='\n')
    # Rotate the BACKGROUND relative to the nozzle, not a precomputed plume.
    if meta['angle']:
        th=math.radians(meta['angle'])
        for f in ['U','k','epsilon']:
            p=folder/'0'/f
            t=p.read_text().replace('flowDir (1 0 0)',f'flowDir ({math.cos(th)} {math.sin(th)} 0)')
            if f=='U':
                u=spec['ambient']['Uref']
                t=t.replace(f'internalField uniform ({u} 0 0);',f'internalField uniform ({u*math.cos(th)} {u*math.sin(th)} 0);')
            p.write_text(t,newline='\n')
    meta.update(signature=signature,engine_sha256=engine_sha,spec=spec,label=label,path=str(folder))
    (folder/'case_info.json').write_text(json.dumps(meta,indent=2),newline='\n')
    result=dict(label=label,path=str(folder),signature=signature,ok=False,commands={})
    for cmd in ['blockMesh','checkMesh','foamRun']:
        start=time.monotonic()
        with (folder/('log.'+cmd)).open('w') as log:
            p=subprocess.run(['cmd','/c',str(OF),cmd],cwd=folder,stdout=log,stderr=subprocess.STDOUT)
        result['commands'][cmd]=dict(returncode=p.returncode,seconds=round(time.monotonic()-start,2))
        if p.returncode:
            break
    else:
        result['ok']=True
    accepted.write_text(json.dumps(result,indent=2),newline='\n')
    print(label, result['ok'],result['commands'],flush=True)
    return result

def jobs(suite):
    out=[]
    for cid in CLIMATES:
        for cls in ['D','M']:
            for on in [False,True]:
                s,m=case_spec(cid,cls,on)
                out.append((f'{cid}_{cls}_{"on" if on else "off"}',s,m))
    if suite=='base': return out
    out=[]
    variants=[('fine',dict(fine=1.4)),('buoyant',dict(gravity=True)),
              ('hydrostatic',dict(gravity=True,hydrostatic=True)),
              ('low',dict(height=1.1)),('sweepneg',dict(angle=-60.)),('sweeppos',dict(angle=60.)),
              ('hotground',dict(ground_delta=10.)),('windlow',dict(wind_factor=.5)),
              ('windhigh',dict(wind_factor=1.5)),('flowlow',dict(flow_factor=.5)),('flowhigh',dict(flow_factor=1.5))]
    for cid,cls in [('bf','D'),('md','M')]:
        for name,kwargs in variants:
            if cls=='D' and name in ['low','sweepneg','sweeppos']: continue
            for on in [False,True]:
                s,m=case_spec(cid,cls,on,**kwargs)
                out.append((f'{cid}_{cls}_{name}_{"on" if on else "off"}',s,m))
    # Consistent buoyant counterfactuals, plus refinement at the accepted
    # hydrostatic boundary; kept separate from rejected legacy buoyancy cases.
    for cid,cls in [('bf','M'),('md','D')]:
        for on in [False,True]:
            s,m=case_spec(cid,cls,on,gravity=True,hydrostatic=True)
            out.append((f'{cid}_{cls}_hydrostatic_{"on" if on else "off"}',s,m))
    for cid,cls in [('bf','D'),('md','M')]:
        for on in [False,True]:
            s,m=case_spec(cid,cls,on,fine=1.4,gravity=True,hydrostatic=True)
            out.append((f'{cid}_{cls}_hydrofine_{"on" if on else "off"}',s,m))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--suite',choices=['base','checks'],default='base');p.add_argument('--filter',default='')
    a=p.parse_args()
    selected=[j for j in jobs(a.suite) if a.filter in j[0]]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,j) for j in selected]
        results=[f.result() for f in as_completed(futures)]
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/f'runs_{a.suite}{"_"+a.filter if a.filter else ""}.json').write_text(json.dumps(results,indent=2),newline='\n')
    if not all(r['ok'] for r in results): raise SystemExit('At least one CFD case failed; inspect logs.')
