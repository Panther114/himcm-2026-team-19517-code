"""Area-weighted venue evaluation using archived, matched CFD kernels.

The equations and physical-field combination follow Task 4. Comparison utility
is dimensionless share of baseline discomfort recovered, integrated over
person-hours. Negative utility from overcooling is retained, not hidden.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from scipy.interpolate import RegularGridInterpolator
from common import ROOT,ENGINE
import climate
from run_cases import enthalpy, CLIMATES, case_spec

CONFIG=json.loads((ROOT/'inputs'/'scenarios.json').read_text())

def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def pair(cid,cls,variant=''):
    label=f'{cid}_{cls}'+('_'+variant if variant else '')
    def find(state):
        found=[]
        for p in (ROOT/'results'/'kernels').glob('*.json'):
            d=load_json(p)
            if d['label']==label+'_'+state: found.append((p,d))
        if len(found)!=1: raise RuntimeError(f'Expected one accepted kernel for {label}_{state}; got {len(found)}')
        p,d=found[0]
        meta=d['metadata']
        kwargs={key:meta[key] for key in ['fine','height','gravity','angle','ground_delta','wind_factor','flow_factor','hydrostatic'] if key in meta}
        expected,_=case_spec(cid,cls,state=='on',**kwargs)
        if json.dumps(expected,sort_keys=True)!=json.dumps(meta['spec'],sort_keys=True):
            raise RuntimeError('CFD inputs changed since generation: '+str(p))
        if hashlib.sha256((ENGINE/'cfd_gen.py').read_bytes()).hexdigest()!=meta['engine_sha256']:
            raise RuntimeError('CFD generator changed since generation: '+str(p))
        if (not d['ended'] or not d['mesh_ok'] or
            d['last_global_continuity'] is None or abs(d['last_global_continuity'])>1e-6 or
            d['max_speed_m_s']>30 or d['body_T_snapshot_rms_K']>.05):
            raise RuntimeError('Case failed acceptance: '+str(p))
        return dict(np.load(p.with_suffix('.npz'))),d
    on,meta=find('on');off,offmeta=find('off')
    valid=on['valid'] & off['valid']
    if any(not np.array_equal(on[k],off[k]) for k in ['x','y','z']):raise ValueError('Unmatched sample coordinates')
    wa=on['Y']/(1-on['Y']);wb=off['Y']/(1-off['Y'])
    k=dict(x=on['x'],y=on['y'],z=on['z'],
           dT=np.where(valid,off['T']-on['T'],0.),
           dw=np.where(valid,wa-wb,0.),
           dv2=np.where(valid,np.sum(on['U']**2,axis=-1)-np.sum(off['U']**2,axis=-1),0.),
           meta=meta,offmeta=offmeta)
    k['valid']=valid
    if 'T_previous' in on and 'T_previous' in off:
        oldvalid=on['valid_previous']&off['valid_previous']
        k['previous']=dict(x=on['x'],y=on['y'],z=on['z'],meta=meta,
            dT=np.where(oldvalid,off['T_previous']-on['T_previous'],0.),
            dw=np.where(oldvalid,on['Y_previous']/(1-on['Y_previous'])-off['Y_previous']/(1-off['Y_previous']),0.),
            dv2=np.where(oldvalid,np.sum(on['U_previous']**2,axis=-1)-np.sum(off['U_previous']**2,axis=-1),0.))
    return k

def geometry(name,resolution=None):
    """Exact polygon-clipped area weights; no credit inside buildings/columns."""
    if name=='bayfront':
        data=load_json(ROOT/'inputs'/'bayfront_site_geometry.json')
        poly=unary_union([Polygon(s['polygon_m']) for s in data['sections'].values()])
        candidates=np.array([[p['x'],p['y']] for p in load_json(ROOT/'inputs'/'bayfront_cooler_positions.json')])
        h=resolution or 1.
        # Reserve device pads identically in baseline and cooled states.
        poly=poly.difference(unary_union([box(x-.4,y-.4,x+.4,y+.4) for x,y in candidates]))
    else:
        data=load_json(ROOT/'inputs'/'medina_umbrella_layout.json')
        candidates=np.array([[p['x'],p['y']] for p in data['umbrellas']])
        poly=unary_union([box(x-12.75,y-12.75,x+12.75,y+12.75) for x,y in candidates])
        buildings=load_json(ROOT/'inputs'/'medina_context.json')['buildings']
        poly=poly.difference(unary_union([Polygon(b['poly']).buffer(0) for b in buildings]))
        poly=poly.difference(unary_union([box(x-.4,y-.4,x+.4,y+.4) for x,y in candidates]))
        h=resolution or 2.
    lo,bo,hi,to=poly.bounds
    xs=np.arange(math.floor(lo/h)*h+h/2,math.ceil(hi/h)*h,h)
    ys=np.arange(math.floor(bo/h)*h+h/2,math.ceil(to/h)*h,h)
    xx,yy=np.meshgrid(xs,ys,indexing='ij');xy=np.column_stack([xx.ravel(),yy.ravel()])
    cells=shapely.box(xy[:,0]-h/2,xy[:,1]-h/2,xy[:,0]+h/2,xy[:,1]+h/2)
    areas=shapely.area(shapely.intersection(cells,poly))
    keep=areas>1e-9
    return dict(xy=xy[keep],area=areas[keep],poly=poly,positions=candidates,h=h,
                x=xs,y=ys,keep=keep,shape=xx.shape)

def select_positions(positions,n):
    """Nested farthest-point scenario layouts; no claim of historical inventory."""
    centre=np.mean(positions,axis=0)
    chosen=[int(np.argmin(np.linalg.norm(positions-centre,axis=1)))]
    d=np.linalg.norm(positions-positions[chosen[0]],axis=1)
    for _ in range(1,n):
        j=int(np.argmax(d));chosen.append(j)
        d=np.minimum(d,np.linalg.norm(positions-positions[j],axis=1));d[chosen]=-1
    return positions[chosen]

def placed(k,xy,positions,angle,weights=None,z=1.1,combination='sum'):
    iz=int(np.argmin(abs(k['z']-z)))
    interp={n:RegularGridInterpolator((k['x'],k['y']),k[n][:,:,iz],bounds_error=False,fill_value=0.) for n in ['dT','dw','dv2']}
    fields={n:np.zeros(len(xy)) for n in interp}
    th=math.radians(angle);cs,sn=math.cos(th),math.sin(th)
    weights=np.ones(len(positions)) if weights is None else np.asarray(weights)
    for (x,y),w in zip(positions,weights):
        dx=xy[:,0]-x;dy=xy[:,1]-y
        local=np.column_stack([cs*dx+sn*dy,-sn*dx+cs*dy])
        sel=(local[:,0]>=k['x'][0])&(local[:,0]<=k['x'][-1])&(local[:,1]>=k['y'][0])&(local[:,1]<=k['y'][-1])
        for name,f in interp.items():
            v=w*f(local[sel])
            if combination=='sum': fields[name][sel]+=v
            elif combination=='max':fields[name][sel]=np.maximum(fields[name][sel],v)
            else:raise ValueError(combination)
    fields['T_floor']=k['meta']['metadata']['spec']['solids'][0]['jet']['T']-273.15
    return fields

def radiation(name,g):
    c=CONFIG[name]
    sp=pvlib.solarposition.get_solarposition(pd.DatetimeIndex([c['time']]),c['latitude'],c['longitude'])
    zen=float(sp.apparent_zenith.iloc[0]);cosz=max(0.,math.cos(math.radians(zen)))
    Q=float(climate.Q_abs(c['clearness'],cosz,c['albedo'])) if cosz>0 else 0.
    # Matched fixed shade scenario. Direct coupling of sun into CFD is absent;
    # exposed radiative load is evaluated afterwards in the comfort equation.
    shade_fraction=c.get('canopy_radiation_fraction',1.)
    return np.full(len(g['xy']),Q*shade_fraction),dict(zenith_deg=zen,Q_exposed_W_m2=Q,Q_applied_W_m2=Q*shade_fraction)

def resource(k,name,n,cop=None,power_factor=1.):
    c=CONFIG[name];m=k['meta']['metadata'];s=m['spec'];cls=m['cls']
    if cls=='D':
        jet=s['solids'][0]['jet'];Ts=jet['T']-273.15
        ya=__import__('cfd_gen').Y_h2o(c['T_C'],c['RH_percent'],c['pressure_Pa']/100)
        wa=ya/(1-ya);ws=jet['Y']/(1-jet['Y'])
        V=jet['U']*.36
        # The specified volume flow is at the SUPPLY outlet. Use its dry-air
        # density for both sides of the enthalpy difference (same dry-air mass).
        esupply=c['pressure_Pa']*ws/(.622+ws)
        rho_da=(c['pressure_Pa']-esupply)/(287.05*(Ts+273.15))
        q=rho_da*V*max(enthalpy(c['T_C'],wa)-enthalpy(Ts,ws),0.)
        C=CONFIG['bayfront'];pump=C['pump_design_flow_L_s']*C['pump_head_kPa']/(C['pump_efficiency']*1000)
        electricity=n*(C['cooler_fan_power_kW']+q/(cop or C['district_COP'])+pump*q/C['district_capacity_kW_thermal'])
        return dict(power_kW=electricity*power_factor,thermal_kW=n*q,thermal_per_unit_kW=q,
                    water_L_h=0.,capacity_feasible=n*q<=C['district_capacity_kW_thermal']+1e-9,
                    capacity_note='490 kW Bayfront connection; hypothetical common capacity for climate counterfactuals')
    C=CONFIG['medina'];water=C['water_L_h_per_fan']
    pump=water/3600/1000*(C['pump_pressure_bar']*1e5)/C['pump_efficiency']/1000
    jet=s['solids'][0]['jet'];ys=jet['Y'];ws=ys/(1-ys)
    e0=float(climate.vapour(c['T_C'],c['RH_percent']));wa=.622*e0/(c['pressure_Pa']/100-e0)
    esupply=c['pressure_Pa']*ws/(.622+ws)
    rho_da=(c['pressure_Pa']-esupply)/(287.05*jet['T'])
    evaporated=max(0.,ws-wa)*rho_da*(jet['U']*.64)*3600
    return dict(power_kW=n*(C['fan_power_kW']+pump)*power_factor,thermal_kW=None,
                water_L_h=n*water,evaporated_L_h=n*evaporated,capacity_feasible=True)

def evaluate(name,g,fields,eps=None,radiation_factor=1.,effect_factor=1.,wind_factor=1.,met=None):
    c=CONFIG[name];n=c['density_persons_m2']*g['area'];tau=CONFIG['equations']['duration_hours']
    m=c['met'] if met is None else met
    eps=climate.eps_fn(c['density_persons_m2'],m) if eps is None else eps
    T0=c['T_C'];e0=float(climate.vapour(T0,c['RH_percent']));p=c['pressure_Pa']/100
    w0=.622*e0/(p-e0)
    dT=effect_factor*fields['dT'];dw=effect_factor*fields['dw'];dv2=effect_factor*fields['dv2']
    T=np.maximum(T0-dT,fields.get('T_floor',-273.15));omega=np.maximum(0.,w0+dw)
    # Humidity-ratio superposition; limit saturation at the resulting T.
    es=climate.vapour(T,100.)
    wsat=.622*es/np.maximum(p-es,1.)
    omega=np.minimum(omega,wsat)
    e=p*omega/(.622+omega)
    ambient_w10=c['wind10_m_s']*wind_factor
    w10=np.sqrt(np.maximum(.05**2,(ambient_w10*climate.PROFILE_BODY)**2+dv2))/climate.PROFILE_BODY
    Q,solar=radiation(name,g);Q*=radiation_factor
    if radiation_factor==0.:
        a0=np.full(len(n),climate.AT_shade(T0,e0,ambient_w10,m))
        af=climate.AT_shade(T,e,w10,m)
    else:
        coef=CONFIG['equations']['radiation_coefficient']
        a0=climate.AT_sun(T0,e0,ambient_w10,Q,m,rad_coef=coef)
        af=climate.AT_sun(T,e,w10,Q,m,rad_coef=coef)
    U=climate.utility(a0,af,eps)
    loss=np.abs(climate.AT_IDEAL-a0)**eps
    share=np.divide(U,loss,out=np.zeros_like(U),where=loss>1e-10)
    dAT=a0-af
    totaln=float(n.sum())
    # Exact polygon areas and identical populations are used for on and off.
    res=dict(area_m2=float(g['area'].sum()),population=totaln,epsilon=float(eps),
             mean_AT_off_C=float(np.average(a0,weights=n)),mean_AT_on_C=float(np.average(af,weights=n)),
             mean_dAT_K=float(np.average(dAT,weights=n)),peak_dAT_K=float(dAT.max()),
             area_gt1K_m2=float(g['area'][dAT>1].sum()),people_gt1K=float(n[dAT>1].sum()),
             area_gt0p5K_m2=float(g['area'][dAT>.5].sum()),
             mean_dAT_gt1K_K=float(np.average(dAT[dAT>1],weights=g['area'][dAT>1])) if np.any(dAT>1) else None,
             protected_off=float(n[a0<=32].sum()/totaln),protected_on=float(n[af<=32].sum()/totaln),
             worst_exceedance_K=float(np.maximum(af-32,0).max()),
             normalised_utility_person_hours=float(np.sum(n*share)*tau),
             raw_utility_person_hours=float(np.sum(n*U)*tau),solar=solar)
    return res,dict(AT_off=a0,AT_on=af,dAT=dAT,T=T,e=e,wind10=w10,share=share)

def main(variant='hydrostatic'):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[];arrays={}
    for name,cid,cls in [('bayfront','bf','D'),('medina','md','M')]:
        k=pair(cid,cls,variant);g=geometry(name)
        if cls=='D':
            q=resource(k,name,1)['thermal_per_unit_kW'];maxn=int(CONFIG['bayfront']['district_capacity_kW_thermal']//q)
            counts=sorted(set([1,4,8,maxn]))
        else: counts=[1,109,218,436]
        for n in counts:
            positions=select_positions(g['positions'],min(n,len(g['positions'])))
            weights=np.full(len(positions),n/len(positions))
            fields=placed(k,g['xy'],positions,CONFIG[name]['flow_angle_deg_from_x'],weights)
            result,spatial=evaluate(name,g,fields)
            r=resource(k,name,n);result.update(venue=name,device=cls,devices=n,resources=r,kernel_variant=variant or 'base',
                 eta_normalised_per_kWh=result['normalised_utility_person_hours']/(r['power_kW']*CONFIG['equations']['duration_hours']))
            rows.append(result)
            if n==max(counts):
                for key,value in spatial.items():arrays[name+'_'+key]=value
                arrays[name+'_xy']=g['xy'];arrays[name+'_area']=g['area'];arrays[name+'_positions']=positions
    (out/'venue_results.json').write_text(json.dumps(rows,indent=2),newline='\n')
    np.savez_compressed(out/'venue_maps.npz',**arrays)
    for r in rows:
        print(r['venue'],r['devices'],'area >1 K',round(r['area_gt1K_m2'],1),
              'eta',round(r['eta_normalised_per_kWh'],4),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['base','hydrostatic'],default='hydrostatic');args=p.parse_args()
    main('' if args.variant=='base' else args.variant)
