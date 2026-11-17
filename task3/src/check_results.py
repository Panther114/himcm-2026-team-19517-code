"""Independent numerical evidence and sensitivity comparisons from saved fields.

Run after extract.py. Missing cases are listed explicitly, never substituted
with historical numbers. Failed buoyant cases remain evidence of rejection.
"""
import json
import numpy as np
from common import ROOT
from analyse import pair, placed, evaluate, geometry, resource, CONFIG
import climate

def one_device(name,cid,cls,variant=''):
    k=pair(cid,cls,variant)
    # Common quadrature at body height; no venue boundary/placement confounding.
    x,y=np.meshgrid(k['x'],k['y'],indexing='ij')
    xy=np.column_stack([x.ravel(),y.ravel()])
    valid=k['valid'][:,:,int(np.argmin(abs(k['z']-1.1)))].ravel()
    g=dict(xy=xy[valid],area=np.full(int(valid.sum()),.25**2))
    # Kernels retain the background-relative angle. For a single-device integral,
    # orientation does not affect area; nonlinear AT is evaluated AFTER averaging.
    fields=placed(k,g['xy'],np.array([[0.,0.]]),0.)
    r,sp=evaluate(name,g,fields,radiation_factor=0.,wind_factor=k['meta']['metadata'].get('wind_factor',1.))
    res=resource(k,name,1)
    r.update(venue=name,device=cls,variant=variant or 'base',resources=res,
             eta_normalised_per_kWh=r['normalised_utility_person_hours']/res['power_kW'],
             shade_only=True)
    if 'previous' in k:
        old=placed(k['previous'],g['xy'],np.array([[0.,0.]]),0.)
        prev,prev_fields=evaluate(name,g,old,radiation_factor=0.,wind_factor=k['meta']['metadata'].get('wind_factor',1.))
        r['iteration_stability']=dict(times=k['meta']['times'],
             dAT_rms_change_K=float(np.sqrt(np.mean((sp['dAT']-prev_fields['dAT'])**2))),
             area_gt1K_change_m2=r['area_gt1K_m2']-prev['area_gt1K_m2'],
             utility_relative_change=(r['normalised_utility_person_hours']-prev['normalised_utility_person_hours'])/max(abs(r['normalised_utility_person_hours']),1e-9))
    return r,k,g,sp

def main():
    rows=[];missing=[]
    variants=['','fine','buoyant','hydrostatic','hydrofine','low','sweepneg','sweeppos','hotground','windlow','windhigh','flowlow','flowhigh']
    for name,cid,classes in [('bayfront','bf',['D','M']),('medina','md',['D','M'])]:
        for cls in classes:
            vv=variants if (cid,cls) in [('bf','D'),('md','M')] else ['','hydrostatic']
            if cls=='D':vv=[v for v in vv if v not in ['low','sweepneg','sweeppos']]
            for v in vv:
                try:r,*_=one_device(name,cid,cls,v)
                except (RuntimeError,FileNotFoundError) as e:
                    missing.append(dict(venue=name,device=cls,variant=v or 'base',reason=str(e)));continue
                rows.append(r)
    comparisons=[]
    for name,cls in [('bayfront','D'),('medina','M')]:
        rr=[r for r in rows if r['venue']==name and r['device']==cls]
        base=next((r for r in rr if r['variant']=='base'),None)
        if base:
            for r in rr:
                comparisons.append(dict(venue=name,device=cls,variant=r['variant'],
                     area_gt1K_m2=r['area_gt1K_m2'],peak_dAT_K=r['peak_dAT_K'],
                     area_change_percent=100*(r['area_gt1K_m2']/base['area_gt1K_m2']-1) if base['area_gt1K_m2']>0 else None,
                     eta_normalised_per_kWh=r['eta_normalised_per_kWh']))
    out=dict(single_device_comparison=rows,physical_checks=comparisons,missing_or_rejected=missing,
             interpretation='Mesh/solver/snapshot checks are numerical verification, not external validation. A zero threshold area does not prove a zero real-world benefit.')
    (ROOT/'results'/'checks.json').write_text(json.dumps(out,indent=2),newline='\n')
    print('Saved',len(rows),'accepted single-device evaluations;',len(missing),'unavailable/rejected variants.')

if __name__=='__main__':main()
