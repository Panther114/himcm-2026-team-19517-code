"""Conditional algebraic and placement sensitivity, separate from CFD reruns."""
import json
import numpy as np
from common import ROOT
from analyse import CONFIG,pair,geometry,placed,evaluate,resource,select_positions

def main(variant='hydrostatic'):
    rows=[]
    for name,cid,cls in [('bayfront','bf','D'),('medina','md','M')]:
        k=pair(cid,cls,variant);g=geometry(name)
        n=14 if name=='bayfront' else 436
        pp=select_positions(g['positions'],min(n,len(g['positions'])))
        weights=np.full(len(pp),n/len(pp));angle=CONFIG[name]['flow_angle_deg_from_x']
        reference=placed(k,g['xy'],pp,angle,weights)
        base,_=evaluate(name,g,reference)
        power=resource(k,name,n)['power_kW']
        def record(parameter,value,res,pwr):
            rows.append(dict(venue=name,kernel_variant=variant,parameter=parameter,value=value,
                area_gt1K_m2=res['area_gt1K_m2'],mean_dAT_K=res['mean_dAT_K'],
                protected_share=res['protected_on'],eta_normalised=res['normalised_utility_person_hours']/pwr))
        record('reference',1.,base,power)
        for eff in [.3,.5,.7]:
            r,_=evaluate(name,g,reference,effect_factor=eff);record('delivery_factor',eff,r,power)
        for eps in [.5,1.,1.5]:
            r,_=evaluate(name,g,reference,eps=eps);record('epsilon',eps,r,power)
        for met in [1.2,2.]:
            r,_=evaluate(name,g,reference,met=met);record('metabolic_rate_met',met,r,power)
        for f in [0.,.5,1.5]:
            r,_=evaluate(name,g,reference,radiation_factor=f);record('radiative_load_factor',f,r,power)
        for delta in [-90.,-45.,45.,90.]:
            # Rotation of a kernel WITH its background tests site alignment,
            # not a change of jet direction relative to the incoming wind.
            fields=placed(k,g['xy'],pp,angle+delta,weights)
            r,_=evaluate(name,g,fields);record('site_flow_rotation_deg',delta,r,power)
        conservative=placed(k,g['xy'],pp,angle,weights,combination='max')
        r,_=evaluate(name,g,conservative);record('overlap_rule','max-envelope scenario',r,power)
        if cls=='D':
            for cop in [2.5,5.5]:record('plant_COP',cop,base,resource(k,name,n,cop=cop)['power_kW'])
        for f in [.5,1.5]:record('system_electricity_factor',f,base,power*f)
    d=dict(method='One input at a time on saved spatial fields. Delivery/radiation factors do not recompute flow. Site rotation rotates the background and plume together. Max-envelope combination is an alternative interaction scenario, not a proven lower bound.',results=rows)
    (ROOT/'results'/'sensitivity.json').write_text(json.dumps(d,indent=2),newline='\n')
    print('Saved',len(rows),'conditional sensitivity evaluations.')

if __name__=='__main__':main()
