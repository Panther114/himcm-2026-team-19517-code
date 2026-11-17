"""Sample saved CFD on matched physical coordinates and audit convergence.

Unlike the legacy gridding code, this samples the volume mesh directly; coarse
cells are not thrown into a single post-processing bin or silently set to zero.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pyvista as pv
from common import ROOT, OF

X=np.arange(-9.875,30.,.25)
Y=np.arange(-11.875,12.,.25)
Z=np.array([.5,1.1,2.,3.,3.5,4.5,5.5])

def times(folder):
    return sorted([p.name for p in folder.iterdir() if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?',p.name) and float(p.name)>0],key=float)

def vtk(folder,t):
    candidates=list((folder/'VTK').glob(f'*_{t}.vtk'))
    if not candidates:
        with (folder/f'log.foamToVTK_{t}').open('w') as log:
            p=subprocess.run(['cmd','/c',str(OF),'foamToVTK','-time',str(t),'-noFaceZones','-excludePatches','(".*")'],cwd=folder,stdout=log,stderr=subprocess.STDOUT)
        candidates=list((folder/'VTK').glob(f'*_{t}.vtk'))
        if p.returncode or not candidates: raise RuntimeError(f'VTK export failed: {folder} time {t}')
    return candidates[0]

def sample(folder,t):
    data=pv.read(vtk(folder,t))
    xx,yy,zz=np.meshgrid(X,Y+12.,Z,indexing='ij')
    points=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])
    sampled=pv.PolyData(points).sample(data.cell_data_to_point_data())
    valid=np.asarray(sampled['vtkValidPointMask'],bool).reshape(xx.shape)
    out={k:np.asarray(sampled[k]).reshape(xx.shape+(() if k!='U' else (3,))) for k in ['T','H2O','U']}
    out['valid']=valid
    out['T']-=273.15
    out['cells']=data.n_cells
    out['min_T']=float(np.min(data['T'])-273.15)
    out['max_T']=float(np.max(data['T'])-273.15)
    out['max_speed']=float(np.linalg.norm(data['U'],axis=-1).max())
    return out

def audit_log(folder):
    text=(folder/'log.foamRun').read_text(errors='replace')
    resid={}
    for field,initial,final in re.findall(r'Solving for (\w+), Initial residual = ([\deE+.-]+), Final residual = ([\deE+.-]+)',text):
        resid[field]={'initial':float(initial),'final':float(final)}
    continuity=re.findall(r'continuity errors : sum local = ([\deE+.-]+), global = ([\deE+.-]+)',text)
    mesh=(folder/'log.checkMesh').read_text(errors='replace')
    return dict(ended='\nEnd\n' in text, last_residuals=resid,
                last_local_continuity=float(continuity[-1][0]) if continuity else None,
                last_global_continuity=float(continuity[-1][1]) if continuity else None,
                mesh_ok='Mesh OK.' in mesh, mesh_failed='Failed' in mesh)

def extract_case(folder):
    meta=json.loads((folder/'case_info.json').read_text())
    run=json.loads((folder/'run.json').read_text())
    if not run['ok']: raise RuntimeError('Solver run failed: '+str(folder))
    out=ROOT/'results'/'kernels';out.mkdir(parents=True,exist_ok=True)
    target=out/(folder.name+'.npz')
    tt=times(folder)
    if len(tt)<2: raise RuntimeError('Two saved solution times required for field stability: '+str(folder))
    audit_path=out/(folder.name+'.json')
    if target.exists() and audit_path.exists() and json.loads(audit_path.read_text()).get('extraction_version')==2: return
    samples=[sample(folder,t) for t in tt[-2:]]
    fields=samples[-1]
    previous=samples[-2]
    np.savez_compressed(target,x=X,y=Y,z=Z,T=fields['T'],Y=fields['H2O'],U=fields['U'],valid=fields['valid'],
                        T_previous=previous['T'],Y_previous=previous['H2O'],U_previous=previous['U'],valid_previous=previous['valid'])
    valid=fields['valid'] & samples[-2]['valid']
    zbody=int(np.where(Z==1.1)[0][0]);v=valid[:,:,zbody]
    dt=fields['T'][:,:,zbody]-samples[-2]['T'][:,:,zbody]
    audit=audit_log(folder)
    speed=np.linalg.norm(fields['U'][:,:,zbody],axis=-1)
    speed_previous=np.linalg.norm(previous['U'][:,:,zbody],axis=-1)
    audit.update(extraction_version=2,label=meta['label'],signature=meta['signature'],times=tt[-2:],cells=fields['cells'],
                 min_T_C=fields['min_T'],max_T_C=fields['max_T'],max_speed_m_s=fields['max_speed'],
                 body_T_snapshot_rms_K=float(np.sqrt(np.mean(dt[v]**2))),
                 body_T_snapshot_max_K=float(np.max(np.abs(dt[v]))),
                 body_speed_snapshot_rms_m_s=float(np.sqrt(np.mean((speed[v]-speed_previous[v])**2))),
                 body_Y_snapshot_rms=float(np.sqrt(np.mean((fields['H2O'][:,:,zbody][v]-previous['H2O'][:,:,zbody][v])**2))),
                 metadata=meta)
    (out/(folder.name+'.json')).write_text(json.dumps(audit,indent=2),newline='\n')
    print('extracted',folder.name,'cells',fields['cells'],'snapshot rms',audit['body_T_snapshot_rms_K'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--filter',default='');a=p.parse_args()
    rejected=[]
    for folder in sorted((ROOT/'cfd').iterdir()):
        if a.filter in folder.name and (folder/'run.json').exists():
            run=json.loads((folder/'run.json').read_text())
            if not run['ok']:
                rejected.append(run);print('rejected failed solver/mesh run',folder.name,flush=True)
                continue
            extract_case(folder)
    (ROOT/'results'/'rejected_runs.json').write_text(json.dumps(rejected,indent=2),newline='\n')
