"""Publication figures F1--F6 from saved Task 4 results (no model runs).

Usage: .venv/Scripts/python src/figures.py [results_folder] [--tariff USD_PER_KWH]
Default folder: project results/. Outputs: figures/fig4_1_framework.{pdf,png}, fig4_2_venue,
fig4_3_pareto, fig4_4_designday, fig4_5_season, fig4_6_robustness.

Exact input interfaces (all temperatures deg C; hours local, beginning of hour):
* maps.npz: xy (N,2), 2 m cell centres in plaza-frame metres; plaza_poly
  (M,2); buildings object array of polygon vertex arrays; slots (S,2);
  design (S,) integer codes 0 none, 1 A fan, 2 M misting fan, 3 E evaporative,
  4 D chilled-air; wind_ratio (N,) body-level U/U10, prevailing NW, roof closed;
  AT_base (N,), AT_cooled (N,) at a representative hot evening hour;
  plaza_angle_deg scalar, long-axis bearing north of east.
* pareto.json: {designs:[{K_usd,H,P_safe,U,E_kWh,W_m3,eta,n_dev,x,mu}],
  dominated:[same], marked:{efficiency_optimum:index or design dict,
  knee:..., recommended:...}, history:[{gen,hv}]}.
  K_usd is annual USD, H is a person-hour-weighted mean in K^2. Native nsga() output is also
  accepted: absent dominated becomes [], absent marked uses maximum eta,
  normalized knee and that knee as recommendation; extra fields are ignored.
* designday.json: {date,hours,T,RH,w10,roof_closed,occupancy,
  on_counts:{A,M,E,D},AT_base_mean,AT_base_p90,AT_cooled_mean,AT_cooled_p90,
  energy_kWh}; each series has 24 entries, roof_closed is boolean.
* season.npz: days (D,) ISO dates; dispatch_kWh,constant_kWh (D,24);
  P_safe_dispatch,P_safe_constant (D,). Full-season person-hour-weighted
  protection is read from season_summary.json. Cost means
  electricity operating cost only: energy times constant --tariff, default
  econ.ECON['tariff'].
* robustness.json: {sobol:{outputs:[names],inputs:[names],S1:[[...]],
  ST:[[...]],ST_conf:[[...]]},scenarios:[names],designs:[names],
  regret:[[...]] (design x scenario, percent of scenario score range),
  robust_choice:name}. Sobol arrays are output x input; ST_conf are supplied
  symmetric half-widths. H is the main output when present, else the first.

Only trusted NPZ files should be used: buildings requires allow_pickle=True.
F1 also reads surrogate_report.json and venue_convergence.json; F3 reads
selection.json for the priority dial and pareto_KP.json for cost–protection.
No synthetic fallback is automatic. A folder named synthetic is explicitly
identified in subtitles. All headlines are calculated from loaded data.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import json
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, Patch, FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import matplotlib.tri as mtri
import numpy as np
import figstyle as fs
from econ import ECON

ROOT = Path(__file__).resolve().parents[1]
SAFE = 32.0


def load_json(folder, name):
    return json.loads((folder/name).read_text(encoding='utf-8'))


def heading(fig, title, result, synthetic):
    fs.title(fig, title, ('Synthetic data: ' if synthetic else '') + result)


def panel(ax, letter, text):
    fs.panel(ax, letter, text)
    ax.title.set_fontsize(fs.PANEL_SIZE)


def legend(fig, handles, y=.025, x=.10, ncol=None):
    return fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(x,y),
                      ncol=ncol or len(handles), borderaxespad=0,
                      handlelength=1.6, handletextpad=.5, columnspacing=1.3)


def validate(fig, name):
    old = fig.dpi
    try:
        for dpi in (100, 300):
            fig.set_dpi(dpi)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always', RuntimeWarning)
                issues = fs.check_text_overlap(fig)
            edges = [str(m.message) for m in messages if 'beyond figure edge' in str(m.message)]
            missing = [str(m.message) for m in messages if 'Glyph' in str(m.message) and 'missing' in str(m.message)]
            if issues or edges or missing:
                pairs = [(a.get_text(), b.get_text() if hasattr(b,'get_text') else type(b).__name__)
                         for a,b in issues]
                raise RuntimeError(f'{name} at {dpi} dpi: {pairs}; {edges}; {missing}')
            print(f'{name}: {dpi} dpi, 0 text overlaps, 0 edge issues')
    finally:
        fig.set_dpi(old)


def finish(fig, name):
    try:
        # fs.save uses tight bounds. Preserve the nominal print width so a
        # compact panel cannot silently enlarge its fonts in LaTeX. This
        # transparent extent guard accounts for the helper's 0.04 in padding.
        width,height = fig.get_size_inches()
        fig.add_artist(Rectangle((.04/width,.04/height),1-.08/width,1-.08/height,
                                 transform=fig.transFigure,facecolor='none',
                                 edgecolor='none',linewidth=0))
        validate(fig,name)
        fs.save(fig,name)
        validate(fig,name)
    finally:
        plt.close(fig)


def fig1(folder, synthetic):
    """Framework with counts and validation scores traced to saved results."""
    report = load_json(folder, 'surrogate_report.json')
    venue = load_json(folder, 'venue_convergence.json')
    selection = load_json(folder, 'selection.json')
    summary = load_json(folder, 'season_summary.json')
    groups = [v for v in report.values() if isinstance(v, dict) and 'targets' in v]
    n_device = sum(v['n'] for v in groups)
    r2 = min(t['loo_r2'] for v in groups for t in v['targets'].values())
    n_venue = len(venue['cases'])
    rec = selection['marked']['recommended']
    fig = fs.new_figure(height=3.2)
    heading(fig, 'Integrated framework for plaza cooling decisions',
            f'{n_venue + n_device:,} primary CFD cases support {summary["hours"]:,} hours of seasonal evaluation.', synthetic)
    # A serpentine reading path provides room for useful text at print size.
    # Everything is native Matplotlib vector geometry, including the connectors.
    ax = fig.add_axes([.025, .045, .95, .72]); ax.set_axis_off()
    positions = [(.005,.56),(.35,.56),(.695,.56),(.695,.04),(.35,.04),(.005,.04)]
    w, h = .30, .43
    boxes = [
        ('Data inputs', 'OSM geometry\nERA5 hourly weather\nEquipment prices and tariffs'),
        ('Layer 1 · CFD', f'{n_venue} venue wind cases\n{n_device} device cases\nBody-level delivery fields'),
        ('Layer 2 · Surrogate', f'POD + Gaussian processes\nMinimum LOO R² = {r2:.3f}\nFast spatial prediction'),
        ('Layer 3 · Simulation', f'{summary["hours"]:,} simulated hours\nSun, shade and roof state\nHourly visitor occupancy'),
        ('Layer 4 · Optimiser', 'NSGA-II design search\nHourly device dispatch\nCost–comfort priority dial'),
        ('Decision outputs', f'{rec["n_dev"]} devices in the selection\n{summary["dispatch"]["P_safe"]:.1%} seasonal protection\nCost, energy and robustness'),
    ]
    for i, ((x,y),(name,body)) in enumerate(zip(positions,boxes)):
        face = fs.BLUE_CMAP(.10) if i in (1,2,3,4) else '#f6f7f9'
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor=face,edgecolor='#c6ccd5',linewidth=.8))
        ax.text(x+.015,y+h-.045,name,fontsize=11,fontweight='bold',va='top')
        ax.text(x+.015,y+h-.165,body,fontsize=8.5,linespacing=1.4,va='top')
    for start,end in [((.305,.775),(.35,.775)),((.65,.775),(.695,.775)),
                      ((.845,.56),(.845,.47)),((.695,.255),(.65,.255)),
                      ((.35,.255),(.305,.255))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,
                                    color=fs.INK_SECONDARY,linewidth=1))
    return fig


def f2(folder, synthetic):
    with np.load(folder/'maps.npz', allow_pickle=True) as z:
        d = {k:z[k] for k in z.files}
    xy, poly = d['xy'], d['plaza_poly']
    delta = d['AT_cooled'] - d['AT_base']
    reduction = -float(np.mean(delta))
    fig = fs.new_figure(height=4.2)
    heading(fig, 'Spatial wind delivery and plaza cooling',
            f'Area-mean AT falls by {reduction:.1f} °C at {int(d["hour"]):02d}:00 on {str(d["date"])}.', synthetic)
    # Crop by the occupied polygon alone; remote buildings cannot set limits.
    lower, upper = poly.min(axis=0)-15, poly.max(axis=0)+15
    delta_limit = max(float(np.percentile(np.abs(delta), 95)), .1)
    at_min = float(min(d['AT_base'].min(), d['AT_cooled'].min()))
    at_max = float(max(d['AT_base'].max(), d['AT_cooled'].max()))
    from matplotlib.colors import Normalize
    at_norm = Normalize(at_min, at_max)
    vals = [d['wind_ratio'], d['AT_base'], delta, d['AT_cooled']]
    norms = [Normalize(0, float(max(d['wind_ratio']))), at_norm,
             fs.diverging_norm(0, -delta_limit, delta_limit), at_norm]
    cmaps = [fs.BLUE_CMAP, fs.ORANGE_CMAP, fs.DIVERGING_CMAP, fs.ORANGE_CMAP]
    names = ['NW reference wind', 'Baseline temperature',
             'Apparent-temperature change', 'Cooled temperature']
    top_row = .44 if np.count_nonzero([np.any(d['design']==code) for code in range(1,5)])>1 else .465
    positions = [[.025,top_row,.37,.255],[.525,top_row,.37,.255],
                 [.025,.105,.37,.255],[.525,.105,.37,.255]]
    axes, maps = [], []
    cells = xy[:,None,:] + np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    for i,(v,norm,cmap,pos,name) in enumerate(zip(vals,norms,cmaps,positions,names)):
        ax = fig.add_axes(pos); axes.append(ax)
        # Rasterise only the sampled field. Adjacent vector cell polygons show
        # white stitching seams in PDF viewers; geometry and type stay vector.
        col = PolyCollection(cells, array=v, cmap=cmap, norm=norm, edgecolors='none',
                             linewidths=0, rasterized=True)
        ax.add_collection(col); maps.append(col)
        col.set_clip_path(Polygon(poly, transform=ax.transData))
        for b in d['buildings']:
            ax.add_patch(Polygon(np.asarray(b,float), facecolor='#dedfe3',
                                 edgecolor=fs.NEUTRAL_GREY, linewidth=.5))
        ax.add_patch(Polygon(poly, fill=False, edgecolor=fs.INK_PRIMARY, linewidth=.8))
        ax.set(xlim=(lower[0],upper[0]), ylim=(lower[1],upper[1]), aspect='equal')
        ax.set_axis_off()
        panel(ax,chr(97+i),name)
        ax.title.set_position((0, 1.015))
        # White rings make empty slots distinct from equipment of any family.
        if i in (0,3):
            pts = d['slots'][d['design']==0]
            ax.scatter(pts[:,0],pts[:,1],s=13,facecolors='white',edgecolors=fs.NEUTRAL_GREY,
                       linewidths=.65,zorder=5)
            for code,c in enumerate('AMED',1):
                pts = d['slots'][d['design']==code]
                ax.scatter(pts[:,0],pts[:,1],s=26,marker=fs.DEVICE_MARKERS[c],
                           facecolor=fs.DEVICE_COLORS[c],edgecolor=fs.INK_PRIMARY,linewidth=.65,zorder=6)
        # All orientation furniture occupies the 15 m margin, outside the field.
        if i==0:
            ax.add_patch(Rectangle(lower+np.array([2,1]),32,12,facecolor='white',edgecolor='none',zorder=3))
            x0, y0 = lower + np.array([5,4])
            ax.plot([x0,x0+20],[y0,y0],color=fs.INK_PRIMARY,lw=1.8,zorder=4)
            ax.text(x0+10,y0+2,'20 m',ha='center',va='bottom',fontsize=8,zorder=4)
            theta=np.deg2rad(float(d['plaza_angle_deg']))
            anchor=np.array([upper[0]-7,upper[1]-13])
            # World angles become local angles by subtracting the plaza bearing;
            # north therefore points (sin(bearing), cos(bearing)) in this frame.
            tip=anchor+np.array([np.sin(theta),np.cos(theta)])*10
            ax.add_patch(Rectangle(upper-np.array([16,13]),16,13,facecolor='white',edgecolor='none',zorder=3))
            ax.annotate('',xy=tip,xytext=anchor,
                        arrowprops=dict(arrowstyle='-|>',lw=.8,color=fs.INK_SECONDARY,
                                        shrinkA=0,shrinkB=0,mutation_scale=8),zorder=4)
            ax.text(tip[0],tip[1]+1,'N',ha='center',va='bottom',fontsize=8,fontweight='bold',zorder=4)
        if i==3 and np.nanmin(d['AT_cooled']) < SAFE < np.nanmax(d['AT_cooled']):
            tri=mtri.Triangulation(xy[:,0],xy[:,1])
            tri.set_mask(~MplPath(poly).contains_points(xy[tri.triangles].mean(axis=1)))
            contour=ax.tricontour(tri,d['AT_cooled'],levels=[SAFE],colors=fs.INK_PRIMARY,linewidths=.8)
            contour.set_clip_path(Polygon(poly,transform=ax.transData))
    for m,pos,label,ticks,extend in [
        (maps[0],[.405,top_row+.009,.012,.235],'Wind ratio (U/U10)',None,'neither'),
        (maps[1],[.905,.25,.012,.39],'AT, baseline and cooled (°C)',None,'neither'),
        (maps[2],[.405,.114,.012,.235],'ΔAT (°C)',[-delta_limit,0,delta_limit],'both'),
    ]:
        cax=fig.add_axes(pos); cb=fig.colorbar(m,cax=cax,extend=extend)
        cb.set_label(label,labelpad=3); cb.outline.set_visible(False); cax.grid(False)
        if ticks is not None: cb.set_ticks(ticks, labels=[f'{v:.1f}' for v in ticks])
        else: cb.locator=MaxNLocator(3); cb.update_ticks()
    active = [c for code,c in enumerate('AMED',1) if np.any(d['design']==code)]
    labels={'A':'Fan','M':'Misting fan','E':'Evaporative','D':'Chilled air'}
    handles=[Line2D([],[],ls='none',marker=fs.DEVICE_MARKERS[c],markerfacecolor=fs.DEVICE_COLORS[c],
                    markeredgecolor=fs.INK_PRIMARY,label=labels[c]+' (used)') for c in active]
    handles += [Line2D([],[],ls='none',marker='o',markerfacecolor='white',markeredgecolor=fs.NEUTRAL_GREY,
                       label='Unused slot'),Patch(facecolor='none',edgecolor=fs.INK_PRIMARY,linewidth=.8,label='Occupied plaza'),
                Line2D([],[],color=fs.INK_PRIMARY,lw=.8,label=f'Cooled AT = {SAFE:g} °C')]
    legend(fig,handles,y=.775 if len(handles)>4 else .80,x=.035,ncol=3 if len(handles)>4 else None)
    fig.text(.035,.025,f'Wind: NW/closed reference; AT: design hour. ΔAT scale: 95th percentile (±{delta_limit:.1f} °C).',
             fontsize=8,color=fs.INK_SECONDARY)
    return fig


def marked_designs(d):
    front = d['designs']
    cost = np.array([v['K_usd'] for v in front]); heat = np.array([v['H'] for v in front])
    c = (cost-cost.min())/max(np.ptp(cost),1e-12)
    h = (heat-heat.min())/max(np.ptp(heat),1e-12)
    knee = int(np.argmax(1-c-h))
    fallback = dict(efficiency_optimum=int(np.argmax([v['eta'] for v in front])),knee=knee,recommended=knee)
    return {k: front[v] if isinstance(v,int) else v
            for k,v in ((k,d.get('marked',{}).get(k,v)) for k,v in fallback.items())}


def f3(folder, synthetic):
    d=load_json(folder,'pareto.json')
    selection=load_json(folder,'selection.json')
    kp=load_json(folder,'pareto_KP.json')
    front=sorted(d['designs'],key=lambda v:v['K_usd'])
    marked=marked_designs(d); rec=marked['recommended']
    fig=fs.new_figure(height=4.2)
    heading(fig,'Cost, heat burden and attainable protection',
            f'The selected {rec["n_dev"]}-device layout protects {rec["P_safe"]:.1%} of representative person-hours.',synthetic)
    ax=fig.add_axes([.10,.405,.465,.365])
    protect=fig.add_axes([.715,.56,.26,.21])
    conv=fig.add_axes([.715,.29,.26,.105])
    panel(ax,'a','Cost–burden front'); panel(protect,'b','Attainable protection')
    panel(conv,'c','Search convergence')
    focus=front+list(marked.values())
    costs=np.array([v['K_usd'] for v in focus]); heats=np.array([v['H'] for v in focus])
    dx=max(np.ptp(costs),1); dy=max(np.ptp(heats),1e-3)
    xlim=(costs.min()-.035*dx,costs.max()+.06*dx)
    ylim=(heats.min()-.09*dy,heats.max()+.10*dy)
    dom=d.get('dominated',[])
    visible=[v for v in dom if xlim[0]<=v['K_usd']<=xlim[1] and ylim[0]<=v['H']<=ylim[1]]
    if visible:
        ax.scatter([v['K_usd']/1000 for v in visible],[v['H'] for v in visible],s=10,
                   color=fs.NEUTRAL_GREY,alpha=.32,edgecolors='none')
    ax.step([v['K_usd']/1000 for v in front],[v['H'] for v in front],where='post',color=fs.FAN)
    ax.legend(handles=[Line2D([],[],color=fs.FAN,label='Pareto front'),
                       Line2D([],[],ls='none',marker='o',markerfacecolor=fs.NEUTRAL_GREY,
                              markeredgecolor='none',alpha=.5,label='Dominated samples')],
              loc='upper right',borderaxespad=.3,handlelength=1.5)
    ax.set(xlim=np.array(xlim)/1000,ylim=ylim,xlabel='Annual cost (thousand USD/year)',
           ylabel='Heat burden (K²)')
    ax.set_xticks([0,2,4,6,8]); ax.yaxis.set_major_locator(MaxNLocator(4))
    # Offset labels stay in the axes and near their exact marked design.
    specs=[('efficiency_optimum','Efficiency','s',(12,4),'left','bottom'),
           ('recommended','Selected','D',(-8,-13),'right','top'),
           ('knee','Knee','^',(14,13),'left','bottom')]
    for key,name,marker,offset,ha,va in specs:
        v=marked[key]
        ax.plot(v['K_usd']/1000,v['H'],marker=marker,color=fs.INK_PRIMARY,
                markeredgecolor='white',linestyle='none',markersize=6,zorder=6)
        ax.annotate(name,xy=(v['K_usd']/1000,v['H']),xytext=offset,textcoords='offset points',
                    ha=ha,va=va,fontsize=8.5,
                    bbox=dict(facecolor='white',edgecolor='none',pad=.6,alpha=.92),
                    arrowprops=dict(arrowstyle='-',lw=.8,color=fs.INK_SECONDARY),zorder=7)
    pf=sorted(kp['designs'],key=lambda v:(v['K_usd'],-v['P_safe']))
    protect.step([v['K_usd']/1000 for v in pf],[100*v['P_safe'] for v in pf],where='post',color=fs.FAN)
    protect.plot(rec['K_usd']/1000,100*rec['P_safe'],marker='D',color=fs.INK_PRIMARY,ls='none',markersize=5)
    protect.set(xlim=np.array(xlim)/1000,xlabel='Cost (thousand USD/year)',ylabel='Protected share (%)')
    pv=np.array([v['P_safe']*100 for v in pf]+[rec['P_safe']*100])
    protect.set_ylim(pv.min()-np.ptp(pv)*.08,pv.max()+np.ptp(pv)*.14)
    protect.set_xticks([0,4,8]); protect.yaxis.set_major_locator(MaxNLocator(3))
    hist=d['history']; hv=np.array([v['hv'] for v in hist]); gen=[v['gen'] for v in hist]
    conv.plot(gen,100*hv/hv.max(),color=fs.INK_PRIMARY,lw=1.6)
    conv.set(xlim=(0,max(gen)),ylim=(0,105),xlabel='Generation',ylabel='HV (%)')
    conv.set_xticks([0,100,200]); conv.set_yticks([0,100]); conv.tick_params(pad=2)
    conv.xaxis.labelpad=1; conv.yaxis.labelpad=2
    # A compact, ruled strip connects priorities to the actual saved designs.
    fig.text(.035,.205,'d  Operator priority dial',fontsize=11,fontweight='bold')
    dial=selection['priority_dial']
    for i,v in enumerate(dial):
        x=.055+i*.19; design=v['design']; n=design['n_dev']
        chosen=(design['x']==rec['x'] and np.isclose(design.get('mu',0),rec.get('mu',0)))
        fig.add_artist(Rectangle((x-.015,.055),.176,.139,transform=fig.transFigure,
                                facecolor=fs.BLUE_CMAP(.10) if chosen else '#f6f7f9',edgecolor='none'))
        fig.text(x,.169,f'p = {v["p"]:g}',fontsize=8.5,fontweight='bold' if chosen else 'normal')
        fig.text(x,.126,f'{n} devices · ${design["K_usd"]:,.0f}/yr',fontsize=8)
        fig.text(x,.083,f'{design["P_safe"]:.1%} protected',fontsize=8)
    # Explain the crop explicitly; the front and every marked design are intact.
    fig.text(.10,.255,f'{len(dom)-len(visible)} dominated samples beyond the shown range are omitted.',
             fontsize=8,color=fs.INK_SECONDARY)
    fig.text(.035,.015,'Priority: p = 0 favours cost; p = 1 favours comfort. Protection uses the saved optimisation evaluation.',
             fontsize=8,color=fs.INK_SECONDARY)
    return fig


def f4(folder, synthetic):
    d = load_json(folder,'designday.json'); h = np.asarray(d['hours'])
    fig = fs.new_figure(height=4.2)
    heading(fig,'Hourly conditions and cooling on the design day',
            f'Hourly cooling uses {sum(d["energy_kWh"]):.0f} kWh on {d["date"]}.',synthetic)
    axes = []
    # Reserve room above each panel for its title and the preceding axis ticks
    # while retaining the 4.2-inch figure height.
    for y,height in [(.73,.07),(.545,.075),(.36,.075),(.17,.10)]:
        axes.append(fig.add_axes([.095,y,.72,height],sharex=axes[0] if axes else None))
    weather,occ,on,at = axes
    for ax in axes:
        ax.set_xlim(0,24); ax.set_xticks([0,6,12,18,24]); ax.yaxis.set_major_locator(MaxNLocator(2))
        ax.tick_params(pad=2)
        if ax is not at: ax.tick_params(labelbottom=False)
    # Values are beginning-of-hour observations; extend the last hour to 24:00.
    weather.plot(np.r_[h,24],np.r_[d['T'],d['T'][-1]],color=fs.INK_PRIMARY)
    weather.set_ylabel('T (°C)'); weather.set_ylim(min(d['T'])-1,max(d['T'])+2)
    panel(weather,'a','Weather')
    weather.text(.48,1.19,f'RH {min(d["RH"]):.0f}–{max(d["RH"]):.0f}%; wind {min(d["w10"]):.1f}–{max(d["w10"]):.1f} m/s',
                 transform=weather.transAxes,fontsize=fs.ANNOTATION_SIZE)
    occ.fill_between(np.r_[h,24],np.r_[d['occupancy'],d['occupancy'][-1]],
                     step='post',color=fs.BLUE_CMAP(.45),linewidth=0)
    occ.step(np.r_[h,24],np.r_[d['occupancy'],d['occupancy'][-1]],where='post',color=fs.FAN)
    for hour,closed in zip(h,d['roof_closed']):
        if closed: occ.axvspan(hour,hour+1,color=fs.NEUTRAL_GREY,alpha=.18,lw=0,zorder=0)
    occ.set_ylabel('People'); occ.set_ylim(0,max(d['occupancy'])*1.12)
    panel(occ,'b','Occupancy')
    occ.text(.55,1.19,'Grey shading: roof closed',transform=occ.transAxes,fontsize=fs.ANNOTATION_SIZE)
    bottom = np.zeros(24)
    for c in 'AMED':
        v = np.asarray(d['on_counts'][c])
        # The hourly gap is 2 pt at print scale; avoid thick vertical white
        # strokes that would erase an entire one-device stack segment.
        gap = 2/(.72*fs.TEXT_WIDTH*72/24)
        on.bar(h+.5,v,bottom=bottom,width=1-gap,align='center',color=fs.DEVICE_COLORS[c],
               hatch='////' if c=='E' else None,
               edgecolor=fs.INK_SECONDARY if c=='D' else 'white',
               linewidth=.65 if c=='D' else .25)
        bottom += v
    on.set_ylabel('Devices on'); on.set_ylim(0,max(max(bottom)*1.12,1))
    on.yaxis.set_major_locator(MaxNLocator(2,integer=True))
    panel(on,'c','Device dispatch')
    device_names = {'A':'Fan','M':'Misting fan','E':'Evaporative cooler','D':'Chilled air'}
    handles = [Patch(facecolor=fs.DEVICE_COLORS[c],hatch='////' if c=='E' else None,
                     edgecolor=fs.INK_SECONDARY if c=='D' else 'white',label=device_names[c])
               for c in 'AMED' if np.any(d['on_counts'][c])]
    fig.legend(handles=handles,bbox_to_anchor=(.46,.455),loc='lower left',ncol=4,
               borderaxespad=0,handlelength=1,columnspacing=1.3)
    for key,col,ls in [('AT_base_mean',fs.NEUTRAL_GREY,'-'),('AT_base_p90',fs.NEUTRAL_GREY,'--'),
                        ('AT_cooled_mean',fs.FAN,'-'),('AT_cooled_p90',fs.FAN,'--')]:
        at.plot(np.r_[h,24],np.r_[d[key],d[key][-1]],color=col,linestyle=ls,lw=1.6)
    lo = min(SAFE-2,min(d['AT_cooled_mean'])-1)
    hi = max(SAFE+2,max(d['AT_base_p90'])+1)
    at.axhspan(lo,SAFE,color=fs.STATUS_COLORS['Safe'],alpha=.09,lw=0)
    at.axhline(SAFE,color=fs.INK_SECONDARY,ls='--',lw=.8)
    at.set(ylim=(lo,hi),ylabel='AT (°C)',xlabel='Local time (h)')
    panel(at,'d','Apparent temperature')
    at.text(.55,1.19,rf'Safe: $AT \leq {SAFE:g}\,^{{\circ}}\mathrm{{C}}$',
            transform=at.transAxes,fontsize=fs.ANNOTATION_SIZE)
    for key,label,y in [('AT_cooled_p90','90th percentile',.80),('AT_cooled_mean','Mean',.25)]:
        at.annotate(label,xy=(24,d[key][-1]),xycoords='data',xytext=(1.025,y),
                    textcoords='axes fraction',fontsize=fs.ANNOTATION_SIZE,va='center',
                    arrowprops=dict(arrowstyle='-',color=fs.INK_SECONDARY,lw=.8))
    legend(fig,[Line2D([],[],color=fs.NEUTRAL_GREY,label='Baseline'),
                Line2D([],[],color=fs.FAN,label='Cooled'),
                Line2D([],[],color=fs.INK_PRIMARY,label='Mean'),
                Line2D([],[],color=fs.INK_PRIMARY,ls='--',label='90th percentile')],y=.012)
    return fig


def f5(folder, synthetic, tariff):
    with np.load(folder/'season.npz',allow_pickle=True) as z:
        d={k:z[k] for k in z.files}
    summary=load_json(folder,'season_summary.json')
    dispatch,constant=float(d['dispatch_kWh'].sum()),float(d['constant_kWh'].sum())
    saving=100*(1-dispatch/constant) if constant else 0
    fig=fs.new_figure(height=3.8)
    heading(fig,'Seasonal dispatch and electricity savings',
            f'Hourly operation saves {saving:.1f}% electricity with {summary["dispatch"]["P_safe"]:.2%} seasonal protection.',synthetic)
    ax=fig.add_axes([.085,.31,.45,.445]); panel(ax,'a','Hourly electricity demand')
    im=ax.imshow(d['dispatch_kWh'],aspect='auto',origin='upper',interpolation='nearest',cmap=fs.BLUE_CMAP,
                 extent=(0,24,len(d['days'])-.5,-.5),vmin=0)
    import datetime
    months=np.array([str(day)[:7] for day in d['days']])
    ticks=np.r_[0,np.flatnonzero(months[1:]!=months[:-1])+1]
    ax.set_yticks(ticks,[datetime.date.fromisoformat(str(d['days'][i])).strftime('%b') for i in ticks])
    ax.set(xlabel='Local time (h)',ylabel='Season date'); ax.set_xticks([0,6,12,18,24]); ax.grid(False)
    cax=fig.add_axes([.085,.17,.45,.022]); cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    cb.set_label('Hourly electricity (kWh)',labelpad=3)
    ticks=np.linspace(0,float(d['dispatch_kWh'].max()),3)
    cb.set_ticks(ticks,labels=[f'{v:.1f}' for v in ticks]); cb.outline.set_visible(False); cax.grid(False)
    e_ax=fig.add_axes([.70,.51,.275,.24]); c_ax=fig.add_axes([.70,.17,.275,.19])
    panel(e_ax,'b','Seasonal electricity')
    panel(c_ax,'c','Electricity cost')
    for a,values,label in [(e_ax,np.array([constant,dispatch])/1000,'Energy (MWh)'),
                            (c_ax,np.array([constant,dispatch])*tariff,'Cost (USD)')]:
        a.bar([0,1],values,color=[fs.NEUTRAL_GREY,fs.FAN],width=.58)
        a.set_ylabel(label,labelpad=3); a.set_ylim(0,values.max()*1.30)
        a.set_xticks([0,1],['Constant','Hourly'])
        a.yaxis.set_major_locator(MaxNLocator(3))
        for x,v in enumerate(values):
            a.text(x,v+values.max()*.035,f'{v:,.1f}' if a is e_ax else f'{v:,.0f}',
                   ha='center',va='bottom',fontsize=8.5)
    fig.text(.035,.021,f'Full-season protection: {summary["constant"]["P_safe"]:.2%} constant; '
             f'{summary["dispatch"]["P_safe"]:.2%} hourly. Electricity tariff: ${tariff:g}/kWh.',
             fontsize=8,color=fs.INK_SECONDARY)
    return fig


def f6(folder, synthetic):
    d=load_json(folder,'robustness.json'); s=d['sobol']
    out=s['outputs'].index('H') if 'H' in s['outputs'] else 0
    st,s1,conf=(np.asarray(s[k],float)[out] for k in ['ST','S1','ST_conf'])
    order=np.argsort(-st,kind='stable'); regret=np.asarray(d['regret'],float)
    robust=d['designs'].index(d['robust_choice'])
    choice=d['robust_choice'].replace('p=.','p=0.')
    fig=fs.new_figure(height=4.0)
    heading(fig,'Input sensitivity and scenario regret',
            f'The {choice} selection has {regret[robust].max():.1f}% worst-case regret across {regret.shape[1]} scenarios.',synthetic)
    ax=fig.add_axes([.265,.20,.235,.55]); heat=fig.add_axes([.665,.285,.315,.465])
    panel(ax,'a','Heat-burden sensitivity'); panel(heat,'b','Scenario regret (%)')
    names={'wind_scale':'Wind speed','T_off':'Air temperature','RH_off':'Relative humidity',
           'occ_scale':'Occupancy','tariff':'Electricity tariff','cop_dc':'Cooling efficiency',
           'device_eff':'Device effectiveness','roof_tau':'Roof transmissivity',
           'AT_safe':'Safety threshold','elasticity_scale':'Comfort elasticity'}
    y=np.arange(len(order))
    ax.barh(y,st[order],height=.65,color=fs.BLUE_CMAP(.4),edgecolor='white',linewidth=2,
            xerr=conf[order],error_kw=dict(ecolor=fs.INK_SECONDARY,lw=.8,capsize=2))
    ax.barh(y,s1[order],height=.20,color=fs.FAN,edgecolor='none',linewidth=0)
    ax.set_yticks(y,[names.get(s['inputs'][i],s['inputs'][i].replace('_',' ')) for i in order]); ax.invert_yaxis()
    ax.set_xlabel('Sobol index (dimensionless)')
    ax.set_xlim(0,max(st+conf)*1.08); ax.set_xticks([0,.2,.4])
    ax.axvline(0,color=fs.INK_SECONDARY,lw=.8); ax.spines['left'].set_visible(False)
    ax.grid(axis='x'); ax.grid(axis='y',visible=False); ax.tick_params(axis='y',length=0,pad=5)
    # Very small indices are explicitly reported rather than looking missing.
    for j,i in enumerate(order):
        if st[i]<.001: ax.text(.008,j,'<0.001',va='center',fontsize=8,color=fs.INK_SECONDARY)
    image=heat.imshow(regret,cmap=fs.BLUE_CMAP,vmin=0,vmax=max(regret.max(),1),aspect='auto',interpolation='nearest')
    scenarios=[v.replace('Humid 14 d','Humid\n14 days').replace('Extreme heat','Extreme\nheat').replace('Season +2 C','Season\n+2 °C') for v in d['scenarios']]
    heat.set_xticks(np.arange(len(scenarios)),scenarios,fontsize=8)
    # The result may include a separately searched minimax design. Wrap its
    # identifier, preserving the saved KH-front index without crowding the bars.
    design_labels=[v.replace('p=.','p=0.').replace(' (','\n(') for v in d['designs']]
    heat.set_yticks(np.arange(len(d['designs'])),design_labels)
    heat.tick_params(length=0,pad=5); heat.grid(False)
    for row in range(regret.shape[0]):
        for col in range(regret.shape[1]):
            t=heat.text(col,row,f'{regret[row,col]:.1f}',ha='center',va='center',fontsize=8,color=fs.INK_PRIMARY)
            t.set_bbox(dict(facecolor='white',edgecolor='none',pad=.7,alpha=.93))
    heat.add_patch(Rectangle((-.5,robust-.5),regret.shape[1],1,fill=False,edgecolor=fs.INK_PRIMARY,linewidth=1.4,clip_on=False))
    heat.spines[['left','bottom']].set_visible(False)
    heat.get_yticklabels()[robust].set_fontweight('bold')
    cax=fig.add_axes([.665,.145,.315,.018]); cb=fig.colorbar(image,cax=cax,orientation='horizontal')
    regret_label=('Regret (% of scenario objective range)' if 'range' in d.get('regret_units','')
                  else 'Regret relative to scenario optimum (%)')
    cb.set_label(regret_label,fontsize=8.5,labelpad=3)
    cb.set_ticks(np.linspace(0,max(regret.max(),1),3)); cb.outline.set_visible(False); cax.grid(False)
    legend(fig,[Patch(facecolor=fs.BLUE_CMAP(.4),label='Total order ± confidence interval'),
                Patch(facecolor=fs.FAN,label='First order')],y=.025,x=.035,ncol=2)
    return fig


# ---------------------------------------------------------------------------
# Paper-layout panels: separate PDFs sized for their printed width so that no
# text is scaled in LaTeX. Output: figures/paper/p_*.{pdf,png}.
# Run: .venv/Scripts/python src/figures.py --paper
# ---------------------------------------------------------------------------
PAPER_DIR = ROOT/'figures'/'paper'
TITLE_PT = 10


def _pfig(width, height):
    fig = plt.figure(figsize=(width,height))
    fig.set_layout_engine('constrained',w_pad=.03,h_pad=.03,wspace=.04,hspace=.04)
    return fig


def _ptitle(ax, text):
    ax.set_title(text, loc='left', fontsize=TITLE_PT, fontweight='bold', pad=4)


def _pfinish(fig, name):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    validate(fig, name)
    with matplotlib.rc_context({'pdf.fonttype':42}):
        fig.savefig(PAPER_DIR/f'{name}.pdf', facecolor='white', bbox_inches=None)
        fig.savefig(PAPER_DIR/f'{name}.png', dpi=300, facecolor='white', bbox_inches=None)
    plt.close(fig)


def _venue_panel(d, which, width=3.4, height=1.75):
    from matplotlib.colors import Normalize
    xy, poly = d['xy'], d['plaza_poly']
    lower, upper = poly.min(axis=0)-9, poly.max(axis=0)+9
    lower[1] -= 9
    delta = d['AT_cooled'] - d['AT_base']
    lim = max(float(np.percentile(np.abs(delta), 95)), .1)
    if which == 'wind':
        v, norm, cmap = d['wind_ratio'], Normalize(0, float(d['wind_ratio'].max())), fs.BLUE_CMAP
        label, title = 'Wind ratio U/U10', 'a  Reference wind and device slots'
    else:
        v, norm, cmap = delta, fs.diverging_norm(0,-lim,lim), fs.DIVERGING_CMAP
        label, title = 'Change in AT (°C)', 'b  AT change, 8 E at 18:00'
    fig = _pfig(width, height); ax = fig.add_subplot(111)
    cells = xy[:,None,:] + np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    col = PolyCollection(cells, array=v, cmap=cmap, norm=norm, edgecolors='none', rasterized=True)
    ax.add_collection(col); col.set_clip_path(Polygon(poly, transform=ax.transData))
    for b in d['buildings']:
        ax.add_patch(Polygon(np.asarray(b,float), facecolor='#dedfe3', edgecolor=fs.NEUTRAL_GREY, linewidth=.5))
    ax.add_patch(Polygon(poly, fill=False, edgecolor=fs.INK_PRIMARY, linewidth=.8))
    ax.set(xlim=(lower[0],upper[0]), ylim=(lower[1],upper[1]), aspect='equal'); ax.set_axis_off()
    pts = d['slots'][d['design']==0]
    ax.scatter(pts[:,0],pts[:,1],s=11,facecolors='white',edgecolors=fs.NEUTRAL_GREY,linewidths=.6,zorder=5)
    for code,c in enumerate('AMED',1):
        p = d['slots'][d['design']==code]
        if len(p): ax.scatter(p[:,0],p[:,1],s=22,marker=fs.DEVICE_MARKERS[c],facecolor=fs.DEVICE_COLORS[c],
                              edgecolor=fs.INK_PRIMARY,linewidth=.6,zorder=6)
    x0,y0 = lower+np.array([2,2])
    ax.add_patch(Rectangle((x0-1,y0-1.5),26,9,facecolor='white',edgecolor='none',zorder=3))
    ax.plot([x0,x0+20],[y0,y0],color=fs.INK_PRIMARY,lw=1.6,zorder=4)
    ax.text(x0+10,y0+1.2,'20 m',ha='center',va='bottom',fontsize=8,zorder=4)
    _ptitle(ax,title)
    cb = fig.colorbar(col, ax=ax, orientation='horizontal', fraction=.07, pad=.02, shrink=.8,
                      extend='both' if which!='wind' else 'neither')
    cb.set_label(label, fontsize=8.5, labelpad=2); cb.outline.set_visible(False); cb.ax.grid(False)
    cb.ax.tick_params(labelsize=8)
    if which != 'wind': cb.set_ticks([-lim,0,lim], labels=[f'{-lim:.1f}','0',f'{lim:.1f}'])
    else:
        cb.locator = MaxNLocator(3); cb.update_ticks()
    return fig


def paper_panels(folder):
    with np.load(folder/'maps.npz', allow_pickle=True) as z:
        maps = {k:z[k] for k in z.files}
    _pfinish(_venue_panel(maps,'wind'), 'p_venue_wind')
    _pfinish(_venue_panel(maps,'dat'), 'p_venue_dat')

    # --- Pareto panels -----------------------------------------------------
    d = load_json(folder,'pareto.json'); kp = load_json(folder,'pareto_KP.json')
    sel = load_json(folder,'selection.json')['marked']
    sup = load_json(folder,'summary_support.json')['representative_hour_attainability']
    front = sorted(d['designs'], key=lambda v:v['K_usd'])
    specs = [('efficiency_optimum','Efficiency','s',(12,-6),'left','top'),
             ('recommended','Balanced','D',(-8,-14),'right','top'),
             ('knee','Knee','^',(12,14),'left','bottom')]
    fig = _pfig(2.45,1.85); ax = fig.add_subplot(111)
    ax.step([v['K_usd']/1000 for v in front],[v['H'] for v in front],where='post',color=fs.FAN)
    for key,name,mk,off,ha,va in specs:
        v = sel[key]
        ax.plot(v['K_usd']/1000,v['H'],marker=mk,color=fs.INK_PRIMARY,markeredgecolor='white',ls='none',markersize=6,zorder=6)
        ax.annotate(name,xy=(v['K_usd']/1000,v['H']),xytext=off,textcoords='offset points',ha=ha,va=va,fontsize=8,
                    arrowprops=dict(arrowstyle='-',lw=.7,color=fs.INK_SECONDARY),zorder=7)
    ax.set(xlabel='Annual cost (thousand USD)',ylabel='Heat burden H (K²)'); ax.set_xticks([0,2,4,6,8])
    ax.yaxis.set_major_locator(MaxNLocator(4)); _ptitle(ax,'a  Cost and burden')
    _pfinish(fig,'p_front_kh')

    pf = sorted(kp['designs'], key=lambda v:(v['K_usd'],-v['P_safe']))
    fig = _pfig(2.45,1.85); ax = fig.add_subplot(111)
    ax.step([v['K_usd']/1000 for v in pf],[100*v['P_safe'] for v in pf],where='post',color=fs.FAN)
    rec = sel['recommended']
    ax.plot(rec['K_usd']/1000,100*rec['P_safe'],marker='D',color=fs.INK_PRIMARY,markeredgecolor='white',ls='none',markersize=6)
    ax.annotate('Balanced',xy=(rec['K_usd']/1000,100*rec['P_safe']),xytext=(10,-14),textcoords='offset points',fontsize=8,
                arrowprops=dict(arrowstyle='-',lw=.7,color=fs.INK_SECONDARY))
    ax.set(xlabel='Annual cost (thousand USD)',ylabel='Protected share (%)',xlim=(-.3,26.8))
    ax.set_xticks([0,10,20]); ax.yaxis.set_major_locator(MaxNLocator(4)); _ptitle(ax,'b  Cost and protection')
    _pfinish(fig,'p_front_kp')

    fig = _pfig(1.95,1.85); ax = fig.add_subplot(111)
    keys = ['none','24_A','24_M','24_E','24_D']; labs = ['None','24 A','24 M','24 E','24 D']
    vals = [100*sup[k]['P_safe'] for k in keys]
    cols = [fs.NEUTRAL_GREY,fs.FAN,fs.EVAPORATIVE,fs.EVAPORATIVE,fs.CHILLED_AIR]
    ax.bar(range(5),vals,color=cols,width=.7)
    ax.axhline(95,color=fs.STATUS_COLORS['Danger'],ls='--',lw=1)
    ax.text(-.4,89.5,'Target 95 %',fontsize=8,color=fs.INK_PRIMARY,va='top')
    for i,v in enumerate(vals): ax.text(i,v+1.5,f'{v:.0f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(range(5),labs,fontsize=7.5); ax.tick_params(axis='x',pad=2)
    ax.set(ylim=(0,100),ylabel='Protected share (%)'); ax.yaxis.set_major_locator(MaxNLocator(4))
    _ptitle(ax,'c  Attainability')
    _pfinish(fig,'p_attain')

    # --- Design day and season --------------------------------------------
    dd = load_json(folder,'designday.json'); h = np.asarray(dd['hours'])
    fig = _pfig(2.3,1.8); ax = fig.add_subplot(111)
    for hour,closed in zip(h,dd['roof_closed']):
        if closed: ax.axvspan(hour,hour+1,color=fs.NEUTRAL_GREY,alpha=.16,lw=0)
    for key,col,ls in [('AT_base_mean',fs.NEUTRAL_GREY,'-'),('AT_base_p90',fs.NEUTRAL_GREY,'--'),
                       ('AT_cooled_mean',fs.FAN,'-'),('AT_cooled_p90',fs.FAN,'--')]:
        ax.plot(np.r_[h,24],np.r_[dd[key],dd[key][-1]],color=col,ls=ls,lw=1.4)
    ax.axhline(SAFE,color=fs.INK_SECONDARY,ls=':',lw=1)
    ax.text(7,SAFE-.6,'Safe limit 32 °C',fontsize=8,va='top')
    ax.set(xlim=(0,24),ylim=(28,56),xlabel='Local time (h)',ylabel='AT (°C)'); ax.set_xticks([0,6,12,18,24])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.legend(handles=[Line2D([],[],color=fs.NEUTRAL_GREY,label='Uncooled'),Line2D([],[],color=fs.FAN,label='Cooled'),
                       Line2D([],[],color=fs.INK_PRIMARY,ls='-',label='Mean'),Line2D([],[],color=fs.INK_PRIMARY,ls='--',label='90th pct.')],
              loc='upper center',ncol=2,fontsize=7.5,handlelength=1.6,columnspacing=.8,borderaxespad=.2)
    _ptitle(ax,'a  Design-day AT'); _pfinish(fig,'p_dd_at')

    fig = _pfig(2.3,1.8); ax = fig.add_subplot(111)
    n_on = np.sum([np.asarray(dd['on_counts'][c]) for c in 'AMED'],axis=0)
    for hour,closed in zip(h,dd['roof_closed']):
        if closed: ax.axvspan(hour,hour+1,color=fs.NEUTRAL_GREY,alpha=.16,lw=0)
    ax.bar(h+.5,n_on,width=.8,color=fs.EVAPORATIVE,linewidth=0)
    ax.set(xlim=(0,24),ylim=(0,10),xlabel='Local time (h)',ylabel='E units on'); ax.set_xticks([0,6,12,18,24])
    ax.yaxis.set_major_locator(MaxNLocator(5,integer=True))
    ax.text(12,9.7,'Grey: roof closed',ha='center',va='top',fontsize=8)
    _ptitle(ax,'b  Dispatch'); _pfinish(fig,'p_dd_dev')

    with np.load(folder/'season.npz',allow_pickle=True) as z:
        s = {k:z[k] for k in z.files}
    import datetime
    fig = _pfig(2.3,1.8); ax = fig.add_subplot(111)
    im = ax.imshow(s['dispatch_kWh'],aspect='auto',origin='upper',interpolation='nearest',cmap=fs.BLUE_CMAP,
                   extent=(0,24,len(s['days'])-.5,-.5),vmin=0)
    months = np.array([str(x)[:7] for x in s['days']]); t = np.r_[0,np.flatnonzero(months[1:]!=months[:-1])+1]
    ax.set_yticks(t,[datetime.date.fromisoformat(str(s['days'][i])).strftime('%b') for i in t])
    ax.set(xlabel='Local time (h)'); ax.set_xticks([0,6,12,18,24]); ax.grid(False)
    cb = fig.colorbar(im,ax=ax,fraction=.06,pad=.02); cb.set_label('kWh per hour',fontsize=8.5,labelpad=2)
    cb.outline.set_visible(False); cb.ax.grid(False); cb.ax.tick_params(labelsize=8); cb.locator=MaxNLocator(3); cb.update_ticks()
    _ptitle(ax,'c  Season dispatch'); _pfinish(fig,'p_season')

    # --- Robustness --------------------------------------------------------
    rb = load_json(folder,'robustness.json'); S = rb['sobol']
    idx = {n:i for i,n in enumerate(S['inputs'])}
    inputs = ['tariff','T_off','AT_safe','RH_off','wind_scale','roof_tau']
    names = {'tariff':'Tariff','T_off':'Air temperature','AT_safe':'Safety limit','RH_off':'Humidity',
             'wind_scale':'Wind speed','roof_tau':'Roof transmittance'}
    fig = _pfig(3.4,2.0); ax = fig.add_subplot(111)
    y = np.arange(len(inputs)); hgt = .36
    for out,col,lab,off in [('K_usd',fs.NEUTRAL_GREY,'Cost K',-hgt/2),('P_safe',fs.FAN,'Protected share',hgt/2)]:
        o = S['outputs'].index(out)
        st = [S['ST'][o][idx[n]] for n in inputs]; cf = [S['ST_conf'][o][idx[n]] for n in inputs]
        ax.barh(y+off,st,height=hgt,color=col,label=lab,linewidth=0,xerr=cf,
                error_kw=dict(ecolor=fs.INK_SECONDARY,lw=.7,capsize=1.5))
    ax.set_yticks(y,[names[n] for n in inputs],fontsize=8); ax.invert_yaxis()
    ax.set(xlabel='Total-order Sobol index',xlim=(0,1.05)); ax.grid(axis='x'); ax.grid(axis='y',visible=False)
    ax.tick_params(axis='y',length=0); ax.legend(loc='lower right',fontsize=8,borderaxespad=.2)
    _ptitle(ax,'a  Sobol sensitivity'); _pfinish(fig,'p_sobol')

    reg = np.asarray(rb['regret'],float)
    keep = [i for i,n in enumerate(rb['designs']) if n in ('p=.25','p=.5','p=.75','Efficiency','Minimax (KH 325)')]
    lab = {'p=.25':'1 A + 4 E (p = 0.25)','p=.5':'8 E (p = 0.5)','p=.75':'1 M + 12 E (p = 0.75)',
           'Efficiency':'1 M (efficiency)','Minimax (KH 325)':'12 E (minimax)'}
    fig = _pfig(3.4,2.0); ax = fig.add_subplot(111)
    sub = reg[keep]
    ax.imshow(sub,cmap=fs.BLUE_CMAP,vmin=0,vmax=100,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(4),['Typical','Humid\n14 days','Extreme\nday','Season\n+2 °C'],fontsize=8)
    ax.set_yticks(range(len(keep)),[lab[rb['designs'][i]] for i in keep],fontsize=8)
    ax.tick_params(length=0); ax.grid(False); ax.spines[:].set_visible(False)
    for r in range(sub.shape[0]):
        for c in range(4):
            ax.text(c,r,f'{sub[r,c]:.0f}',ha='center',va='center',fontsize=8,
                    color='white' if sub[r,c]>60 else fs.INK_PRIMARY)
    ri = [i for i,k in enumerate(keep) if rb['designs'][k]=='Minimax (KH 325)'][0]
    ax.add_patch(Rectangle((-.5,ri-.5),4,1,fill=False,edgecolor=fs.INK_PRIMARY,lw=1.2,clip_on=False))
    _ptitle(ax,'b  Scenario regret (%)'); _pfinish(fig,'p_regret')


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('results',nargs='?',type=Path,default=ROOT/'results')
    p.add_argument('--paper',action='store_true',help='write only the paper-layout panels to figures/paper/')
    p.add_argument('--tariff',type=float,default=ECON['tariff'],help='Electricity cost in USD/kWh (F5 only)')
    args = p.parse_args()
    if not np.isfinite(args.tariff) or args.tariff < 0: p.error('--tariff must be finite and nonnegative')
    folder = args.results.resolve(); synthetic = folder.name.lower()=='synthetic'
    if args.paper:
        paper_panels(folder); return
    for build,name in [(fig1,'fig4_1_framework'),(f2,'fig4_2_venue'),(f3,'fig4_3_pareto'),(f4,'fig4_4_designday'),
                       (f5,'fig4_5_season'),(f6,'fig4_6_robustness')]:
        fig = build(folder,synthetic,args.tariff) if build is f5 else build(folder,synthetic)
        finish(fig,name)
    print(f'Wrote six PDF/PNG pairs to {fs.FIGURES_DIR}; font: {fs.FONT_FAMILY}')


if __name__ == '__main__':
    main()
