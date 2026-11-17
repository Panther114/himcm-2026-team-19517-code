"""Publication figures from saved Task 3 results, reusing Task 4's serif style."""
import json
from pathlib import Path
import numpy as np
from common import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator
import figstyle as fs
from analyse import geometry

OUT=ROOT/'figures'
COLORBAR_LABEL='Apparent-temperature reduction (K)'


def export(fig,name):
    OUT.mkdir(exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[];entries=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text().strip():continue
        bbox=text.get_window_extent(renderer)
        if not np.all(np.isfinite(bbox.extents)) or bbox.width<=0 or bbox.height<=0:continue
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1>fig.bbox.width+1 or bbox.y1>fig.bbox.height+1:
            outside.append(text.get_text())
        entries.append((text,bbox))
    if outside:raise RuntimeError('Figure text outside page: '+str(outside))
    overlaps=[]
    for i,(a,ba) in enumerate(entries):
        for b,bb in entries[i+1:]:
            if ba.x0 < bb.x1 and bb.x0 < ba.x1 and ba.y0 < bb.y1 and bb.y0 < ba.y1:
                overlaps.append((a.get_text(),b.get_text()))
    if overlaps:raise RuntimeError('Figure text boxes overlap: '+str(overlaps[:8]))
    # Keep the nominal print width despite tight export bounds, as in Task 4.
    width,height=fig.get_size_inches()
    fig.add_artist(Rectangle((.04/width,.04/height),1-.08/width,1-.08/height,
                             transform=fig.transFigure,facecolor='none',
                             edgecolor='none',linewidth=0))
    for suffix in ['pdf','png']:
        fig.savefig(OUT/f'{name}.{suffix}',dpi=300)
    print(f'{name}: 0 text overlaps, 0 text objects outside the page')
    plt.close(fig)

def outline(ax,poly):
    for p in (poly.geoms if hasattr(poly,'geoms') else [poly]):
        x,y=p.exterior.xy;ax.plot(x,y,color=fs.INK_SECONDARY,lw=.65)
        for ring in p.interiors:
            x,y=ring.xy;ax.plot(x,y,color=fs.INK_MUTED,lw=.35)

def _bbox_union(artists,renderer):
    boxes=[]
    for artist in artists:
        if artist is not None and artist.get_visible():
            boxes.append(artist.get_window_extent(renderer))
    if not boxes:raise RuntimeError('No visible orientation artist')
    x0=min(b.x0 for b in boxes);y0=min(b.y0 for b in boxes)
    x1=max(b.x1 for b in boxes);y1=max(b.y1 for b in boxes)
    return matplotlib.transforms.Bbox.from_extents(x0,y0,x1,y1)

def _cell_boxes(ax,g,values,threshold):
    mask=np.abs(values)>threshold
    xy=g['xy'][mask];h=g['h']
    corners=np.stack([xy+np.array([-h/2,-h/2]),xy+np.array([h/2,-h/2]),
                      xy+np.array([h/2,h/2]),xy+np.array([-h/2,h/2])],axis=1)
    pts=ax.transData.transform(corners.reshape(-1,2)).reshape(-1,4,2)
    return np.column_stack([pts[:,:,0].min(axis=1),pts[:,:,1].min(axis=1),
                            pts[:,:,0].max(axis=1),pts[:,:,1].max(axis=1)])

def check_furniture(fig,ax,g,values,scale,name,furniture):
    """Raise if a scale bar or north arrow covers a non-negligible field cell."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    threshold=.05*scale
    cells=_cell_boxes(ax,g,values,threshold)
    for label,artists in furniture:
        box=_bbox_union(artists,renderer)
        hit=((box.x0 < cells[:,2]) & (cells[:,0] < box.x1) &
             (box.y0 < cells[:,3]) & (cells[:,1] < box.y1))
        if hit.any():
            raise RuntimeError(f'{name}: {label} covers {int(hit.sum())} mesh cells above '
                               f'5 % of the panel scale')
    print(f'{name}: scale bar and north arrow clear of {int((np.abs(values)>threshold).sum())} '
          f'cells with |value| above {threshold:.3g} K')

def scale_bar(ax,length,xlim,ylim,ymin,height_in):
    """Place a true-length bar in the bottom margin, outside the occupied polygon."""
    dx=xlim[1]-xlim[0];dy=ylim[1]-ylim[0]
    m_per_pt=dy/(height_in*72)
    text_height=fs.ANNOTATION_SIZE*1.2*m_per_pt
    x0=xlim[0]+.08*dx
    y=ymin-.42*text_height
    line=ax.plot([x0,x0+length],[y,y],color=fs.INK_PRIMARY,lw=fs.LINE_WIDTH,
                 solid_capstyle='butt',clip_on=False,zorder=10)[0]
    text=ax.text(x0+length/2,y-2*m_per_pt,f'{length:g} m',ha='center',va='top',
                 fontsize=fs.ANNOTATION_SIZE,color=fs.INK_PRIMARY,clip_on=False,zorder=10)
    return line,text

def maps(rows,z):
    fig=fs.new_figure(height=5.6)
    fs.title(fig,'Apparent-Temperature Reduction at the Two Venues',
             'Separate colour scales; matched 1.1 m fields with buildings and device footprints excluded')
    W,H=fig.get_size_inches()
    specs=[
        dict(name='bayfront',vmin=-.25,vmax=4.,ticks=[-.25,0,1,2,3,4],
             labels=['\u22120.25','0','1','2','3','4'],mx=18.,my=24.,
             bar=20.,angle=33.,anchor=(.93,.92),letter='a'),
        dict(name='medina',vmin=-.05,vmax=.5,ticks=[-.05,0,.1,.2,.3,.4,.5],
             labels=['\u22120.05','0','0.1','0.2','0.3','0.4','0.5'],mx=40.,my=55.,
             bar=100.,angle=0.,anchor=(.985,.93),letter='b'),
    ]
    for s in specs:
        s['g']=geometry(s['name'])
        lo,bo,hi,to=s['g']['poly'].bounds
        s['xlim']=(lo-s['mx'],hi+s['mx']);s['ylim']=(bo-s['my'],to+s['my'])
        s['aspect']=(s['xlim'][1]-s['xlim'][0])/(s['ylim'][1]-s['ylim'][0])
        s['h']=2.55;s['w']=s['h']*s['aspect']
        s['poly_ymin']=bo
    gap=.65
    left=(W-sum(s['w'] for s in specs)-gap)/2
    s0=specs[0]
    origin=[left,left+s0['w']+gap]
    for s,x0 in zip(specs,origin):
        s['x0']=x0
        s['ax']=fig.add_axes([x0/W,1.72/H,s['w']/W,s['h']/H])
    for s in specs:
        ax=s['ax'];g=s['g'];name=s['name']
        values=z[name+'_dAT']
        grid=np.full(np.prod(g['shape']),np.nan);grid[g['keep']]=values
        h=g['h'];xx=np.r_[g['x']-h/2,g['x'][-1]+h/2];yy=np.r_[g['y']-h/2,g['y'][-1]+h/2]
        norm=TwoSlopeNorm(vmin=s['vmin'],vcenter=0.,vmax=s['vmax'])
        mesh=ax.pcolormesh(xx,yy,grid.reshape(g['shape']).T,norm=norm,
                           cmap=fs.DIVERGING_CMAP.reversed(),rasterized=True,shading='flat')
        s['mesh']=mesh
        outline(ax,g['poly'])
        p=z[name+'_positions']
        if name=='bayfront':
            dist=np.linalg.norm(g['positions'][:,None,:]-p[None,:,:],axis=2).min(axis=1)
            other=g['positions'][dist>1e-6]
            ax.scatter(other[:,0],other[:,1],s=6.,marker='s',color=fs.NEUTRAL_GREY,
                       linewidths=0,zorder=3)
            ax.scatter(p[:,0],p[:,1],s=16.,marker='x',color=fs.INK_PRIMARY,linewidths=.7,zorder=4)
        else:
            ax.scatter(p[:,0],p[:,1],s=4.,marker='o',facecolors='none',
                       edgecolors=fs.INK_SECONDARY,linewidths=.35,zorder=3)
        ax.set_xlim(*s['xlim']);ax.set_ylim(*s['ylim']);ax.set_aspect('equal')
        ax.set_xlabel('Local x (m)');ax.set_ylabel('Local y (m)');ax.grid(False)
        ax.set_xticks([0,25,50,75,100,125] if name=='bayfront' else [-300,-200,-100,0,100,200,300])
        ax.set_yticks([0,25,50,75,100,125,150] if name=='bayfront' else [-200,-100,0,100,200])
        r=max([r for r in rows if r['venue']==name],key=lambda r:r['devices'])
        fs.panel(ax,s['letter'],('Bayfront' if name=='bayfront' else 'Medina')+f" ({r['devices']} devices)")
        bar=scale_bar(ax,s['bar'],s['xlim'],s['ylim'],s['poly_ymin'],s['h'])
        tail,nlabel=fs.north_arrow(ax,angle_deg=s['angle'],location='upper right')
        tail.xy=s['anchor'];nlabel.xy=s['anchor']
        s['furniture']=[('scale bar',bar),('north arrow',[tail.arrow_patch,nlabel])]
    for s in specs:
        cax=fig.add_axes([s['x0']/W,1.10/H,s['w']/W,.13/H])
        cb=fig.colorbar(s['mesh'],cax=cax,orientation='horizontal',extend='both')
        cb.set_ticks(s['ticks']);cb.set_ticklabels(s['labels'])
        cb.set_label(COLORBAR_LABEL,fontsize=fs.AXIS_LABEL_SIZE,labelpad=4)
        cb.ax.tick_params(labelsize=fs.TICK_SIZE,width=fs.ANNOTATION_WIDTH,pad=2)
        cb.outline.set_visible(False);cax.grid(False)
    for s in specs:
        check_furniture(fig,s['ax'],s['g'],z[s['name']+'_dAT'],s['vmax']-s['vmin'],
                        s['name'],s['furniture'])
    handles=[Line2D([],[],ls='none',marker='x',color=fs.INK_PRIMARY,
                    markeredgecolor=fs.INK_PRIMARY,markeredgewidth=.7,
                    markersize=5,label='Operating unit (14)'),
             Line2D([],[],ls='none',marker='s',markerfacecolor=fs.NEUTRAL_GREY,
                    markeredgecolor='none',markersize=3.2,label='Candidate position')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(origin[0]/W,.055),
               ncol=1,frameon=False,fontsize=fs.LEGEND_SIZE,handletextpad=.5,
               borderaxespad=0)
    fig.text((origin[1])/W,.055,'Open dots mark 243 fan columns',
             fontsize=fs.LEGEND_SIZE,color=fs.INK_SECONDARY,ha='left',va='bottom')
    export(fig,'fig3_1_venue_fields')

def _check_row(rr,venue,device,variant):
    matches=[r for r in rr if r['venue']==venue and r['device']==device and r['variant']==variant]
    if len(matches)!=1:raise RuntimeError(f'Expected one {venue}/{device}/{variant} row')
    return matches[0]

def _eta(rr,venue,device,variant):
    if variant=='sweep':
        a=_check_row(rr,venue,device,'sweepneg');b=_check_row(rr,venue,device,'sweeppos')
        return .5*(a['eta_normalised_per_kWh']+b['eta_normalised_per_kWh']),'sweepneg+sweeppos'
    row=_check_row(rr,venue,device,variant)
    return row['eta_normalised_per_kWh'],variant

def comparison(rows,checks):
    fig=fs.new_figure(height=8.0)
    fs.title(fig,'Cooling Efficiency Across Climates and Layout Sizes',
             'Dimensionless discomfort recovery per kWh with device counts and base-case sensitivity')
    ax=fig.add_axes([.10,.585,.34,.245]);bx=fig.add_axes([.60,.585,.36,.245])
    rr=checks['single_device_comparison']
    for j,(cls,color,label) in enumerate([('M',fs.EVAPORATIVE,'Elevated mist'),('D',fs.CHILLED_AIR,'Chilled air')]):
        values=[]
        for name in ['bayfront','medina']:
            variant=rows[0].get('kernel_variant','base')
            matches=[r for r in rr if r['venue']==name and r['device']==cls and r['variant']==variant]
            if len(matches)!=1:raise RuntimeError('Four matched climate/device cases required')
            values.append(matches[0]['eta_normalised_per_kWh'])
        ax.bar(np.arange(2)+(j-.5)*.33,values,width=.3,color=color,edgecolor=fs.INK_SECONDARY,lw=.6,label=label)
    ax.set_xticks([0,1],['Bayfront\nHot-humid','Medina\nHot-dry']);ax.set_ylim(0,4.9)
    ax.set_yticks([0,1,2,3,4])
    ax.set_ylabel('Discomfort recovery (person-hours/kWh)')
    fs.panel(ax,'a','Device comparison in shade')
    ax.legend(loc='upper left',frameon=False,fontsize=fs.LEGEND_SIZE,borderaxespad=.3,
              handlelength=1.5)
    offsets={('bayfront',1):(6,6,'left','bottom'),('bayfront',4):(-4,-11,'right','top'),
             ('bayfront',8):(4,7,'left','bottom'),('bayfront',14):(6,5,'left','bottom'),
             ('medina',1):(6,6,'left','bottom'),('medina',109):(6,6,'left','bottom'),
             ('medina',218):(6,5,'left','bottom'),('medina',436):(6,-9,'left','top')}
    for name,color,marker in [('bayfront',fs.CHILLED_AIR,'D'),('medina',fs.EVAPORATIVE,'^')]:
        r=sorted([r for r in rows if r['venue']==name],key=lambda r:r['devices'])
        bx.plot([v['devices'] for v in r],[v['eta_normalised_per_kWh'] for v in r],
                color=color,marker=marker,label=name.title())
        for v in r:
            off=offsets[(name,v['devices'])]
            bx.annotate(str(v['devices']),xy=(v['devices'],v['eta_normalised_per_kWh']),
                        xytext=(off[0],off[1]),textcoords='offset points',ha=off[2],va=off[3],
                        fontsize=fs.ANNOTATION_SIZE,color=fs.INK_PRIMARY,zorder=8,
                        bbox=dict(facecolor='white',edgecolor='none',pad=.5,alpha=.9))
    bx.set_xscale('log');bx.set_xlabel('Operating device count (log scale)')
    bx.set_ylabel('Discomfort recovery (person-hours/kWh)');bx.set_ylim(0,3.0)
    bx.set_yticks([0,1,2,3])
    bx.xaxis.set_major_locator(FixedLocator([1,10,100,1000]))
    bx.xaxis.set_major_formatter(FixedFormatter(['1','10','100','1000']))
    bx.set_xlim(.7,1100)
    fs.panel(bx,'b','Venue scaling')
    bx.legend(loc='lower right',frameon=False,fontsize=fs.LEGEND_SIZE,borderaxespad=.3)
    # Panel c: ratios for one changed input at a time, against each venue's base case.
    groups=[('Bayfront chilled air','bayfront','D',fs.CHILLED_AIR,
             [('Wind halved','windlow'),('Wind +50 %','windhigh'),
              ('Flow halved','flowlow'),('Flow +50 %','flowhigh'),
              ('Ground +10 K','hotground'),('Refined mesh','fine')]),
            ('Medina misting fans','medina','M',fs.EVAPORATIVE,
             [('Outlet lowered to 1.1 m','low'),('Aimed 60\u00b0 off wind','sweep'),
              ('Wind halved','windlow'),('Wind +50 %','windhigh'),
              ('Flow halved','flowlow'),('Flow +50 %','flowhigh'),
              ('Ground +10 K','hotground'),('Refined mesh','fine')])]
    built=[]
    y=1
    for group,venue,device,color,items in reversed(groups):
        base,_=_eta(rr,venue,device,'base')
        start=len(built)
        for label,variant in items:
            eta,source=_eta(rr,venue,device,variant)
            built.append(dict(group=group,venue=venue,device=device,label=label,variant=variant,
                              source=source,y=y,eta=eta,ratio=eta/base,color=color))
            y+=1
        for item in built[start:]:item['heading_y']=y
        y+=1
    axc=fig.add_axes([.27,.075,.70,.39])
    axc.barh([v['y'] for v in built],[v['ratio'] for v in built],height=.62,
             color=[v['color'] for v in built],edgecolor='white',linewidth=.8)
    axc.axvline(1,color=fs.INK_SECONDARY,lw=.8,ls='--',zorder=1)
    axc.set_yticks([v['y'] for v in built],[v['label'] for v in built])
    axc.tick_params(axis='y',length=0,pad=4)
    axc.set_xlim(0,3.35);axc.set_xticks([0,1,2,3])
    axc.set_ylim(-1.0,17.8)
    axc.set_xlabel('Efficiency ratio to base case (dimensionless)')
    axc.grid(False);axc.grid(axis='x');axc.set_axisbelow(True)
    fs.panel(axc,'c','Single-device sensitivity')
    for group,venue,device,color,items in groups:
        row=next(v for v in built if v['group']==group)
        axc.text(0.,row['heading_y'],group,transform=axc.get_yaxis_transform(),
                 ha='left',va='center',fontsize=9.5,fontweight='bold',color=fs.INK_PRIMARY)
    for v in built:
        axc.text(v['ratio']+.03,v['y'],f"{v['ratio']:.2f}",ha='left',va='center',
                 fontsize=fs.ANNOTATION_SIZE,color=fs.INK_PRIMARY)
    axc.text(1.03,-.55,'Base case = 1',ha='left',va='center',fontsize=fs.ANNOTATION_SIZE,
             color=fs.INK_SECONDARY,bbox=dict(facecolor='white',edgecolor='none',pad=.8))
    axc.legend(handles=[Patch(facecolor=fs.CHILLED_AIR,label='Bayfront chilled air'),
                        Patch(facecolor=fs.EVAPORATIVE,label='Medina misting fans')],
               loc='upper right',frameon=False,fontsize=fs.LEGEND_SIZE,borderaxespad=.3,
               handlelength=1.5)
    print('Panel c ratios from checks.json single_device_comparison:')
    for v in sorted(built,key=lambda v:-v['y']):
        print(f"  {v['group']:22s} {v['label']:26s} eta={v['eta']:.6f} "
              f"ratio={v['ratio']:.4f} row={v['venue']}/{v['device']}/{v['source']}")
    export(fig,'fig3_2_efficiency_scaling')

def main():
    rows=json.loads((ROOT/'results'/'venue_results.json').read_text())
    checks=json.loads((ROOT/'results'/'checks.json').read_text())
    with np.load(ROOT/'results'/'venue_maps.npz') as z:maps(rows,z)
    comparison(rows,checks)

if __name__=='__main__':main()
