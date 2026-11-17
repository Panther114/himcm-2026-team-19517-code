"""Venue wind field at Barahat Msheireb: extruded OSM buildings (+ closed roof), snappyHexMesh, steady incompressible RANS (k-epsilon)
with an atmospheric boundary-layer inlet. One case per wind direction and roof state; the geometry is rotated so that the wind blows along +x.

Frames: 'plaza frame' = origin at the plaza centroid, x along the plaza long axis (about 19 deg north of east), y to the north-west side.
Results are stored back in the plaza frame as body-level (1.1 m) wind-speed ratios U/U10 (U10 = reference wind at 10 m at the inlet).
"""
import os, sys, json, math, shutil
import numpy as np
from shapely.geometry import Polygon, Point
from shapely import affinity
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
_tutorials = os.environ.get('FOAM_TUTORIALS') or (os.path.join(os.environ['BLUECFD_ROOT'], 'OpenFOAM-12', 'tutorials') if os.environ.get('BLUECFD_ROOT') else '')
TUT = os.path.join(_tutorials, 'incompressibleFluid', 'windAroundBuildings') if _tutorials else ''
LAT0, LON0 = 25.2862, 51.5262
PLAZA_ID = 524110885
DEFAULT_H = 25.0      # m, Msheireb blocks are 8-9 storeys; OSM-tagged neighbours 20-25 m
ROOF_Z = (29.5, 30.5) # m, membrane roof about 30 m above ground (closed state), modelled as a 1 m slab


def lonlat_xy(g):
    return [((p['lon'] - LON0) * 111320 * math.cos(math.radians(LAT0)), (p['lat'] - LAT0) * 110540) for p in g]


def load_scene(radius=200.0):
    d = json.load(open(os.path.join(ROOT, 'data', 'osm_msheireb.json')))
    plaza = Polygon(lonlat_xy([e for e in d['elements'] if e['id'] == PLAZA_ID][0]['geometry']))
    c = plaza.centroid
    # long-axis angle from the minimum rotated rectangle
    mrr = list(plaza.minimum_rotated_rectangle.exterior.coords)
    edges = [(mrr[i], mrr[i + 1]) for i in range(4)]
    a, b = max(edges, key=lambda e: math.dist(*e))
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
    if ang > 90: ang -= 180
    to_plaza = lambda P: affinity.rotate(affinity.translate(P, -c.x, -c.y), -ang, origin=(0, 0))
    P = to_plaza(plaza)
    blds = []
    for e in d['elements']:
        t = e.get('tags', {})
        if e['type'] != 'way' or 'building' not in t or 'geometry' not in e: continue
        B = Polygon(lonlat_xy(e['geometry']))
        if not B.is_valid: B = B.buffer(0)
        if B.area < 20 or B.centroid.distance(c) > radius: continue
        h = None
        try:
            h = float(str(t.get('height', '')).split()[0]) if t.get('height') else None
        except ValueError: h = None
        if h is None and t.get('building:levels'):
            h = 3.5 * float(t['building:levels'])
        hs = 'tag' if h else 'default'
        blds.append(dict(id=e['id'], poly=to_plaza(B), h=h or DEFAULT_H, hsrc=hs, name=t.get('name')))
    return dict(plaza=P, angle_deg=ang, centroid_xy=(c.x, c.y), buildings=blds)


def extrude(poly, z0, z1):
    m = trimesh.creation.extrude_polygon(poly, z1 - z0)
    m.apply_translation([0, 0, z0])
    return m


def write_obj(scene, rot_deg, roof, path):
    """Rotate the plaza-frame scene by rot_deg (counter-clockwise) and write buildings and roof as named OBJ groups."""
    parts = {}
    meshes = [extrude(affinity.rotate(b['poly'], rot_deg, origin=(0, 0)), 0, b['h']) for b in scene['buildings']]
    parts['buildings'] = trimesh.util.concatenate(meshes)
    if roof:
        parts['roof'] = extrude(affinity.rotate(scene['plaza'], rot_deg, origin=(0, 0)), *ROOF_Z)
    for k, m in parts.items():
        m.export(path.replace('.obj', f'_{k}.obj'))
    return list(parts)


def hdr(cls, obj, loc='0'):
    return ('FoamFile\n{\n    format      ascii;\n    class       %s;\n    location    "%s";\n    object      %s;\n}\n' % (cls, loc, obj))


def wr(path, txt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w', newline='\n').write(txt)


def make_case(case, scene, wind_from_deg, roof, U10=1.0, z0_inlet=0.5, cell=4.0, endTime=1500, lvl_bld=1, lvl_plaza=2):
    if not TUT or not os.path.isdir(TUT):
        raise RuntimeError('Set FOAM_TUTORIALS or BLUECFD_ROOT to an OpenFOAM 12 installation containing the required tutorial case.')
    """wind_from_deg: meteorological direction (0 = from north, 90 = from east)."""
    if os.path.exists(case): shutil.rmtree(case)
    for d in ('0', 'constant/geometry', 'system'): os.makedirs(os.path.join(case, d), exist_ok=True)
    # flow direction in the world frame (towards), then in the plaza frame
    to_deg_world = (270 - wind_from_deg) % 360            # math angle of the vector the wind blows towards
    flow_plaza = to_deg_world - scene['angle_deg']        # in plaza frame
    rot = -flow_plaza                                     # rotate scene so that the flow points to +x
    names = write_obj(scene, rot, roof, os.path.join(case, 'constant', 'geometry', 'scene.obj'))
    xmin, xmax, ymin, ymax, zmax = -260, 340, -240, 240, 120
    nx, ny, nz = int((xmax - xmin) / cell), int((ymax - ymin) / cell), int(zmax / cell)
    wr(f'{case}/system/blockMeshDict', hdr('dictionary', 'blockMeshDict', 'system') + f"""convertToMeters 1;
vertices ( ({xmin} {ymin} 0) ({xmax} {ymin} 0) ({xmax} {ymax} 0) ({xmin} {ymax} 0)
           ({xmin} {ymin} {zmax}) ({xmax} {ymin} {zmax}) ({xmax} {ymax} {zmax}) ({xmin} {ymax} {zmax}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
boundary (
  inlet  {{ type patch; faces ((0 3 7 4)); }}
  outlet {{ type patch; faces ((1 5 6 2)); }}
  ground {{ type wall; faces ((0 1 2 3)); }}
  sides  {{ type symmetry; faces ((0 4 5 1) (3 2 6 7)); }}
  top    {{ type symmetry; faces ((4 7 6 5)); }}
);
""")
    # plaza refinement box in the rotated frame (bounding box of the rotated plaza, 6 m margin, up to 6 m height)
    Pr = affinity.rotate(scene['plaza'], rot, origin=(0, 0))
    bx = Pr.bounds
    geo = ''.join(f'    {n} {{ type triSurfaceMesh; file "scene_{n}.obj"; }}\n' for n in names)
    rs = ''.join(f'        {n} {{ level ({lvl_bld} {lvl_bld}); patchInfo {{ type wall; }} }}\n' for n in names)
    feats = ''.join(f'        {{ file "scene_{n}.obj"; level {lvl_bld}; }}\n' for n in names)
    wr(f'{case}/system/surfaceFeaturesDict', hdr('dictionary', 'surfaceFeaturesDict', 'system') +
       f'surfaces ({" ".join(chr(34) + "scene_" + n + ".obj" + chr(34) for n in names)});\nincludedAngle 150;\n')
    wr(f'{case}/system/snappyHexMeshDict', hdr('dictionary', 'snappyHexMeshDict', 'system') + f"""castellatedMesh true;
snap true;
addLayers false;
geometry
{{
{geo}    plazaBox {{ type searchableBox; min ({bx[0]-6:.2f} {bx[1]-6:.2f} 0); max ({bx[2]+6:.2f} {bx[3]+6:.2f} 6); }}
}};
castellatedMeshControls
{{
    maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; maxLoadUnbalance 0.1; nCellsBetweenLevels 3;
    features
    (
{feats}    );
    refinementSurfaces
    {{
{rs}    }}
    resolveFeatureAngle 30;
    refinementRegions {{ plazaBox {{ mode inside; level {lvl_plaza}; }} }}
    insidePoint (-250.3 -230.7 60.1);
    allowFreeStandingZoneFaces true;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; explicitFeatureSnap true; implicitFeatureSnap false; }}
addLayersControls {{ layers {{}} relativeSizes true; expansionRatio 1.2; finalLayerThickness 0.5; minThickness 1e-3; }}
meshQualityControls {{ #includeEtc "caseDicts/mesh/generation/meshQualityDict" }}
mergeTolerance 1e-6;
""")
    for f in ('fvSchemes', 'fvSolution'):
        shutil.copy(f'{TUT}/system/{f}', f'{case}/system/{f}')
    # robustness for the building array (several cases diverged with the tutorial settings)
    sch = open(f'{case}/system/fvSchemes').read()
    sch = sch.replace('turbulence      bounded Gauss limitedLinear 1;', 'turbulence      bounded Gauss upwind;')
    sch = sch.replace('default         Gauss linear corrected;', 'default         Gauss linear limited corrected 0.5;')
    sch = sch.replace('    default         corrected;', '    default         limited corrected 0.5;')
    open(f'{case}/system/fvSchemes', 'w', newline='\n').write(sch)
    sol = open(f'{case}/system/fvSolution').read()
    sol = sol.replace('        p               0.3;', '        p               0.2;').replace('        U               0.7;', '        U               0.5;')
    sol = sol.replace('"(k|omega|epsilon).*" 0.7;', '"(k|omega|epsilon).*" 0.5;')
    sol = sol.replace('    "(U|k|omega|epsilon)"', '    Phi\n    {\n        $p;\n    }\n\n    "(U|k|omega|epsilon)"', 1)   # solver for potentialFoam
    sol = sol.replace('SIMPLE\n{\n', 'SIMPLE\n{\n    nNonOrthogonalCorrectors 1;\n', 1) if 'nNonOrthogonalCorrectors 0;' not in sol.split('potentialFlow')[0] else \
        sol.replace('nNonOrthogonalCorrectors 0;', 'nNonOrthogonalCorrectors 1;', 1)
    open(f'{case}/system/fvSolution', 'w', newline='\n').write(sol)
    for f in ('momentumTransport', 'physicalProperties'):
        shutil.copy(f'{TUT}/constant/{f}', f'{case}/constant/{f}')
    wr(f'{case}/system/controlDict', hdr('dictionary', 'controlDict', 'system') + f"""application foamRun;
solver incompressibleFluid;
startFrom latestTime; startTime 0; stopAt endTime; endTime {endTime}; deltaT 1;
writeControl timeStep; writeInterval 400; purgeWrite 0; writeFormat binary; writePrecision 7;
timeFormat general; timePrecision 6; runTimeModifiable true;
libs ("libatmosphericModels.so");
""")
    abl = f"""    Uref {U10}; Zref 10; zDir (0 0 1); flowDir (1 0 0); z0 uniform {z0_inlet}; zGround uniform 0.0;\n"""
    kin = 1.5 * (0.1 * U10) ** 2; ein = 0.09 ** 0.75 * kin ** 1.5 / 10
    walls = 'ground|buildings|roof'
    wr(f'{case}/0/U', hdr('volVectorField', 'U') + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform ({U10} 0 0);
boundaryField
{{
    inlet {{ type atmBoundaryLayerInletVelocity;
{abl}    }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value $internalField; }}
    "({walls})" {{ type noSlip; }}
    "(sides|top)" {{ type symmetry; }}
}}
""")
    wr(f'{case}/0/k', hdr('volScalarField', 'k') + f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform {kin:.6g};
boundaryField
{{
    inlet {{ type atmBoundaryLayerInletK;
{abl}    }}
    outlet {{ type inletOutlet; inletValue uniform {kin:.6g}; value $internalField; }}
    "({walls})" {{ type kqRWallFunction; value $internalField; }}
    "(sides|top)" {{ type symmetry; }}
}}
""")
    wr(f'{case}/0/epsilon', hdr('volScalarField', 'epsilon') + f"""dimensions [0 2 -3 0 0 0 0];
internalField uniform {ein:.6g};
boundaryField
{{
    inlet {{ type atmBoundaryLayerInletEpsilon;
{abl}    }}
    outlet {{ type inletOutlet; inletValue uniform {ein:.6g}; value $internalField; }}
    "({walls})" {{ type epsilonWallFunction; value $internalField; }}
    "(sides|top)" {{ type symmetry; }}
}}
""")
    wr(f'{case}/0/nut', hdr('volScalarField', 'nut') + f"""dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    ground {{ type nutkAtmRoughWallFunction; z0 uniform 0.05; value uniform 0; }}
    "(buildings|roof)" {{ type nutkWallFunction; value uniform 0; }}
    "(sides|top)" {{ type symmetry; }}
}}
""")
    wr(f'{case}/0/p', hdr('volScalarField', 'p') + """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    "(ground|buildings|roof)" { type zeroGradient; }
    "(sides|top)" { type symmetry; }
}
""")
    meta = dict(wind_from_deg=wind_from_deg, roof=roof, rot_deg=rot, U10=U10, z0_inlet=z0_inlet, cell=cell, plaza_angle_deg=scene['angle_deg'])
    json.dump(meta, open(f'{case}/venue_meta.json', 'w'))
    return meta


def run_case(case):
    from runner import sh
    steps = [('surfaceFeatures',), ('blockMesh',), ('snappyHexMesh', '-overwrite'), ('checkMesh',), ('potentialFoam',), ('foamRun',)]
    out = {}
    for s in steps:
        rc, t = sh(case, *s, log=f'log.{s[0]}'); out[s[0]] = [rc, round(t)]
        if rc and s[0] != 'checkMesh':
            break
    out['ok'] = out.get('foamRun', [1])[0] == 0
    json.dump(out, open(f'{case}/run_status.json', 'w'))
    return out


if __name__ == '__main__':
    sc = load_scene()
    print('plaza area', round(sc['plaza'].area), 'angle', round(sc['angle_deg'], 1), 'buildings', len(sc['buildings']),
          'default heights', sum(b['hsrc'] == 'default' for b in sc['buildings']))
    case = os.path.join(ROOT, 'cfd', 'venue', sys.argv[1] if len(sys.argv) > 1 else 'test_NW_closed')
    print(make_case(case, sc, wind_from_deg=float(sys.argv[2]) if len(sys.argv) > 2 else 315, roof=(sys.argv[3] == 'closed') if len(sys.argv) > 3 else True,
                    endTime=int(sys.argv[4]) if len(sys.argv) > 4 else 1500))
    print(run_case(case))


def sample_case(case, sc, z=1.1, dz=0.4, time=None):
    """Body-level wind at the plaza grid points: speed ratio U/U10_ref and flow direction (plaza-frame math degrees)."""
    import pyvista as pv, subprocess
    from runner import OF
    meta = json.load(open(os.path.join(case, 'venue_meta.json')))
    import post
    cd = pv.read(post.latest_vtk(case, time))
    c = cd.cell_centers().points
    sel = np.abs(c[:, 2] - z) <= dz
    fld = 'UMean' if 'UMean' in cd.cell_data else 'U'      # iteration-averaged field when available
    pts = c[sel][:, :2]; U = cd.cell_data[fld][sel]
    # case frame = plaza frame rotated by rot_deg; map grid points into case frame
    a = math.radians(meta['rot_deg'])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    g = sc['xy'] @ R.T
    from scipy.spatial import cKDTree
    d, idx = cKDTree(pts).query(g, k=4)
    wts = 1 / np.maximum(d, 1e-3) ** 2; wts /= wts.sum(1, keepdims=True)
    Uc = (U[idx][:, :, :2] * wts[:, :, None]).sum(1)
    Up = Uc @ R            # back to the plaza frame (R orthonormal: inverse = transpose)
    speed = np.linalg.norm(Up, axis=1) / meta['U10']
    ang = np.degrees(np.arctan2(Up[:, 1], Up[:, 0]))
    return dict(ratio=speed, dir_deg=ang, meta=meta)
