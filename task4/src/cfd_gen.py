"""OpenFOAM 12 case generator for the two venue subdomains (steady RAS, multicomponent air/H2O, optional Lagrangian mist).

A case is specified by a `spec` dictionary. The mesh is a multi-block blockMesh built from axis breakpoints; blocks that fall inside
solid boxes are omitted, so solids become wall patches and one solid face may carry the jet (fan / cooler outlet) as an inlet patch.
Frame: x downwind, y span (cyclic), z up. Units: metres.  Everything marked ASSUMED in the spec is a scenario parameter.
"""
import os, math, shutil, itertools, textwrap
HEADER = """FoamFile
{{
    format      ascii;
    class       {cls};
    {loc}object      {obj};
}}
"""
_tutorials = os.environ.get('FOAM_TUTORIALS') or (os.path.join(os.environ['BLUECFD_ROOT'], 'OpenFOAM-12', 'tutorials') if os.environ.get('BLUECFD_ROOT') else '')
TUT = os.path.join(_tutorials, 'multicomponentFluid', 'verticalChannelSteady') if _tutorials else ''


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w', newline='\n').write(text)


def hdr(cls, obj, loc=None):
    return HEADER.format(cls=cls, obj=obj, loc=(f'location    "{loc}";\n    ' if loc else ''))


# ---------------------------------------------------------------- psychrometrics
def es_hpa(T): return 6.105 * math.exp(17.27 * T / (237.7 + T))
def Y_h2o(T, RH, p_hpa):
    e = RH / 100 * es_hpa(T); om = 0.622 * e / (p_hpa - e); return om / (1 + om)


# ---------------------------------------------------------------- mesh
def mesh(spec):
    xb, yb, zb = spec['xb'], spec['yb'], spec['zb']
    xn, yn, zn = spec['xn'], spec['yn'], spec['zn']
    nx, ny, nz = len(xn), len(yn), len(zn)
    def cen(b, i): return 0.5 * (b[i] + b[i + 1])
    def solid(i, j, k):
        x, y, z = cen(xb, i), cen(yb, j), cen(zb, k)
        for s in spec['solids']:
            x0, x1, y0, y1, z0, z1 = s['box']
            if x0 < x < x1 and y0 < y < y1 and z0 < z < z1: return s
        return None
    verts = {}
    def vid(i, j, k):
        if (i, j, k) not in verts: verts[(i, j, k)] = len(verts)
        return verts[(i, j, k)]
    blocks, patches = [], {}
    def add(name, face): patches.setdefault(name, []).append(face)
    FACES = {'xm': ((0, 4, 7, 3), (-1, 0, 0)), 'xp': ((1, 2, 6, 5), (1, 0, 0)), 'ym': ((0, 1, 5, 4), (0, -1, 0)),
             'yp': ((3, 7, 6, 2), (0, 1, 0)), 'zm': ((0, 3, 2, 1), (0, 0, -1)), 'zp': ((4, 5, 6, 7), (0, 0, 1))}
    bnd = {'xm': 'inlet', 'xp': 'outlet', 'ym': 'cycY0', 'yp': 'cycY1', 'zm': 'ground', 'zp': 'top'}
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        if solid(i, j, k): continue
        c = [vid(i + a, j + b, k + cc) for cc in (0, 1) for (a, b) in ((0, 0), (1, 0), (1, 1), (0, 1))]
        # c = v000 v100 v110 v010 v001 v101 v111 v011
        blocks.append((c, (xn[i], yn[j], zn[k])))
        for f, (loc, d) in FACES.items():
            ii, jj, kk = i + d[0], j + d[1], k + d[2]
            inside = 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz
            if not inside: add(bnd[f], tuple(c[m] for m in loc)); continue
            s = solid(ii, jj, kk)
            if s:
                jet = s.get('jet')
                name = jet['patch'] if jet and jet['face'] == f_opp(f) else s['name']
                add(name, tuple(c[m] for m in loc))
    inv = sorted(verts.items(), key=lambda kv: kv[1])
    L = ['FoamFile\n{\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\nconvertToMeters 1;\nvertices\n(']
    for (i, j, k), _ in inv: L.append(f'    ({xb[i]:.6g} {yb[j]:.6g} {zb[k]:.6g})')
    L.append(');\nblocks\n(')
    for c, n in blocks: L.append(f'    hex ({" ".join(map(str, c))}) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)')
    L.append(');\nedges\n();\nboundary\n(')
    for name, faces in patches.items():
        typ = {'cycY0': 'cyclic', 'cycY1': 'cyclic'}.get(name)
        L.append(f'    {name}\n    {{')
        if typ: L.append(f'        type cyclic;\n        neighbourPatch {"cycY1" if name == "cycY0" else "cycY0"};')
        elif name == 'top' or name == 'outlet' or name == 'inlet' or name in [s['jet']['patch'] for s in spec['solids'] if s.get('jet')]: L.append('        type patch;')
        else: L.append('        type wall;')
        L.append('        faces\n        (')
        for f in faces: L.append(f'            ({" ".join(map(str, f))})')
        L.append('        );\n    }')
    L.append(');\nmergePatchPairs\n();\n')
    return '\n'.join(L), list(patches), len(blocks)


def f_opp(f): return {'xm': 'xp', 'xp': 'xm', 'ym': 'yp', 'yp': 'ym', 'zm': 'zp', 'zp': 'zm'}[f]


# ---------------------------------------------------------------- fields
def fields(case, spec, patches):
    a = spec['ambient']; p_hpa = a['p'] / 100
    Ya = Y_h2o(a['T'] - 273.15, a['RH'], p_hpa)
    jetp = [s['jet']['patch'] for s in spec['solids'] if s.get('jet')]
    walls = [p for p in patches if p not in ('inlet', 'outlet', 'top', 'cycY0', 'cycY1', 'ground', *jetp)]
    abl = f"""Uref {a['Uref']}; Zref {a['Zref']}; zDir (0 0 1); flowDir (1 0 0); z0 uniform {a['z0']}; zGround uniform 0.0;"""
    def cyc(): return '    cycY0 { type cyclic; }\n    cycY1 { type cyclic; }\n'
    def block(name, dims, internal, per, cls='volScalarField'):
        s = hdr(cls, name) + f'dimensions {dims};\ninternalField {internal};\nboundaryField\n{{\n' + cyc()
        for pat, body in per.items():
            for p in (pat if isinstance(pat, tuple) else (pat,)):
                if p in patches: s += f'    {p}\n    {{\n        {body}\n    }}\n'
        return s + '}\n'
    wallsT = tuple(walls)
    # velocity
    per = {'inlet': f'type atmBoundaryLayerInletVelocity; {abl} value uniform (0 0 0);',
           'outlet': 'type pressureInletOutletVelocity; value uniform (0 0 0);', 'top': 'type slip;',
           'ground': 'type noSlip;', wallsT: 'type noSlip;'}
    for s_ in spec['solids']:
        if s_.get('jet'): per[s_['jet']['patch']] = f"type fixedValue; value uniform ({s_['jet']['U']} 0 0);"
    w(f'{case}/0/U', block('U', '[0 1 -1 0 0 0 0]', f"uniform ({a['Uref']} 0 0)", per, 'volVectorField'))   # start from the ambient wind
    if not spec.get('gravity', False):
        # forced convection (g = 0): plain p formulation with a total-pressure outlet (uniform outlet pressure is consistent when g = 0)
        per = {'inlet': 'type zeroGradient;', 'outlet': f"type totalPressure; p0 uniform {a['p']}; value uniform {a['p']};", 'top': 'type zeroGradient;',
               'ground': 'type zeroGradient;', wallsT: 'type zeroGradient;'}
        for s_ in spec['solids']:
            if s_.get('jet'): per[s_['jet']['patch']] = 'type zeroGradient;'
        w(f'{case}/0/p', block('p', '[1 -1 -2 0 0 0 0]', f"uniform {a['p']}", per))
    else:
        # buoyant formulation with p_rgh (p = p_rgh + rho*g*h + pRef); p itself is calculated (unstable in tests)
        per = {('inlet', 'outlet', 'top', 'ground', *walls, *jetp): f"type calculated; value uniform {a['p']};"}
        w(f'{case}/0/p', block('p', '[1 -1 -2 0 0 0 0]', f"uniform {a['p']}", per))
        per = {'inlet': 'type fixedFluxPressure; value uniform 0;', 'outlet': 'type prghPressure; p uniform 0; value uniform 0;',
               'top': 'type fixedFluxPressure; value uniform 0;', 'ground': 'type fixedFluxPressure; value uniform 0;',
               wallsT: 'type fixedFluxPressure; value uniform 0;'}
        for s_ in spec['solids']:
            if s_.get('jet'): per[s_['jet']['patch']] = 'type fixedFluxPressure; value uniform 0;'
        w(f'{case}/0/p_rgh', block('p_rgh', '[1 -1 -2 0 0 0 0]', 'uniform 0', per))
        w(f'{case}/constant/pRef', hdr('uniformDimensionedScalarField', 'pRef', 'constant') + f"dimensions [1 -1 -2 0 0 0 0];\nvalue {a['p']};\n")
    # temperature
    per = {'inlet': f"type fixedValue; value uniform {a['T']};", 'outlet': f"type inletOutlet; inletValue uniform {a['T']}; value uniform {a['T']};",
           'top': 'type zeroGradient;', 'ground': f"type fixedValue; value uniform {spec['T_ground']};", wallsT: 'type zeroGradient;'}
    for s_ in spec['solids']:
        if s_.get('jet'): per[s_['jet']['patch']] = f"type fixedValue; value uniform {s_['jet']['T']};"
    w(f'{case}/0/T', block('T', '[0 0 0 1 0 0 0]', f"uniform {a['T']}", per))
    # species
    for sp in ('H2O', 'air'):
        def val(Y): return Y if sp == 'H2O' else 1 - Y
        per = {'inlet': f'type fixedValue; value uniform {val(Ya):.8g};', 'outlet': f'type inletOutlet; inletValue uniform {val(Ya):.8g}; value uniform {val(Ya):.8g};',
               'top': 'type zeroGradient;', 'ground': 'type zeroGradient;', wallsT: 'type zeroGradient;'}
        for s_ in spec['solids']:
            if s_.get('jet'): per[s_['jet']['patch']] = f"type fixedValue; value uniform {val(s_['jet']['Y']):.8g};"
        w(f'{case}/0/{sp}', block(sp, '[]', f'uniform {val(Ya):.8g}', per))
    # turbulence (kEpsilon with ABL inlet)
    kin = 1.5 * (0.1 * a['Uref']) ** 2; ein = 0.09 ** 0.75 * kin ** 1.5 / (0.41 * a['Zref'])
    per = {'inlet': f'type atmBoundaryLayerInletK; {abl} value uniform {kin:.4g};', 'outlet': f'type inletOutlet; inletValue uniform {kin:.4g}; value uniform {kin:.4g};',
           'top': 'type zeroGradient;', ('ground', *walls): 'type kqRWallFunction; value uniform 0.1;'}
    for s_ in spec['solids']:
        if s_.get('jet'): per[s_['jet']['patch']] = f'type fixedValue; value uniform {kin:.4g};'
    w(f'{case}/0/k', block('k', '[0 2 -2 0 0 0 0]', f'uniform {kin:.4g}', per))
    per = {'inlet': f'type atmBoundaryLayerInletEpsilon; {abl} value uniform {ein:.4g};', 'outlet': f'type inletOutlet; inletValue uniform {ein:.4g}; value uniform {ein:.4g};',
           'top': 'type zeroGradient;', ('ground', *walls): f'type epsilonWallFunction; value uniform {ein:.4g};'}
    for s_ in spec['solids']:
        if s_.get('jet'): per[s_['jet']['patch']] = f'type fixedValue; value uniform {ein:.4g};'
    w(f'{case}/0/epsilon', block('epsilon', '[0 2 -3 0 0 0 0]', f'uniform {ein:.4g}', per))
    per = {('inlet', 'outlet', 'top', *jetp): 'type calculated; value uniform 0;', 'ground': f"type nutkAtmRoughWallFunction; z0 uniform {a['z0']}; value uniform 0;", tuple(walls): 'type nutkWallFunction; value uniform 0;'}
    w(f'{case}/0/nut', block('nut', '[0 2 -1 0 0 0 0]', 'uniform 0', per))
    per = {('inlet', 'outlet', 'top', *jetp): 'type calculated; value uniform 0;', ('ground', *walls): 'type compressible::alphatWallFunction; Prt 0.85; value uniform 0;'}
    w(f'{case}/0/alphat', block('alphat', '[1 -1 -1 0 0 0 0]', 'uniform 0', per))
    return Ya


# ---------------------------------------------------------------- constant + system
def constant_system(case, spec):
    shutil.copy(f'{TUT}/constant/speciesThermo', f'{case}/constant/speciesThermo')
    shutil.copy(f'{TUT}/constant/physicalProperties', f'{case}/constant/physicalProperties')
    w(f'{case}/constant/g', hdr('uniformDimensionedVectorField', 'g', 'constant') + 'dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 %s);\n' % ('-9.81' if spec.get('gravity', False) else '0'))   # Task 4 default: forced convection, no buoyancy
    w(f'{case}/constant/momentumTransport', hdr('dictionary', 'momentumTransport') + 'simulationType RAS;\nRAS\n{\n    model kEpsilon;\n    turbulence on;\n    printCoeffs on;\n}\n')
    fv = hdr('dictionary', 'fvModels')   # buoyancy through p_rgh (constant/g + 0/p_rgh), not a buoyancyForce source
    if spec.get('cloud'): fv += 'clouds\n{\n    type clouds;\n    libs ("liblagrangianParcel.so");\n}\n'
    w(f'{case}/constant/fvModels', fv)
    w(f'{case}/constant/fvConstraints', hdr('dictionary', 'fvConstraints'))   # no pressure limiter with the p_rgh formulation
    shutil.copy(f'{TUT}/system/fvSchemes', f'{case}/system/fvSchemes')
    sch = open(f'{case}/system/fvSchemes').read()
    sch = sch.replace('div(phi,k) bounded Gauss upwind;', 'div(phi,k) bounded Gauss upwind;')
    open(f'{case}/system/fvSchemes', 'w', newline='\n').write(sch)
    sol = open(f'{TUT}/system/fvSolution').read().replace('"(U|Yi|h|k|omega).*"', '"(U|Yi|h|k|epsilon|omega).*"').replace('        p               0.3;', '        p               0.3;\n        p_rgh           0.2;').replace('        U               0.7;', '        U               0.5;').replace('        h               0.7;', '        h               0.5;')
    sol = sol.replace('relTol          0.1;\n        maxIter         20;', 'relTol          0.1;\n        maxIter         20;')
    open(f'{case}/system/fvSolution', 'w', newline='\n').write(sol)
    ctl = hdr('dictionary', 'controlDict', 'system') + f"""application foamRun;
solver multicomponentFluid;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {spec['endTime']};
deltaT 1;
writeControl timeStep;
writeInterval {spec['writeInterval']};
purgeWrite 2;
writeFormat ascii;
writePrecision 7;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
libs ("libatmosphericModels.so");
functions
{{
    outletFlux
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        writeFields no;
        regionType patch;
        name outlet;
        operation sum;
        fields (phi);
    }}
    inletFlux
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        writeFields no;
        regionType patch;
        name inlet;
        operation sum;
        fields (phi);
    }}
}}
"""
    w(f'{case}/system/controlDict', ctl)
    if spec.get('cloud'):
        c = spec['cloud']
        mdot = c['mdot']
        w(f'{case}/constant/cloudProperties', hdr('dictionary', 'cloudProperties') + f"""type reactingMultiphaseCloud;
solution
{{
    calcFrequency   10;
    maxTrackTime    5.0;
    maxCo           0.3;
    coupled         true;
    cellValueSourceCorrection off;
    sourceTerms
    {{
        resetOnStartup  false;
        schemes
        {{
            rho semiImplicit 1;
            U semiImplicit 1;
            Yi semiImplicit 1;
            h semiImplicit 1;
            radiation semiImplicit 1;
        }}
    }}
    interpolationSchemes
    {{
        rho cell; U cellPoint; mu cell; T cell; Cp cell; kappa cell; p cell;
    }}
    integrationSchemes
    {{
        U Euler;
        T analytical;
    }}
}}
constantProperties
{{
    rho0 1000;
    T0 {c['T_water']};
    Cp0 4100;
    constantVolume false;
}}
subModels
{{
    particleForces
    {{
        sphereDrag;
        gravity;
    }}
    injectionModels
    {{
        model1
        {{
            type coneInjection;
            massFlowRate {mdot:.6g};
            parcelsPerSecond {c['parcels_per_s']};
            uniformParcelSize volume;
            injectionMethod point;
            flowType constantVelocity;
            position ({c['pos'][0]} {c['pos'][1]} {c['pos'][2]});
            direction (1 0 0);
            Umag constant {c['Umag']};
            thetaInner constant 0;
            thetaOuter constant {c['half_angle']};
            flowRateProfile constant 1;
            sizeDistribution
            {{
                type RosinRammler;
                Q 3;
                min {c['dmin']};
                max {c['dmax']};
                d {c['rr_lambda']};
                n {c['n']};
            }}
        }}
    }}
    dispersionModel stochasticDispersionRAS;
    patchInteractionModel standardWallInteraction;
    heatTransferModel RanzMarshall;
    compositionModel singleMixtureFraction;
    phaseChangeModel liquidEvaporation;
    devolatilisationModel none;
    surfaceReactionModel none;
    stochasticCollisionModel none;
    surfaceFilmModel none;
    radiation off;
    standardWallInteractionCoeffs
    {{
        type rebound;
    }}
    RanzMarshallCoeffs
    {{
        BirdCorrection off;
    }}
    singleMixtureFractionCoeffs
    {{
        phases
        (
            gas
            {{
            }}
            liquid
            {{
                H2O 1;
            }}
            solid
            {{
            }}
        );
        YGasTot0 0;
        YLiquidTot0 1;
        YSolidTot0 0;
    }}
    liquidEvaporationCoeffs
    {{
        enthalpyTransfer enthalpyDifference;
        activeLiquids ( H2O );
    }}
}}
cloudFunctions
{{
}}
""")


def make(case, spec):
    if not TUT or not os.path.isdir(TUT):
        raise RuntimeError('Set FOAM_TUTORIALS or BLUECFD_ROOT to an OpenFOAM 12 installation containing the required tutorial case.')
    if os.path.exists(case): shutil.rmtree(case)
    os.makedirs(f'{case}/system'); os.makedirs(f'{case}/constant'); os.makedirs(f'{case}/0')
    txt, patches, nb = mesh(spec)
    w(f'{case}/system/blockMeshDict', txt)
    Ya = fields(case, spec, patches)
    constant_system(case, spec)
    ncell = 0
    xn, yn, zn = spec['xn'], spec['yn'], spec['zn']
    return dict(patches=patches, blocks=nb, Ya=Ya)
