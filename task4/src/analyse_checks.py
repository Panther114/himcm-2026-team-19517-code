"""Analyse completed verification cases without running CFD.

Run with .venv/Scripts/python src/analyse_checks.py. Only done.json cases
and completed dependencies are read. post.body_level may export missing VTK;
no solver, mesh generator, or other CFD-case mutation is invoked.
Outputs are replaced atomically so this can be rerun as cases finish.
"""
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import climate
import devices
import effect
import post
import surrogate
from run_checks import MESH_POINTS

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / 'cfd' / 'checks'
STANDARD = ROOT / 'cfd' / 'devices'
AREA = 0.25
XX, YY = np.meshgrid(post.GX, post.GY)
WINDOW = (XX >= 0) & (XX < 30) & (YY >= 0) & (YY < 24)


def relative(value, reference):
    """Signed fractional change; undefined zero-reference changes are null."""
    if value is None or reference is None or abs(reference) < 1e-12:
        return None
    return float((value - reference) / abs(reference))


def metrics(dat, mask=None):
    a = dat if mask is None else dat[mask]
    cooled = a[a > 0.5]
    return dict(area_gt_1_K_m2=float(np.count_nonzero(a > 1) * AREA),
                area_gt_0_5_K_m2=float(len(cooled) * AREA),
                peak_dAT_K=float(a.max()), mean_dAT_gt_0_5_K=float(cooled.mean()) if len(cooled) else None,
                mean_dAT_K=float(a.mean()))


def apparent(fields):
    return climate.AT_shade(fields['T'], fields['e'],
                            fields['speed'] / climate.PROFILE_BODY, m=1.2)


def ambient(kw, speed):
    return dict(T=np.full_like(speed, kw['Ta']),
                e=np.full_like(speed, climate.vapour(kw['Ta'], kw['RH'])), speed=speed)


def shifted(a, layout):
    """Exact 6 m translation on the 0.5 m grid; zero outside source support."""
    b = np.zeros_like(a)
    if layout == 'side':
        b[12:, :] = a[:-12, :]
    elif layout == 'line':
        b[:, 12:] = a[:, :-12]
    else:
        raise ValueError(layout)
    return b


def plot_fields(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figstyle
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.4 * len(rows)),
                             squeeze=False, layout='constrained')
    extent = (post.GX[0] - .25, post.GX[-1] + .25, post.GY[0] - .25, post.GY[-1] + .25)
    for axs, (label, left, right, titles) in zip(axes, rows):
        if left is None:
            for ax, title in zip(axs, titles + ('Difference',)):
                ax.set_title(f'{label}\n{title}', fontsize=11)
                ax.text(.5, .5, 'Pending completed dependencies', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
            continue
        difference = right - left
        vmax = max(float(left.max()), float(right.max()), .1)
        vmin = min(float(left.min()), float(right.min()), 0.)
        lim = max(float(abs(difference).max()), .01)
        for j, (ax, a, title) in enumerate(zip(axs, (left, right, difference), titles + ('Second − first',))):
            im = ax.imshow(a, origin='lower', extent=extent, aspect='equal',
                           cmap=figstyle.DIVERGING_CMAP if j == 2 else figstyle.BLUE_CMAP,
                           vmin=-lim if j == 2 else vmin, vmax=lim if j == 2 else vmax)
            ax.set_title(f'{label}\n{title}', fontsize=11)
            ax.set_xlabel('Downwind distance (m)')
            if j == 0:
                ax.set_ylabel('Crosswind distance (m)')
            if label.startswith('Superposition'):
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((0, 0), 30, 24, fill=False, ec='#14161a', lw=.8))
            fig.colorbar(im, ax=ax, shrink=.8, pad=.02, label='ΔAT (K)')
    tmp = ROOT / 'work' / 'checks_fields.tmp.png'
    fig.savefig(tmp, dpi=160)
    plt.close(fig)
    tmp.replace(ROOT / 'work' / 'checks_fields.png')


def main():
    for folder in ('results', 'work'):
        (ROOT / folder).mkdir(exist_ok=True)
    # Snapshot eligibility at startup; never read a partially finished case.
    finished = {p.resolve() for folder in (CHECKS, STANDARD) for p in folder.iterdir()
                if p.is_dir() and (p / 'done.json').exists()}
    cache, sources, pending, rows = {}, {}, [], []

    def ready(paths, label):
        missing = [str(p.relative_to(ROOT)).replace('\\', '/') for p in paths if p.resolve() not in finished]
        if missing:
            pending.append(dict(comparison=label, missing_done=missing))
        return not missing

    def read(path):
        key = str(path.relative_to(ROOT)).replace('\\', '/')
        if key not in cache:
            t = post.latest_time(str(path))
            cache[key] = post.body_level(str(path), p_hpa=surrogate.P_HPA, time=t)
            if not all(np.isfinite(a).all() for a in cache[key].values()):
                raise ValueError(f'Nonfinite body-level fields: {path}')
            sources[key] = dict(time=t)
        return cache[key]

    def std_ref(cls, wind):
        body = 'A' if cls in ('A', 'M') else cls
        ref = min(devices.OFF_REF, key=lambda r: abs(r - wind))
        return STANDARD / f'off_{body}_{ref:.1f}', ref

    mesh, off, superposition = {}, {}, {}
    doe = dict(devices.doe())
    for name in MESH_POINTS:
        kw = doe[name]; cls = kw['cls']; body = 'A' if cls == 'M' else cls
        fine = CHECKS / f'mesh_{name}_on'; standard = STANDARD / f'{name}_on'
        fine_off = CHECKS / f'mesh_off_{body}_1.0'
        standard_off, ref = std_ref(cls, kw['w_b'])
        titles = ('Standard mesh', 'Fine mesh (1.6×)')
        if not ready([fine, standard, fine_off, standard_off], fine.name):
            rows.append((f'Mesh {name}', None, None, titles)); continue
        # Exactly surrogate.off_reference's nearest-reference and linear rule,
        # using post.body_level directly for every field on the common grid.
        s0 = read(standard_off)['speed'] * (kw['w_b'] / ref)
        f0 = read(fine_off)['speed'] * kw['w_b']
        sdat = apparent(ambient(kw, s0)) - apparent(read(standard))
        fdat = apparent(ambient(kw, f0)) - apparent(read(fine))
        sm, fm = metrics(sdat), metrics(fdat)
        changes = {k: relative(fm[k], sm[k]) for k in sm}
        required = ('area_gt_1_K_m2', 'area_gt_0_5_K_m2', 'peak_dAT_K', 'mean_dAT_gt_0_5_K')
        passed = {k: abs(changes[k]) < .1 if changes[k] is not None else None for k in required}
        mesh[name] = dict(parameters={k: float(v) if isinstance(v, np.number) else v for k, v in kw.items()},
                          standard=sm, fine=fm, relative_change=changes,
                          metric_pass_lt_10_percent=passed,
                          passes_all_required=all(v is True for v in passed.values()),
                          dAT_field_rms_difference_K=float(np.sqrt(np.mean((fdat-sdat)**2))),
                          standard_off_reference=standard_off.name, standard_off_speed_scale=float(kw['w_b']/ref),
                          fine_off_reference=fine_off.name, fine_off_speed_scale=float(kw['w_b']))
        rows.append((f'Mesh {name}', sdat, fdat, titles))

    for body in ('A', 'E', 'D'):
        fine = CHECKS / f'mesh_off_{body}_1.0'; standard = STANDARD / f'off_{body}_1.0'
        if not ready([fine, standard], fine.name):
            continue
        sm, fm = (float(read(p)['speed'].mean()) for p in (standard, fine))
        change = relative(fm, sm)
        zero_metrics = metrics(np.zeros_like(XX))
        off[body] = dict(standard_mean_speed_m_s=sm, fine_mean_speed_m_s=fm,
                         relative_change=change, passes_lt_10_percent=abs(change) < .1,
                         standard_dAT_metrics=zero_metrics, fine_dAT_metrics=zero_metrics,
                         dAT_relative_change={k: None for k in zero_metrics},
                         dAT_K=0.0, area_gt_1_K_m2=0.0, area_gt_0_5_K_m2=0.0,
                         mean_dAT_gt_0_5_K=None,
                         note='Each off field is its own device-off baseline; dAT is identically zero, so relative dAT changes are undefined.')

    for cls in ('E', 'D'):
        kw = dict(cls=cls, Ta=40., RH=30., w_b=1., setting=1., T_sup=18.)
        single = CHECKS / f'super_{cls}_single'; refpath, ref = std_ref(cls, 1.)
        for layout in ('side', 'line'):
            case = CHECKS / f'super_{cls}_{layout}'; titles = ('Two-unit CFD', 'Superposed single-unit CFD')
            label = f'Superposition {cls} {layout}'
            if not ready([case, single, refpath], case.name):
                rows.append((label, None, None, titles)); continue
            base = ambient(kw, read(refpath)['speed'] / ref)
            one = read(single)
            dt = base['T'] - one['T']
            dw = effect.w_of_e(one['e']) - effect.w_of_e(base['e'])
            v2 = np.maximum(one['speed']**2 - base['speed']**2, 0.) / climate.PROFILE_BODY**2
            ht = effect.HourType(base['T'], effect.w_of_e(base['e']),
                                 base['speed']/climate.PROFILE_BODY,
                                 np.zeros_like(dt), np.zeros_like(dt, dtype=bool), np.ones_like(dt), m=1.2)
            cap = surrogate.supply_state(cls, kw)[0]
            predicted_at = ht.at(dt + shifted(dt, layout), dw + shifted(dw, layout),
                                 v2 + shifted(v2, layout), cap)
            truth = apparent(base) - apparent(read(case))
            prediction = apparent(base) - predicted_at
            tm, pm = metrics(truth, WINDOW), metrics(prediction, WINDOW)
            error = prediction[WINDOW] - truth[WINDOW]
            rms = float(np.sqrt(np.mean(error**2)))
            truth_rms = float(np.sqrt(np.mean(truth[WINDOW]**2)))
            superposition[f'{cls}_{layout}'] = dict(cfd=tm, superposed=pm,
                relative_difference={k: relative(pm[k], tm[k]) for k in tm},
                dAT_rms_error_K=rms, cfd_dAT_rms_K=truth_rms,
                relative_rms_error=rms/truth_rms if truth_rms > 1e-12 else None,
                supply_drop_cap_K=float(cap), baseline_off_reference=refpath.name)
            rows.append((label, truth, prediction, titles))

    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        methodology=dict(cell_area_m2=AREA, sampling='post.body_level: z=1.1 +/- 0.25 m, 0.5 m grid',
            mesh_window_m=dict(x=[-10, 30], y=[0, 24]), superposition_window_m=dict(x=[0, 30], y=[0, 24]),
            window_cell_count=int(WINDOW.sum()), wind_profile_body=climate.PROFILE_BODY, metabolic_rate=1.2,
            mesh_resolution_factor=1.6, acceptance_relative_change_strictly_below=.1,
            relative_difference='(fine-standard)/abs(standard), or (superposed-CFD)/abs(CFD); null for zero denominator',
            relative_rms_error='RMS(superposed dAT - CFD dAT) / RMS(CFD dAT) in downstream window',
            superposition='Raw single-CFD dT and signed humidity-ratio dW; positive added speed squared as targets_for; shifted increments zero outside support; effect.HourType.at applies joint supply cap, saturation and quadrature.',
            baseline='Ambient T and e; mesh uses matching-resolution off speed. Fine off speed scaled from 1 m/s. Standard nearest-wind reference scaled as surrogate.off_reference. Superposition uses the standard single-body off reference for both CFD and prediction; no two-body off CFD exists.',
            off_metrics='Off-state dAT is identically zero against itself; only mean speed provides a nontrivial mesh test.'),
        completed_check_cases=sorted(p.name for p in finished if p.parent == CHECKS.resolve()),
        sources=sources, mesh=mesh, mesh_off=off, superposition=superposition, pending=pending)
    plot_fields(rows)
    tmp = ROOT / 'results' / 'checks.json.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(ROOT / 'results' / 'checks.json')
    print(json.dumps({k: report[k] for k in ('mesh', 'mesh_off', 'superposition', 'pending')}, indent=2))


if __name__ == '__main__':
    main()
