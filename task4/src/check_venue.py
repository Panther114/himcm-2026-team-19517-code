"""Recompute completed venue fields and snapshot convergence without running solvers.

Only post.latest_vtk may write inside cfd (missing VTK exports). Outputs are
replaced on every run; directions are plaza-frame mathematical degrees.
"""
import json
import math
from pathlib import Path
import re

import numpy as np
import pyvista as pv

import post
import venue
import venue_cfd


ROOT = Path(__file__).resolve().parent.parent


def solution_times(case):
    """Positive numeric solution directories, sorted numerically; exclude initial 0."""
    times = []
    for folder in case.iterdir():
        if not folder.is_dir():
            continue
        try:
            value = float(folder.name)
        except ValueError:
            continue
        if math.isfinite(value) and value > 0 and (folder / 'U').is_file():
            times.append((value, folder.name))
    return [name for _, name in sorted(times)]


def check_case(case, sc):
    available = solution_times(case)
    times = available[-3:]
    if not times:
        raise ValueError(f'{case.name}: no saved positive solution times with U')
    ratios, directions, maxima = [], [], []
    for time in times:
        print(f'Sampling {case.name} at {time}', flush=True)
        sampled = venue_cfd.sample_case(str(case), sc, time=str(time))
        ratio = np.asarray(sampled['ratio'], dtype=float)
        direction = np.asarray(sampled['dir_deg'], dtype=float)
        mesh = pv.read(post.latest_vtk(str(case), time=str(time)))
        velocity = np.asarray(mesh.cell_data['U'])
        if (ratio.shape != (len(sc['xy']),) or direction.shape != ratio.shape
                or not np.isfinite(ratio).all() or not np.isfinite(direction).all()
                or not np.isfinite(velocity).all()):
            raise ValueError(f'{case.name} at {time}: invalid field values or shape')
        # Include any exported point velocities as well as all volume cells.
        maximum = float(np.linalg.norm(velocity, axis=-1).max())
        if 'U' in mesh.point_data:
            point_velocity = np.asarray(mesh.point_data['U'])
            if not np.isfinite(point_velocity).all():
                raise ValueError(f'{case.name} at {time}: invalid point velocities')
            maximum = max(maximum, float(np.linalg.norm(point_velocity, axis=-1).max()))
        ratios.append(ratio)
        directions.append(direction)
        maxima.append(maximum)
        del mesh

    means = [float(r.mean()) for r in ratios]
    mad = relative = p95 = None
    if len(times) >= 2:
        change = np.abs(ratios[-1] - ratios[-2])
        mad = float(change.mean())
        relative = mad / means[-1] if means[-1] > 0 else (0.0 if mad == 0 else None)
        p95 = float(np.percentile(change, 95))
        verdict = 'converged' if relative is not None and relative < 0.05 else 'oscillating'
    else:
        verdict = 'insufficient_snapshots'

    count = len(times) if verdict == 'oscillating' else 1
    final_ratio = np.mean(ratios[-count:], axis=0)
    angles = np.radians(np.stack(directions[-count:]))
    weights = np.stack(ratios[-count:])
    final_direction = np.degrees(np.arctan2(
        np.sum(weights * np.sin(angles), axis=0),
        np.sum(weights * np.cos(angles), axis=0)))
    report = dict(
        available_times=available, times_used=times, mean_ratio=means,
        mean_absolute_change=mad, relative_mean_change=relative,
        relative_mean_change_percent=None if relative is None else 100 * relative,
        change_percentile_95=p95, max_speed_by_time_m_s=maxima,
        max_speed_last_m_s=maxima[-1], U10_m_s=float(sampled['meta']['U10']),
        verdict=verdict, final_times=times[-count:],
        final_method='last_snapshot' if count == 1 else f'mean_last_{count}_snapshots',
        final_mean_ratio=float(final_ratio.mean()),
    )
    return report, final_ratio, final_direction


def print_table(reports):
    print('\nCase          Times used       Mean ratios                Mean |change|  Rel. %   P95 change  Max U (m/s)  Verdict')
    for name, r in reports.items():
        def fmt(value, digits=4):
            return 'n/a' if value is None else f'{value:.{digits}f}'
        print(f"{name:<13} {','.join(r['times_used']):<16} "
              f"{','.join(f'{v:.4f}' for v in r['mean_ratio']):<25} "
              f"{fmt(r['mean_absolute_change']):>13} "
              f"{fmt(r['relative_mean_change_percent'], 2):>7} "
              f"{fmt(r['change_percentile_95']):>12} "
              f"{r['max_speed_last_m_s']:>12.3f}  {r['verdict']}")


def main():
    sc = venue.build()
    reports, fields, skipped = {}, {}, []
    for case in sorted((ROOT / 'cfd' / 'venue').iterdir()):
        if not case.is_dir() or not re.fullmatch(r'w\d{3}_(open|closed)', case.name):
            continue
        status = case / 'run_status.json'
        if not status.is_file() or json.loads(status.read_text()).get('ok') is not True:
            skipped.append(case.name)
            continue
        report, ratio, direction = check_case(case, sc)
        reports[case.name] = report
        fields[f'ratio_{case.name}'] = ratio
        fields[f'dir_{case.name}'] = direction

    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    # Stage complete replacements so a failed sampling pass preserves prior results.
    npz_tmp = output / 'venue_fields.tmp.npz'
    json_tmp = output / 'venue_convergence.tmp.json'
    np.savez_compressed(npz_tmp, **fields)
    payload = dict(
        relative_change_denominator='mean ratio at latest sampled time',
        direction_units='plaza-frame mathematical degrees',
        grid_cell_m=sc['cell'], grid_points=len(sc['xy']),
        cases=reports, skipped_cases=skipped,
    )
    json_tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    npz_tmp.replace(output / 'venue_fields.npz')
    json_tmp.replace(output / 'venue_convergence.json')
    print(f'\nSaved {len(reports)} cases; skipped {len(skipped)} unfinished cases.', flush=True)
    print_table(reports)


if __name__ == '__main__':
    main()
