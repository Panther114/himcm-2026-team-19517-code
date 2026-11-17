"""Compare saved device surrogates with Section B field evidence, without CFD writes.

Run: .venv/Scripts/python src/validate.py
Only results/validation.md is written. Scenario choices are assumptions because
the research register does not supply numerical field-test ambient conditions.
"""
from pathlib import Path
import pickle
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import climate
import devices
import post
import surrogate

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (len(post.GY), len(post.GX))
MIST_URL = 'https://transsolar.com/media/pages/projects/cayman-islands-dry-mist-testing-at-camana-bay/36f5203325-1686137593/plea2018_extended-abstract_final-version.pdf'
NUS_URL = 'https://cde.nus.edu.sg/news/a-greener-solution-for-cooler-outdoor-dining/'
AIR_URL = 'https://www.stengg.com/getmedia/1791fe57-516c-4774-ae49-511a7cf9258c/Airbitat-Urban-Cooler-datasheet-UBC01.pdf'


def fields(models, cls, Ta, RH, wb, supply=18.0):
    """Use the same mixing and added-speed convention as precompute.py."""
    inputs = {'M': [wb, Ta, RH], 'D': [wb, 1.0, supply, Ta]}[cls]
    pr = {k: model.predict([inputs])[0].reshape(SHAPE)
          for k, model in models[cls].items()}
    off_speed = surrogate.off_reference(cls, wb)
    ea = climate.vapour(Ta, RH)
    if cls == 'M':
        dt, e_on = pr['dT'], ea + pr['de']
    else:
        drop, dw = surrogate.supply_state('D', dict(Ta=Ta, RH=RH, T_sup=supply))
        dt = pr['phiT'] * drop
        omega_on = surrogate.omega(ea) + pr['phiT'] * dw
        e_on = surrogate.P_HPA * omega_on / (0.622 + omega_on)
    on_speed = np.sqrt(off_speed**2 + pr['v2'])
    dat = (climate.AT_shade(Ta, ea, off_speed / climate.PROFILE_BODY, m=1.2)
           - climate.AT_shade(Ta - dt, e_on, on_speed / climate.PROFILE_BODY, m=1.2))
    assert all(np.isfinite(v).all() for v in (dt, dat))
    return dt, dat


def centreline(field, x):
    x = np.atleast_1d(x)
    return RegularGridInterpolator((post.GY, post.GX), field)(
        np.column_stack([np.full(len(x), 12.0), x]))


def reach(dt):
    """First downward 1 K crossing after the centreline maximum, in metres."""
    x = np.linspace(0.25, post.GX[-1], int(round((post.GX[-1] - 0.25) / 0.01)) + 1)
    t = centreline(dt, x)
    peak = int(np.argmax(t))
    if t[peak] < 1:
        return 0.0, False
    below = np.flatnonzero(t[peak:] < 1)
    if not len(below):
        return float(post.GX[-1]), True
    j = peak + below[0]
    return float(x[j-1] + (1-t[j-1]) * (x[j]-x[j-1]) / (t[j]-t[j-1])), False


def area(dat):
    return float(np.count_nonzero((dat > 1) & (post.GX[None, :] >= 0)) * 0.25)


def error(value, reference):
    return f'{value-reference:+.3f}', f'{value/reference:.3f}'


def main():
    with (ROOT / 'results/surrogate.pkl').open('rb') as f:
        models = pickle.load(f)
    dt, dat = fields(models, 'M', 33.0, 75.0, 0.3)
    mt, ma = centreline(dt, [2, 4, 9]), centreline(dat, [2, 4, 9])
    observed = np.array([5.5, 4.5, 1.0])
    ddt, ddat = fields(models, 'D', 33.0, 45.0, 0.3)
    (dr, censored), da = reach(ddt), area(ddat)
    lines = [
        '# Comparison of Simulated Outdoor Cooling Fields with Published Field Evidence', '',
        'Inputs: `results/surrogate.pkl` and device-off reference fields. '
        'No model fitting or CFD rerun is performed. Published comparison values are approximate '
        'and the cited pages have not been reverified for this report.', '',
        '## Method and condition matching', '',
        'The stored surrogate is sampled in device coordinates: outlet face x = 0 m, centreline y = 12 m. '
        'Fields represent the 1.1 ± 0.25 m body slab on the 0.5 m grid in `src/post.py`. '
        'Centreline reductions are bilinearly interpolated from that grid. The off state has ambient '
        'temperature and vapour pressure, with the matching body speed field scaled from the 0.3 m/s off reference '
        '(fan body A for M, body D for D). On-state speed is sqrt(speed_off² + v2), following the production model. '
        'Both on and off speeds are divided by `climate.PROFILE_BODY` = '
        f'{climate.PROFILE_BODY:.6f} before shaded AT is evaluated at 1.2 met. '
        'M humidity is reconstructed from predicted vapour-pressure increase; D humidity follows its thermal mixing fraction '
        'and the production supply-state rule. Positive reductions mean cooling.', '',
        'Camana Bay is described only as very humid tropical, fan setting 3, and sheltered from breeze; '
        '**no numerical ambient temperature, RH or wind is stated in the register**. '
        'The primary M scenario therefore assumes 33 °C, 75% RH and 0.3 m/s body-level wind. '
        'Full model airflow is 2.3 m³/s with 5 L/h mist at 25 °C; equivalence to the field fan setting 3 '
        'is unestablished. D uses the requested 33 °C ambient, 18 °C supply, 0.3 m/s body wind, '
        '1.57 m³/s full airflow and the DOE ambient RH of 45%; the field RH and flow are unavailable. '
        'The 18 °C supply is a requested model scenario, not a measured NUS/Airbitat condition. '
        'Inputs are within the nominal DOE design domain. D wind 0.3 m/s is slightly below the realised '
        'training minimum of 0.316 m/s, and full setting 1 is slightly above its realised maximum of 0.998. '
        'The comparisons therefore include small endpoint extrapolations and do not establish local training support.', '',
        'Differences below are model minus reference; ratios are model/reference. '
        'Cross-index and vendor comparisons are descriptive discrepancies, not calibrated measurement errors or confidence intervals.', '',
        '## Misting fan: Camana Bay distance profile', '',
        '| Distance from outlet (m) | Published ΔUTCI (K) | Model ΔT air (K) | ΔT − ΔUTCI (K) | ΔT / ΔUTCI | Model ΔAT shade (K) | ΔAT − ΔUTCI (K) | ΔAT / ΔUTCI |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x, obs, t, a in zip([2, 4, 9], observed, mt, ma):
        te, tr = error(t, obs); ae, ar = error(a, obs)
        lines.append(f'| {x} | {obs:.2f} | {t:.3f} | {te} | {tr} | {a:.3f} | {ae} | {ar} |')
    lines += ['', '| Normalised decay | Published ΔUTCI ratio | Model ΔT ratio | Difference | Model/reference | Model ΔAT ratio | Difference | Model/reference |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for j in [1, 2]:
        ref, t, a = observed[j]/observed[0], mt[j]/mt[0], ma[j]/ma[0]
        te, tr = error(t, ref); ae, ar = error(a, ref)
        lines.append(f'| {[2,4,9][j]} m / 2 m | {ref:.3f} | {t:.3f} | {te} | {tr} | {a:.3f} | {ae} | {ar} |')
    lines += ['', f'Source: [Camana Bay, PLEA 2018]({MIST_URL}). All three reported values are approximate, '
              'abstract-level, low-confidence evidence; the register records a failed PDF fetch. '
              'UTCI and shaded AT are different thermal indices, while air temperature is not a comfort index. '
              'Consequently none of the differences in the first table is a same-index error. '
              'Normalisation permits a limited comparison of spatial decay, but does not remove differing '
              'physiology, radiation, wind response or measurement-height assumptions. '
              'The very large air-temperature ratios arise from the near-zero 2 m denominator and are numerically sensitive; '
              'they do not imply large absolute air cooling.', '',
              'Water comparison: model 5.000 L/h versus the reported maximum 5.400 L/h: '
              'difference −0.400 L/h and ratio 0.926. This establishes compatibility with an upper limit, '
              'not agreement with a measured flow. No measured air-temperature profile is supplied.', '',
              '## Chilled-air unit: reach and footprint', '',
              'Reach is the first downstream crossing below ΔT air = 1 K after the centreline maximum, '
              'located by linear interpolation when resolved. If no crossing occurs, reach is reported as a lower bound '
              'at the final sampled centre, 29.75 m; no unsaved downstream tail is extrapolated. '
              'Footprint is the number of downstream grid cells (x ≥ 0) '
              'with shaded ΔAT > 1 K times 0.25 m², over 0–30 m downstream and 0–24 m span. '
              'It includes the fan contribution and modelled dehumidification. It is not a circular area inferred from reach.', '',
              '| Comparison | Model | Reference | Difference | Model/reference |',
              '|---|---:|---:|---:|---:|']
    for label, value, ref, unit in [('NUS reach, lower endpoint', dr, 3.5, 'm'),
                                  ('NUS reach, upper endpoint', dr, 4.0, 'm'),
                                  ('Airbitat coverage, sourced vendor claim', da, 40, 'm²'),
                                  ('25 m² coverage, unconfirmed scenario only', da, 25, 'm²')]:
        diff, ratio = error(value, ref)
        bound = '>' if unit == 'm' and censored else ''
        lines.append(f'| {label} | {bound}{value:.3f} {unit} | {ref:g} {unit} | {bound}{diff} {unit} | {bound}{ratio} |')
    boundary = np.any(ddat[:, -1] > 1) or np.any(ddat[[0, -1]][:, post.GX >= 0] > 1)
    lines += ['', f'Sources: [NUS outdoor dining report]({NUS_URL}); [Airbitat Urban Cooler datasheet]({AIR_URL}). '
              'Both are low-confidence, abstract-level entries in the register. NUS does not define a 1 K '
              'air-temperature threshold, and the vendor does not define coverage as ΔAT > 1 K. '
              'The NUS device is not confirmed as Airbitat. The 40 m² value is a vendor claim; '
              '**25 m² was not confirmed by the research file** and is shown only to bound the requested comparison. '
              'The register separately cites Airbitat output as low as 24 °C, warmer than the prescribed 18 °C model supply '
              '(difference −6 K; a Celsius ratio is not physically meaningful). '
              'Thus these comparisons cannot establish like-for-like reach or coverage errors. '
              f'The >1 K AT footprint {"touches" if boundary else "does not touch"} the downstream or lateral domain boundary. '
              f'Centreline air cooling at 29.75 m is {centreline(ddt, [post.GX[-1]])[0]:.3f} K. '
              'Where the footprint reaches the boundary, the reported area is domain-limited and may undercount the total footprint.', '',
              '## Sensitivity to assumed ambient conditions', '',
              'The following corner-scenario ranges indicate sensitivity to missing test conditions; they are not statistical uncertainty bounds. '
              'Nominal-domain corner inputs can lie outside realised DOE marginal bounds and include endpoint extrapolation.', '',
              '| Model | Scenario envelope | Computed range |', '|---|---|---|']
    samples = [centreline(fields(models, 'M', t, rh, w)[1], [2,4,9])
               for t in [30.,35.] for rh in [70.,80.] for w in [.2,.5]]
    sample = np.asarray(samples)
    lines.append('| M | 30/35 °C; 70/80% RH; 0.2/0.5 m/s (8 corners) | ΔAT at 2, 4, 9 m: '
                 + '; '.join(f'{lo:.3f}–{hi:.3f} K' for lo,hi in zip(sample.min(0),sample.max(0)))+' |')
    ds = [fields(models, 'D', 33., rh, w) for rh in [45.,75.] for w in [.2,.5]]
    rs, ars = [reach(t) for t,a in ds], [area(a) for t,a in ds]
    reach_text = ('all >29.750 m (censored)' if all(c for r,c in rs)
                  else ', '.join(('>' if c else '') + f'{r:.3f}' for r,c in rs) + ' m')
    lines.append(f'| D | 33 °C; supply 18 °C; 45/75% RH; 0.2/0.5 m/s (4 corners) | Reach {reach_text}; domain footprint {min(ars):.2f}–{max(ars):.2f} m² |')
    lines += ['', 'D RH is absent from the fitted input vector; its RH sensitivity here enters only through '
              'supply humidity and AT reconstruction, not through independently fitted flow fields.', '',
              '## Evaporative cooler: available evidence and missing comparator', '',
              'The [Abu Dhabi courtyard study](https://research.uaeu.ac.ae/en/publications/evaluating-the-effectiveness-of-outdoor-evaporative-cooling-in-a-/) '
              'reports a shift from very strong heat stress to no/moderate stress, but the register supplies no '
              'numeric UTCI reduction, outlet temperature or throw profile. The [Saudi pad study](https://iieta.org/download/file/fid/136143) '
              'has an ambiguous 12–32.5 °C drop range under arid conditions, with an unclear definition. '
              'Neither provides a comparable numerical outdoor body-level measurement for class E. '
              'The Abu Dhabi R² values concern simulated UTCI versus occupancy and are not device temperature errors. '
              'Difference: not estimable; ratio: not estimable. No E magnitude or decay validation is established.', '',
              '## Validation conclusion', '',
              f'M AT decreases from {ma[0]:.3f} K at 2 m to {ma[2]:.3f} K at 9 m, but its '
              f'9 m/2 m ratio ({ma[2]/ma[0]:.3f}) greatly exceeds the reported UTCI ratio '
              f'({observed[2]/observed[0]:.3f}); the reported decay shape is not reproduced. '
              'Model air cooling increases between these three stations rather than following the comfort-index decay. '
              'Agreement in absolute magnitude is '
              'not established because the indices and numerical ambient conditions are unmatched. '
              'The water input is consistent with the stated upper limit. D reach and footprint are '
              'substantially larger than the reported scales at the assumed low wind; neither magnitude nor decay shape is independently '
              'validated: source thresholds, supply and device identity are insufficiently specified. '
              'E has no comparable numerical measurement in the supplied evidence. These comparisons '
              'provide limited external plausibility checks, not independent quantitative validation '
              'of device classes or whole-plaza predictions. The gravity-off flow approximation, coarse '
              'body-slab sampling and approximately 10–15% near-field mesh-peak uncertainty remain relevant '
              'when interpreting the results.', '']
    (ROOT / 'results/validation.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
