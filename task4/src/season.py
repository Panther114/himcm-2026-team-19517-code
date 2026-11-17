"""Season assembly: hourly baseline state of every plaza cell (no devices).

For each hour: roof state (published rule), sun or shade per cell (roof, buildings), absorbed radiation, local body-level wind from the
venue CFD (interpolated between the two nearest simulated directions), 10 m-equivalent wind for the AT formula, occupancy.
Outputs are cached in results/season_base.npz.
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import climate, venue, venue_cfd

ROOT = climate.ROOT
DIRS = [0, 45, 90, 135, 180, 225, 270, 315]


def load_wind_fields(sc):
    out = {}
    for roof in ('closed', 'open'):
        for d in DIRS:
            case = os.path.join(ROOT, 'cfd', 'venue', f'w{d:03d}_{roof}')
            r = venue_cfd.sample_case(case, sc)
            out[(roof, d)] = r
    return out


def local_wind(fields, roof, wdir):
    """Interpolate the speed ratio and the flow direction between the two nearest simulated directions."""
    d0 = int(wdir // 45) * 45 % 360; d1 = (d0 + 45) % 360; f = (wdir % 45) / 45
    a, b = fields[(roof, d0)], fields[(roof, d1)]
    ratio = (1 - f) * a['ratio'] + f * b['ratio']
    va = np.stack([np.cos(np.radians(a['dir_deg'])), np.sin(np.radians(a['dir_deg']))], 1)
    vb = np.stack([np.cos(np.radians(b['dir_deg'])), np.sin(np.radians(b['dir_deg']))], 1)
    v = (1 - f) * va * a['ratio'][:, None] + f * vb * b['ratio'][:, None]
    return ratio, np.degrees(np.arctan2(v[:, 1], v[:, 0]))


def build(params=None, cache=True):
    p = dict(roof_tau=0.15, albedo=0.35, occ_scale=1.0, edge_share=0.5, roof_policy='published', T_off=0.0, RH_off=0.0, wind_scale=1.0)
    if params: p.update(params)
    path = os.path.join(ROOT, 'results', 'season_base.npz')
    if cache and params is None and os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    sc = venue.build(); df = climate.load_weather()
    fields = load_wind_fields(sc)
    wsp = venue.spatial_weights(sc, seat_frac=p['edge_share'])
    nH, nC = len(df), len(sc['xy'])
    T = np.zeros(nH); e = np.zeros(nH); RH = np.zeros(nH)
    w_eq = np.zeros((nH, nC), np.float32); wdir_loc = np.zeros((nH, nC), np.float32); wb = np.zeros((nH, nC), np.float32)
    Q = np.zeros((nH, nC), np.float32); sun = np.zeros((nH, nC), bool); n = np.zeros((nH, nC), np.float32); roof = np.zeros(nH, bool)
    for i, r in enumerate(df.itertuples()):
        w10 = r.w10 * p['wind_scale']
        if p['roof_policy'] == 'published':
            closed = bool(r.sun_up and w10 < 10)
        elif p['roof_policy'] == 'always_open':
            closed = False
        else:
            closed = w10 < 10
        roof[i] = closed
        T[i] = r.T + p['T_off']; RH[i] = min(100, max(1, r.RH + p['RH_off'])); e[i] = climate.vapour(T[i], RH[i])
        ratio, ang = local_wind(fields, 'closed' if closed else 'open', r.wdir)
        wb[i] = np.maximum(ratio * w10, 0.05)
        w_eq[i] = wb[i] / climate.PROFILE_BODY
        wdir_loc[i] = ang
        if r.sun_up:
            Qs = climate.Q_abs(r.kt, r.cos_z, p['albedo'])
            if closed:
                Q[i] = p['roof_tau'] * Qs; sun[i] = True          # diffuse light through the membrane on every cell
            else:
                sh = venue.shadow_mask(sc, r.azimuth, r.zenith)
                sun[i] = ~sh; Q[i] = np.where(sh, 0.0, Qs)
        n[i] = venue.occupancy_total(r.hour, r.dow, p['occ_scale']) * wsp
    out = dict(T=T, e=e, RH=RH, w_eq=w_eq, wb=wb, wdir_loc=wdir_loc, Q=Q, sun=sun, n=n, roof=roof,
               hour=df['hour'].values, dow=df['dow'].values, w10=df['w10'].values * p['wind_scale'], xy=sc['xy'], slots=sc['slots'])
    if cache and params is None:
        os.makedirs(os.path.dirname(path), exist_ok=True); np.savez_compressed(path, **out)
    return out
