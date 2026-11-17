"""Layer 3 physics shared by every Task 4 calculation: weather, sun, absorbed radiation, apparent temperature, utility.

Equations follow Section 3.1 of main.tex:
  e = RH/100 * 6.105 exp(17.27 T / (237.7 + T))
  AT_shade = T + 0.33 e - 0.70 w + 58.2 m / (w + 10) - 12.15
  AT_sun   = T + 0.348 e - 0.70 w + (c_Q Q + 58.2 m) / (w + 10) - 12.40      (c_Q = 1; Q already includes alpha_h)
  Q = alpha_h k_t S [f_p (1 - r) + F_sky r cos(th) + F_gnd alpha_g cos(th)]  (Section 3.1.1, Erbs diffuse ratio r(k_t))
  U = |AT_ideal - AT_i|^eps - |AT_ideal - AT_f|^eps                          (Equation 3.3, R = 1)
The wind w in the AT formula is the 10 m-equivalent wind, as in Task 2a (Steadman calibration); local body-level wind from the CFD
is converted back to its 10 m equivalent with the same profile factor used in Task 2a: w_b = w_s (z_b / 10)^0.25.
"""
import os, json, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

AT_IDEAL = 23.17
AT_SAFE = 32.0
S0 = 1360.0
ALPHA_H, F_P, F_SKY, F_GND = 0.7, 0.25, 0.5, 0.5
PROFILE = (1.5 / 10) ** 0.25          # w_b / w_s (Task 2a, kappa_w = 0.25, z_b = 1.5 m)
PROFILE_BODY = (1.1 / 10) ** 0.25     # same power law at the CFD sampling height 1.1 m (used for every CFD-derived wind)
LAT, LON = 25.2862, 51.5262


def eps_fn(rho, m):
    rho = min(max(rho, 0.02), 0.16)    # fitted density range 0.02-0.16 persons/m2 (Task 2a cap); ln(rho) needs rho > 0
    # main.tex boxed elasticity equation, checked 2026-11-16.
    return math.exp(1.27825 - 1.79203 * m / (1 + m) - 0.01036 * math.log(rho))


def vapour(T, RH):
    return RH / 100.0 * 6.105 * np.exp(17.27 * T / (237.7 + T))


def AT_shade(T, e, w10, m=1.2):
    return T + 0.33 * e - 0.70 * w10 + 58.2 * m / (w10 + 10) - 12.15


def AT_sun(T, e, w10, Q, m=1.2, rad_coef=1.0):
    return T + 0.348 * e - 0.70 * w10 + (rad_coef * Q + 58.2 * m) / (w10 + 10) - 12.40


def erbs_r(kt):
    kt = np.asarray(kt, float)
    return np.where(kt <= 0.22, 1 - 0.09 * kt,
                    np.where(kt <= 0.80, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4, 0.165))


def Q_abs(kt, cos_z, albedo):
    r = erbs_r(kt)
    return ALPHA_H * kt * S0 * (F_P * (1 - r) + F_SKY * r * cos_z + F_GND * albedo * cos_z)


def utility(at0, atf, eps):
    return np.abs(AT_IDEAL - at0) ** eps - np.abs(AT_IDEAL - atf) ** eps


def load_weather(path=None):
    path = path or os.path.join(ROOT, 'data', 'weather_doha_2025_JJAS.json')
    h = json.load(open(path))['hourly']
    df = pd.DataFrame(h)
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize('Asia/Qatar')
    df = df.rename(columns={'temperature_2m': 'T', 'relative_humidity_2m': 'RH', 'wind_speed_10m': 'w10', 'wind_direction_10m': 'wdir',
                            'shortwave_radiation': 'GHI', 'surface_pressure': 'p_hpa'})
    import pvlib
    # ERA5 radiation is the mean over the preceding hour: evaluate the sun at mid-interval
    sp = pvlib.solarposition.get_solarposition(df['time'] - pd.Timedelta(minutes=30), LAT, LON)
    df['zenith'] = sp['apparent_zenith'].values
    df['azimuth'] = sp['azimuth'].values
    df['cos_z'] = np.clip(np.cos(np.radians(df['zenith'])), 0, 1)
    df['sun_up'] = (df['zenith'] < 88) & (df['GHI'] > 5)
    df['kt'] = np.where(df['sun_up'], np.clip(df['GHI'] / (S0 * np.maximum(df['cos_z'], 0.05)), 0, 0.9), 0.0)
    df['e'] = vapour(df['T'], df['RH'])
    df['hour'] = df['time'].dt.hour
    df['dow'] = df['time'].dt.dayofweek          # Friday = 4, Saturday = 5 (Qatar weekend)
    return df
