"""Precompute the optimisation inputs: representative hour types and the device fields at every slot for every hour type.

Hour types: k-medoids-like clustering (k-means, then the member hour nearest the centroid) over standardised hourly features
(air temperature, vapour pressure, 10 m wind, wind direction (sin, cos), roof state, mean absorbed radiation, occupancy).
Each type is represented by a real hour, so its spatial fields are physically consistent; weight = number of hours represented.
Device fields: surrogate prediction at the slot's local body-level wind, aimed along the local flow direction, converted to
(dT, de, v2_eq) at the plaza cells, where v2_eq is the added squared speed in 10 m-equivalent terms (divided by PROFILE^2).
"""
import os, sys, json, pickle, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import climate, econ, devices

ROOT = climate.ROOT
CLASSES = ['A', 'M', 'E', 'D']


def hour_types(base, K=60, seed=0):
    from sklearn.cluster import KMeans
    occ = base['n'].sum(1); Qm = base['Q'].mean(1)
    # Equal cell weighting; component means respect the angular wrap at +/-180.
    wd = np.radians(base['wdir_loc'].astype(float))
    F = np.column_stack([base['T'], base['e'], base['w10'], np.sin(wd).mean(1), np.cos(wd).mean(1), base['roof'].astype(float), Qm, occ])
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    km = KMeans(K, n_init=10, random_state=seed).fit(Z)
    reps, W = [], []
    for k in range(K):
        idx = np.where(km.labels_ == k)[0]
        j = idx[np.argmin(((Z[idx] - km.cluster_centers_[k]) ** 2).sum(1))]
        reps.append(int(j)); W.append(len(idx))
    return np.array(reps), np.array(W, float), km.labels_


def slot_cells(base):
    xy = base['xy']; s = base['slots']
    return np.array([np.argmin(np.linalg.norm(xy - p, axis=1)) for p in s])


def e_from_w(w, p=1008.0):
    return p * w / (0.622 + w)


def device_fields(models, base, hours, setting=1.0, T_sup=18.0, design=None):
    """Return dT, dW (humidity ratio), v2 arrays of shape (len(hours), n_slots, n_classes, n_cells) (float32) and caps (H, S, C)."""
    import surrogate
    xy = base['xy']; slots = base['slots']; sc = slot_cells(base)
    H, S, C, N = len(hours), len(slots), len(CLASSES), len(xy)
    dT = np.zeros((H, S, C, N), np.float32); dW = np.zeros_like(dT); v2 = np.zeros_like(dT); caps = np.zeros((H, S, C), np.float32)
    # Batch GP/POD predictions instead of one BLAS call for every slot/hour.
    # Optional installed design avoids predicting unused classes in real-hour runs.
    predictions = {}
    for ci, cls in enumerate(CLASSES):
        if not models.get(cls): continue
        pairs = [(hi, s) for hi in range(H) for s in range(S) if design is None or int(design[s]) == ci+1]
        if not pairs: continue
        X = []
        for hi, s in pairs:
            h = hours[hi]; Ta, RH = float(base['T'][h]), float(base['RH'][h])
            wb = float(np.clip(base['wb'][h, sc[s]], .2, 3.0))
            X.append({'A':[wb,setting], 'E':[wb,setting,Ta,RH], 'D':[wb,setting,T_sup,Ta], 'M':[wb,Ta,RH]}[cls])
        shp = (len(surrogate.post.GY), len(surrogate.post.GX))
        for start in range(0, len(X), 64):
            pr = {t: np.maximum(g.predict(np.array(X[start:start+64])), 0.0).reshape(-1, *shp) for t,g in models[cls].items()}
            for j, pair in enumerate(pairs[start:start+64]):
                predictions[pair+(ci,)] = {t:a[j] for t,a in pr.items()}
    for hi, h in enumerate(hours):
        Ta, RH = float(base['T'][h]), float(base['RH'][h])
        e_a = float(base['e'][h]); w_a = 0.622 * e_a / (1008.0 - e_a)
        for s in range(S):
            c0 = sc[s]; flow = float(base['wdir_loc'][h, c0])
            for ci, cls in enumerate(CLASSES):
                m = models.get(cls)
                if not m or (hi, s, ci) not in predictions: continue
                pr = predictions.pop((hi, s, ci))
                vv = surrogate.place(pr['v2'], slots[s], flow, xy) / climate.PROFILE_BODY ** 2
                if cls == 'A':
                    tt = np.zeros(N); ww = np.zeros(N); cap = 0.0
                elif cls == 'E':
                    dTs, dws = surrogate.supply_state('E', dict(Ta=Ta, RH=RH))
                    tt = surrogate.place(pr['phiT'], slots[s], flow, xy) * dTs
                    ww = surrogate.place(pr['phiW'], slots[s], flow, xy) * dws; cap = dTs
                elif cls == 'D':
                    dTs, dws = surrogate.supply_state('D', dict(Ta=Ta, RH=RH, T_sup=T_sup))
                    ph = surrogate.place(pr['phiT'], slots[s], flow, xy)
                    tt = ph * dTs; ww = ph * dws; cap = dTs       # humidity follows the same delivery fraction
                else:  # M
                    tt = surrogate.place(pr['dT'], slots[s], flow, xy); ee = surrogate.place(pr['de'], slots[s], flow, xy)
                    ww = 0.622 * (e_a + ee) / (1008.0 - e_a - ee) - w_a; cap = float(tt.max())
                dT[hi, s, ci] = tt; dW[hi, s, ci] = ww; v2[hi, s, ci] = vv; caps[hi, s, ci] = cap
    return dT, dW, v2, caps


def run(K=60):
    import season
    base = season.build()
    reps, W, labels = hour_types(base, K)
    models = pickle.load(open(os.path.join(ROOT, 'results', 'surrogate.pkl'), 'rb'))
    dT, dW, v2, caps = device_fields(models, base, reps)
    run_e = np.zeros((K, len(CLASSES))); run_w = np.zeros((K, len(CLASSES)))
    for i, h in enumerate(reps):
        for ci, cls in enumerate(CLASSES):
            run_e[i, ci], run_w[i, ci] = econ.hourly_running(cls, float(base['T'][h]), float(base['RH'][h]))
    np.savez_compressed(os.path.join(ROOT, 'results', 'precomputed.npz'), reps=reps, W=W, labels=labels, dT=dT, dW=dW, v2=v2, caps=caps,
                        run_e=run_e, run_w=run_w)
    return reps, W


if __name__ == '__main__':
    r, W = run(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
    print('hour types', len(r), 'weights sum', W.sum())
