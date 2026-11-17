"""Numerical toolkit for the two-stage thermal-comfort elasticity workflow.

Pure Python standard library, no third-party dependencies.  ``main.py`` uses:

* :func:`fit_wls` -- Householder-QR weighted least squares with HC1 robust or
  cluster-robust sandwich standard errors (Stata ``vce(robust)`` /
  ``vce(cluster)`` layout);
* :func:`powerlaw_fit` -- profiled-WLS fit of ``logit p = a + R*delta**eps``,
  which extracts one elasticity per random group;
* :func:`jacobi_eigen` -- symmetric eigensolver used for correlation-matrix PCA;
* :func:`stata_table` -- regression report formatting.

All estimation constants (deviation-bin edges, minimum cell sizes, the 5%
elimination level) live here so that the workflow has a single source of truth.
"""
from __future__ import annotations

import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# shared constants
# ---------------------------------------------------------------------------
DELTA_EDGES = (0.10, 0.35, 0.70, 1.20, 2.00, 3.30, 5.50, 8.00)  # K
NB = len(DELTA_EDGES) - 1
MET_CUTS = (1.05, 1.35)
MIN_BIN_N = 15.0        # minimum weight per deviation bin
MIN_BINS = 4            # minimum usable deviation bins per group
ALPHA = 0.05            # backward-elimination significance level


# ---------------------------------------------------------------------------
# classical statistics (incomplete beta -> t / F p-values)
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    fpmin, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, df: float) -> float:
    """Two-sided p-value of a t statistic."""
    if df <= 0 or not math.isfinite(t):
        return 1.0
    return betai(df / 2.0, 0.5, df / (df + t * t))


def f_pvalue(f: float, d1: float, d2: float) -> float:
    if f <= 0 or not math.isfinite(f):
        return 1.0
    return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def t_critical(df: float, two_sided: float = 0.05) -> float:
    lo, hi = 0.0, 100.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if t_pvalue(mid, df) > two_sided:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stars(p: float) -> str:
    if not math.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------------------------------------------------------------------------
# tiny dense linear algebra
# ---------------------------------------------------------------------------
def householder_qr(A, b):
    """QR of A (n x k) applied to b; return (R, Q'b[:k])."""
    A = [row[:] for row in A]
    b = list(b)
    n, k = len(A), len(A[0])
    for j in range(k):
        norm = math.sqrt(sum(A[i][j] ** 2 for i in range(j, n)))
        if norm == 0.0:
            continue
        alpha = -norm if A[j][j] >= 0 else norm
        v = [0.0] * n
        for i in range(j, n):
            v[i] = A[i][j]
        v[j] -= alpha
        vv = sum(v[i] ** 2 for i in range(j, n))
        if vv <= 0.0:
            continue
        for col in range(j, k):
            s = 2.0 * sum(v[i] * A[i][col] for i in range(j, n)) / vv
            for i in range(j, n):
                A[i][col] -= s * v[i]
        s = 2.0 * sum(v[i] * b[i] for i in range(j, n)) / vv
        for i in range(j, n):
            b[i] -= s * v[i]
    R = [[A[i][j] if i <= j else 0.0 for j in range(k)] for i in range(k)]
    return R, b[:k]


def invert_upper(R):
    k = len(R)
    inv = [[0.0] * k for _ in range(k)]
    for j in range(k):
        x = [0.0] * k
        for i in range(k - 1, -1, -1):
            s = (1.0 if i == j else 0.0) - sum(R[i][l] * x[l] for l in range(i + 1, k))
            x[i] = s / R[i][i]
        for i in range(k):
            inv[i][j] = x[i]
    return inv


def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][l] * B[l][j] for l in range(m)) for j in range(p)] for i in range(n)]


def transpose(A):
    return [list(col) for col in zip(*A)]


def matvec(A, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in A]


# ---------------------------------------------------------------------------
# WLS with sandwich covariance
# ---------------------------------------------------------------------------
def fit_wls(X, y, w, groups, names, cov_kind="robust"):
    """Weighted least squares with robust / cluster-robust standard errors.

    ``cov_kind="robust"``  -> HC1, each row its own cluster.
    ``cov_kind="cluster"`` -> clusters supplied in ``groups``.

    Slope columns are standardised internally for numerical stability, then
    coefficients and covariances are transformed back to the original units.
    Exactly-collinear columns are dropped Stata-style instead of producing
    arbitrary coefficients.
    """
    n = len(y)
    if n == 0:
        raise ValueError("no observations")
    if cov_kind == "robust":
        groups = [str(i) for i in range(n)]
    k = len(X[0])
    wmean = sum(w) / n
    w = [max(wi / wmean, 1e-12) for wi in w]
    sw = [math.sqrt(wi) for wi in w]

    mu = [0.0] * k
    sd = [1.0] * k
    for j in range(1, k):
        col = [row[j] for row in X]
        mu[j] = sum(col) / n
        v = sum((c - mu[j]) ** 2 for c in col) / max(n - 1, 1)
        sd[j] = math.sqrt(v) if v > 1e-14 else 1.0
    Z = [[1.0] + [(row[j] - mu[j]) / sd[j] for j in range(1, k)] for row in X]

    A = [[sw[i] * Z[i][j] for j in range(k)] for i in range(n)]
    yw = [sw[i] * y[i] for i in range(n)]
    R, qty = householder_qr(A, yw)
    diag = [abs(R[j][j]) for j in range(k)]
    tol = 1e-9 * max(diag)
    dropped = [names[j] for j in range(1, k) if diag[j] <= tol]
    if dropped:
        keep = [0] + [j for j in range(1, k) if diag[j] > tol]
        if len(keep) < 2:
            raise ValueError("design matrix has no usable slope columns")
        return fit_wls([[row[j] for j in keep] for row in X], y, w, groups,
                       [names[j] for j in keep], cov_kind)

    b = [0.0] * k
    for i in range(k - 1, -1, -1):
        b[i] = (qty[i] - sum(R[i][l] * b[l] for l in range(i + 1, k))) / R[i][i]

    fit_z = [sum(Z[i][j] * b[j] for j in range(k)) for i in range(n)]
    resid = [y[i] - fit_z[i] for i in range(n)]
    uw = [sw[i] * resid[i] for i in range(n)]

    gscores = defaultdict(lambda: [0.0] * k)
    for i, g in enumerate(groups):
        s = gscores[g]
        for j in range(k):
            s[j] += A[i][j] * uw[i]
    G = len(gscores)
    meat = [[0.0] * k for _ in range(k)]
    for s in gscores.values():
        for i in range(k):
            for j in range(k):
                meat[i][j] += s[i] * s[j]

    # H = A'A = R'R, hence H^-1 = R^-1 (R^-1)'; cov = H^-1 meat H^-1.
    Rinv = invert_upper(R)
    Hinv = matmul(Rinv, transpose(Rinv))
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov_b = matmul(matmul(Hinv, meat), Hinv)
    cov_b = [[corr * v for v in row] for row in cov_b]

    S = [[0.0] * k for _ in range(k)]
    S[0][0] = 1.0
    for j in range(1, k):
        S[0][j] = -mu[j] / sd[j]
        S[j][j] = 1.0 / sd[j]
    beta = matvec(S, b)
    cov = matmul(matmul(S, cov_b), transpose(S))
    se = [math.sqrt(max(cov[j][j], 0.0)) for j in range(k)]

    df = max(G - 1, 1)
    tcrit = t_critical(df)
    t = [beta[j] / se[j] if se[j] > 0 else 0.0 for j in range(k)]
    p = [t_pvalue(t[j], df) for j in range(k)]
    ci = [(beta[j] - tcrit * se[j], beta[j] + tcrit * se[j]) for j in range(k)]

    rss = sum(w[i] * resid[i] ** 2 for i in range(n))
    ybar = sum(w[i] * y[i] for i in range(n)) / sum(w)
    tss = sum(w[i] * (y[i] - ybar) ** 2 for i in range(n))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)
    mse = rss / max(n - k, 1)
    fstat = (r2 / max(k - 1, 1)) / ((1.0 - r2) / max(n - k, 1)) if r2 < 1 else float("inf")
    aic = n * math.log(max(rss / n, 1e-12)) + 2 * k
    bic = n * math.log(max(rss / n, 1e-12)) + k * math.log(n)

    dmax, dmin = max(diag[1:] or [1.0]), min(diag[1:] or [1.0])
    cond = dmax / dmin if dmin > 0 else float("inf")
    zcols = [[Z[i][j] for i in range(n)] for j in range(1, k)]
    zmean = [sum(c) / n for c in zcols]
    zs = [[v - zmean[j] for v in zcols[j]] for j in range(len(zcols))]
    maxcorr = 0.0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            num = sum(zs[i][l] * zs[j][l] for l in range(n))
            den = math.sqrt(sum(v * v for v in zs[i]) * sum(v * v for v in zs[j]))
            if den > 0:
                maxcorr = max(maxcorr, abs(num / den))

    return dict(names=list(names), coef=beta, se=se, t=t, p=p, ci=ci,
                n=n, k=k, G=G, df=df, cov_kind=cov_kind,
                r2=r2, adj_r2=adj_r2, rss=rss, tss=tss,
                root_mse=math.sqrt(max(mse, 0.0)), f=fstat,
                f_p=f_pvalue(fstat, max(k - 1, 1), max(n - k, 1)),
                aic=aic, bic=bic, cond=cond, maxcorr=maxcorr, dropped=dropped,
                tcrit=tcrit)


# ---------------------------------------------------------------------------
# stage-1 fit: one elasticity per group
# ---------------------------------------------------------------------------
def new_bin():
    """Empty accumulator for one deviation bin."""
    return dict(sw=0.0, yta=0.0, ytsv=0.0, sx=0.0, srho=0.0,
                smet=0.0, stop=0.0, swarm=0.0)


def logit(p):
    return math.log(p / (1.0 - p))


def powerlaw_fit(bins, outcome, usable, a_global, free_baseline):
    """Fit ``logit P(unacceptable) = a + R*delta**eps`` by profiled WLS.

    ``D = U_max - U = R*delta**eps`` is the utility shortfall; a is the
    log-odds baseline at delta = 0.  For each trial eps the model is linear in
    (a, R), so both are profiled out in closed form and eps is chosen by a
    coarse grid plus golden-section search.  Boundary solutions (unidentified)
    are returned as ``None``.
    """
    ykey = "yta" if outcome == "ta" else "ytsv"
    pts = []
    for i in usable:
        b = bins[i]
        p = (b[ykey] + 0.5) / (b["sw"] + 1.0)
        p = min(max(p, 1e-4), 0.999)
        d = math.exp(b["sx"] / b["sw"])             # geometric mean deviation
        v = b["sw"] * p * (1.0 - p)                 # inverse Var(logit p_hat)
        pts.append((d, logit(p), max(v, 1e-9)))
    if len(pts) < MIN_BINS:
        return None

    def wls(e):
        xs = [d ** e for d, _, _ in pts]
        sw = sum(v for _, _, v in pts)
        if free_baseline:
            xb = sum(v * x for x, (_, _, v) in zip(xs, pts)) / sw
            zb = sum(v * z for _, z, v in pts) / sw
            sxx = sum(v * (x - xb) ** 2 for x, (_, _, v) in zip(xs, pts))
            if sxx <= 1e-12:
                return None
            R = sum(v * (x - xb) * (z - zb)
                    for x, (_, z, v) in zip(xs, pts)) / sxx
            a = zb - R * xb
        else:
            a = a_global
            den = sum(v * x * x for x, (_, _, v) in zip(xs, pts))
            if den <= 1e-12:
                return None
            R = sum(v * x * (z - a) for x, (_, z, v) in zip(xs, pts)) / den
        rss = sum(v * (z - a - R * x) ** 2
                  for x, (_, z, v) in zip(xs, pts))
        return rss, a, R

    lo, hi, grid_n = 0.05, 4.0, 79
    step = (hi - lo) / grid_n
    candidates = []
    for i in range(grid_n + 1):
        e = lo + i * step
        res = wls(e)
        if res is not None and math.isfinite(res[0]):
            candidates.append((res[0], e))
    if not candidates:
        return None
    e_best = min(candidates)[1]
    a0, b0 = max(lo, e_best - step), min(hi, e_best + step)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b0 - phi * (b0 - a0), a0 + phi * (b0 - a0)
    fc, fd = wls(c)[0], wls(d)[0]
    for _ in range(50):
        if fc < fd:
            b0, d, fd = d, c, fc
            c = b0 - phi * (b0 - a0)
            fc = wls(c)[0]
        else:
            a0, c, fc = c, d, fd
            d = a0 + phi * (b0 - a0)
            fd = wls(d)[0]
        if b0 - a0 < 1e-5:
            break
    eps = 0.5 * (a0 + b0)
    if eps < lo + 0.03 or eps > hi - 0.03:
        return None
    res = wls(eps)
    if res is None:
        return None
    rss, a, R = res
    if R <= 0:
        return None
    h = 0.01
    up, dn = wls(min(eps + h, hi)), wls(max(eps - h, lo))
    if up is None or dn is None:
        return None
    curv = (up[0] - 2.0 * rss + dn[0]) / (h * h)
    if curv <= 0:
        return None
    npar = 3 if free_baseline else 2
    sigma2 = rss / max(len(pts) - npar, 1)
    se = math.sqrt(max(2.0 * sigma2 / curv, 1e-12))
    se_nom = math.sqrt(2.0 / curv)
    return eps, se, se_nom, math.log(R), len(pts), a


def reference_rates(obs):
    """P(unacceptable | delta = 0), extrapolated from the near-neutral bins.

    The smallest observed deviation is 0.1 K, so the raw near-neutral rate is
    an upper bound.  A pooled ``logit(p_b) = a + b*delta_b`` fit over the first
    four bins is extrapolated to delta = 0 instead.
    """
    cnt = [0.0] * NB
    ys = {"ta": [0.0] * NB, "tsv": [0.0] * NB}
    for o in obs:
        cnt[o["db"]] += 1.0
        ys["ta"][o["db"]] += o["yta"]
        ys["tsv"][o["db"]] += o["ytsv"]
    out = {}
    for key in ("ta", "tsv"):
        pts = []
        for i in range(min(4, NB)):
            if cnt[i] < 30.0:
                continue
            p = min(max((ys[key][i] + 0.5) / (cnt[i] + 1.0), 1e-4), 0.999)
            mid = math.sqrt(DELTA_EDGES[i] * DELTA_EDGES[i + 1])
            pts.append((mid, logit(p), cnt[i] * p * (1.0 - p)))
        if len(pts) < 2:
            out[key] = 0.05
            continue
        sw = sum(v for _, _, v in pts)
        xb = sum(v * x for x, _, v in pts) / sw
        zb = sum(v * z for _, z, v in pts) / sw
        sxx = sum(v * (x - xb) ** 2 for x, _, v in pts)
        b = sum(v * (x - xb) * (z - zb) for x, z, v in pts) / sxx if sxx > 0 else 0.0
        a = zb - b * xb
        out[key] = min(max(1.0 / (1.0 + math.exp(-a)), 0.005), 0.5)
    return out


# ---------------------------------------------------------------------------
# PCA helper: eigenvalues/eigenvectors of a small symmetric matrix
# ---------------------------------------------------------------------------
def jacobi_eigen(A, tol=1e-11, iters=2000):
    n = len(A)
    a = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iters):
        p, q, big = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > big:
                    big, p, q = abs(a[i][j]), i, j
        if big < tol:
            break
        theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]) if a[p][q] else 0.0
        t = (math.copysign(1.0, theta) /
             (abs(theta) + math.sqrt(theta * theta + 1.0))) if theta else 1.0
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            if k not in (p, q):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], V


# ---------------------------------------------------------------------------
# generic step-2 helpers (used for PC-score regressions)
# ---------------------------------------------------------------------------
def fit_custom(cells, extra, names, cov_kind="robust"):
    """WLS of epsilon on arbitrary precomputed columns (e.g. PC scores)."""
    X = [[1.0] + [extra[m][i] for m in range(len(extra))]
         for i in range(len(cells))]
    y = [c["eps"] for c in cells]
    w = [1.0 / (c["se"] ** 2) for c in cells]
    cap = sorted(w)[max(int(0.90 * (len(w) - 1)), 0)]
    w = [min(x, cap) for x in w]
    return fit_wls(X, y, w, [c["cluster"] for c in cells],
                   ["_cons"] + list(names), cov_kind)


def backward_custom(cells, extra, names, cov_kind="robust", alpha=ALPHA):
    """Drop the least significant column until all p-values <= alpha."""
    cols, nms, dropped = list(extra), list(names), []
    while True:
        res = fit_custom(cells, cols, nms, cov_kind)
        survived = res["names"][1:]
        sync = [nms.index(nm) for nm in survived if nm in nms]
        cols = [cols[i] for i in sync]
        nms = [nms[i] for i in sync]
        cand = [(res["p"][j + 1], j) for j in range(len(nms))]
        if not cand:
            break
        worst_p, idx = max(cand)
        if worst_p <= alpha:
            break
        dropped.append((nms[idx], worst_p))
        cols.pop(idx)
        nms.pop(idx)
    return cols, nms, dropped, res


# ---------------------------------------------------------------------------
# Stata-style reporting
# ---------------------------------------------------------------------------
def stata_table(res, title, notes):
    w = 32
    n, k, G = res["n"], res["k"], res["G"]
    dfm, dfr = k - 1, max(n - k, 1)
    ssm, ssr = max(res["tss"] - res["rss"], 0.0), res["rss"]
    msm, msr = ssm / max(dfm, 1), ssr / dfr
    rule = "-" * 79
    sep = "-" * 14 + "+" + "-" * 35
    fval = msm / msr if msr else float("inf")
    L = [title, "",
         f"{'Source':<14}|{'SS':>13}{'df':>9}{'MS':>13}   Number of obs   = {n:>10,}",
         f"{sep}   {'F(' + str(dfm) + ', ' + str(dfr) + ')':<16}= {fval:>10.2f}",
         f"{'Model':<14}|{ssm:>13.4f}{dfm:>9}{msm:>13.4f}   Prob > F       = {res['f_p']:>10.4f}",
         f"{'Residual':<14}|{ssr:>13.4f}{dfr:>9}{msr:>13.4f}   R-squared      = {res['r2']:>10.4f}",
         f"{'-' * 14}+{'-' * 35}   Adj R-squared  = {res['adj_r2']:>10.4f}",
         f"{'Total':<14}|{res['tss']:>13.4f}{max(n - 1, 1):>9}{res['tss'] / max(n - 1, 1):>13.4f}"
         f"   Root MSE      = {res['root_mse']:>10.4f}",
         "", rule,
         f"{'epsilon':>{w}} | {'Coef.':>11} {'Robust SE':>11} {'t':>8} {'P>|t|':>8}"
         f" {'[95% Conf. Interval]':>24}",
         "-" * (w + 1) + "+" + "-" * (79 - w - 2)]
    for j, name in enumerate(res["names"]):
        lab = name if len(name) <= w else name[:w - 1] + "~"
        L.append(f"{lab:>{w}} | {res['coef'][j]:>11.5f} {res['se'][j]:>11.5f}"
                 f" {res['t'][j]:>8.2f} {res['p'][j]:>8.3f}"
                 f" {res['ci'][j][0]:>11.5f} {res['ci'][j][1]:>11.5f}  {stars(res['p'][j])}")
    se_note = (f"Std. Err. adjusted for {G} clusters in target (met,rho) cells"
               if res["cov_kind"] == "cluster" else
               "Heteroskedasticity-robust Std. Err. (Stata vce(robust), HC1)")
    L += [rule, se_note,
          f"Collinearity: condition index (QR) = {res['cond']:.1f}, "
          f"max |corr| among regressors = {res['maxcorr']:.4f}"]
    if res["dropped"]:
        L.append("omitted because of collinearity: " + ", ".join(res["dropped"]))
    L += [""] + notes
    return "\n".join(L)
