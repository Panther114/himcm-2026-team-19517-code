"""Two-stage thermal-comfort elasticity workflow (single entry point).

 S0  Each survey row gets
       AT_f      Steadman apparent temperature from ta, rh, vel
       AT_ideal  ONE GLOBAL value: the maximiser of the deviation-weighted
                 dissatisfaction loss L(T) = sum(n*d)/sum(d), d = |AT_f - T|,
                 n = 1 for an unacceptable vote, on a 0.01 K grid
       delta     |AT_f - AT_ideal|
       rho       ONE 24-h mean density per building type (hour is gone)
       vote      dissat_ta (1 = unacceptable), dissat_tsv (robustness)
       met
     NO hour, NO GBOBD post-stratification.

 S1  Stratify rows by building type, then split each type into 5 equal
     groups ordered by met (lowest 20%, next 20%, ...).  No random sampling.

 S2  Within each stratum use 1 K bins of apparent-temperature deviation to
     get P(acceptable), the bin's empirical centre AT, and
     delta_b = |mean(AT in bin) - AT_ideal|.  The central bin is excluded
     from the slope regression.  U_max is the utility of the GLOBAL 1 K bin
     with the HIGHEST P(acceptable), searched over all bins (not just bin 0).
     U_max policy is selected with --u-max:
       global-max  ONE GLOBAL highest-P 1 K bin [default]
       ideal       ONE GLOBAL value: utility of the AT_ideal (bin 0) bin
       legacy      per-stratum bin-0 utility, raised to that stratum's
                   highest bin utility (the shipped baseline)
     Logit utility is unbounded above, so a small near-unanimous stratum bin
     can exceed a global U_max.  Two transformations keep every non-central
     bin -- including the U_max bin itself -- valid:
       (1) cap  U_b* = min(U_b, U_global(b) + cap_margin), i.e. no stratum
           bin may be more comfortable than the pooled global 1 K curve
           at the same temperature, up to a small allowance;
       (2) hybrid shortfall [default].  Let g = U_max - U_b*, let
           s = SE[logit(P_max)] be the sampling resolution of the peak, and
           let s_eff = z*s with z=1.645 (one-sided 95% band).  g0 = s_eff is
           the switch point: for g >= g0 the shortfall is EXACTLY the raw
           D = g, so no ordinary bin is perturbed.  Only for extremely high
           utilities (g < g0) is a positive branch used:
             D*(g) = (g0 - s_eff) + s_eff*exp((g - g0)/s_eff).
           D*(g0) = g0 and D*'(g0) = 1, so the join is C1-continuous with
           the raw branch.  D*(g) > 0 for every finite g, and because the
           utility cap bounds g >= -cap_margin, D* >= s_eff*exp((-cap_margin
           - g0)/s_eff) > 0 in these data.  --shortfall-z moves z;
           --shortfall-switch moves g0; --shortfall softplus|floored|raw
           retain the older global forms.
     From P = 1/(1+exp(-U)) invert U = -ln(1/P-1) = logit(P), set
     D = R*delta_b**eps, and fit
     ln D = ln R + eps*ln delta_b by weighted OLS.  The slope IS that
     stratum's elasticity eps.  Every survey row in a usable stratum becomes
     one (met, rho, eps) point.

 S3  Ten indicators: x, ln x, e^x, x/(1+x), x ln x for met AND for rho.
     WLS of eps on all ten (aweight 1/se^2), a diagnostic univariate R2 /
     cross-validated R2 screen, then PCA on the FULL 10x10 correlation
     matrix, PC scores, and 5% backward elimination of insignificant PCs.

 Density scale comes from encoded ASHRAE 62.1-2022 Table 6-1 and 2025
 California Title 24 Part 6 Table 120.1-A rows (src/standards.py).

Outputs: out/ by default (15 result files; see README.md for the file map).
Use --out to write a candidate run elsewhere without touching out/.

Usage:  python main.py
        python main.py --ideal-lo 10 --ideal-hi 35
        python main.py --bin-axis at --cov cluster
        python main.py --out out_candidate --shortfall-switch 0.05
Python 3.9+ standard library only, no third-party dependencies.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from src import core as reg
from src import standards

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
CLEAN = DATA / "ashrae_clean.csv"
PROFILE = DATA / "density_profile.csv"
STANDARDS = DATA / "standards"

# ---------------------------------------------------------------------------
# knobs (all deliberately capped; see complexity_check)
# ---------------------------------------------------------------------------
N_MET_BINS = 5                      # met-ordered split within each building type
MIN_STRATUM_ROWS = 200
MIN_GROUP_BIN_ROWS = 8              # min rows in one stratum's 1 K bin
MIN_GROUP_BINS = 4                  # min usable bins for the log-log WLS
MIN_USABLE_STRATA = 5               # min strata whose elasticity fit succeeds
MIN_BIN_ROWS = 200
IDEAL_STEP = 0.01                   # K, resolution of the global L(T) grid
MAX_IDEAL_GRID = 6000               # <= 60 K interval at 0.01 K (complexity cap)
MIN_DELTA, MAX_DELTA = 0.10, 8.00
MET_LO, MET_HI = 0.7, 3.0
GRIFFITHS = 0.5
PCA_CUM_TARGET = 0.70
PC_LOADING_CUT = 0.50
SE_FLOOR, SE_CAP = 0.05, 5.0
GRID_RHO = (0.011, 0.03, 0.084, 0.20, 0.50, 1.00, 2.00)
GRID_MET = (0.8, 1.2, 1.7, 2.5)
SAFE_GRID_RHO = (0.03, 0.084, 0.20, 0.50)   # practical, near-observed range
SAFE_GRID_MET = (0.8, 1.2, 1.7)
DELTA_EDGES = reg.DELTA_EDGES
NB = reg.NB

FORMS = [
    ("x", lambda v: v),
    ("ln_x", lambda v: math.log(v)),
    ("exp_x", lambda v: math.exp(v)),
    ("sat_x", lambda v: v / (1.0 + v)),          # x/(1+x)
    ("x_ln_x", lambda v: v * math.log(v)),
]
TERMS = [(var, name, fn) for var in ("met", "rho") for name, fn in FORMS]
LABELS = [f"{name}({var})" for var, name, _ in TERMS]
FORM_BY_LABEL = {f"{name}({var})": (var, fn)
                 for var, name, fn in TERMS}

LOG: list[str] = []
U_MAX_POLICY = "global-max"    # global-max | ideal | legacy
SHRINK_ALPHA = 0.0             # optional pseudo-rows at the global 1 K rate
SHORTFALL_MODE = "hybrid"      # hybrid | softplus | floored | raw
CAP_MARGIN: float | None = 0.10   # cap U at U_global(bin) + margin; None = off
CAP_RULE = "bin"               # bin | bin-z | absolute
CAP_Z = 1.645                  # z multiplier for the s-based ceilings
CAP_LEVEL: float | None = None # resolved absolute cap level
CAP_HITS = 0                   # stratum bins whose utility was capped
SHORTFALL_Z = 1.645            # auto scale = SHORTFALL_Z * SE[logit(P_max)]
SE_PEAK = 0.0                  # raw peak SE, recorded in summary.json
SHORTFALL_SCALE = 0.0          # 0 = auto = SHORTFALL_Z * SE[logit(P_max)]
SHORTFALL_SWITCH = 0.0         # 0 = auto = SHORTFALL_SCALE (g0 = s_eff)
P_GLOBAL_BIN: dict[int, float] = {}
U_GLOBAL_BIN: dict[int, float] = {}


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def fnum(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    i = min(max(int(q * (len(sorted_vals) - 1)), 0), len(sorted_vals) - 1)
    return sorted_vals[i]


def steadman(ta, rh, vel):
    """Steadman / Australian BoM apparent temperature in degrees C.

    e = rh/100 * 6.105 * exp(17.27 Ta / (237.7 + Ta))   [hPa]
    AT = Ta + 0.33 e - 0.70 v - 4.00
    `v` is the occupied-zone air speed (an approximation to 10 m wind).
    """
    e = (rh / 100.0) * 6.105 * math.exp(17.27 * ta / (237.7 + ta))
    return ta + 0.33 * e - 0.70 * vel - 4.00


def acceptable_rate(n_acc, n):
    """p_acc = n_acc / sqrt(n^2 + 1).

    The sqrt(n^2+1) denominator keeps p strictly below 1.  A 1e-6 guard is
    still applied at both ends so logit(p) stays finite when a bin is
    unanimous (the raw value would be 0 or ~1 - 1/(2n^2)).
    """
    if n <= 0:
        return 1e-6
    p = n_acc / math.sqrt(n * n + 1.0)
    return min(max(p, 1e-6), 1.0 - 1e-6)


def variance(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    mu = sum(vals) / n
    return sum((v - mu) ** 2 for v in vals) / (n - 1)


# ---------------------------------------------------------------------------
# S0  load once, derive AT / AT_ideal / delta / rho
# ---------------------------------------------------------------------------
def load_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ta = fnum(r.get("ta"))
            if ta is None:
                continue
            rows.append(dict(
                bid=(r.get("building_id") or "").strip(),
                bt=(r.get("building_type") or "").strip().lower(),
                ta=ta,
                rh=fnum(r.get("rh")),
                vel=fnum(r.get("vel")),
                met=fnum(r.get("met")),
                yta=fnum(r.get("dissat_ta")),       # 1 = unacceptable
                ytsv=fnum(r.get("dissat_tsv")),
                season=(r.get("season") or "").strip(),
            ))
    return rows


def add_apparent_temperature(rows):
    rhs = [r["rh"] for r in rows if r["rh"] is not None]
    vels = [r["vel"] for r in rows if r["vel"] is not None]
    med_rh = statistics.median(rhs) if rhs else 50.0
    med_vel = statistics.median(vels) if vels else 0.10
    n_imp = 0
    for r in rows:
        if r["rh"] is None or r["vel"] is None:
            n_imp += 1
        rh = min(max(r["rh"] if r["rh"] is not None else med_rh, 10.0), 90.0)
        vel = min(max(r["vel"] if r["vel"] is not None else med_vel, 0.0), 1.0)
        r["rh_used"], r["vel_used"] = rh, vel
        r["at"] = steadman(r["ta"], rh, vel)
    return med_rh, med_vel, n_imp


def _loss_grid_max(at, bad, lo, hi, step):
    """Global grid maximum of L(T) = sum(n_i*|x_i - T|) / sum(|x_i - T|).

    Numerator and denominator are both sums of absolute deviations.  With
    sorted x, prefix sums P and total S,

        sum_i |x_i - T| = T*(2*j - N) + S - 2*P[j],   j = #{x_i <= T},

    so each grid point costs two binary searches instead of an O(N) pass.
    This is algebraically identical to the naive loop, just O(log N) per T.
    """
    pts = sorted(zip(at, bad))
    xs = [p[0] for p in pts]
    xb = [p[0] for p in pts if p[1] > 0.5]
    n, m = len(xs), len(xb)
    if n == 0 or hi <= lo:
        raise SystemExit("no usable votes / empty interval for AT_ideal")
    pa, pb = [0.0] * (n + 1), [0.0] * (m + 1)
    for i, x in enumerate(xs):
        pa[i + 1] = pa[i] + x
    for i, x in enumerate(xb):
        pb[i + 1] = pb[i] + x
    k = int(math.floor((hi - lo) / step))
    if k + 1 > MAX_IDEAL_GRID:
        raise SystemExit("guardrail: AT_ideal interval exceeds "
                         f"{MAX_IDEAL_GRID} grid points at step {step}")
    best_L, best_T, best_den = -1.0, lo, 0.0
    for i in range(k + 1):
        T = lo + i * step
        j = bisect.bisect_right(xs, T)
        jb = bisect.bisect_right(xb, T)
        den = T * (2 * j - n) + pa[n] - 2.0 * pa[j]
        num = T * (2 * jb - m) + pb[m] - 2.0 * pb[jb]
        if den <= 1e-9:
            continue
        L = num / den
        if L > best_L:
            best_L, best_T, best_den = L, T, den
    return best_T, best_L, best_den, k + 1


def add_ideal_temperature(rows, lo=None, hi=None):
    """One GLOBAL AT_ideal: the value maximising the loss L on a 0.01 K grid.

    n = 1 when the vote is unacceptable, 0 otherwise; d = |AT_f - T| is the
    deviation from the candidate T.  The search interval defaults to the
    observed apparent-temperature range and can be set with --ideal-lo /
    --ideal-hi.  Every row gets the SAME AT_ideal; nothing varies by building.
    """
    pts = [(r["at"], r["yta"]) for r in rows
           if r["yta"] is not None and math.isfinite(r["at"])]
    if not pts:
        raise SystemExit("no usable votes to locate AT_ideal")
    at = [p[0] for p in pts]
    bad = [p[1] for p in pts]
    lo = min(at) if lo is None else float(lo)
    hi = max(at) if hi is None else float(hi)
    T, L, den, n_grid = _loss_grid_max(at, bad, lo, hi, IDEAL_STEP)
    for r in rows:
        r["ideal"] = T
        r["delta"] = abs(r["at"] - T)
    return dict(temperature=T, loss=L, interval=[lo, hi], step=IDEAL_STEP,
                grid_points=n_grid, n_votes=len(at),
                n_unacceptable=sum(1 for v in bad if v > 0.5),
                sum_deviation_at_max=den)


def prepare(rows, rho_by_type, bin_axis):
    """Filter to analysis rows, attach rho / deviation-bin / 1 K bin."""
    for r in rows:
        r["axis"] = r["at"] if bin_axis == "at" else (r["at"] - r["ideal"])
    axes = sorted(r["axis"] for r in rows
                  if r["yta"] is not None and r["met"] is not None)
    lo, hi = pct(axes, 0.01), pct(axes, 0.99)
    edge = math.floor(lo)
    rho_vals = sorted({v for v in rho_by_type.values()})
    cuts = reg.MET_CUTS
    kept, dropped = [], defaultdict(int)
    for r in rows:
        if r["yta"] is None or r["ytsv"] is None:
            dropped["no_vote"] += 1
            continue
        if r["met"] is None or not (MET_LO <= r["met"] <= MET_HI):
            dropped["met"] += 1
            continue
        if r["bt"] not in rho_by_type:
            dropped["unknown_building_type"] += 1
            continue
        d = r["delta"]
        if not (MIN_DELTA <= d < MAX_DELTA):
            dropped["delta_range"] += 1
            continue
        bi = bisect.bisect_right(DELTA_EDGES, d) - 1
        if bi < 0 or bi >= NB:
            dropped["delta_bin"] += 1
            continue
        if not (lo <= r["axis"] <= hi):
            dropped["at_tail"] += 1
            continue
        r["rho"] = rho_by_type[r["bt"]]
        r["db"] = bi
        r["ld"] = math.log(d)
        r["atbin"] = int(math.floor((r["at"] - r["ideal"]) + 0.5))
        r["mb"] = 0 if r["met"] < cuts[0] else (1 if r["met"] < cuts[1] else 2)
        r["rb"] = min(bisect.bisect_left(rho_vals, r["rho"]), len(rho_vals) - 1)
        kept.append(r)
    return kept, edge, lo, hi, dict(dropped)


# ---------------------------------------------------------------------------
# S1  utility logistic function from the rate of ACCEPTABLE votes
# ---------------------------------------------------------------------------
def utility_bins(rows):
    acc = defaultdict(lambda: dict(n=0.0, ta=0.0, tsv=0.0,
                                   d=0.0, at=0.0, ideal=0.0,
                                   met=0.0, rho=0.0))
    for r in rows:
        a = acc[r["atbin"]]
        a["n"] += 1.0
        a["ta"] += 1.0 - r["yta"]           # acceptable (direct utility vote)
        a["tsv"] += 1.0 - r["ytsv"]         # acceptable (PPD proxy)
        a["d"] += r["delta"]
        a["at"] += r["at"]
        a["ideal"] += r["ideal"]
        a["met"] += r["met"]
        a["rho"] += r["rho"]
    out = []
    for b, a in sorted(acc.items()):
        if a["n"] < MIN_BIN_ROWS:
            continue
        p_ta = acceptable_rate(a["ta"], a["n"])
        p_tsv = acceptable_rate(a["tsv"], a["n"])
        center_at = a["at"] / a["n"]
        center_delta = abs((a["at"] - a["ideal"]) / a["n"])
        out.append(dict(bin=b, axis_center=float(b), n=a["n"],
                        p_acceptable=p_ta, U_logit=reg.logit(p_ta),
                        p_acceptable_tsv=p_tsv, U_logit_tsv=reg.logit(p_tsv),
                        mean_delta=a["d"] / a["n"],
                        center_at=center_at, center_delta=center_delta,
                        mean_met=a["met"] / a["n"],
                        mean_rho=a["rho"] / a["n"]))
    return out


def utility_curve(bins, key="U_logit"):
    """Global utility logistic curve: logit P(acceptable) = c0+c1*x+c2*x^2."""
    X = [[1.0, b["axis_center"], b["axis_center"] ** 2] for b in bins]
    y = [b[key] for b in bins]
    pkey = "p_acceptable" if key == "U_logit" else "p_acceptable_tsv"
    w = [b["n"] * b[pkey] * (1.0 - b[pkey]) for b in bins]
    return reg.fit_wls(X, y, w, [str(i) for i in range(len(bins))],
                       ["_cons", "axis", "axis^2"], "robust")


# ---------------------------------------------------------------------------
# S1/S2  building type x met-bin strata -> one elasticity per stratum
# ---------------------------------------------------------------------------
def stratify(rows):
    """Split by building type, then into 5 equal met-ordered groups per type."""
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["bt"]].append(r)
    strata = []
    for bt, items in sorted(by_type.items()):
        items = sorted(items, key=lambda r: r["met"])
        n = len(items)
        for d in range(N_MET_BINS):
            lo, hi = n * d // N_MET_BINS, n * (d + 1) // N_MET_BINS
            part = items[lo:hi]
            if len(part) < MIN_STRATUM_ROWS:
                continue
            strata.append(dict(building_type=bt, met_bin=d,
                               rows=part, cluster=f"{bt}|met_bin_{d}"))
    return strata


def bin_utility(b, n, n_acc):
    """logit(P) for a 1 K bin, optionally shrunk toward the global rate."""
    global CAP_HITS
    P = acceptable_rate(n_acc, n)
    if SHRINK_ALPHA > 0.0 and b in P_GLOBAL_BIN:
        pg = P_GLOBAL_BIN[b]
        P = min(max((n_acc + SHRINK_ALPHA * pg) / (n + SHRINK_ALPHA),
                    1e-6), 1.0 - 1e-6)
    U = -math.log(1.0 / P - 1.0)
    if CAP_RULE == "absolute" and CAP_LEVEL is not None:
        if U > CAP_LEVEL:
            CAP_HITS += 1
            U = CAP_LEVEL
    elif CAP_RULE == "bin-z" and b in U_GLOBAL_BIN:
        cap = U_GLOBAL_BIN[b] + CAP_Z * SHORTFALL_SCALE
        if U > cap:
            CAP_HITS += 1
            U = cap
    elif CAP_MARGIN is not None and b in U_GLOBAL_BIN:
        cap = U_GLOBAL_BIN[b] + CAP_MARGIN
        if U > cap:
            CAP_HITS += 1
            U = cap
    return P, U


def stratum_bins(stratum):
    """1 K bins -> ((ln delta, U, weight) points, stratum bin-0 utility).

    The central bin is never used in the slope regression, but the legacy
    U_max policy needs its utility.  With any non-legacy policy the
    independent variable is ln|mean(AT in bin) - AT_ideal|; the legacy
    policy uses the integer bin centre ln|b| instead.
    """
    acc = defaultdict(lambda: dict(n=0.0, acc=0.0, dev=0.0))
    for r in stratum["rows"]:
        a = acc[r["atbin"]]
        a["n"] += 1.0
        a["acc"] += 1.0 - r["yta"]
        a["dev"] += r["at"] - r["ideal"]
    pts, u_centre = [], None
    for b, a in sorted(acc.items()):
        if a["n"] < MIN_GROUP_BIN_ROWS:
            continue
        P, U = bin_utility(b, a["n"], a["acc"])
        if b == 0:
            u_centre = U
            continue
        if U_MAX_POLICY == "legacy":
            ln_d = math.log(abs(b))
        else:
            centre_delta = abs(a["dev"] / a["n"])
            if centre_delta <= 0.0:
                continue
            ln_d = math.log(centre_delta)
        pts.append((ln_d, U, a["n"] * P * (1.0 - P)))
    return pts, u_centre


def softplus_gap(gap, scale):
    """s*ln(1+exp(gap/s)), numerically stable for very negative gaps."""
    x = gap / scale
    if x > 30.0:
        return gap
    if x < -30.0:
        return scale * math.exp(x)
    return scale * (x + math.log1p(math.exp(-x)))


def hybrid_gap(gap, scale, switch):
    """Raw shortfall above `switch`; C1 positive branch below it.

    For g >= g0 the result is the EXACT raw shortfall g.  For g < g0,

        D*(g) = (g0 - s) + s*exp((g - g0)/s)

    so D*(g0) = g0, D*'(g0) = 1 (C1 join), and D* -> g0 - s >= 0 as
    g -> -infinity.  For every finite gap D* > 0; a 1e-12 numerical floor
    guards underflow.  Only extremely high utilities (U near or above
    U_max) enter the transformed branch; every ordinary bin is untouched.
    """
    if gap >= switch:
        return gap
    x = (gap - switch) / scale
    floor = max(switch - scale, 1e-12)
    if x < -700.0:
        return floor
    return max(floor + scale * math.exp(x), 1e-12)


def elasticity_points(bin_pts, u_max):
    """D = R*delta_b**eps with a strictly positive shortfall.

    hybrid    D = g for g >= g0 (EXACT raw shortfall); D*(g) below g0,
              where g = U_max - U*.  g0 = s by default, s = SE[logit(Pmax)].
              D*(g0) = g0 with matching slope 1; D* > 0 for finite g.
    softplus  D = s*ln(1 + exp(g/s)) for every g (older global transform)
    floored   D = max(g, s)
    raw       D = g; bins with g <= 0 are dropped
    Returns (points, number of bins whose shortfall was transformed).
    """
    out, n_limited = [], 0
    for ln_d, U, weight in bin_pts:
        gap = u_max - U
        if SHORTFALL_MODE == "raw":
            if gap <= 0.0:
                continue
            D = gap
        elif SHORTFALL_MODE == "floored":
            D = max(gap, SHORTFALL_SCALE)
            if gap < SHORTFALL_SCALE:
                n_limited += 1
        elif SHORTFALL_MODE == "softplus":
            D = max(softplus_gap(gap, SHORTFALL_SCALE), 1e-12)
            if gap < SHORTFALL_SCALE:
                n_limited += 1
        else:  # hybrid
            D = max(hybrid_gap(gap, SHORTFALL_SCALE, SHORTFALL_SWITCH),
                    1e-12)
            if gap < SHORTFALL_SWITCH:
                n_limited += 1
        out.append((ln_d, math.log(D), weight))
    return out, n_limited


def wls_log_log(pts):
    """Weighted OLS of ln D on ln centre_delta; the slope is elasticity."""
    n = len(pts)
    if n < MIN_GROUP_BINS:
        return None
    sw = sum(p[2] for p in pts)
    xb = sum(p[2] * p[0] for p in pts) / sw
    yb = sum(p[2] * p[1] for p in pts) / sw
    sxx = sum(p[2] * (p[0] - xb) ** 2 for p in pts)
    if sxx <= 1e-12:
        return None
    eps = sum(p[2] * (p[0] - xb) * (p[1] - yb) for p in pts) / sxx
    lnR = yb - eps * xb
    rss = sum(p[2] * (p[1] - lnR - eps * p[0]) ** 2 for p in pts)
    sigma2 = rss / max(n - 2, 1)
    se = math.sqrt(max(sigma2 / sxx, 1e-12))
    se_nom = math.sqrt(1.0 / sxx)
    return eps, se, se_nom, lnR, n, rss


def derive_cells(strata, bin_pts_all, u_global, u_max_bin=None):
    """One (met, rho, eps) regression point per SURVEY ROW.

    With a non-legacy U_max policy, ONE global value is used for every
    stratum; it is never re-estimated or raised within a stratum.  The
    bounded shortfall keeps every non-central bin (including the U_max
    bin) in the fit.  The legacy policy instead uses the stratum's own
    bin-0 utility raised to that stratum's highest bin utility, with the
    raw shortfall.  Every vote inside a usable stratum keeps its own met
    (and rho).
    """
    cells, spreads, stratum_records = [], [], []
    diag = dict(n_strata=len(strata), too_few_bins=0, fit_failed=0,
                skipped_nonpositive_shortfall=0,
                shortfall_limited_bins=0,
                central_bin_excluded=True,
                u_max_policy=U_MAX_POLICY,
                shortfall_mode=SHORTFALL_MODE,
                shortfall_scale=SHORTFALL_SCALE,
                shortfall_z=SHORTFALL_Z,
                se_peak=SE_PEAK,
                shortfall_switch=(SHORTFALL_SWITCH
                                  if SHORTFALL_MODE == "hybrid" else None),
                shortfall_floor=(SHORTFALL_SWITCH - SHORTFALL_SCALE
                                 if SHORTFALL_MODE == "hybrid" else None),
                cap_margin=CAP_MARGIN,
                cap_rule=CAP_RULE,
                cap_z=(CAP_Z if CAP_RULE in ("absolute", "bin-z") else None),
                cap_level=CAP_LEVEL,
                cap_hits=CAP_HITS,
                shrink_alpha=SHRINK_ALPHA,
                u_max_global=u_global, u_max_bin=u_max_bin,
                u_max_bin_is_ideal=(u_max_bin == 0))
    umax_vals = []
    for s, (bin_pts, u_centre) in zip(strata, bin_pts_all):
        rows = s["rows"]
        ats = [r["at"] for r in rows]
        spreads.append(max(ats) - min(ats))
        if U_MAX_POLICY == "legacy":
            u_used = max([u_centre if u_centre is not None else u_global]
                         + [p[1] for p in bin_pts])
        else:
            u_used = u_global
        pts, n_limited = elasticity_points(bin_pts, u_used)
        diag["skipped_nonpositive_shortfall"] += len(bin_pts) - len(pts)
        diag["shortfall_limited_bins"] += n_limited
        fit = wls_log_log(pts)
        if fit is None:
            diag["too_few_bins" if len(pts) < MIN_GROUP_BINS
                 else "fit_failed"] += 1
            continue
        eps, se, se_nom, lnR, k, rss = fit
        if not (math.isfinite(eps) and 0.0 < eps < 5.0):
            diag["fit_failed"] += 1
            continue
        umax_vals.append(u_used)
        n = len(rows)
        sid = s["cluster"]
        met_mean = sum(r["met"] for r in rows) / n
        rho_mean = sum(r["rho"] for r in rows) / n
        at_spread = spreads[-1]
        se_capped = min(max(se, SE_FLOOR), SE_CAP)
        R = math.exp(lnR)
        stratum_records.append(dict(
            stratum=sid, building_type=s["building_type"],
            met_bin=s["met_bin"], n_rows=n, k_bins=k,
            met_mean=met_mean, rho_mean=rho_mean, at_spread=at_spread,
            eps=eps, se=se_capped, se_nom=se_nom, R=R,
            U_max=u_used))
        for r in rows:
            cells.append(dict(
                building_type=s["building_type"], met_bin=s["met_bin"],
                stratum=sid, cluster=sid, n_rows=n, k_bins=k,
                met=r["met"], rho=r["rho"], at=r["at"], atbin=r["atbin"],
                stratum_met=met_mean, stratum_rho=rho_mean,
                at_spread=at_spread, eps=eps, se=se_capped,
                se_nom=se_nom, R=R, rss=rss, U_max=u_used))
    diag["u_max_median"] = (statistics.median(umax_vals)
                            if umax_vals else u_global)
    diag["n_usable_strata"] = len(stratum_records)
    diag["n_row_points"] = len(cells)
    diag["at_spread_min"] = min(spreads) if spreads else 0.0
    diag["at_spread_median"] = statistics.median(spreads) if spreads else 0.0
    diag["at_spread_max"] = max(spreads) if spreads else 0.0
    return cells, diag, stratum_records


# ---------------------------------------------------------------------------
# S4  WLS + diagnostic R2 screen + PCA + PC elimination
# ---------------------------------------------------------------------------
def design(cells, idx=None):
    use = list(range(len(TERMS))) if idx is None else idx
    cols = [[TERMS[j][2](c[TERMS[j][0]]) for c in cells] for j in use]
    X = [[1.0] + [cols[m][i] for m in range(len(use))]
         for i in range(len(cells))]
    names = ["_cons"] + [LABELS[j] for j in use]
    return X, cols, names


def weights(cells, log_link=False):
    if log_link:
        w = [c["eps"] ** 2 / max(c["se"] ** 2, 1e-12) for c in cells]
    else:
        w = [1.0 / max(c["se"] ** 2, 1e-12) for c in cells]
    cap = sorted(w)[max(int(0.90 * (len(w) - 1)), 0)]
    return [min(x, cap) for x in w]


def cv_r2_simple(x, y, w, groups):
    n = len(y)
    wm = sum(w) / n
    ww = [wi / wm for wi in w]
    ybar = sum(ww[i] * y[i] for i in range(n)) / sum(ww)
    tss = sum(ww[i] * (y[i] - ybar) ** 2 for i in range(n))
    if tss <= 0:
        return 0.0
    sse = 0.0
    for g in sorted(set(groups)):
        tr = [i for i in range(n) if groups[i] != g]
        te = [i for i in range(n) if groups[i] == g]
        if len(tr) < 5 or not te:
            continue
        sw = sum(ww[i] for i in tr)
        xb = sum(ww[i] * x[i] for i in tr) / sw
        yb = sum(ww[i] * y[i] for i in tr) / sw
        sxx = sum(ww[i] * (x[i] - xb) ** 2 for i in tr)
        b = (sum(ww[i] * (x[i] - xb) * (y[i] - yb) for i in tr) / sxx
             if sxx > 0 else 0.0)
        a = yb - b * xb
        for i in te:
            sse += ww[i] * (y[i] - a - b * x[i]) ** 2
    return 1.0 - sse / tss


def pca_columns(cols, labels):
    n, p = len(cols[0]), len(cols)
    mu = [sum(c) / n for c in cols]
    sd = [math.sqrt(sum((v - mu[j]) ** 2 for v in cols[j]) / (n - 1)) or 1.0
          for j in range(p)]
    Z = [[(cols[j][i] - mu[j]) / sd[j] for j in range(p)] for i in range(n)]
    R = [[sum(Z[i][a] * Z[i][b] for i in range(n)) / (n - 1)
          for b in range(p)] for a in range(p)]
    ev, V = reg.jacobi_eigen(R)
    order = sorted(range(p), key=lambda k: -ev[k])
    ev = [max(ev[k], 0.0) for k in order]
    V = [[V[i][k] for k in order] for i in range(p)]
    share = [e / p for e in ev]
    cum, run = [], 0.0
    for s in share:
        run += s
        cum.append(run)
    k = max(1, sum(1 for e in ev if e > 1.0))
    while k < p and cum[k - 1] < PCA_CUM_TARGET:
        k += 1
    scores = [[sum(V[j][m] * Z[i][j] for j in range(p)) /
               math.sqrt(max(ev[m], 1e-12)) for m in range(k)] for i in range(n)]
    loadings = [[V[j][m] * math.sqrt(ev[m]) for m in range(k)] for j in range(p)]
    return dict(labels=list(labels), ev=ev, share=share, cum=cum, k=k,
                scores=scores, loadings=loadings)


def predict(m, met, rho, log_link=False):
    val = m["coef"][0]
    for j in range(1, len(m["names"])):
        nm = m["names"][j]
        if nm not in FORM_BY_LABEL:
            continue
        var, fn = FORM_BY_LABEL[nm]
        val += m["coef"][j] * fn(met if var == "met" else rho)
    if not log_link:
        return val
    return math.exp(max(min(val, 60.0), -60.0))


def safe_log_expression(cells, cols, ylog, wlog, groups_id, cov_kind):
    """Highest-R2 (met form, rho form) log-link pair that stays in band.

    Screened on the PRACTICAL extrapolation grid (near-observed rho/met);
    the wider 28-point grid is only used for reporting.  25 candidate pairs,
    each a 3-column WLS on the row-level points, i.e. O(25*n*9) with no
    optimisation loop.
    """
    grid = [(r, m) for r in SAFE_GRID_RHO for m in SAFE_GRID_MET]
    best, n_safe = None, 0
    for jm in range(5):                       # met forms
        for jr in range(5, 10):               # rho forms
            names = ["_cons", LABELS[jm], LABELS[jr]]
            Xp = [[1.0, cols[jm][i], cols[jr][i]] for i in range(len(cells))]
            try:
                m = reg.fit_wls(Xp, ylog, wlog, groups_id, names, cov_kind)
            except (ValueError, ZeroDivisionError):
                continue
            vals = [predict(m, met, rho, True) for rho, met in grid]
            if not all(0.05 <= v <= 10.0 for v in vals):
                continue
            n_safe += 1
            if best is None or m["r2"] > best["r2"]:
                best = m
    return best, n_safe, len(grid)


def positivity_check(cells, m_level, m_log):
    grid = []
    for r in GRID_RHO:
        for a in GRID_MET:
            lv = predict(m_level, a, r)
            lg = predict(m_log, a, r, True)
            grid.append(dict(rho=r, met=a, eps_level=lv, eps_log_link=lg,
                             level_positive=lv > 0,
                             level_in_band=0.05 <= lv <= 10.0,
                             log_in_band=0.05 <= lg <= 10.0))
    return dict(grid=grid,
                n=len(grid),
                level_negative=sum(1 for g in grid if not g["level_positive"]),
                level_out_of_band=sum(1 for g in grid if not g["level_in_band"]),
                log_out_of_band=sum(1 for g in grid if not g["log_in_band"]),
                observed_rho_range=[min(c["rho"] for c in cells),
                                    max(c["rho"] for c in cells)])


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def write_csv(name, rows, fieldnames=None):
    if not rows:
        return
    keys = fieldnames or list(rows[0].keys())
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def fmt_table(res, title, notes):
    return reg.stata_table(res, title, notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-axis", choices=("dev", "at"), default="dev",
                    help="dev = 1 K bins of (AT - AT_ideal) [default]; "
                         "at = literal 1 K bins of absolute apparent temperature")
    ap.add_argument("--cov", choices=("robust", "cluster"), default="robust")
    ap.add_argument("--out", default="out",
                    help="output directory (relative to the project root)")
    ap.add_argument("--ideal-lo", type=float, default=None,
                    help="lower end of the AT_ideal loss search interval "
                         "(default: minimum observed apparent temperature)")
    ap.add_argument("--ideal-hi", type=float, default=None,
                    help="upper end of the AT_ideal loss search interval "
                         "(default: maximum observed apparent temperature)")
    ap.add_argument("--u-max", choices=("global-max", "ideal", "legacy"),
                    default="global-max",
                    help="U_max policy: global-max = ONE global highest-P "
                         "1 K bin [default]; ideal = ONE global AT_ideal "
                         "bin; legacy = per-stratum bin 0 raised to that "
                         "stratum's highest bin (shipped baseline)")
    ap.add_argument("--shortfall",
                    choices=("hybrid", "softplus", "floored", "raw"),
                    default="hybrid",
                    help="shortfall transform; hybrid [default] uses the "
                         "EXACT raw D above the switch point and a positive "
                         "branch only for extremely high utilities; "
                         "softplus/floored transform every bin; raw drops "
                         "D <= 0")
    ap.add_argument("--cap-margin", type=float, default=0.10,
                    help="cap stratum U at U_global(same 1 K bin) + margin; "
                         "negative disables the cap (--cap-rule bin only)")
    ap.add_argument("--cap-rule", choices=("bin", "bin-z", "absolute"),
                    default="bin",
                    help="bin = cap at U_global(same 1 K bin)+margin "
                         "[default]; bin-z = same, with margin cap_z*s; "
                         "absolute = cap every U at U_max + cap_z*s")
    ap.add_argument("--cap-z", type=float, default=1.645,
                    help="z multiplier for --cap-rule bin-z|absolute; 1 = one "
                         "peak SE, 1.645 = one-sided 95%% upper bound [default]")
    ap.add_argument("--shortfall-z", type=float, default=1.645,
                    help="auto shortfall multiplier: s_eff = z*SE[logit(P_max)]; "
                         "1.645 = one-sided 95%% band [default]")
    ap.add_argument("--shortfall-scale", type=float, default=0.0,
                    help="shortfall scale s_eff; 0 = auto = z*SE[logit(P_max)]")
    ap.add_argument("--shortfall-switch", type=float, default=0.0,
                    help="hybrid switch gap g0; 0 = auto = s "
                         "(must be >= s; larger g0 raises the positive floor)")
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="optional pseudo-rows at the global 1 K rate used "
                         "to shrink noisy stratum bins (0 disables)")
    a = ap.parse_args()
    global OUT, U_MAX_POLICY, SHRINK_ALPHA, P_GLOBAL_BIN, U_GLOBAL_BIN
    global SHORTFALL_MODE, CAP_MARGIN, SHORTFALL_SCALE, SHORTFALL_SWITCH
    global SHORTFALL_Z, SE_PEAK, CAP_RULE, CAP_Z, CAP_LEVEL, CAP_HITS
    U_MAX_POLICY = a.u_max
    SHRINK_ALPHA = a.shrink
    CAP_RULE = a.cap_rule
    CAP_Z = a.cap_z
    CAP_LEVEL = None
    CAP_HITS = 0
    CAP_MARGIN = a.cap_margin if a.cap_margin >= 0 else None
    SHORTFALL_Z = a.shortfall_z
    SE_PEAK = 0.0
    SHORTFALL_SCALE = a.shortfall_scale
    SHORTFALL_SWITCH = a.shortfall_switch
    if (SHRINK_ALPHA < 0.0 or SHORTFALL_SCALE < 0.0
            or SHORTFALL_SWITCH < 0.0 or CAP_Z < 0.0
            or SHORTFALL_Z < 0.0):
        raise SystemExit("--shrink, --shortfall-scale, --shortfall-switch, "
                         "--shortfall-z and --cap-z must be nonnegative")
    if U_MAX_POLICY == "legacy":
        SHORTFALL_MODE = "raw"
        CAP_MARGIN = None
        CAP_RULE = "bin"
    else:
        SHORTFALL_MODE = a.shortfall
    out_path = Path(a.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    OUT = out_path.resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---- complexity check BEFORE any heavy work -------------------------
    n_terms = len(TERMS)
    n_cap = 110_000                     # cap on row-level regression points
    n_stage4_fits = 2 + n_terms + 4 + 2 + 25   # full, univariate, PCA, final, safety
    ops = (6 * n_cap                                # CSV + derive passes
           + n_cap * 17                              # sort + prefix sums for L grid
           + MAX_IDEAL_GRID * 2 * 17                 # 0.01 K grid, 2 bisections / T
           + n_cap * 17                              # met sort within building types
           + n_cap                                   # strata -> 1 K bins + log-log WLS
           + n_cap                                   # expand strata to row points
           + n_cap * (n_stage4_fits * 25            # all stage-4 WLS fits
                      + 2 * n_terms ** 2)           # full-correlation PCA
           + 10 * 3 * n_cap)                        # leave-one-stratum-out CV screen
    log("=" * 79)
    log("Two-stage thermal-comfort elasticity workflow")
    log("=" * 79)
    log(f"complexity estimate: ~{ops:,} elementary operations "
        f"(cap 250,000,000); met bins={N_MET_BINS}, forms={n_terms}")
    if ops > 250_000_000:
        raise SystemExit("REFUSING to run: estimated work exceeds the cap")
    log("no MLE / scipy.minimize / statsmodels; single CSV read; "
        "no random sampling, no O(N) loop inside the elasticity fit")

    # ---- S0 -------------------------------------------------------------
    log("\n[S0] reading data/ashrae_clean.csv once ...")
    rows = load_rows(CLEAN)
    log(f"  rows with air temperature: {len(rows):,}")
    med_rh, med_vel, n_imp = add_apparent_temperature(rows)
    log(f"  Steadman AT: median rh={med_rh:.1f}, median vel={med_vel:.3f}, "
        f"rows with imputed rh/vel={n_imp:,}")
    ideal = add_ideal_temperature(rows, a.ideal_lo, a.ideal_hi)
    log(f"  AT_ideal = {ideal['temperature']:.2f} C (GLOBAL, one value for "
        f"all {len(rows):,} rows); L_max={ideal['loss']:.4f}")
    log(f"    loss L(T)=sum(n*d)/sum(d), n=1 if unacceptable, d=|AT-T|; "
        f"grid [{ideal['interval'][0]:.2f}, {ideal['interval'][1]:.2f}] "
        f"step {IDEAL_STEP} K ({ideal['grid_points']:,} points, "
        f"{ideal['n_votes']:,} votes, "
        f"{ideal['n_unacceptable']:,} unacceptable)")
    density = standards.build_density(STANDARDS, PROFILE)
    rho_by_type = density["rho_by_type"]
    log("  one density per building type (standards-derived): " +
        ", ".join(f"{k}={v:.4f}" for k, v in sorted(rho_by_type.items())))
    log("  density provenance: " +
        ", ".join(f"{p['building_type']}=[{p['ref_number']}] {p['table']}"
                  for p in density["provenance"]))
    log(f"  profile validation: {density['validation']['n_types']} types, "
        f"table rows {density['validation']['tables']}, "
        f"max |CSV - standard| = "
        f"{density['validation']['max_abs_diff']:.2e}")
    kept, edge, lo, hi, dropped = prepare(rows, rho_by_type, a.bin_axis)
    log(f"  usable rows: {len(kept):,}; 1 K axis range [{lo:.2f}, {hi:.2f}] "
        f"(axis = {a.bin_axis}); dropped: {dropped}")
    if len(kept) < 1000:
        raise SystemExit("too few usable rows")

    # ---- S1 -------------------------------------------------------------
    bins = utility_bins(kept)
    bin_ids = [b["bin"] for b in bins]
    if len(bin_ids) < 4:
        raise SystemExit("too few usable 1 K bins")
    P_GLOBAL_BIN = {b["bin"]: b["p_acceptable"] for b in bins}
    U_GLOBAL_BIN = {b["bin"]: b["U_logit"] for b in bins}
    u_bin = max(bins, key=lambda b: b["p_acceptable"])
    u_ideal_bin = next((b for b in bins if b["bin"] == 0), None)
    u_ideal = u_ideal_bin["U_logit"] if u_ideal_bin else u_bin["U_logit"]
    if U_MAX_POLICY == "global-max":
        u_global = u_bin["U_logit"]
        u_global_bin = u_bin["bin"]
    else:
        u_global = u_ideal
        u_global_bin = 0
    p_pk = u_bin["p_acceptable"]
    se_pk = 1.0 / math.sqrt(max(u_bin["n"] * p_pk * (1.0 - p_pk), 1.0))
    SE_PEAK = se_pk
    if SHORTFALL_SCALE <= 0.0:
        SHORTFALL_SCALE = SHORTFALL_Z * se_pk
        scale_source = (f"auto = z*SE[logit(P_max)], z={SHORTFALL_Z:g}, "
                        f"SE={se_pk:.6f} = 1/sqrt(n*P*(1-P)), "
                        f"n={u_bin['n']:.0f}, P={p_pk:.6f}")
    else:
        scale_source = "user-supplied"
    if SHORTFALL_SWITCH <= 0.0:
        SHORTFALL_SWITCH = SHORTFALL_SCALE
        switch_source = "auto = s_eff = z*SE (one-sided band)"
    else:
        switch_source = "user-supplied"
    if SHORTFALL_SWITCH < SHORTFALL_SCALE:
        raise SystemExit("--shortfall-switch must be >= s_eff so that the "
                         "hybrid join stays monotone and positive")
    if CAP_RULE == "absolute":
        CAP_LEVEL = u_global + CAP_Z * SHORTFALL_SCALE
        cap_bound = CAP_Z * SHORTFALL_SCALE
    elif CAP_RULE == "bin-z":
        CAP_LEVEL = None
        cap_bound = CAP_Z * SHORTFALL_SCALE
    else:
        CAP_LEVEL = None
        cap_bound = CAP_MARGIN if CAP_MARGIN is not None else 0.0
    hybrid_floor_bound = max(
        1e-12,
        max(SHORTFALL_SWITCH - SHORTFALL_SCALE, 0.0)
        + SHORTFALL_SCALE * math.exp(
            (-cap_bound - SHORTFALL_SWITCH) / SHORTFALL_SCALE))
    curve = utility_curve(bins)
    curve_tsv = utility_curve(bins, "U_logit_tsv")
    log(f"\n[S1] {len(bins)} global 1 K bins (>= {MIN_BIN_ROWS} rows each); "
        f"bin 0 = [-0.5,0.5) K centred on AT_ideal, excluded from regressions")
    log(f"  U_max policy = {U_MAX_POLICY}; shortfall = {SHORTFALL_MODE}; "
        f"shrink alpha = {SHRINK_ALPHA:g}")
    if U_MAX_POLICY == "legacy":
        log(f"  global fallback U_max (AT_ideal bin 0) = {u_global:.4f}; "
            "each stratum raises it to its own highest bin utility")
    else:
        log(f"  ONE global U_max = {u_global:.4f} from highest-P bin "
            f"{u_global_bin} (P={u_bin['p_acceptable']:.4f}), "
            f"empirical centre AT = {u_bin['center_at']:.4f}")
    log(f"  AT_ideal bin 0: P={u_ideal_bin['p_acceptable']:.4f}, "
        f"U={u_ideal:.4f}; AT_ideal = {ideal['temperature']:.4f} C")
    if CAP_RULE == "absolute":
        cap_text = f"absolute U_max+{CAP_Z:g}*s = {CAP_LEVEL:.6f}"
    elif CAP_RULE == "bin-z":
        cap_text = f"bin U_global+{CAP_Z:g}*s"
    else:
        cap_text = ("off" if CAP_MARGIN is None
                    else f"bin U_global+{CAP_MARGIN:g}")
    log(f"  shortfall transform: mode={SHORTFALL_MODE}, "
        f"cap={cap_text}, s_eff={SHORTFALL_SCALE:.6f} "
        f"({scale_source})")
    if SHORTFALL_MODE == "hybrid":
        log(f"    z={SHORTFALL_Z:g}, raw SE={SE_PEAK:.6f}, "
            f"s_eff=z*SE={SHORTFALL_SCALE:.6f}")
        log(f"    switch g0={SHORTFALL_SWITCH:.6f} ({switch_source}); "
            f"asymptotic floor g0-s_eff="
            f"{SHORTFALL_SWITCH-SHORTFALL_SCALE:.6f}")
        log(f"    guaranteed D* >= {hybrid_floor_bound:.3e} "
            f"(cap bounds g >= -{cap_bound:g}); 1e-12 numerical floor")
        log("    g >= g0 -> D = g EXACTLY (normal utilities unchanged); "
            "g < g0 -> D* = (g0-s_eff)+s_eff*exp((g-g0)/s_eff) > 0")
        log("    D*(g0)=g0 and D*'(g0)=1, so the join is C1")
    elif SHORTFALL_MODE == "softplus":
        log("    D = s*ln(1+exp((U_max - U*)/s)) > 0 for every bin")
    elif SHORTFALL_MODE == "floored":
        log("    D = max(U_max - U*, s) > 0 for every bin")
    else:
        log("    D = U_max - U*; bins with D <= 0 are dropped")
    if u_bin["bin"] != 0:
        log("  NOTE: the global highest-P bin is NOT centred on AT_ideal; "
            "AT_ideal comes from max L(T), not from max P(acceptable)")
    log("  global utility curve logit P(acceptable) = "
        + " + ".join(f"({curve['coef'][j]:.4f})*{nm}"
                     for j, nm in enumerate(curve["names"]))
        + f"   [R2={curve['r2']:.4f}]")

    # ---- S2 -------------------------------------------------------------
    strata = stratify(kept)
    bin_pts_all = [stratum_bins(s) for s in strata]
    type_counts = defaultdict(int)
    for s in strata:
        type_counts[s["building_type"]] += 1
    log(f"\n[S2] {len(strata)} strata = {len(type_counts)} building types "
        f"x {N_MET_BINS} met bins (>= {MIN_STRATUM_ROWS} rows each)")
    log("  strata per type: " +
        ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))
    log("  independent variable: "
        + ("ln|integer 1 K bin centre|" if U_MAX_POLICY == "legacy"
           else "ln|mean(AT in 1 K bin) - AT_ideal|")
        + "; central bin excluded; U_max policy = " + U_MAX_POLICY)

    # ---- S3 -------------------------------------------------------------
    cells, cell_diag, stratum_records = derive_cells(
        strata, bin_pts_all, u_global, u_global_bin)
    n_usable = len(stratum_records)
    log(f"\n[S3] {len(cells):,} row-level (met, rho, eps) points from "
        f"{n_usable}/{len(strata)} usable strata "
        f"(too few bins={cell_diag['too_few_bins']}, "
        f"fit failed={cell_diag['fit_failed']}, "
        f"skipped nonpositive shortfall="
        f"{cell_diag['skipped_nonpositive_shortfall']}, "
        f"shortfall-limited bins="
        f"{cell_diag['shortfall_limited_bins']}, "
        f"capped bins={cell_diag['cap_hits']})")
    log(f"  U_max: policy={U_MAX_POLICY}, global={u_global:.4f} "
        f"from bin {u_global_bin}, median used="
        f"{cell_diag['u_max_median']:.4f}; "
        f"central bin excluded from all slope fits")
    log(f"  eps > 1: "
        f"{sum(1 for s in stratum_records if s['eps'] > 1.0)}/"
        f"{n_usable} usable strata (law of decreasing marginal utility)")
    log(f"  AT spread inside a stratum (K): "
        f"min={cell_diag['at_spread_min']:.2f}, "
        f"median={cell_diag['at_spread_median']:.2f}, "
        f"max={cell_diag['at_spread_max']:.2f}")
    if n_usable < MIN_USABLE_STRATA:
        raise SystemExit("too few usable strata")
    eps_sorted = sorted(s["eps"] for s in stratum_records)
    log(f"  n_strata={n_usable}, mean eps={sum(eps_sorted)/n_usable:.4f}, "
        f"p10-p90={eps_sorted[n_usable//10]:.4f}-"
        f"{eps_sorted[9*n_usable//10]:.4f}, "
        f"median R={sorted(s['R'] for s in stratum_records)[n_usable//2]:.4f}")

    # ---- S4 -------------------------------------------------------------
    X, cols, names = design(cells)
    y = [c["eps"] for c in cells]
    w = weights(cells)
    ylog = [math.log(c["eps"]) for c in cells]
    wlog = weights(cells, log_link=True)
    groups_id = [c["cluster"] for c in cells]
    m_full = reg.fit_wls(X, y, w, groups_id, names, a.cov)
    m_cluster = reg.fit_wls(X, y, w, groups_id, names, "cluster")
    m_full_log = reg.fit_wls(X, ylog, wlog, groups_id, names, a.cov)
    m_full_log_cluster = reg.fit_wls(X, ylog, wlog, groups_id,
                                     names, "cluster")

    singles = []
    for j, lab in enumerate(LABELS):
        Xi = [[1.0, cols[j][i]] for i in range(len(cells))]
        m = reg.fit_wls(Xi, y, w, groups_id, ["_cons", lab], a.cov)
        singles.append(dict(form=lab, variable=TERMS[j][0],
                            r2=m["r2"], cv_r2=cv_r2_simple(
                                cols[j], y, w, groups_id),
                            coef=m["coef"][1], p=m["p"][1]))
    singles.sort(key=lambda s: -s["r2"])

    pca = pca_columns(cols, LABELS)
    pc_cols = [[pca["scores"][i][m] for i in range(len(cells))]
               for m in range(pca["k"])]
    pc_names = [f"PC{m+1}" for m in range(pca["k"])]
    Xpc = [[1.0] + [pc_cols[m][i] for m in range(pca["k"])]
           for i in range(len(cells))]
    m_pc_log = reg.fit_wls(Xpc, ylog, wlog, groups_id,
                           ["_cons"] + pc_names, a.cov)
    m_pc = reg.fit_custom(cells, pc_cols, pc_names, "robust")
    pc_kept, pc_names_kept, pc_dropped, m_pc_sel = reg.backward_custom(
        cells, pc_cols, pc_names, "robust")
    surviving = [pc_names.index(nm) for nm in pc_names_kept if nm in pc_names]
    load_pass = [LABELS[j] for j in range(len(LABELS))
                 if any(abs(pca["loadings"][j][m]) >= PC_LOADING_CUT
                        for m in surviving)]

    best_met = max((s for s in singles if s["variable"] == "met"),
                   key=lambda s: s["r2"])
    best_rho = max((s for s in singles if s["variable"] == "rho"),
                   key=lambda s: s["r2"])
    final_idx = [LABELS.index(best_met["form"]), LABELS.index(best_rho["form"])]
    Xf, _, fnames = design(cells, final_idx)
    m_final = reg.fit_wls(Xf, y, w, groups_id, fnames, a.cov)
    m_log_naive = reg.fit_wls(Xf, ylog, wlog, groups_id, fnames, a.cov)
    m_safe, n_safe, n_grid_pts = safe_log_expression(
        cells, cols, ylog, wlog, groups_id, a.cov)
    if m_safe is not None:
        m_log = m_safe
        log_expr_source = "extrapolation-safe screen over 25 (met,rho) pairs"
    else:
        m_log = m_log_naive
        log_expr_source = "R2-winner pair (no fully safe pair found)"
    log(f"  log-link selection: {log_expr_source}; "
        f"{n_safe}/25 pairs safe on {n_grid_pts} grid points; "
        f"chosen R2(ln eps)={m_log['r2']:.4f}")
    pos = positivity_check(cells, m_final, m_log)

    var_eps = variance(y)
    ceiling_nom = 1.0 - (sum(c["se_nom"] ** 2 for c in cells) / len(cells)) / var_eps \
        if var_eps > 0 else 0.0
    ceiling_cons = 1.0 - (sum(c["se"] ** 2 for c in cells) / len(cells)) / var_eps \
        if var_eps > 0 else 0.0

    log(f"\n[S4] full 10-form model: R2={m_full['r2']:.4f}, "
        f"adj R2={m_full['adj_r2']:.4f}, cond={m_full['cond']:.1f}, "
        f"max|corr|={m_full['maxcorr']:.4f}")
    log(f"  full 10-form model on ln(eps): R2={m_full_log['r2']:.4f}, "
        f"k={m_full_log['k']}, cond={m_full_log['cond']:.1f}")
    log(f"  PCA-score model on ln(eps): R2={m_pc_log['r2']:.4f}, "
        f"PCs={pca['k']}")
    log(f"  best single form: {singles[0]['form']} "
        f"(R2={singles[0]['r2']:.4f}, CV R2={singles[0]['cv_r2']:.4f})")
    log(f"  PCA: {pca['k']} components, cumulative={pca['cum'][pca['k']-1]:.4f}; "
        f"significant PCs after 5% elimination: "
        f"{', '.join(pc_names_kept) or 'none'}")
    log(f"  final expression: {best_met['form']} + {best_rho['form']} "
        f"(R2={m_final['r2']:.4f})")
    log(f"  positivity grid: level negative {pos['level_negative']}/{pos['n']}, "
        f"level out-of-band {pos['level_out_of_band']}/{pos['n']}, "
        f"log-link out-of-band {pos['log_out_of_band']}/{pos['n']}")

    # ---- outputs --------------------------------------------------------
    write_csv("density_provenance.csv", density["provenance"])
    (OUT / "citations.txt").write_text(
        standards.format_plain(density["citations"], density["provenance"]),
        encoding="utf-8")
    (OUT / "references.bib").write_text(
        standards.format_bibtex(density["citations"]), encoding="utf-8")
    write_csv("cells.csv", cells)
    write_csv("utility_bins.csv", bins)
    write_csv("single_forms.csv", singles)
    write_csv("pca_variance.csv",
              [dict(component=f"PC{i+1}", eigenvalue=pca["ev"][i],
                    share=pca["share"][i], cumulative=pca["cum"][i])
               for i in range(len(LABELS))])
    write_csv("pca_loadings.csv",
              [dict(form=LABELS[j], **{f"PC{m+1}": pca["loadings"][j][m]
                                       for m in range(pca["k"])})
               for j in range(len(LABELS))])
    write_csv("coef_table.csv",
              [dict(model=tag, term=m["names"][j], coef=m["coef"][j],
                    se=m["se"][j], t=m["t"][j], p=m["p"][j],
                    ci_lo=m["ci"][j][0], ci_hi=m["ci"][j][1])
               for tag, m in (("full_10_forms", m_full),
                              ("full_cluster", m_cluster),
                              ("full_10_forms_ln_eps", m_full_log),
                              ("full_10_forms_ln_eps_cluster",
                               m_full_log_cluster),
                              ("pc_scores_ln_eps", m_pc_log),
                              ("pc_scores", m_pc),
                              ("pc_selected", m_pc_sel),
                              ("final_level", m_final),
                              ("final_log_link", m_log))
               for j in range(len(m["names"]))])
    write_csv("fit_stats.csv",
              [dict(model=tag, n=m["n"], k=m["k"], r2=m["r2"],
                    adj_r2=m["adj_r2"], aic=m["aic"], bic=m["bic"],
                    root_mse=m["root_mse"], cond=m["cond"],
                    maxcorr=m["maxcorr"], G=m["G"])
               for tag, m in (("full_10_forms", m_full),
                              ("full_cluster", m_cluster),
                              ("full_10_forms_ln_eps", m_full_log),
                              ("full_10_forms_ln_eps_cluster",
                               m_full_log_cluster),
                              ("pc_scores_ln_eps", m_pc_log),
                              ("pc_scores", m_pc),
                              ("pc_selected", m_pc_sel),
                              ("final_level", m_final),
                              ("final_log_link", m_log))])
    by_bt = defaultdict(list)
    for c in cells:
        by_bt[c["building_type"]].append(c)
    by_bt_strata = defaultdict(int)
    for s in stratum_records:
        by_bt_strata[s["building_type"]] += 1
    write_csv("robustness.csv",
              [dict(building_type=bt, n_rows=len(v),
                    n_strata=len({c["stratum"] for c in v}),
                    rho=v[0]["rho"],
                    mean_met=sum(c["met"] for c in v) / len(v),
                    mean_eps=sum(c["eps"] for c in v) / len(v),
                    median_R=sorted(c["R"] for c in v)[len(v) // 2],
                    mean_at_spread=sum(c["at_spread"] for c in v) / len(v))
               for bt, v in sorted(by_bt.items())])
    write_csv("positivity_grid.csv", pos["grid"])

    # ---- Stata-style report --------------------------------------------
    T = ["=" * 79,
         "TWO-STAGE THERMAL-COMFORT ELASTICITY WORKFLOW (Stata-style output)",
         "=" * 79,
         "Data    ASHRAE Global Thermal Comfort Database II v2.1.0 [5].",
         "Rho     phi(h) from GBOBD [3,4]; rho_peak from ASHRAE 62.1-2022",
         "        Table 6-1 [1] and 2025 CA Title 24 Part 6 Table 120.1-A [2].",
         "        See out/density_provenance.csv and out/citations.txt.",
         "Step 0  AT_f = Steadman apparent temperature; ONE GLOBAL AT_ideal",
         "        maximises L(T)=sum(n*d)/sum(d), d=|AT_f-T|, n=1 if the",
         "        vote is unacceptable; exhaustive 0.01 K grid. delta =",
         "        |AT_f - AT_ideal|; one rho per building type (24-h mean).",
         "Step 1  Stratify by building type, then 5 met bins per type.",
         "        Within each stratum: 1 K bins -> P = n_acc/sqrt(n^2+1),",
         f"        U* = min(U, U_global(bin) + {cap_text}); central bin",
         "        b=0 excluded from the fit.",
         "Step 2  P = 1/(1+exp(-U)) => U = -ln(1/P-1) = logit(P).",
         f"        U_max = {u_global:.4f} from the GLOBAL highest-P 1 K bin",
         f"        {u_global_bin} (P={u_bin['p_acceptable']:.4f}); "
         f"AT_ideal bin 0 U={u_ideal:.4f}.",
         f"        global-max-is-ideal = {u_bin['bin'] == 0}; "
         f"median U_max used = {cell_diag['u_max_median']:.4f}.",
         f"        shortfall = {SHORTFALL_MODE}, s = "
         f"{SHORTFALL_SCALE:.6f} = SE[logit(P_max)],",
         f"        switch g0 = s = {SHORTFALL_SWITCH:.6f}; "
         f"guaranteed D* >= {hybrid_floor_bound:.3e}.",
         "        g >= g0 -> D = g EXACTLY (raw); g < g0 -> D* = (g0-s)",
         f"        + s*exp((g-g0)/s) > 0, C1 at g0. "
         f"{cell_diag['shortfall_limited_bins']} bins transformed.",
         "        D = R*delta_b**eps; weighted OLS of ln D on",
         f"        {'ln|integer bin centre|' if U_MAX_POLICY == 'legacy' else 'ln|mean(AT in bin) - AT_ideal|'}"
         " gives eps (the slope) for that stratum.",
         "Step 3  One (met, rho, eps) point per survey row, with eps from",
         "        that row's stratum; 10 indicators, WLS, PCA + 5% PC",
         "        elimination.  SEs are clustered on the stratum.",
         f"Note    Recommended log-link expression: {log_expr_source}.",
         "=" * 79, ""]
    T += [fmt_table(m_full, "TABLE 1  Full model: all 10 functional forms",
                    ["aweight = 1/se(eps)^2 (capped at 90th pct)",
                     f"vce({a.cov}); clusters = met-bin x building-type strata"]), ""]
    T += [fmt_table(m_cluster, "TABLE 2  Full model, vce(cluster) sensitivity",
                    ["Same coefficients, cluster-robust standard errors."]), ""]
    T += [fmt_table(m_full_log,
                    "TABLE 2A  Full 10-form model with ln(epsilon) as dependent variable",
                    ["Dependent variable is ln(epsilon); eps = exp(X*b) > 0.",
                     "aweight = eps^2 / se(eps)^2 (capped at 90th pct).",
                     "PCA is on the predictor correlation matrix, so the PCs",
                     "are identical to those in TABLE 4; only the response",
                     "scale changes.  Do not interpret individual coefficients:",
                     "the forms are near-collinear (see condition index)."]), ""]
    T += [fmt_table(m_full_log_cluster,
                    "TABLE 2B  Full 10-form ln(epsilon) model, vce(cluster)",
                    ["Same coefficients, cluster-robust standard errors."]), ""]
    hdr = (f"{'form':<24}{'family':<10}{'R2':>10}{'CV R2':>10}"
           f"{'coef':>12}{'p':>10}")
    T += ["TABLE 3  Univariate functional-form verification (diagnostic only)",
          "", hdr, "-" * 79]
    T += [f"{s['form']:<24}{s['variable']:<10}{s['r2']:>10.4f}"
          f"{s['cv_r2']:>10.4f}{s['coef']:>12.4f}{s['p']:>10.4f}"
          for s in singles]
    T += ["", "Note    This screen is NOT used to pre-select PCA inputs; PCA is",
          "        run on the FULL 10x10 correlation matrix so p-values stay valid.",
          ""]
    T += ["TABLE 4  PCA of all 10 forms (correlation matrix)", "",
          f"{'component':<12}{'eigenvalue':>14}{'share':>12}{'cumulative':>14}",
          "-" * 79]
    T += [f"{'PC'+str(i+1):<12}{pca['ev'][i]:>14.5f}{pca['share'][i]:>12.5f}"
          f"{pca['cum'][i]:>14.5f}" for i in range(len(LABELS))]
    T += ["", f"Retained: {pca['k']} components (Kaiser > 1, then 70% cumulative).",
          ""]
    T += ["TABLE 5  PCA loadings", "",
          f"{'form':<24}" + "".join(f"{'PC'+str(m+1):>12}"
                                    for m in range(pca["k"])), "-" * 79]
    T += [f"{LABELS[j]:<24}" + "".join(f"{pca['loadings'][j][m]:>12.4f}"
                                       for m in range(pca["k"]))
          for j in range(len(LABELS))]
    T += ["", f"|loading| >= {PC_LOADING_CUT:.2f} on a surviving PC: "
          f"{len(load_pass)}/{len(LABELS)} forms", ""]
    T += [fmt_table(m_pc, "TABLE 6  Regression of epsilon on retained PCs",
                    ["PC scores are orthogonal by construction."]), ""]
    T += [fmt_table(m_pc_log,
                    "TABLE 6A  Regression of ln(epsilon) on retained PCs",
                    ["Dependent variable is ln(epsilon); eps = exp(X*b) > 0.",
                     "aweight = eps^2 / se(eps)^2.  PC scores are orthogonal",
                     "by construction, so this is the PCA-reduced ln-eps fit."]), ""]
    T += [fmt_table(m_pc_sel, "TABLE 7  PCs after 5% backward elimination",
                    ["Eliminated: " + (", ".join(n for n, _ in pc_dropped)
                                       or "none")]),
          "", f"Significant PCs: {', '.join(pc_names_kept) or 'none'}", ""]
    T += [fmt_table(m_final, "TABLE 8  Final level expression (R2-screen winners)",
                    ["Forms chosen by univariate R2; PCA validates the common",
                     "factor and removes insignificant components."]), ""]
    T += [fmt_table(m_log, "TABLE 9  Recommended POSITIVE expression (log-link)",
                    ["Dependent variable is ln(epsilon); eps = exp(X*b) > 0.",
                     "aweight = eps^2 / se(eps)^2."]), ""]
    T += ["TABLE 10  Positivity / extrapolation grid", "",
          f"{'rho':>10}{'met':>10}{'eps_level':>14}{'eps_log_link':>16}"
          f"{'level>0':>10}", "-" * 79]
    T += [f"{g['rho']:>10.3f}{g['met']:>10.2f}{g['eps_level']:>14.4f}"
          f"{g['eps_log_link']:>16.4f}{str(g['level_positive']):>10}"
          for g in pos["grid"]]
    T += ["", "=" * 79, "IS THIS LOW R2 FEASIBLE?  (plain English)", "=" * 79,
          "epsilon is a log-log slope estimated from a few noisy 1 K bins",
          "per stratum, so each epsilon carries its own standard error.  A",
          f"simple noise ceiling is R2_max = 1 - mean(se^2)/Var(eps).  Here",
          f"Var(eps) = {var_eps:.4f}, the nominal ceiling is {ceiling_nom:.4f}",
          f"and the residual-based conservative ceiling is {ceiling_cons:.4f}.",
          "The 10 candidate forms are smooth functions of one another, so the",
          "full-model R2 is inflated by collinearity.  A modest R2 for a single",
          "shape is therefore expected, not a bug.", "",
          "=" * 79, "BOTTOM LINE (plain English)", "=" * 79,
          f"From {len(kept):,} comfort votes we built {n_usable} usable strata,",
          f"and kept {len(cells):,} row-level (met, rho, eps) points.  The",
          "recommended expression is the log-link model in TABLE 9:",
          "    epsilon = exp(b0 + b_met*f(met) + b_rho*g(rho))",
          "which is positive by construction and stays in a plausible range",
          f"on the extrapolation grid ({pos['log_out_of_band']}/{pos['n']} "
          "out of band).",
          "Caveats: rho is a class-level proxy; strata are 5 met bins within",
          "building types (SEs clustered by stratum); the 10 forms are",
          "collinear; eps is stratum-level, so rows are not independent.",
          f"U_max policy = {U_MAX_POLICY}; global value {u_global:.4f} "
          f"from the highest-P bin {u_global_bin}; median used = "
          f"{cell_diag['u_max_median']:.4f}.",
          f"The {SHORTFALL_MODE} shortfall (s_eff={SHORTFALL_SCALE:.6f}, "
          f"z={SHORTFALL_Z:g}, g0={SHORTFALL_SWITCH:.6f}) is the EXACT raw "
          "shortfall above g0;",
          "only extremely high utilities use the positive C1 branch, so",
          "ordinary bins are unmodified. The central bin is never used in a",
          "slope fit. eps near the peak still depends on s_eff and g0.",
          "Only non-legacy policies use ONE global U_max; legacy raises it",
          "within each stratum and uses the raw shortfall.",
          "=" * 79]
    (OUT / "stata_tables.txt").write_text("\n".join(T) + "\n",
                                                encoding="utf-8")

    delta_text = ("ln|integer 1 K bin centre|" if U_MAX_POLICY == "legacy"
                  else "|mean(AT in 1 K bin) - AT_ideal|")
    umax_text = {
        "legacy": "per-stratum bin 0 raised to that stratum's highest bin",
        "ideal": "ONE global AT_ideal (bin 0) value",
        "global-max": "ONE global highest-P 1 K bin",
    }[U_MAX_POLICY]
    shortfall_text = {
        "raw": "D=U_max-U (raw; bins with D<=0 are dropped)",
        "floored": f"D=max(U_max-U*, {SHORTFALL_SCALE:.6f})",
        "softplus": (f"D=s_eff*ln(1+exp((U_max-U*)/s_eff)), "
                     f"s_eff={SHORTFALL_SCALE:.6f}"),
        "hybrid": (f"g=U_max-U*; D=g for g>=g0={SHORTFALL_SWITCH:.6f} "
                   f"(EXACT raw); D*=(g0-s_eff)+s_eff*exp((g-g0)/s_eff)>0 "
                   f"for g<g0, s_eff=z*SE={SHORTFALL_SCALE:.6f}, "
                   f"z={SHORTFALL_Z:g}, floor=g0-s_eff="
                   f"{SHORTFALL_SWITCH-SHORTFALL_SCALE:.6f}, C1 at g0"),
    }[SHORTFALL_MODE]
    if CAP_RULE == "absolute":
        cap_text_s = (f"min(U, U_max+{CAP_Z:g}*s) = "
                      f"min(U, {CAP_LEVEL:.6f})")
    elif CAP_RULE == "bin-z":
        cap_text_s = f"min(U, U_global(bin)+{CAP_Z:g}*s)"
    else:
        cap_text_s = ("off" if CAP_MARGIN is None
                      else f"min(U, U_global(bin)+{CAP_MARGIN:g})")
    shortfall_kind = {
        "raw": "raw shortfall ",
        "hybrid": "hybrid (raw above g0, positive C1 branch below g0) ",
        "softplus": "strictly positive shortfall ",
        "floored": "floored shortfall ",
    }[SHORTFALL_MODE]
    summary = dict(
        method=("Steadman AT; ONE GLOBAL AT_ideal from the 0.01 K grid "
                "maximum of L(T)=sum(n*d)/sum(d); stratify by building type "
                "then 5 met-ordered bins per type; per-stratum 1 K bins -> "
                "P=n_acc/sqrt(n^2+1); U=-ln(1/P-1); U_max from "
                + umax_text + "; stratum U cap " + cap_text_s + "; "
                "central bin excluded from regressions; "
                "delta_b=" + delta_text + "; " + shortfall_kind
                + shortfall_text + "; D=R*delta_b^eps; weighted OLS of "
                "ln D on ln delta_b -> one epsilon per stratum; "
                "every survey row in a usable stratum becomes one "
                "(met, rho, eps) point carrying its own met; "
                "WLS of epsilon AND of ln(epsilon) on 10 forms with "
                "full-correlation PCA and 5% PC elimination; recommended "
                "expression is the positivity-screened log link "
                "epsilon = exp(X*b); SEs clustered on the stratum"),
        bin_axis=a.bin_axis, n_strata=len(strata),
        n_usable_strata=n_usable,
        strata_per_type=dict(sorted(type_counts.items())),
        n_survey_records=len(kept), n_1k_bins=len(bins),
        n_regression_rows=len(cells), cluster_groups=m_cluster["G"],
        complexity_ops=ops, runtime_seconds=None,
        apparent_temperature=dict(formula="Ta + 0.33*e - 0.70*v - 4.00, "
                                         "e = rh/100*6.105*exp(17.27Ta/(237.7+Ta))",
                                  median_rh=med_rh, median_vel=med_vel,
                                  n_imputed=n_imp),
        at_ideal=ideal,
        density_by_type=rho_by_type,
        density_provenance=density["provenance"],
        density_validation=density["validation"],
        citations=[{k: c[k] for k in ("ref_number", "key", "short",
                                      "bibtex_key", "role", "doi", "url")}
                   for c in sorted(density["citations"].values(),
                                   key=lambda r: int(r["ref_number"]))],
        utility_curve=dict(coef=dict(zip(curve["names"], curve["coef"])),
                           r2=curve["r2"],
                           tsv_coef=dict(zip(curve_tsv["names"],
                                             curve_tsv["coef"])),
                           tsv_r2=curve_tsv["r2"]),
        utility_max_global=u_global,
        utility_max_bin=u_global_bin,
        utility_max_bin_p=u_bin["p_acceptable"],
        utility_max_bin_n=u_bin["n"],
        utility_max_bin_center_at=u_bin["center_at"],
        utility_max_bin_is_ideal=(u_global_bin == 0),
        utility_max_source=umax_text,
        utility_max_median=cell_diag["u_max_median"],
        at_ideal_bin_p=u_ideal_bin["p_acceptable"],
        at_ideal_bin_U=u_ideal,
        global_highest_p_bin=u_bin["bin"],
        global_highest_p_bin_U=u_bin["U_logit"],
        global_highest_p_bin_is_ideal=(u_bin["bin"] == 0),
        u_max_policy=U_MAX_POLICY,
        shrink_alpha=SHRINK_ALPHA,
        shortfall=dict(mode=SHORTFALL_MODE,
                       z=SHORTFALL_Z,
                       se_peak=SE_PEAK,
                       scale=SHORTFALL_SCALE,
                       scale_source=scale_source,
                       switch_gap=(SHORTFALL_SWITCH
                                   if SHORTFALL_MODE == "hybrid" else None),
                       switch_source=(switch_source
                                      if SHORTFALL_MODE == "hybrid"
                                      else None),
                       positive_floor=(SHORTFALL_SWITCH - SHORTFALL_SCALE
                                       if SHORTFALL_MODE == "hybrid"
                                       else None),
                       guaranteed_min_shortfall=(
                           hybrid_floor_bound
                           if SHORTFALL_MODE == "hybrid" else None),
                       numerical_floor=1e-12,
                       raw_above_switch=(SHORTFALL_MODE == "hybrid"),
                       cap_margin=CAP_MARGIN,
                       cap_rule=CAP_RULE,
                       cap_z=(CAP_Z if CAP_RULE in ("absolute", "bin-z") else None),
                       cap_level=CAP_LEVEL,
                       cap_hits=CAP_HITS,
                       formula=shortfall_text,
                       strictly_positive=SHORTFALL_MODE != "raw"),
        elasticity_fit=dict(min_group_bin_rows=MIN_GROUP_BIN_ROWS,
                            min_group_bins=MIN_GROUP_BINS,
                            min_stratum_rows=MIN_STRATUM_ROWS,
                            n_met_bins=N_MET_BINS,
                            delta_definition=delta_text,
                            u_max_policy=U_MAX_POLICY,
                            shortfall_mode=SHORTFALL_MODE,
                            shortfall_switch=(
                                SHORTFALL_SWITCH
                                if SHORTFALL_MODE == "hybrid" else None),
                            utility_cap=cap_text_s,
                            bin_rate_shrink_alpha=SHRINK_ALPHA,
                            central_bin_excluded=True,
                            row_level_points=True,
                            weights="n_bin * P * (1-P)"),
        cell_diagnostics=cell_diag,
        strata_by_building_type=dict(sorted(by_bt_strata.items())),
        rows_by_building_type={bt: len(v) for bt, v in sorted(by_bt.items())},
        epsilon_mean=sum(eps_sorted) / n_usable,
        epsilon_median=eps_sorted[n_usable // 2],
        epsilon_p10=eps_sorted[n_usable // 10],
        epsilon_p90=eps_sorted[9 * n_usable // 10],
        R_median=sorted(s["R"] for s in stratum_records)[n_usable // 2],
        full_model=dict(r2=m_full["r2"], adj_r2=m_full["adj_r2"],
                        cond_index=m_full["cond"],
                        max_abs_corr=m_full["maxcorr"]),
        ln_eps_full_model=dict(
            dependent_variable="ln(epsilon)",
            link="log -> epsilon = exp(X*b)",
            weight="epsilon^2 / se(epsilon)^2 (capped at 90th pct)",
            n=m_full_log["n"], k=m_full_log["k"],
            r2=m_full_log["r2"], adj_r2=m_full_log["adj_r2"],
            cond_index=m_full_log["cond"],
            max_abs_corr=m_full_log["maxcorr"],
            coef=dict(zip(m_full_log["names"], m_full_log["coef"])),
            se=dict(zip(m_full_log["names"], m_full_log["se"])),
            p=dict(zip(m_full_log["names"], m_full_log["p"]))),
        ln_eps_pc_model=dict(
            dependent_variable="ln(epsilon)",
            link="log -> epsilon = exp(X*b)",
            n=m_pc_log["n"], k=m_pc_log["k"],
            r2=m_pc_log["r2"], adj_r2=m_pc_log["adj_r2"],
            coef=dict(zip(m_pc_log["names"], m_pc_log["coef"])),
            se=dict(zip(m_pc_log["names"], m_pc_log["se"])),
            p=dict(zip(m_pc_log["names"], m_pc_log["p"]))),
        best_single_form=singles[0]["form"],
        best_single_r2=singles[0]["r2"],
        best_single_cv_r2=singles[0]["cv_r2"],
        final_forms=[best_met["form"], best_rho["form"]],
        final_r2=m_final["r2"],
        pca=dict(n_components=pca["k"],
                 cumulative=pca["cum"][pca["k"] - 1],
                 eigenvalues=pca["ev"][:pca["k"]],
                 significant_components=pc_names_kept,
                 dropped_components=[n for n, _ in pc_dropped],
                 n_forms_passing_loading_cut=len(load_pass),
                 note="PCA eliminates insignificant COMPONENTS, not individual "
                      "near-collinear forms"),
        low_r2_diagnosis=dict(var_epsilon=var_eps,
                              r2_ceiling_nominal=ceiling_nom,
                              r2_ceiling_conservative=ceiling_cons),
        positivity=dict(observed_rho_range=pos["observed_rho_range"],
                        n_grid_points=pos["n"],
                        level_negative=pos["level_negative"],
                        level_out_of_band=pos["level_out_of_band"],
                        log_link_out_of_band=pos["log_out_of_band"]),
        recommended_function=dict(
            link="log", source=log_expr_source,
            forms=m_log["names"][1:],
            coef=dict(zip(m_log["names"], m_log["coef"])),
            r2_on_ln_eps=m_log["r2"],
            naive_r2_winner_forms=fnames[1:],
            n_extrapolation_safe_pairs=n_safe,
            n_grid_points=n_grid_pts,
            safety_grid=dict(rho=list(SAFE_GRID_RHO), met=list(SAFE_GRID_MET)),
            note="log-link guarantees eps>0; the [0.05,10] band is a "
                 "plausibility check on the practical extrapolation grid"),
        robustness=dict(sorted(by_bt_strata.items())),
        caveats=["Steadman indoor vel is an approximation to 10 m wind.",
                 "rho is one number per building type, not a measured occupancy.",
                 "eps is estimated once per stratum and repeated on every row; "
                 "rows are not independent, so SEs are clustered on the stratum.",
                 "Strata are 5 met-ordered bins within building type.",
                 "rho_peak uses the SI reading of ASHRAE 62.1-2022 Table 6-1 "
                 "('#/1000 ft2 or #/100 m2'); Title 24 rows apply the same "
                 "design number per 100 m2, a modelling choice documented in "
                 "out/density_provenance.csv.",
                 "The 10 forms are strongly collinear; full-model coefficients "
                 "are not separately interpretable."],
    )
    summary["caveats"] += (
        ["U_max is the per-stratum bin-0 utility raised to that stratum's "
         "highest bin utility (legacy baseline); it is re-estimated inside "
         "each stratum and the global highest-P bin is not used.",
         "delta_b is " + delta_text + "; the central bin is excluded "
         "from every slope fit.",
         "The raw shortfall drops any bin with D<=0; no utility cap or "
         "shortfall transformation is applied."]
        if U_MAX_POLICY == "legacy" else [
            f"U_max policy is '{U_MAX_POLICY}': {umax_text}. AT_ideal "
            "maximises the dissatisfaction loss L(T), NOT P(acceptable); "
            "the two constants optimise different functionals. In these "
            "data the highest-P 1 K bin is not bin 0, so U_max sits "
            "about 1 K away from AT_ideal.",
            "delta_b is " + delta_text + "; the central bin is excluded "
            "from every slope fit.",
            "Stratum utility is capped at " + cap_text_s + " so that a "
            "small near-unanimous bin cannot exceed the ceiling implied by "
            "the pooled peak bin.",
            {
                "hybrid": (
                    f"The hybrid shortfall ({shortfall_text}) leaves the raw "
                    "shortfall EXACT for every bin above the switch gap and "
                    "applies a strictly positive C1 branch only to extremely "
                    "high utilities, so the U_max bin stays in the regression."),
                "softplus": (
                    f"The softplus shortfall ({shortfall_text}) is strictly "
                    "positive for every bin, including the U_max bin, and "
                    "approaches the raw gap away from the ceiling."),
                "floored": (
                    f"The floored shortfall ({shortfall_text}) is strictly "
                    "positive for every bin."),
                "raw": "The raw shortfall drops bins with D<=0.",
            }[SHORTFALL_MODE],
            f"s_eff = z*SE[logit(P_max)] with z={SHORTFALL_Z:g} is the "
            "one-sided resolution band of the peak; it is data-derived, "
            "not tuned. The utility cap bounds the gap below, so D > 0 for "
            "every retained bin. eps near the peak is still sensitive to z "
            "and the cap rule, so the absolute magnitude of eps is "
            "model-dependent.",
            f"Bin rates are shrunk toward the global 1 K rate with "
            f"{SHRINK_ALPHA:g} pseudo-rows (0 = no shrinkage)."])
    summary["runtime_seconds"] = round(time.time() - t0, 2)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"\ndone in {summary['runtime_seconds']} s; wrote 15 files to {OUT}")


if __name__ == "__main__":
    main()
