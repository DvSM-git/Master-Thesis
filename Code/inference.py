"""
Inference procedures for the MoM-IV thesis (Paper/iteration4/inference.tex).

Implements, faithfully to the tex:

  * the MoM-AR test (Algorithm `alg:mom_ar`, Theorem `thm:mom_ar_size`),
    with either the oracle sigma_Ze or a feasible robust (MoM) scale estimate
    computed at a beta0-free preliminary estimator, per the Feasibility remark;
  * the exact MoM-AR confidence set via closed-form breakpoint enumeration
    (Algorithm `alg:mom_ar_cs`, Theorem `thm:piecewise`, Cor. `cor:endpoints`);
  * the self-normalised AR test (Def. `def:sn_stat`, Prop. `prop:sn_pivotal`),
    with simulated R_k critical values, and its confidence set
    (Cor. `cor:sn_cs`) via grid + bisection refinement;
  * the standard (non-robust) Anderson-Rubin test and its closed-form
    confidence set (quadratic inversion, Dufour geometry) as baseline;
  * the monotonicity / single-interval diagnostics
    (Prop. `prop:mono_det`, `prop:mono_cheby`).

Median convention
-----------------
Theorem `thm:piecewise` and Algorithm `alg:mom_ar_cs` define the median as the
order statistic of rank r = ceil(k/2) (for even k, the *lower* of the two
middle values), which is what makes the median coincide with a single block
mean on each affine piece. All inference procedures in this module therefore
use that rank-based median (`rank_median`), including the simulation of the
R_k critical values, so statistic and critical value are always consistent.
The point estimators in simulation.py keep numpy's midpoint convention; the
counting argument in the proofs is valid for either.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INF = np.inf

# ----------------------------------------------------------------------------
# Basic building blocks
# ----------------------------------------------------------------------------


def k_blocks(delta: float) -> int:
    """Number of blocks for the MoM-AR test, k = ceil(8 ln(1/delta)) (thm:mom_ar_size)."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return int(np.ceil(8.0 * np.log(1.0 / delta)))


def rank_median(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Median as the order statistic of rank r = ceil(k/2) (1-indexed), the
    convention of Theorem `thm:piecewise`. Equals the usual median for odd k;
    for even k it is the lower of the two middle values.
    """
    v = np.asarray(v)
    k = v.shape[axis]
    r = (k + 1) // 2  # ceil(k/2), 1-indexed
    return np.take(np.partition(v, r - 1, axis=axis), r - 1, axis=axis)


def rank_mad(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """MAD(v) = med_j |v_j - med_l v_l| with the rank-based median (def:sn_stat)."""
    v = np.asarray(v)
    med = np.expand_dims(rank_median(v, axis=axis), axis=axis)
    return rank_median(np.abs(v - med), axis=axis)


def block_means(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Partition into k blocks of size m = floor(n/k) and return the block means

        a_j = mu_hat_ZY^(j),   b_j = mu_hat_ZX^(j),

    so that W_bar_j(beta0) = a_j - beta0 * b_j (eq:Wbar). Rows are shuffled
    first by default (the sample is i.i.d., so any partition is valid;
    shuffling removes dependence on row order). The last n - k*m rows are
    dropped, as in the algorithms.
    """
    n = len(Y)
    m = n // k
    if m == 0:
        raise ValueError(f"k={k} too large for n={n}: block size floor(n/k) is 0")
    ZY = Z * Y
    ZX = Z * X
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        perm = rng.permutation(n)
        ZY = ZY[perm]
        ZX = ZX[perm]
    a = ZY[: k * m].reshape(k, m).mean(axis=1)
    b = ZX[: k * m].reshape(k, m).mean(axis=1)
    return a, b, m


def tau_n(sigma_Ze: float, n: int, delta: float) -> float:
    """
    MoM-AR threshold tau_n(delta) = sigma_Ze * sqrt(32 ln(1/delta) / n) (eq:tau).

    Note (rounding): the proof-exact threshold is 2 sigma_Ze / sqrt(m) with
    m = floor(n/k); eq:tau equals it only when k = 8 ln(1/delta) exactly and
    k | n, and is otherwise very slightly smaller (anti-conservative). We
    implement eq:tau as stated in the thesis.
    """
    return sigma_Ze * np.sqrt(32.0 * np.log(1.0 / delta) / n)


def robust_sigma_Ze(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    delta: float,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> float:
    """
    Feasible, beta0-free robust estimate of sigma_Ze, per the Feasibility
    remark in inference.tex: residuals are formed at a preliminary consistent
    estimator (the MoR point estimate, which is beta0-free), and the scale is
    read off the block means of Z*eps_hat:

        sigma_hat = 1.4826 * MAD_j( bar(W)_j ) * sqrt(m),

    where 1.4826 = 1/Phi^{-1}(3/4) makes the MAD consistent for the standard
    deviation of the (asymptotically normal) block means. This uses the same
    robust scale functional as the self-normalised test and stays stable
    under heavy tails.

    (A scalar-MoM estimate of E[(Z eps)^2] was rejected: for tail index 2.1
    the squared products have tail index ~1.05, and the median of their block
    means sits far below the mean, severely underestimating sigma_Ze.)
    """
    n = len(Y)
    k = k_blocks(delta)
    m = n // k
    if m == 0:
        raise ValueError(f"n={n} too small for k={k}")
    if rng is None:
        rng = np.random.default_rng()

    # Preliminary beta0-free estimator: Median-of-Ratios (alg:mor).
    a, b, _ = block_means(Y, X, Z, k, rng=rng, shuffle=shuffle)
    if np.any(b == 0):
        raise ValueError("a block mean(Z*X) is zero; instrument not relevant")
    beta_mr = float(np.median(a / b))

    W = Z * (Y - beta_mr * X)
    if shuffle:
        W = W[rng.permutation(n)]
    bm = W[: k * m].reshape(k, m).mean(axis=1)
    sigma_hat = 1.4826 * float(rank_mad(bm)) * np.sqrt(m)
    return max(sigma_hat, 1e-150)


# ----------------------------------------------------------------------------
# MoM-AR test (Algorithm alg:mom_ar)
# ----------------------------------------------------------------------------


def mom_ar_statistic(a: np.ndarray, b: np.ndarray, beta0) -> np.ndarray:
    """
    W_tilde(beta0) = med_j(a_j - beta0 * b_j) (eq:Wtilde), vectorised over a
    scalar or 1-d array of null values beta0.
    """
    beta0 = np.asarray(beta0, dtype=float)
    W = a[None, :] - beta0.reshape(-1, 1) * b[None, :]
    out = rank_median(W, axis=1)
    return out if beta0.ndim else out[0]


def mom_ar_test(
    data: pd.DataFrame,
    beta0,
    delta: float = 0.05,
    sigma_Ze: float | None = None,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> dict:
    """
    Median-of-Means Anderson-Rubin test (alg:mom_ar, thm:mom_ar_size).

    Rejects H0: beta = beta0 when |W_tilde(beta0)| > tau_n(delta), with
    tau_n(delta) = sigma_Ze * sqrt(32 ln(1/delta)/n) (eq:tau).

    sigma_Ze : oracle value of sigma_Ze. If None, the feasible robust
               estimate of `robust_sigma_Ze` is used (computed once, free of
               beta0, as required by the Feasibility remark).
    beta0    : scalar or array of null values (vectorised; one threshold).
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n = len(Y)
    k = k_blocks(delta)
    if rng is None:
        rng = np.random.default_rng()

    a, b, m = block_means(Y, X, Z, k, rng=rng, shuffle=shuffle)
    if sigma_Ze is None:
        sigma_Ze = robust_sigma_Ze(Y, X, Z, delta, rng=rng, shuffle=shuffle)
    thr = tau_n(sigma_Ze, n, delta)
    W_tilde = mom_ar_statistic(a, b, beta0)
    return {
        "reject": np.abs(W_tilde) > thr,
        "W_tilde": W_tilde,
        "threshold": thr,
        "k": k,
        "m": m,
        "n": n,
    }


# ----------------------------------------------------------------------------
# Exact MoM-AR confidence set (Algorithm alg:mom_ar_cs)
# ----------------------------------------------------------------------------


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge closed intervals that overlap or share an endpoint (fp tolerance)."""
    merged: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if merged:
            plo, phi = merged[-1]
            tol = 1e-12 * max(1.0, abs(phi) if np.isfinite(phi) else 1.0)
            if lo <= phi + tol:
                merged[-1] = (plo, max(phi, hi))
                continue
        merged.append((lo, hi))
    return merged


def mom_ar_cs_exact(a: np.ndarray, b: np.ndarray, tau: float) -> list[tuple[float, float]]:
    """
    Exact MoM-AR confidence set via closed-form breakpoint enumeration
    (Algorithm alg:mom_ar_cs).

    1. Enumerate all pairwise crossings of the block-mean lines
       W_bar_j(beta) = a_j - beta * b_j  (at most C(k,2), Theorem thm:piecewise).
    2. On each open interval between consecutive crossings the median
       coincides with a single active line (rank r = ceil(k/2)).
    3. Intersect that line's band condition |W_bar_a(beta)| <= tau with the
       interval; endpoints are among the 2k candidates of Cor. cor:endpoints.

    Returns a list of closed intervals (lo, hi), possibly infinite, merged.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = a.size
    r = (k + 1) // 2

    i, j = np.triu_indices(k, 1)
    db = b[j] - b[i]
    nz = db != 0
    crossings = np.unique((a[j][nz] - a[i][nz]) / db[nz])
    edges = np.concatenate(([-INF], crossings, [INF]))

    pieces: list[tuple[float, float]] = []
    for l in range(len(edges) - 1):
        lo, hi = edges[l], edges[l + 1]
        if lo == hi:
            continue
        # test point inside the open interval
        if np.isinf(lo) and np.isinf(hi):
            beta_star = 0.0
        elif np.isinf(lo):
            beta_star = hi - 1.0
        elif np.isinf(hi):
            beta_star = lo + 1.0
        else:
            beta_star = 0.5 * (lo + hi)

        W = a - beta_star * b
        active = int(np.argpartition(W, r - 1)[r - 1])
        aa, bb = a[active], b[active]

        if bb != 0.0:
            e1 = (aa - tau) / bb
            e2 = (aa + tau) / bb
            Jlo, Jhi = (e1, e2) if e1 <= e2 else (e2, e1)
        else:
            if abs(aa) <= tau:
                Jlo, Jhi = -INF, INF
            else:
                continue

        s_lo, s_hi = max(lo, Jlo), min(hi, Jhi)
        if s_lo <= s_hi:
            pieces.append((s_lo, s_hi))

    return _merge_intervals(pieces)


def mom_ar_cs(
    data: pd.DataFrame,
    delta: float = 0.05,
    sigma_Ze: float | None = None,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> dict:
    """
    MoM-AR confidence set (def:cs) computed exactly by Algorithm alg:mom_ar_cs.

    Also reports the single-interval diagnostics:
      - `all_same_sign`: all block means b_j share a sign (prop:mono_det);
        when True, the CS is guaranteed a single interval.
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n = len(Y)
    k = k_blocks(delta)
    if rng is None:
        rng = np.random.default_rng()

    a, b, m = block_means(Y, X, Z, k, rng=rng, shuffle=shuffle)
    if sigma_Ze is None:
        sigma_Ze = robust_sigma_Ze(Y, X, Z, delta, rng=rng, shuffle=shuffle)
    thr = tau_n(sigma_Ze, n, delta)
    intervals = mom_ar_cs_exact(a, b, thr)
    return {
        "intervals": intervals,
        "threshold": thr,
        "k": k,
        "m": m,
        "n": n,
        "all_same_sign": bool(np.all(b > 0) or np.all(b < 0)),
        "block_means": (a, b),
    }


def cs_contains(intervals: list[tuple[float, float]], x: float, tol: float = 1e-9) -> bool:
    """Membership check for a list of closed intervals."""
    return any(lo - tol <= x <= hi + tol for lo, hi in intervals)


def aggregate_cs(cs_list: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """
    Aggregated confidence set of cor:agg_cs: the beta0 covered by at least half
    of the B per-permutation confidence sets.

    The aggregated test of eq:agg_test rejects when strictly MORE than half of
    the replicates reject, so beta0 survives when at least B/2 of them fail to
    reject, i.e. when it lies in at least B/2 of the sets `cs_list`.

    Each input set is a finite union of closed intervals, so the coverage count
    is piecewise constant with breakpoints only at their endpoints, and the
    result is exact rather than a grid approximation. The count is a sum of
    indicators of closed sets, hence upper semi-continuous, so {count >= B/2}
    is itself closed: every piece of the answer begins and ends at one of those
    endpoints (or at an infinity), which is what the reconstruction below uses.

    The intervals within one set are disjoint after merging, so a point lies in
    at most one of them and the coverage count is simply the number of
    intervals containing it. Stacking every interval of every set into two
    arrays turns each count into one vectorised comparison, which is what makes
    the B = 1000 of the case studies tractable.

    Parameters
    ----------
    cs_list : the B confidence sets, each as returned by mom_ar_cs_exact,
              sn_ar_cs or standard_ar_cs (an empty list is the empty set)

    Returns
    -------
    list of closed intervals, merged and in increasing order
    """
    B = len(cs_list)
    if B == 0:
        raise ValueError("aggregate_cs needs at least one confidence set")
    need = B / 2.0
    tol = 1e-9                                   # as in cs_contains

    flat = [iv for cs in cs_list for iv in _merge_intervals(sorted(cs))]
    if not flat:
        return []
    los = np.array([lo for lo, _ in flat])
    his = np.array([hi for _, hi in flat])

    finite = np.concatenate([los[np.isfinite(los)], his[np.isfinite(his)]])
    edges = np.unique(finite)

    # Regions in increasing order as (probe point, lower end, upper end): the
    # open stretch below the first edge, then each edge as a singleton and the
    # open stretch that follows it.
    if edges.size == 0:
        regions = [(0.0, -INF, INF)]             # every set is empty or all of R
    else:
        span = max(float(edges[-1] - edges[0]), 1.0)
        regions = [(float(edges[0]) - span, -INF, float(edges[0]))]
        for i, e in enumerate(edges):
            e = float(e)
            regions.append((e, e, e))
            nxt = float(edges[i + 1]) if i + 1 < edges.size else INF
            probe = 0.5 * (e + nxt) if np.isfinite(nxt) else e + span
            regions.append((probe, e, nxt))

    probes = np.array([r[0] for r in regions])
    counts = np.empty(probes.size, dtype=np.int64)
    step = max(1, int(4e6 // los.size))          # cap the boolean block at ~4M
    for s in range(0, probes.size, step):
        x = probes[s:s + step, None]
        counts[s:s + step] = ((los[None, :] - tol <= x)
                              & (x <= his[None, :] + tol)).sum(axis=1)

    out: list[tuple[float, float]] = []
    run: tuple[float, float] | None = None
    for (_, lo, hi), c in zip(regions, counts):
        if c >= need:
            run = (lo, hi) if run is None else (run[0], hi)
        elif run is not None:
            out.append(run)
            run = None
    if run is not None:
        out.append(run)
    return _merge_intervals(out)


def cs_summary(intervals: list[tuple[float, float]]) -> dict:
    """Length / component summaries of a confidence set."""
    n_comp = len(intervals)
    unbounded = any(np.isinf(lo) or np.isinf(hi) for lo, hi in intervals)
    length = INF if unbounded else float(sum(hi - lo for lo, hi in intervals))
    return {"n_components": n_comp, "unbounded": unbounded, "length": length}


# ----------------------------------------------------------------------------
# Self-normalised AR test (def:sn_stat, prop:sn_pivotal)
# ----------------------------------------------------------------------------

_RK_CACHE_PATH = Path(__file__).resolve().parent / "_rk_cache.json"


def sn_statistic(a: np.ndarray, b: np.ndarray, beta0) -> np.ndarray:
    """T(beta0) = |W_tilde(beta0)| / MAD_j(W_bar_j(beta0)), vectorised over beta0."""
    beta0 = np.asarray(beta0, dtype=float)
    W = a[None, :] - beta0.reshape(-1, 1) * b[None, :]
    med = rank_median(W, axis=1)
    mad = rank_median(np.abs(W - med[:, None]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.abs(med) / mad
    out = np.where(mad > 0, T, INF)
    return out if beta0.ndim else out[0]


def simulate_rk(
    k: int,
    n_sims: int = 1_000_000,
    seed: int = 20260710,
    batch: int = 100_000,
) -> np.ndarray:
    """
    Simulate n_sims draws of R_k = |med(xi_1..xi_k)| / MAD(xi_1..xi_k) with
    xi_j iid N(0,1) (prop:sn_pivotal), using the same rank-based med/MAD as
    the test statistic so critical values and statistic are consistent.
    """
    rng = np.random.default_rng(seed + k)  # distinct stream per k
    out = np.empty(n_sims)
    pos = 0
    while pos < n_sims:
        nb = min(batch, n_sims - pos)
        xi = rng.standard_normal((nb, k))
        med = rank_median(xi, axis=1)
        mad = rank_median(np.abs(xi - med[:, None]), axis=1)
        out[pos : pos + nb] = np.abs(med) / mad
        pos += nb
    return out


def rk_critical_value(k: int, delta: float, n_sims: int = 1_000_000) -> float:
    """
    c_{k,delta}: the (1-delta) quantile of R_k, simulated once per k and cached
    on disk (Code/_rk_cache.json). Feeds both the SN test and the appendix table.
    """
    key = f"k={k}|delta={delta:g}|sims={n_sims}"
    cache: dict[str, float] = {}
    if _RK_CACHE_PATH.exists():
        cache = json.loads(_RK_CACHE_PATH.read_text())
    if key in cache:
        return cache[key]
    draws = simulate_rk(k, n_sims=n_sims)
    val = float(np.quantile(draws, 1.0 - delta))
    cache[key] = val
    _RK_CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return val


def sn_ar_test(
    data: pd.DataFrame,
    beta0,
    delta: float = 0.05,
    k: int | None = None,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
    n_sims: int = 1_000_000,
) -> dict:
    """
    Self-normalised AR test (def:sn_stat): reject when T(beta0) > c_{k,delta},
    the simulated (1-delta) quantile of R_k. Scale-free by prop:scale_invariance;
    k defaults to ceil(8 ln(1/delta)) to match the MoM-AR test.
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    if k is None:
        k = k_blocks(delta)
    if rng is None:
        rng = np.random.default_rng()

    a, b, m = block_means(Y, X, Z, k, rng=rng, shuffle=shuffle)
    c = rk_critical_value(k, delta, n_sims=n_sims)
    T = sn_statistic(a, b, beta0)
    return {"reject": T > c, "T": T, "critical_value": c, "k": k, "m": m}


def sn_ar_cs(
    a: np.ndarray,
    b: np.ndarray,
    c_crit: float,
    span_factor: float = 5.0,
    n_bisect: int = 80,
) -> list[tuple[float, float]]:
    """
    Self-normalised AR confidence set (cor:sn_cs):
        { beta0 : |W_tilde(beta0)| <= c_crit * MAD_j(W_bar_j(beta0)) }

    f(beta) = |med| - c*MAD is continuous piecewise affine. Its breakpoints
    are all contained in the enumerable set

      * pairwise crossings W_i = W_j (breakpoints of the median, thm:piecewise),
      * block ratios a_j / b_j (zeros of the active median line, kinks of |med|),
      * per median-segment "anti-crossings" W_i + W_j = 2 * W_active
        (reorderings of the absolute deviations entering the MAD),

    so f is affine between consecutive anchor points. Evaluating f on the
    anchors plus segment midpoints therefore finds every sign change; each
    boundary is then refined to ~machine precision by bisection.
    Unboundedness is detected from the sign of f far outside the anchor range.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = a.size
    r = (k + 1) // 2

    def f(beta: np.ndarray) -> np.ndarray:
        beta = np.atleast_1d(np.asarray(beta, dtype=float))
        W = a[None, :] - beta[:, None] * b[None, :]
        med = rank_median(W, axis=1)
        mad = rank_median(np.abs(W - med[:, None]), axis=1)
        return np.abs(med) - c_crit * mad

    # --- anchor points ---
    i, j = np.triu_indices(k, 1)
    db = b[j] - b[i]
    nz = db != 0
    crossings = np.unique((a[j][nz] - a[i][nz]) / db[nz])
    anchor_parts = [crossings]
    bnz = b != 0
    anchor_parts.append(a[bnz] / b[bnz])

    # Anti-crossings: on each median segment with active line (aa, bb), the
    # deviation reorderings solve (a_i + a_j - 2 aa) = beta (b_i + b_j - 2 bb).
    seg_edges = np.concatenate(([-INF], crossings, [INF]))
    sum_a = a[i] + a[j]
    sum_b = b[i] + b[j]
    for l in range(len(seg_edges) - 1):
        lo, hi = seg_edges[l], seg_edges[l + 1]
        if np.isinf(lo) and np.isinf(hi):
            beta_star = 0.0
        elif np.isinf(lo):
            beta_star = hi - 1.0
        elif np.isinf(hi):
            beta_star = lo + 1.0
        else:
            if hi - lo <= 0:
                continue
            beta_star = 0.5 * (lo + hi)
        W = a - beta_star * b
        act = int(np.argpartition(W, r - 1)[r - 1])
        num = sum_a - 2.0 * a[act]
        den = sum_b - 2.0 * b[act]
        dnz = den != 0
        roots = num[dnz] / den[dnz]
        inside = roots[(roots > lo) & (roots < hi)]
        if inside.size:
            anchor_parts.append(inside)

    anchors = np.concatenate(anchor_parts) if anchor_parts else np.array([0.0])
    anchors = anchors[np.isfinite(anchors)]
    if anchors.size == 0:
        anchors = np.array([0.0])
    lo0, hi0 = float(np.min(anchors)), float(np.max(anchors))
    span = max(hi0 - lo0, 1.0)
    lo0 -= span_factor * span
    hi0 += span_factor * span

    anchors = np.unique(np.concatenate([anchors, [lo0, hi0]]))
    midpoints = 0.5 * (anchors[:-1] + anchors[1:])
    grid = np.unique(np.concatenate([anchors, midpoints]))
    fg = f(grid)
    inside = fg <= 0.0

    # unboundedness: asymptotic sign of f
    big = abs(lo0) + abs(hi0) + 1e8 * span
    unb_left = bool(f(np.array([-big]))[0] <= 0.0)
    unb_right = bool(f(np.array([big]))[0] <= 0.0)

    def bisect(x_out: float, x_in: float) -> float:
        """Refine boundary between f(x_out) > 0 and f(x_in) <= 0."""
        for _ in range(n_bisect):
            mid = 0.5 * (x_out + x_in)
            if f(np.array([mid]))[0] <= 0.0:
                x_in = mid
            else:
                x_out = mid
            if abs(x_in - x_out) < 1e-13 * (1.0 + abs(mid)):
                break
        return x_in

    intervals: list[tuple[float, float]] = []
    idx = 0
    G = len(grid)
    while idx < G:
        if not inside[idx]:
            idx += 1
            continue
        # start of an inside-run
        start = idx
        while idx + 1 < G and inside[idx + 1]:
            idx += 1
        end = idx
        lo = -INF if (start == 0 and unb_left) else (
            grid[start] if start == 0 else bisect(grid[start - 1], grid[start])
        )
        hi = INF if (end == G - 1 and unb_right) else (
            grid[end] if end == G - 1 else bisect(grid[end + 1], grid[end])
        )
        intervals.append((lo, hi))
        idx += 1

    return _merge_intervals(intervals)


# ----------------------------------------------------------------------------
# Standard (non-robust) Anderson-Rubin test — baseline
# ----------------------------------------------------------------------------


def _ar_moments(Y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> tuple:
    """First/second sample moments of (ZY, ZX) needed by the standard AR test."""
    ZY = Z * Y
    ZX = Z * X
    n = len(Y)
    A = ZY.mean()
    B = ZX.mean()
    Syy = ZY.var(ddof=1)
    Sxx = ZX.var(ddof=1)
    Sxy = np.cov(ZY, ZX, ddof=1)[0, 1]
    return n, A, B, Syy, Sxx, Sxy


def standard_ar_test(data: pd.DataFrame, beta0, delta: float = 0.05) -> dict:
    """
    Standard just-identified Anderson-Rubin test:

        T(beta0) = sqrt(n) * W_bar(beta0) / s(beta0),   W_i = Z_i (Y_i - beta0 X_i),

    with s^2(beta0) the sample variance (ddof=1) of the W_i, rejected when
    T^2 > chi2_{1, 1-delta}. Variance is estimated at the tested beta0, as is
    standard for AR. Vectorised over beta0 via the quadratic representation
    s^2(beta) = Syy - 2 beta Sxy + beta^2 Sxx.
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n, A, B, Syy, Sxx, Sxy = _ar_moments(Y, X, Z)
    beta0 = np.asarray(beta0, dtype=float)
    Wbar = A - beta0 * B
    s2 = Syy - 2.0 * beta0 * Sxy + beta0 ** 2 * Sxx
    c2 = stats.chi2.ppf(1.0 - delta, df=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        T2 = n * Wbar ** 2 / s2
    reject = T2 > c2
    return {"reject": reject if beta0.ndim else bool(reject), "T2": T2, "chi2_crit": c2}


def standard_ar_cs(data: pd.DataFrame, delta: float = 0.05) -> list[tuple[float, float]]:
    """
    Closed-form confidence set of the standard AR test: solve

        n (A - beta B)^2 <= c2 (Syy - 2 beta Sxy + beta^2 Sxx)

    i.e. q(beta) = p2 beta^2 + p1 beta + p0 <= 0 with

        p2 = n B^2 - c2 Sxx,  p1 = -2 (n A B - c2 Sxy),  p0 = n A^2 - c2 Syy.

    Dufour geometry: an interval (p2 > 0), the whole line or the complement of
    an interval (p2 < 0), or a half-line (p2 = 0).
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n, A, B, Syy, Sxx, Sxy = _ar_moments(Y, X, Z)
    c2 = stats.chi2.ppf(1.0 - delta, df=1)

    p2 = n * B ** 2 - c2 * Sxx
    p1 = -2.0 * (n * A * B - c2 * Sxy)
    p0 = n * A ** 2 - c2 * Syy

    scale = max(abs(p2), abs(p1), abs(p0), 1e-300)
    if abs(p2) < 1e-14 * scale:
        # linear: p1 beta + p0 <= 0
        if abs(p1) < 1e-14 * scale:
            return [(-INF, INF)] if p0 <= 0 else []
        root = -p0 / p1
        return [(-INF, root)] if p1 > 0 else [(root, INF)]

    disc = p1 ** 2 - 4.0 * p2 * p0
    if p2 > 0:
        if disc < 0:
            return []
        rt = np.sqrt(disc)
        return [((-p1 - rt) / (2 * p2), (-p1 + rt) / (2 * p2))]
    # p2 < 0: opens downward
    if disc < 0:
        return [(-INF, INF)]
    rt = np.sqrt(disc)
    r1 = (-p1 + rt) / (2 * p2)  # p2 < 0: this is the smaller root
    r2 = (-p1 - rt) / (2 * p2)
    return [(-INF, min(r1, r2)), (max(r1, r2), INF)]


# ----------------------------------------------------------------------------
# Monotonicity diagnostics (prop:mono_det, prop:mono_cheby)
# ----------------------------------------------------------------------------


def mono_cheby_condition(m: int, k: int, delta: float, mu_ZX: float, sigma2_ZX: float) -> bool:
    """Chebyshev single-interval condition m >= k sigma2_ZX / (delta mu_ZX^2) (eq:mono_cheby)."""
    return m >= k * sigma2_ZX / (delta * mu_ZX ** 2)


# ----------------------------------------------------------------------------
# Self-checks (exactness of the closed-form CS against brute force)
# ----------------------------------------------------------------------------


def _brute_force_cs(a, b, tau, grid) -> np.ndarray:
    """Grid membership of |W_tilde(beta)| <= tau, same median convention."""
    W = a[None, :] - grid[:, None] * b[None, :]
    return np.abs(rank_median(W, axis=1)) <= tau


def verify_cs_exactness(n_cases: int = 200, k: int = 24, seed: int = 1) -> None:
    """
    Verify mom_ar_cs_exact against brute-force grid membership on random
    instances (both using the rank-based median). Raises AssertionError on
    any disagreement away from CS boundaries.
    """
    rng = np.random.default_rng(seed)
    for case in range(n_cases):
        a = rng.standard_normal(k)
        b = rng.standard_normal(k) + rng.choice([0.0, 1.0])  # mix sign patterns
        tau = abs(rng.standard_normal()) * 0.5 + 0.05
        cs = mom_ar_cs_exact(a, b, tau)
        grid = np.linspace(-20, 20, 4001)
        member = _brute_force_cs(a, b, tau, grid)
        for x, inside in zip(grid, member):
            claimed = cs_contains(cs, x)
            if claimed != inside:
                # tolerate disagreement only within fp distance of a boundary
                near = any(
                    (np.isfinite(lo) and abs(x - lo) < 1e-6)
                    or (np.isfinite(hi) and abs(x - hi) < 1e-6)
                    for lo, hi in cs
                )
                assert near, (
                    f"case {case}: exact CS and brute force disagree at beta={x}"
                    f" (exact={claimed}, grid={inside}, cs={cs}, tau={tau})"
                )
    print(f"verify_cs_exactness: {n_cases} random instances OK (k={k})")


if __name__ == "__main__":
    verify_cs_exactness()
