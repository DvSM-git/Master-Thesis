"""
Simulation Study: Median-of-Means Two-Stage Least Squares Regression
=====================================================================
Tests the theoretical claims from the MoM-IV theorems.

Key improvements over initial version:
  - Correct theoretical thresholds from the actual theorems
  - Empirical survival function comparison (the core plot)
  - Coverage study using actual confidence intervals
  - Proper RNG handling for reproducibility
  - Unclipped quantile-based comparisons
  - Both Pareto and Student-t DGPs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto, t as t_dist
from joblib import Parallel, delayed
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Core estimators
# ─────────────────────────────────────────────

def mom_mean(x, k):
    """Median-of-Means estimator for the mean of x using k blocks."""
    n = len(x)
    m = n // k
    if m == 0:
        return np.median(x)
    block_means = [x[j * m:(j + 1) * m].mean() for j in range(k)]
    return np.median(block_means)


def rom_estimator(Y, X, Z, k):
    """
    Ratio-of-Medians (RoM) estimator.
    beta_RoM = MoM(ZY) / MoM(ZX)
    """
    ZY = Z * Y
    ZX = Z * X
    denom = mom_mean(ZX, k)
    if abs(denom) < 1e-15:
        return np.nan
    return mom_mean(ZY, k) / denom


def mor_estimator(Y, X, Z, k):
    """
    Median-of-Ratios (MoR) estimator.
    beta_MoR = median over blocks of (block_mean(ZY) / block_mean(ZX))
    """
    n = len(Y)
    m = n // k
    if m == 0:
        return np.nan
    ZY = Z * Y
    ZX = Z * X
    ratios = []
    for j in range(k):
        sl = slice(j * m, (j + 1) * m)
        szx = ZX[sl].mean()
        if abs(szx) > 1e-15:
            ratios.append(ZY[sl].mean() / szx)
    if len(ratios) == 0:
        return np.nan
    return np.median(ratios)


def iv_estimator(Y, X, Z):
    """Standard IV / Wald estimator."""
    denom = Z @ X
    if abs(denom) < 1e-15:
        return np.nan
    return (Z @ Y) / denom


# ─────────────────────────────────────────────
# DGP
# ─────────────────────────────────────────────

def generate_data(n, beta, mu_ZX, sigma_v, error_dist="pareto",
                  alpha_tail=2.1, df_student=3.0, rng=None):
    """
    DGP:  Y = beta*X + eps
          X = mu_ZX * Z + v
          Z ~ {-1, +1} uniform (symmetric, E[Z]=0, E[Z^2]=1)
          v, eps independent of Z

    Parameters
    ----------
    sigma_v : std dev of v (controls Var(ZX) via Var(ZX) = mu_ZX^2 * Var(Z^2) + Var(Zv))
    """
    if rng is None:
        rng = np.random.default_rng()

    Z = rng.choice([-1.0, 1.0], size=n)

    if error_dist == "pareto":
        raw_eps = pareto.rvs(alpha_tail, size=n, random_state=rng)
        eps = raw_eps - alpha_tail / (alpha_tail - 1)
        raw_v = pareto.rvs(alpha_tail, size=n, random_state=rng)
        v = raw_v - alpha_tail / (alpha_tail - 1)
    elif error_dist == "student":
        eps = t_dist.rvs(df_student, size=n, random_state=rng)
        v = t_dist.rvs(df_student, size=n, random_state=rng)
    else:  # Gaussian
        eps = rng.standard_normal(n)
        v = rng.standard_normal(n)

    v *= sigma_v
    X = mu_ZX * Z + v
    Y = beta * X + eps
    return Y, X, Z


def estimate_population_quantities(beta, mu_ZX, sigma_v, error_dist="pareto",
                                    alpha_tail=2.1, df_student=3.0, n_large=500000):
    """Estimate sigma_ZX, sigma_ZY, sigma_Ze from a very large sample."""
    rng_pop = np.random.default_rng(999)
    Y, X, Z = generate_data(n_large, beta, mu_ZX, sigma_v,
                             error_dist=error_dist, alpha_tail=alpha_tail,
                             df_student=df_student, rng=rng_pop)
    eps = Y - beta * X
    sigma_ZX = np.std(Z * X, ddof=1)
    sigma_ZY = np.std(Z * Y, ddof=1)
    sigma_Ze = np.std(Z * eps, ddof=1)
    mu_ZX_emp = np.mean(Z * X)
    return {
        "sigma_ZX": sigma_ZX, "sigma_ZY": sigma_ZY, "sigma_Ze": sigma_Ze,
        "mu_ZX": mu_ZX_emp
    }


# ═══════════════════════════════════════════════
# STUDY 1: Empirical survival function comparison
# ═══════════════════════════════════════════════
# P(|beta_hat - beta| > t) as a function of t
# for IV, RoM, MoR, overlaid with theoretical bounds.

def _run_trial(trial_seed, n, beta, mu_ZX, sigma_v, error_dist, alpha_tail, k_rom, k_mor):
    rng = np.random.default_rng(trial_seed)
    Y, X, Z = generate_data(n, beta, mu_ZX, sigma_v,
                             error_dist=error_dist, alpha_tail=alpha_tail, rng=rng)
    iv_est = iv_estimator(Y, X, Z)
    rom_est = rom_estimator(Y, X, Z, k_rom)
    mor_est = mor_estimator(Y, X, Z, k_mor)
    return iv_est, rom_est, mor_est


def study1_survival_function(beta=2.0, n=2000, mu_ZX=1.0, sigma_v=0.8,
                              error_dist="pareto", alpha_tail=2.1,
                              n_trials=10000, seed=0, n_jobs=-1):
    print("\n" + "=" * 60)
    print("STUDY 1: Empirical survival functions")
    print("=" * 60)

    # Prescribed k for delta ~ 0.05 (but we plot the full survival function)
    k_rom = 16
    k_mor = 16

    # Generate independent seeds for each trial via SeedSequence
    ss = np.random.SeedSequence(seed)
    trial_seeds = [s.generate_state(1)[0] for s in ss.spawn(n_trials)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_trial)(trial_seeds[i], n, beta, mu_ZX, sigma_v,
                            error_dist, alpha_tail, k_rom, k_mor)
        for i in range(n_trials)
    )

    # Collect estimates
    iv_errors = []
    rom_errors = []
    mor_errors = []

    for iv_est, rom_est, mor_est in results:
        if not np.isnan(iv_est):
            iv_errors.append(abs(iv_est - beta))
        if not np.isnan(rom_est):
            rom_errors.append(abs(rom_est - beta))
        if not np.isnan(mor_est):
            mor_errors.append(abs(mor_est - beta))

    iv_errors = np.sort(iv_errors)
    rom_errors = np.sort(rom_errors)
    mor_errors = np.sort(mor_errors)

    # Population quantities for theoretical bounds
    pop = estimate_population_quantities(beta, mu_ZX, sigma_v,
                                          error_dist=error_dist, alpha_tail=alpha_tail)
    sigma_Ze = pop["sigma_Ze"]
    sigma_ZX = pop["sigma_ZX"]
    sigma_ZY = pop["sigma_ZY"]
    mu_zx = pop["mu_ZX"]

    m = n // k_mor

    print(f"  n={n}, k={k_mor}, m={m}")
    print(f"  sigma_Ze={sigma_Ze:.4f}, sigma_ZX={sigma_ZX:.4f}, mu_ZX={mu_zx:.4f}")
    print(f"  IV trials: {len(iv_errors)}, RoM: {len(rom_errors)}, MoR: {len(mor_errors)}")

    # Theoretical bounds as functions of t
    # Standard IV: P(|beta_IV - beta| > t) <= 4*sigma_Ze^2/(n*t^2*mu_ZX^2)
    #   + 4*sigma_ZX^2/(n*mu_ZX^2)
    # MoR: P(|beta_MoR - beta| > t) <= exp(-k/8) when t = 4*sqrt(32)*sigma_Ze/(|mu_ZX|*sqrt(m))

    t_grid = np.linspace(0.01, np.percentile(iv_errors, 99), 500)

    # IV Chebyshev bound: from the two-event decomposition
    # P(error > t) <= 4*sigma_ZX^2/(n*mu_zx^2) + 4*sigma_Ze^2/(n*t^2*mu_zx^2)
    iv_theory = (4 * sigma_ZX**2 / (n * mu_zx**2)
                 + 4 * sigma_Ze**2 / (n * t_grid**2 * mu_zx**2))
    iv_theory = np.clip(iv_theory, 0, 1)

    # MoR bound: for each t, the required k is determined by the Chebyshev step
    # The bound is: P(error > t) <= exp(-k/8)
    # where the threshold t = 4*sqrt(32)*sigma_Ze / (|mu_ZX|*sqrt(m))
    # Inverting: for a given t, the bound holds when the per-block
    # failure prob is <= 1/4, which requires m >= 32*sigma_Ze^2/(t^2*mu_zx^2)
    # and m >= 32*sigma_ZX^2/mu_zx^2
    # The failure probability is e^{-k/8}
    mor_theory = np.full_like(t_grid, np.exp(-k_mor / 8))

    return {
        "iv_errors": iv_errors, "rom_errors": rom_errors, "mor_errors": mor_errors,
        "t_grid": t_grid, "iv_theory": iv_theory, "mor_theory": mor_theory,
        "pop": pop, "k": k_mor, "m": m, "n": n
    }


# ─────────────────────────────────────────────
# Study 2A: Coverage validation
# ─────────────────────────────────────────────
 
def _run_trial_2a(trial_seed, n, beta, mu_ZX, sigma_v, error_dist, alpha_tail):
    rng = np.random.default_rng(trial_seed)
    Y, X, Z = generate_data(n, beta, mu_ZX, sigma_v,
                             error_dist=error_dist, alpha_tail=alpha_tail, rng=rng)
    iv_est = iv_estimator(Y, X, Z)
    return Y, X, Z, iv_est


def study2a_coverage_validation(beta=2.0, n=2000, mu_ZX=1.0, sigma_v=0.8,
                                 error_dist="pareto", alpha_tail=2.1,
                                 n_trials=10000, seed=123, n_jobs=-1):
    """
    At moderate delta, verify empirically that both theorems hold:
    coverage >= 1 - delta at the full theoretical width.
    Also compute c* (fraction of theoretical width actually needed).
    """
    print("\n" + "=" * 60)
    print("STUDY 2A: Coverage validation (moderate delta)")
    print("=" * 60)

    deltas = np.array([0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01])

    pop = estimate_population_quantities(beta, mu_ZX, sigma_v,
                                          error_dist=error_dist, alpha_tail=alpha_tail)
    sigma_Ze = pop["sigma_Ze"]
    sigma_ZX = pop["sigma_ZX"]
    sigma_ZY = pop["sigma_ZY"]
    mu_zx = pop["mu_ZX"]

    print(f"  n={n}, sigma_Ze={sigma_Ze:.4f}, sigma_ZX={sigma_ZX:.4f}, mu_ZX={mu_zx:.4f}")
    print()
    print(f"  {'delta':>6}  {'k':>4}  {'m':>5}  "
          f"{'IV width':>10}  {'MoR width':>10}  {'RoM width':>10}  "
          f"{'IV cover':>10}  {'MoR cover':>10}  {'RoM cover':>10}  "
          f"{'IV c*':>8}  {'MoR c*':>8}  {'RoM c*':>8}  "
          f"{'IV tight':>10}  {'MoR tight':>10}  {'RoM tight':>10}")

    # Generate independent seeds and run all trials in parallel
    ss = np.random.SeedSequence(seed)
    trial_seeds = [s.generate_state(1)[0] for s in ss.spawn(n_trials)]

    trial_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_trial_2a)(trial_seeds[i], n, beta, mu_ZX, sigma_v,
                               error_dist, alpha_tail)
        for i in range(n_trials)
    )

    # Unpack: keep datasets for per-delta MoR, collect IV errors
    all_data = [(Y, X, Z) for Y, X, Z, _ in trial_results]
    iv_errors = np.array([abs(iv_est - beta)
                          for _, _, _, iv_est in trial_results
                          if not np.isnan(iv_est)])
 
    c_grid = np.linspace(0.001, 1.0, 1000)
 
    results = []
    for delta in deltas:
        k_mor = int(np.ceil(8 * np.log(1 / delta)))
        m_mor = n // k_mor
 
        # Theoretical widths
        iv_width = 2 * np.sqrt(2) * sigma_Ze / (abs(mu_zx) * np.sqrt(delta * n))
        mor_width = 4 * np.sqrt(2) * sigma_Ze / (abs(mu_zx) * np.sqrt(m_mor))
 
        # MoR errors for this delta's k
        mor_ests = Parallel(n_jobs=n_jobs)(
            delayed(mor_estimator)(Y, X, Z, k_mor) for Y, X, Z in all_data
        )
        mor_errors = np.array([abs(e - beta) for e in mor_ests if not np.isnan(e)])

        target = 1 - delta

        # RoM errors for this delta's k
        k_rom = int(np.ceil(8 * np.log(2 / delta)))
        m_rom = n // k_rom

        # RoM theoretical width
        rom_denom = abs(mu_zx) * np.sqrt(m_rom) - 2 * sigma_ZX
        if rom_denom > 0:
            rom_width = 2 * (sigma_ZY + abs(beta) * sigma_ZX) / rom_denom
        else:
            rom_width = np.inf

        rom_errors = []
        for Y, X, Z in all_data:
            rom_est = rom_estimator(Y, X, Z, k_rom)
            if not np.isnan(rom_est):
                rom_errors.append(abs(rom_est - beta))
        rom_errors = np.array(rom_errors)

        rom_coverage = np.mean(rom_errors <= rom_width) if rom_width < np.inf else 1.0

        rom_cov_curve = np.array([np.mean(rom_errors <= c * rom_width)
                                   for c in c_grid]) if rom_width < np.inf else np.ones_like(c_grid)

        rom_idx = np.searchsorted(rom_cov_curve, target)
        rom_cstar = c_grid[rom_idx] if rom_idx < len(c_grid) else 1.0

        # Full-width coverage
        iv_coverage = np.mean(iv_errors <= iv_width)
        mor_coverage = np.mean(mor_errors <= mor_width)
 
        # Sweep c to find c*
        iv_cov_curve = np.array([np.mean(iv_errors <= c * iv_width) for c in c_grid])
        mor_cov_curve = np.array([np.mean(mor_errors <= c * mor_width) for c in c_grid])

        # Find smallest c where coverage >= target
        iv_idx = np.searchsorted(iv_cov_curve, target)
        mor_idx = np.searchsorted(mor_cov_curve, target)
        iv_cstar = c_grid[iv_idx] if iv_idx < len(c_grid) else 1.0
        mor_cstar = c_grid[mor_idx] if mor_idx < len(c_grid) else 1.0
 
        results.append({
            "delta": delta, "k_mor": k_mor, "m_mor": m_mor,
            "k_rom": k_rom, "m_rom": m_rom,
            "iv_width": iv_width, "mor_width": mor_width, "rom_width": rom_width,
            "iv_coverage": iv_coverage, "mor_coverage": mor_coverage, "rom_coverage": rom_coverage,
            "iv_cstar": iv_cstar, "mor_cstar": mor_cstar, "rom_cstar": rom_cstar,
            "iv_cov_curve": iv_cov_curve,
            "mor_cov_curve": mor_cov_curve,
            "rom_cov_curve": rom_cov_curve,
            "c_grid": c_grid, "target": target
        })

        rom_width_str = f"{rom_width:>10.4f}" if rom_width < np.inf else f"{'inf':>10}"
        print(f"  {delta:>6.2f}  {k_mor:>4}  {m_mor:>5}  "
              f"{iv_width:>10.4f}  {mor_width:>10.4f}  {rom_width_str}  "
              f"{iv_coverage:>10.4f}  {mor_coverage:>10.4f}  {rom_coverage:>10.4f}  "
              f"{iv_cstar:>8.3f}  {mor_cstar:>8.3f}  {rom_cstar:>8.3f}  "
              f"{1/iv_cstar:>10.1f}x  {1/mor_cstar:>10.1f}x  {1/rom_cstar:>10.1f}x")
 
    return results
 
 
# ─────────────────────────────────────────────
# Study 2B: Width comparison at extreme delta
# ─────────────────────────────────────────────
 
def study2b_width_comparison(beta=2.0, n=2000, mu_ZX=1.0, sigma_v=0.8,
                              error_dist="pareto", alpha_tail=2.1):
    """
    Purely theoretical: compute CI widths for IV and MoR across
    a wide range of delta values, including extreme ones where
    the polynomial vs logarithmic gap is clearly visible.
    No simulation needed — just plug into the formulas.
    """
    print("\n" + "=" * 60)
    print("STUDY 2B: Theoretical width comparison (extreme delta)")
    print("=" * 60)
 
    pop = estimate_population_quantities(beta, mu_ZX, sigma_v,
                                          error_dist=error_dist, alpha_tail=alpha_tail)
    sigma_Ze = pop["sigma_Ze"]
    sigma_ZX = pop["sigma_ZX"]
    mu_zx = pop["mu_ZX"]
 
    print(f"  n={n}, sigma_Ze={sigma_Ze:.4f}, mu_ZX={mu_zx:.4f}")
    print()
 
    # Fine grid for plotting
    deltas_plot = np.logspace(-1, -8, 200)
 
    # Coarse grid for table
    deltas_table = np.array([0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8])
 
    iv_widths_plot = np.zeros_like(deltas_plot)
    mor_widths_plot = np.zeros_like(deltas_plot)
    rom_widths_plot = np.zeros_like(deltas_plot)
 
    for i, delta in enumerate(deltas_plot):
        # IV: t = 2*sqrt(2)*sigma_Ze / (|mu_ZX|*sqrt(delta*n))
        iv_widths_plot[i] = 2 * np.sqrt(2) * sigma_Ze / (abs(mu_zx) * np.sqrt(delta * n))
 
        # MoR: k = ceil(8*ln(1/delta)), m = n/k, t = 4*sqrt(2)*sigma_Ze / (|mu_ZX|*sqrt(m))
        k_mor = int(np.ceil(8 * np.log(1 / delta)))
        m_mor = max(n // k_mor, 1)
        mor_widths_plot[i] = 4 * np.sqrt(2) * sigma_Ze / (abs(mu_zx) * np.sqrt(m_mor))
 
        # RoM: k = ceil(8*ln(2/delta)), m = n/k
        # t = 2*(sigma_ZY + |beta|*sigma_ZX) / (|mu_ZX|*sqrt(m) - 2*sigma_ZX)
        sigma_ZY = pop["sigma_ZY"]
        k_rom = int(np.ceil(8 * np.log(2 / delta)))
        m_rom = max(n // k_rom, 1)
        denom = abs(mu_zx) * np.sqrt(m_rom) - 2 * sigma_ZX
        if denom > 0:
            rom_widths_plot[i] = 2 * (sigma_ZY + abs(beta) * sigma_ZX) / denom
        else:
            rom_widths_plot[i] = np.inf
 
    # Find crossover: where MoR width < IV width
    crossover_idx = np.where(mor_widths_plot < iv_widths_plot)[0]
    if len(crossover_idx) > 0:
        crossover_delta = deltas_plot[crossover_idx[0]]
        print(f"  Crossover (MoR < IV): delta ≈ {crossover_delta:.2e}")
    else:
        crossover_delta = None
        print(f"  Crossover not reached in range")
 
    # Print table
    print()
    print(f"  {'delta':>10}  {'IV width':>12}  {'RoM width':>12}  {'MoR width':>12}  "
          f"{'IV/MoR':>8}  {'k_MoR':>6}  {'m_MoR':>6}")
 
    for delta in deltas_table:
        k_mor = int(np.ceil(8 * np.log(1 / delta)))
        m_mor = max(n // k_mor, 1)
        k_rom = int(np.ceil(8 * np.log(2 / delta)))
        m_rom = max(n // k_rom, 1)
 
        iv_w = 2 * np.sqrt(2) * sigma_Ze / (abs(mu_zx) * np.sqrt(delta * n))
        mor_w = 4 * np.sqrt(2) * sigma_Ze / (abs(mu_zx) * np.sqrt(m_mor))
 
        sigma_ZY = pop["sigma_ZY"]
        denom = abs(mu_zx) * np.sqrt(m_rom) - 2 * sigma_ZX
        rom_w = 2 * (sigma_ZY + abs(beta) * sigma_ZX) / denom if denom > 0 else np.inf
 
        ratio = iv_w / mor_w
 
        print(f"  {delta:>10.0e}  {iv_w:>12.4f}  {rom_w:>12.4f}  {mor_w:>12.4f}  "
              f"{ratio:>8.2f}  {k_mor:>6}  {m_mor:>6}")
 
    results = {
        "deltas_plot": deltas_plot,
        "iv_widths": iv_widths_plot,
        "mor_widths": mor_widths_plot,
        "rom_widths": rom_widths_plot,
        "crossover_delta": crossover_delta,
        "pop": pop
    }
    return results

# ═══════════════════════════════════════════════
# STUDY 3: Consistency and quantile comparison
# ═══════════════════════════════════════════════

def study3_consistency(beta=2.0, mu_ZX=1.0, sigma_v=0.8,
                       n_list=None, k=16, n_trials=3000,
                       error_dist="pareto", alpha_tail=2.1, seed=456):
    if n_list is None:
        n_list = [200, 500, 1000, 2000, 5000]

    print("\n" + "=" * 60)
    print(f"STUDY 3: Consistency and quantiles (dist={error_dist})")
    print("=" * 60)
    print(f"  {'n':>6}  {'IV med':>8}  {'RoM med':>8}  {'MoR med':>8}  "
          f"{'IV q95':>8}  {'RoM q95':>8}  {'MoR q95':>8}  "
          f"{'IV q99':>8}  {'RoM q99':>8}  {'MoR q99':>8}")

    rng = np.random.default_rng(seed)
    records = []

    for n in n_list:
        iv_errs, rom_errs, mor_errs = [], [], []

        for _ in range(n_trials):
            Y, X, Z = generate_data(n, beta, mu_ZX, sigma_v,
                                     error_dist=error_dist, alpha_tail=alpha_tail, rng=rng)
            iv_est = iv_estimator(Y, X, Z)
            rom_est = rom_estimator(Y, X, Z, k)
            mor_est = mor_estimator(Y, X, Z, k)

            if not np.isnan(iv_est):
                iv_errs.append(abs(iv_est - beta))
            if not np.isnan(rom_est):
                rom_errs.append(abs(rom_est - beta))
            if not np.isnan(mor_est):
                mor_errs.append(abs(mor_est - beta))

        iv_errs = np.array(iv_errs)
        rom_errs = np.array(rom_errs)
        mor_errs = np.array(mor_errs)

        rec = {
            "n": n,
            "iv_median": np.median(iv_errs), "rom_median": np.median(rom_errs),
            "mor_median": np.median(mor_errs),
            "iv_q95": np.percentile(iv_errs, 95), "rom_q95": np.percentile(rom_errs, 95),
            "mor_q95": np.percentile(mor_errs, 95),
            "iv_q99": np.percentile(iv_errs, 99), "rom_q99": np.percentile(rom_errs, 99),
            "mor_q99": np.percentile(mor_errs, 99),
        }
        records.append(rec)

        print(f"  {n:>6}  {rec['iv_median']:>8.4f}  {rec['rom_median']:>8.4f}  "
              f"{rec['mor_median']:>8.4f}  "
              f"{rec['iv_q95']:>8.4f}  {rec['rom_q95']:>8.4f}  {rec['mor_q95']:>8.4f}  "
              f"{rec['iv_q99']:>8.4f}  {rec['rom_q99']:>8.4f}  {rec['mor_q99']:>8.4f}")

    # The key comparison: at the 99th percentile, MoR should dominate IV
    print(f"\n  [CHECK] MoR q99 < IV q99 for all n: "
          f"{all(r['mor_q99'] < r['iv_q99'] for r in records)}")
    print(f"  [CHECK] The tail advantage of MoM grows with n: "
          f"ratio IV_q99/MoR_q99 at n={n_list[0]}: "
          f"{records[0]['iv_q99']/records[0]['mor_q99']:.2f}, "
          f"at n={n_list[-1]}: {records[-1]['iv_q99']/records[-1]['mor_q99']:.2f}")
    return records


# ═══════════════════════════════════════════════
# STUDY 4: Instrument strength boundary
# ═══════════════════════════════════════════════

def study4_instrument_strength(beta=2.0, n=3000, k=16,
                                sigma_v=1.0, n_trials=5000,
                                error_dist="pareto", alpha_tail=2.1, seed=789):
    print("\n" + "=" * 60)
    print("STUDY 4: Instrument strength boundary")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    m = n // k

    # Vary mu_ZX to cross the instrument strength thresholds
    # RoM condition: m > 4 * sigma_ZX^2 / mu_ZX^2
    # MoR condition: m >= 32 * sigma_ZX^2 / mu_ZX^2
    # With Z in {-1,1} and v ~ sigma_v, sigma_ZX^2 = Var(ZX) = Var(mu_ZX*Z^2 + Zv) = sigma_v^2
    # (since Z^2=1 always, so mu_ZX*Z^2 = mu_ZX is constant)
    sigma_ZX_approx = sigma_v  # approximate

    rom_critical_mu = np.sqrt(4 * sigma_ZX_approx**2 / m)
    mor_critical_mu = np.sqrt(32 * sigma_ZX_approx**2 / m)

    mu_values = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]) * rom_critical_mu

    print(f"  n={n}, k={k}, m={m}")
    print(f"  Approx sigma_ZX = {sigma_ZX_approx:.4f}")
    print(f"  RoM critical mu_ZX = {rom_critical_mu:.4f}")
    print(f"  MoR critical mu_ZX = {mor_critical_mu:.4f}")
    print()
    print(f"  {'mu_ZX':>8}  {'r':>8}  {'IV MAE':>10}  {'RoM MAE':>10}  "
          f"{'MoR MAE':>10}  {'MoR nan%':>10}")

    results = []
    for mu_ZX in mu_values:
        iv_errs, rom_errs, mor_errs = [], [], []
        mor_nans = 0

        for _ in range(n_trials):
            Y, X, Z = generate_data(n, beta, mu_ZX, sigma_v,
                                     error_dist=error_dist, alpha_tail=alpha_tail, rng=rng)
            iv_est = iv_estimator(Y, X, Z)
            rom_est = rom_estimator(Y, X, Z, k)
            mor_est = mor_estimator(Y, X, Z, k)

            if not np.isnan(iv_est):
                iv_errs.append(abs(iv_est - beta))
            if not np.isnan(rom_est):
                rom_errs.append(abs(rom_est - beta))
            if np.isnan(mor_est):
                mor_nans += 1
            else:
                mor_errs.append(abs(mor_est - beta))

        r = m * mu_ZX**2 / sigma_ZX_approx**2

        rec = {
            "mu_ZX": mu_ZX, "r": r,
            "iv_mae": np.median(iv_errs) if iv_errs else np.inf,
            "rom_mae": np.median(rom_errs) if rom_errs else np.inf,
            "mor_mae": np.median(mor_errs) if mor_errs else np.inf,
            "mor_nan_pct": 100 * mor_nans / n_trials
        }
        results.append(rec)

        print(f"  {mu_ZX:>8.4f}  {r:>8.2f}  {rec['iv_mae']:>10.4f}  "
              f"{rec['rom_mae']:>10.4f}  {rec['mor_mae']:>10.4f}  "
              f"{rec['mor_nan_pct']:>10.1f}%")

    return results, rom_critical_mu, mor_critical_mu


# ═══════════════════════════════════════════════
# STUDY 5: Tail heaviness comparison
# ═══════════════════════════════════════════════

def study5_tail_heaviness(beta=2.0, n=1500, k=16, mu_ZX=1.0, sigma_v=0.8,
                           n_trials=5000, seed=321):
    print("\n" + "=" * 60)
    print("STUDY 5: Performance across distributions")
    print("=" * 60)

    configs = [
        ("Gaussian", "gaussian", {}),
        ("Student t(3)", "student", {"df_student": 3.0}),
        ("Pareto(3.5)", "pareto", {"alpha_tail": 3.5}),
        ("Pareto(2.5)", "pareto", {"alpha_tail": 2.5}),
        ("Pareto(2.1)", "pareto", {"alpha_tail": 2.1}),
    ]

    print(f"  n={n}, k={k}")
    print()
    print(f"  {'Distribution':>14}  {'IV q50':>8}  {'RoM q50':>8}  {'MoR q50':>8}  "
          f"{'IV q95':>8}  {'RoM q95':>8}  {'MoR q95':>8}  "
          f"{'IV q99':>8}  {'RoM q99':>8}  {'MoR q99':>8}")

    results = []
    for label, dist, kwargs in configs:
        rng = np.random.default_rng(seed)
        iv_errs, rom_errs, mor_errs = [], [], []

        for _ in range(n_trials):
            Y, X, Z = generate_data(n, beta, mu_ZX, sigma_v,
                                     error_dist=dist, rng=rng, **kwargs)
            iv_est = iv_estimator(Y, X, Z)
            rom_est = rom_estimator(Y, X, Z, k)
            mor_est = mor_estimator(Y, X, Z, k)

            if not np.isnan(iv_est):
                iv_errs.append(abs(iv_est - beta))
            if not np.isnan(rom_est):
                rom_errs.append(abs(rom_est - beta))
            if not np.isnan(mor_est):
                mor_errs.append(abs(mor_est - beta))

        iv_errs = np.array(iv_errs)
        rom_errs = np.array(rom_errs)
        mor_errs = np.array(mor_errs)

        rec = {
            "label": label,
            "iv_q50": np.median(iv_errs), "rom_q50": np.median(rom_errs),
            "mor_q50": np.median(mor_errs),
            "iv_q95": np.percentile(iv_errs, 95), "rom_q95": np.percentile(rom_errs, 95),
            "mor_q95": np.percentile(mor_errs, 95),
            "iv_q99": np.percentile(iv_errs, 99), "rom_q99": np.percentile(rom_errs, 99),
            "mor_q99": np.percentile(mor_errs, 99),
        }
        results.append(rec)

        print(f"  {label:>14}  {rec['iv_q50']:>8.4f}  {rec['rom_q50']:>8.4f}  "
              f"{rec['mor_q50']:>8.4f}  "
              f"{rec['iv_q95']:>8.4f}  {rec['rom_q95']:>8.4f}  {rec['mor_q95']:>8.4f}  "
              f"{rec['iv_q99']:>8.4f}  {rec['rom_q99']:>8.4f}  {rec['mor_q99']:>8.4f}")

    # Key check: MoM advantage grows with tail heaviness
    iv_ratios = [r['iv_q99'] / r['mor_q99'] for r in results]
    print(f"\n  [CHECK] IV_q99/MoR_q99 ratio across distributions: "
          f"{', '.join(f'{r:.2f}' for r in iv_ratios)}")
    print(f"  [CHECK] Ratio increases with tail heaviness: "
          f"{iv_ratios[-1] > iv_ratios[0]}")
    return results


# ═══════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════

def make_plots(res1, res2a, res2b, res3, res4, res5, output_path="simulation_results.png"):
    out_dir = os.path.dirname(output_path)
    base = os.path.splitext(os.path.basename(output_path))[0]

    def save(fig, suffix):
        path = os.path.join(out_dir, f"{base}_{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {path}")
        plt.close(fig)

    # ── Plot 1: Survival function ──────────────
    fig, ax1 = plt.subplots(figsize=(6, 5))
    for errors, label, color, ls in [
        (res1["iv_errors"],  "Standard IV", "red", "--"),
        (res1["rom_errors"], "RoM",         "blue", "-"),
        (res1["mor_errors"], "MoR",         "green", "-"),
    ]:
        sorted_e = np.sort(errors)
        survival = 1 - np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax1.plot(sorted_e, survival, color=color, linestyle=ls, label=label, linewidth=1.2)
    ax1.set_xlabel(r"$t = |\hat{\beta} - \beta|$")
    ax1.set_ylabel(r"$P(|\hat{\beta} - \beta| > t)$")
    ax1.set_title("Empirical Survival Functions\n(log-log scale)")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylim(1e-4, 1)
    ax1.set_xlim(1e-3, 10)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    save(fig, "1_survival")

    # ── Plot 2a: Coverage study ────────────────────────────────
    fig, ax2a = plt.subplots(figsize=(6, 5))
    idx = next(i for i, r in enumerate(res2a) if abs(r["delta"] - 0.05) < 0.001)
    r = res2a[idx]
    delta = r["delta"]
    c_grid = r["c_grid"]
    ax2a.plot(c_grid, r["iv_cov_curve"], "r-", linewidth=1.5, label="IV")
    ax2a.plot(c_grid, r["mor_cov_curve"], "g-", linewidth=1.5, label="MoR")
    ax2a.plot(c_grid, r["rom_cov_curve"], "b-", linewidth=1.5, label="RoM")
    ax2a.axhline(1 - delta, color="k", linestyle="--", linewidth=1,
                label=f"Target {1-delta:.2f}")
    ax2a.axvline(r["iv_cstar"], color="r", linestyle=":", alpha=0.6)
    ax2a.axvline(r["mor_cstar"], color="g", linestyle=":", alpha=0.6)
    ax2a.axvline(r["rom_cstar"], color="b", linestyle=":", alpha=0.6)
    ax2a.annotate(f'IV c*={r["iv_cstar"]:.2f}\n({1/r["iv_cstar"]:.1f}x conserv.)',
                 xy=(r["iv_cstar"], 1 - delta), fontsize=7,
                 xytext=(r["iv_cstar"] + 0.08, 1 - delta - 0.10),
                 arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax2a.annotate(f'MoR c*={r["mor_cstar"]:.2f}\n({1/r["mor_cstar"]:.1f}x conserv.)',
                 xy=(r["mor_cstar"], 1 - delta), fontsize=7,
                 xytext=(r["mor_cstar"] + 0.08, 1 - delta - 0.20),
                 arrowprops=dict(arrowstyle="->", color="green"), color="green")
    ax2a.annotate(f'RoM c*={r["rom_cstar"]:.2f}\n({1/r["rom_cstar"]:.1f}x conserv.)',
                 xy=(r["rom_cstar"], 1 - delta), fontsize=7,
                 xytext=(r["rom_cstar"] + 0.08, 1 - delta - 0.30),
                 arrowprops=dict(arrowstyle="->", color="blue"), color="blue")
    ax2a.set_xlabel("Fraction c of theoretical width")
    ax2a.set_ylabel("Coverage probability")
    ax2a.set_title(f"Bound Tightness (δ = {delta})\nHow much of theoretical width is needed?")
    ax2a.legend(fontsize=7, loc="lower right")
    ax2a.set_xlim(0, 1.05)
    ax2a.set_ylim(0.5, 1.02)
    ax2a.grid(True, alpha=0.3)
    save(fig, "2a_coverage")

    # ── Plot 2b: CI width comparison ──────────────────────────
    fig, ax2b = plt.subplots(figsize=(6, 5))
    deltas_a = [r["delta"] for r in res2a]
    iv_cstars = [r["iv_cstar"] for r in res2a]
    mor_cstars = [r["mor_cstar"] for r in res2a]
    rom_cstars = [r["rom_cstar"] for r in res2a]
    ax2b.plot(deltas_a, [1/c for c in iv_cstars], "r-s", label="IV conservativeness",
             markersize=5)
    ax2b.plot(deltas_a, [1/c for c in mor_cstars], "g-^", label="MoR conservativeness",
             markersize=5)
    ax2b.plot(deltas_a, [1/c for c in rom_cstars], "b-o", label="RoM conservativeness",
             markersize=5)
    ax2b.set_xlabel("δ")
    ax2b.set_ylabel("Conservativeness (1/c*)\n(how many × too wide)")
    ax2b.set_title("Bound Conservativeness vs δ\n(higher = more conservative)")
    ax2b.legend(fontsize=7)
    ax2b.grid(True, alpha=0.3)
    save(fig, "2b_conservativeness")

    # ── Plot 3: Consistency quantiles ─────────────────────────
    fig, ax3 = plt.subplots(figsize=(6, 5))
    ns = [r["n"] for r in res3]
    ax3.plot(ns, [r["iv_q99"] for r in res3], "r--s", label="IV q99", alpha=0.7)
    ax3.plot(ns, [r["rom_q99"] for r in res3], "b-o", label="RoM q99")
    ax3.plot(ns, [r["mor_q99"] for r in res3], "g-^", label="MoR q99")
    ax3.plot(ns, [r["iv_median"] for r in res3], "r--s", label="IV median", alpha=0.3, markersize=3)
    ax3.plot(ns, [r["mor_median"] for r in res3], "g-^", label="MoR median", alpha=0.3, markersize=3)
    ax3.set_xlabel("n")
    ax3.set_ylabel(r"$|\hat{\beta} - \beta|$")
    ax3.set_title("Consistency: Error Quantiles vs n\n(99th percentile and median)")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.3)
    save(fig, "3_consistency")

    # ── Plot 4: Instrument strength ───────────────────────────
    fig, ax4 = plt.subplots(figsize=(6, 5))
    res4_data, rom_crit, mor_crit = res4
    rs = [r["r"] for r in res4_data]
    ax4.plot(rs, [r["iv_mae"] for r in res4_data], "r--s", label="IV", alpha=0.7)
    ax4.plot(rs, [r["rom_mae"] for r in res4_data], "b-o", label="RoM")
    ax4.plot(rs, [r["mor_mae"] for r in res4_data], "g-^", label="MoR")
    ax4.axvline(4, color="blue", linestyle=":", label="RoM threshold (r=4)")
    ax4.axvline(32, color="green", linestyle=":", label="MoR threshold (r=32)")
    ax4.set_xlabel(r"$r = m \cdot \mu_{ZX}^2 / \sigma_{ZX}^2$")
    ax4.set_ylabel("Median absolute error")
    ax4.set_title("Instrument Strength\n(performance vs strength ratio)")
    ax4.set_xscale("log")
    ax4.legend(fontsize=6)
    ax4.grid(True, alpha=0.3)
    save(fig, "4_instrument_strength")

    # ── Plot 5: Tail heaviness ────────────────────────────────
    fig, ax5 = plt.subplots(figsize=(6, 5))
    labels = [r["label"] for r in res5]
    x_pos = range(len(labels))
    width = 0.25
    ax5.bar([x - width for x in x_pos], [r["iv_q99"] for r in res5],
            width, label="IV q99", color="red", alpha=0.7)
    ax5.bar(x_pos, [r["rom_q99"] for r in res5],
            width, label="RoM q99", color="blue", alpha=0.7)
    ax5.bar([x + width for x in x_pos], [r["mor_q99"] for r in res5],
            width, label="MoR q99", color="green", alpha=0.7)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(labels, rotation=25, fontsize=7)
    ax5.set_ylabel("99th percentile of |error|")
    ax5.set_title("Tail Heaviness Comparison\n(99th percentile across distributions)")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3, axis="y")
    save(fig, "5_tail_heaviness")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("MoM-2SLS Simulation Study")
    print("=" * 60)

    res1  = study1_survival_function()
    res2a = study2a_coverage_validation()
    res2b = study2b_width_comparison()
    res3  = study3_consistency()
    res4  = study4_instrument_strength()
    res5  = study5_tail_heaviness()

    print("\n" + "=" * 60)
    print("ALL STUDIES COMPLETE — generating figures...")

    make_plots(res1, res2a, res2b, res3, res4, res5,
               output_path="Paper\images\graphs\sim.png")