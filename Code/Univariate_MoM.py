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
import matplotlib.gridspec as gridspec
from scipy.stats import pareto, t as t_dist
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
        eps = raw_eps - alpha_tail / (alpha_tail - 1)  # center at zero
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
# This is the CORE plot: P(|beta_hat - beta| > t) as a function of t
# for IV, RoM, MoR, overlaid with theoretical bounds.

def study1_survival_function(beta=2.0, n=2000, mu_ZX=1.0, sigma_v=0.8,
                              error_dist="pareto", alpha_tail=2.1,
                              n_trials=10000, seed=42):
    print("\n" + "=" * 60)
    print("STUDY 1: Empirical survival functions")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    # Prescribed k for delta ~ 0.05 (but we plot the full survival function)
    k_rom = 16
    k_mor = 16

    # Collect estimates
    iv_errors = []
    rom_errors = []
    mor_errors = []

    for _ in range(n_trials):
        Y, X, Z = generate_data(n, beta, mu_ZX, sigma_v,
                                 error_dist=error_dist, alpha_tail=alpha_tail, rng=rng)
        iv_est = iv_estimator(Y, X, Z)
        rom_est = rom_estimator(Y, X, Z, k_rom)
        mor_est = mor_estimator(Y, X, Z, k_mor)

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
    # Standard IV (Chebyshev): P(|beta_IV - beta| > t) <= 4*sigma_Ze^2/(n*t^2*mu_ZX^2)
    #   + 4*sigma_ZX^2/(n*mu_ZX^2)   [denominator event, constant]
    # MoR: P(|beta_MoR - beta| > t) <= exp(-k/8) when t = 4*sqrt(2)*sigma_Ze/(|mu_ZX|*sqrt(m))

    t_grid = np.linspace(0.01, np.percentile(iv_errors, 99), 500)

    # IV Chebyshev bound: from the two-event decomposition
    # P(error > t) <= 4*sigma_ZX^2/(n*mu_zx^2) + 4*sigma_Ze^2/(n*t^2*mu_zx^2)
    iv_theory = (4 * sigma_ZX**2 / (n * mu_zx**2)
                 + 4 * sigma_Ze**2 / (n * t_grid**2 * mu_zx**2))
    iv_theory = np.clip(iv_theory, 0, 1)

    # MoR bound: for each t, the required k is determined by the Chebyshev step
    # The bound is: P(error > t) <= exp(-k/8)
    # where the threshold t = 4*sqrt(2)*sigma_Ze / (|mu_ZX|*sqrt(m))
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


# ═══════════════════════════════════════════════
# STUDY 2: Coverage of theoretical confidence intervals
# ═══════════════════════════════════════════════
# For each delta, compute the theoretical CI width and check empirical coverage.

def study2_coverage(beta=2.0, n=2000, mu_ZX=1.0, sigma_v=0.8,
                    error_dist="pareto", alpha_tail=2.1,
                    n_trials=8000, seed=123):
    print("\n" + "=" * 60)
    print("STUDY 2: Coverage of theoretical confidence intervals")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    deltas = np.array([0.30, 0.20, 0.10, 0.05, 0.02, 0.01])

    pop = estimate_population_quantities(beta, mu_ZX, sigma_v,
                                          error_dist=error_dist, alpha_tail=alpha_tail)
    sigma_Ze = pop["sigma_Ze"]
    sigma_ZX = pop["sigma_ZX"]
    sigma_ZY = pop["sigma_ZY"]
    mu_zx = pop["mu_ZX"]

    print(f"  n={n}, sigma_Ze={sigma_Ze:.4f}, sigma_ZX={sigma_ZX:.4f}, mu_ZX={mu_zx:.4f}")
    print()
    print(f"  {'delta':>6}  {'k_MoR':>6}  {'m':>5}  "
          f"{'IV width':>10}  {'MoR width':>10}  "
          f"{'IV cover':>10}  {'MoR cover':>10}  "
          f"{'IV ok':>6}  {'MoR ok':>6}")

    results = []
    for delta in deltas:
        # MoR: k = ceil(8*ln(1/delta)), threshold = 4*sqrt(2)*sigma_Ze/(|mu_ZX|*sqrt(m))
        k_mor = int(np.ceil(8 * np.log(1 / delta)))
        m_mor = n // k_mor

        # MoR theoretical width (from Theorem 2)
        mor_width = 4 * np.sqrt(2) * sigma_Ze / (abs(mu_zx) * np.sqrt(m_mor))

        # Standard IV theoretical width (from Theorem in standard_iv.tex, equal split)
        # t = 2*sqrt(2)*sigma_Ze / (|mu_ZX|*sqrt(delta*n))
        iv_width = 2 * np.sqrt(2) * sigma_Ze / (abs(mu_zx) * np.sqrt(delta * n))

        # Simulate
        iv_covered = 0
        mor_covered = 0
        iv_valid = 0
        mor_valid = 0

        for _ in range(n_trials):
            Y, X, Z = generate_data(n, beta, mu_ZX, sigma_v,
                                     error_dist=error_dist, alpha_tail=alpha_tail, rng=rng)

            iv_est = iv_estimator(Y, X, Z)
            mor_est = mor_estimator(Y, X, Z, k_mor)

            if not np.isnan(iv_est):
                iv_valid += 1
                if abs(iv_est - beta) <= iv_width:
                    iv_covered += 1

            if not np.isnan(mor_est):
                mor_valid += 1
                if abs(mor_est - beta) <= mor_width:
                    mor_covered += 1

        iv_coverage = iv_covered / max(iv_valid, 1)
        mor_coverage = mor_covered / max(mor_valid, 1)

        # Coverage should be >= 1 - delta
        target = 1 - delta
        iv_ok = iv_coverage >= target - 0.02   # 2% tolerance for MC noise
        mor_ok = mor_coverage >= target - 0.02

        results.append({
            "delta": delta, "k_mor": k_mor, "m_mor": m_mor,
            "iv_width": iv_width, "mor_width": mor_width,
            "iv_coverage": iv_coverage, "mor_coverage": mor_coverage,
            "target": target
        })

        print(f"  {delta:>6.2f}  {k_mor:>6}  {m_mor:>5}  "
              f"{iv_width:>10.4f}  {mor_width:>10.4f}  "
              f"{iv_coverage:>10.4f}  {mor_coverage:>10.4f}  "
              f"{str(iv_ok):>6}  {str(mor_ok):>6}")

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

def make_plots(res1, res2, res3, res4, res5, output_path="simulation_results.png"):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── Plot 1: Survival function (THE key plot) ──────────────
    ax1 = fig.add_subplot(gs[0, 0])

    for errors, label, color, ls in [
        (res1["iv_errors"],  "Standard IV", "red", "--"),
        (res1["rom_errors"], "RoM",         "blue", "-"),
        (res1["mor_errors"], "MoR",         "green", "-"),
    ]:
        sorted_e = np.sort(errors)
        survival = 1 - np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax1.plot(sorted_e, survival, color=color, linestyle=ls, label=label, linewidth=1.2)

    # Overlay IV Chebyshev bound
    ax1.plot(res1["t_grid"], res1["iv_theory"], "r:", alpha=0.7,
             label="IV Chebyshev bound", linewidth=1.5)

    ax1.set_xlabel(r"$t = |\hat{\beta} - \beta|$")
    ax1.set_ylabel(r"$P(|\hat{\beta} - \beta| > t)$")
    ax1.set_title("Empirical Survival Functions\n(log-log scale)")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylim(1e-4, 1)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Coverage study ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    deltas = [r["delta"] for r in res2]
    targets = [r["target"] for r in res2]
    iv_cov = [r["iv_coverage"] for r in res2]
    mor_cov = [r["mor_coverage"] for r in res2]

    ax2.plot(deltas, targets, "k--", label="Target (1-δ)", linewidth=1.5)
    ax2.plot(deltas, iv_cov, "r-s", label="IV coverage", markersize=5)
    ax2.plot(deltas, mor_cov, "g-^", label="MoR coverage", markersize=5)
    ax2.set_xlabel("δ")
    ax2.set_ylabel("Coverage probability")
    ax2.set_title("Coverage of Theoretical CIs\n(should be above dashed line)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Plot 2b: CI width comparison ──────────────────────────
    ax2b = fig.add_subplot(gs[0, 2])
    iv_widths = [r["iv_width"] for r in res2]
    mor_widths = [r["mor_width"] for r in res2]
    ax2b.plot(deltas, iv_widths, "r-s", label="IV width ~ 1/√δ")
    ax2b.plot(deltas, mor_widths, "g-^", label="MoR width ~ √ln(1/δ)")
    ax2b.set_xlabel("δ")
    ax2b.set_ylabel("CI half-width")
    ax2b.set_title("Confidence Interval Widths\n(polynomial vs logarithmic)")
    ax2b.set_yscale("log")
    ax2b.legend(fontsize=7)
    ax2b.grid(True, alpha=0.3)

    # ── Plot 3: Consistency quantiles ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
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

    # ── Plot 4: Instrument strength ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
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

    # ── Plot 5: Tail heaviness ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
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

    fig.suptitle("MoM-2SLS Simulation Study: Verifying Theoretical Claims",
                 fontsize=13, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("MoM-2SLS Simulation Study")
    print("=" * 60)

    res1 = study1_survival_function()
    res2 = study2_coverage()
    res3 = study3_consistency()
    res4 = study4_instrument_strength()
    res5 = study5_tail_heaviness()

    print("\n" + "=" * 60)
    print("ALL STUDIES COMPLETE — generating figures...")

    make_plots(res1, res2, res3, res4, res5,
               output_path="simulation_results.png")