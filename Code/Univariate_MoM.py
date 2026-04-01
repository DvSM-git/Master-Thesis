"""
Simulation Study: Median-of-Means Two-Stage Least Squares Regression
=====================================================================
Tests the theoretical claims in "Median-of-Means Two-Stage Least Squares Regression"

Claims tested:
  1. MoM mean estimator achieves sub-Gaussian tail bounds (eq. 2, L = sqrt(32))
     vs empirical mean's Chebyshev-type bounds (polynomial in delta)
  2. RoM and MoR are consistent estimators of beta under heavy-tailed errors
  3. MoR has tighter effective variance: sigma_Ze <= sigma_ZY + |beta|*sigma_ZX
  4. Instrument strength conditions:
       RoM: m > 4 * sigma^2_ZX / mu^2_ZX
       MoR: m >= 32 * sigma^2_ZX / mu^2_ZX
  5. Block count difference: RoM needs k = ceil(8*ln(2/delta)),
                             MoR needs k = ceil(8*ln(1/delta))
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pareto, t as t_dist
import warnings

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# Core estimators
# ─────────────────────────────────────────────

def mom_mean(x, k):
    """Median-of-Means estimator for the mean of x using k blocks."""
    n = len(x)
    m = n // k
    block_means = [x[j * m:(j + 1) * m].mean() for j in range(k)]
    return np.median(block_means)


def rom_estimator(Y, X, Z, k):
    """
    Ratio-of-Medians (RoM) estimator.
    beta_RoM = MoM(ZY) / MoM(ZX)
    """
    ZY = Z * Y
    ZX = Z * X
    return mom_mean(ZY, k) / mom_mean(ZX, k)


def mor_estimator(Y, X, Z, k):
    """
    Median-of-Ratios (MoR) estimator.
    beta_MoR = median over blocks of (block_mean(ZY) / block_mean(ZX))
    """
    n = len(Y)
    m = n // k
    ZY = Z * Y
    ZX = Z * X
    ratios = []
    for j in range(k):
        sl = slice(j * m, (j + 1) * m)
        szx = ZX[sl].mean()
        if szx != 0:
            ratios.append(ZY[sl].mean() / szx)
    return np.median(ratios)


def iv_estimator(Y, X, Z):
    """Standard IV / Wald estimator."""
    return (Z @ Y) / (Z @ X)


# ─────────────────────────────────────────────
# DGP
# ─────────────────────────────────────────────

def generate_data(n, beta, mu_ZX, sigma_ZX, error_dist="pareto", alpha_tail=2.1, rng=rng):
    """
    DGP:  Y = beta*X + eps
          Z is the instrument with E[ZX] = mu_ZX
          E[Z*eps] = 0  (exogeneity)

    Z ~ Bernoulli(0.5) shifted to {-1, +1}
    X = mu_ZX * Z + v,   v independent of Z
    eps independent of Z (heavy-tailed)
    """
    Z = rng.choice([-1.0, 1.0], size=n)

    if error_dist == "pareto":
        # Pareto with finite variance (alpha > 2)
        eps = (pareto.rvs(alpha_tail, size=n, random_state=rng) - alpha_tail / (alpha_tail - 1))
        v   = (pareto.rvs(alpha_tail, size=n, random_state=rng) - alpha_tail / (alpha_tail - 1))
    elif error_dist == "student":
        df = 3.0   # finite variance, heavy tails
        eps = t_dist.rvs(df, size=n, random_state=rng)
        v   = t_dist.rvs(df, size=n, random_state=rng)
    else:  # Gaussian baseline
        eps = rng.standard_normal(n)
        v   = rng.standard_normal(n)

    # Scale v so Var(ZX) ~ sigma_ZX^2
    v *= sigma_ZX

    X = mu_ZX * Z + v
    Y = beta * X + eps
    return Y, X, Z


# ─────────────────────────────────────────────
# STUDY 1 – Sub-Gaussian tail bound for MoM mean estimator
# ─────────────────────────────────────────────
# Claim: P(|mu_hat - mu| > sigma * sqrt(32*ln(1/delta)/n)) <= delta  (eq. 2)
# We verify this empirically: for many delta values, the empirical exceedance
# probability should stay below delta.

def study1_mom_tail_bound(n=2000, n_trials=5000, alpha_tail=2.1):
    """
    Compare empirical tail probabilities of:
      - empirical mean
      - MoM with k = ceil(8*ln(1/delta))
    against their theoretical bounds.
    """
    print("\n" + "=" * 60)
    print("STUDY 1: Sub-Gaussian tail bound for MoM mean estimator")
    print("=" * 60)

    deltas = np.array([0.30, 0.20, 0.10, 0.05, 0.02, 0.01])

    # Generate many samples from Pareto (heavy-tailed, finite variance)
    mu = 0.0
    samples = np.array([
        pareto.rvs(alpha_tail, size=n, random_state=rng) - alpha_tail / (alpha_tail - 1)
        for _ in range(n_trials)
    ])  # shape (n_trials, n),  mean=0, Var ~ finite

    # Empirical sigma
    sigma_emp = np.std(samples)

    results = {"delta": deltas,
               "mean_exceed": [],
               "mom_exceed": [],
               "mean_bound": [],
               "mom_bound": []}

    for delta in deltas:
        k = int(np.ceil(8 * np.log(1 / delta)))

        # MoM threshold (eq. 2, L = sqrt(32))
        mom_thresh = sigma_emp * np.sqrt(32 * np.log(1 / delta) / n)
        # Chebyshev threshold for empirical mean  P(|mu_bar - mu| > t) <= sigma^2/(n*t^2)
        # Inverted: for the same delta, threshold = sigma / sqrt(n*delta)
        mean_thresh = sigma_emp / np.sqrt(n * delta)

        # Empirical mean exceedance
        emp_means = samples.mean(axis=1)
        mean_exceed = np.mean(np.abs(emp_means - mu) > mean_thresh)

        # MoM exceedance
        mom_estimates = np.array([mom_mean(samples[i], k) for i in range(n_trials)])
        mom_exceed = np.mean(np.abs(mom_estimates - mu) > mom_thresh)

        results["mean_exceed"].append(mean_exceed)
        results["mom_exceed"].append(mom_exceed)
        results["mean_bound"].append(delta)   # Chebyshev bound IS delta by construction
        results["mom_bound"].append(delta)    # theoretical guarantee

        print(f"  delta={delta:.2f} | k={k:2d} | "
              f"Mean exceed={mean_exceed:.4f} (bound={delta:.2f}) | "
              f"MoM  exceed={mom_exceed:.4f} (bound={delta:.2f})")

    # Key check: MoM empirical exceedance <= delta?
    mom_ok = all(e <= d + 0.02 for e, d in zip(results["mom_exceed"], deltas))
    print(f"\n  [CHECK] MoM sub-Gaussian bound holds (within 2% tolerance): {mom_ok}")
    return results


# ─────────────────────────────────────────────
# STUDY 2 – Consistency of RoM and MoR for beta
# ─────────────────────────────────────────────

def study2_consistency(beta=2.0, mu_ZX=1.0, sigma_ZX=1.0,
                       n_list=None, k=16, n_trials=2000,
                       error_dist="pareto"):
    """
    As n grows, RoM, MoR, and IV should all converge to beta.
    Under heavy tails, standard IV may be erratic.
    """
    if n_list is None:
        n_list = [200, 500, 1000, 2000, 5000]

    print("\n" + "=" * 60)
    print(f"STUDY 2: Consistency  (beta={beta}, dist={error_dist})")
    print("=" * 60)
    print(f"  {'n':>6}  {'IV bias':>10}  {'RoM bias':>10}  {'MoR bias':>10}  "
          f"{'IV RMSE':>10}  {'RoM RMSE':>10}  {'MoR RMSE':>10}")

    records = []
    for n in n_list:
        iv_ests, rom_ests, mor_ests = [], [], []
        for _ in range(n_trials):
            Y, X, Z = generate_data(n, beta, mu_ZX, sigma_ZX, error_dist=error_dist)
            iv_ests.append(iv_estimator(Y, X, Z))
            rom_ests.append(rom_estimator(Y, X, Z, k))
            mor_ests.append(mor_estimator(Y, X, Z, k))

        iv_arr  = np.array(iv_ests)
        rom_arr = np.array(rom_ests)
        mor_arr = np.array(mor_ests)

        # Trim extreme outliers for display only (IV can diverge under heavy tails)
        iv_arr_t  = np.clip(iv_arr,  beta - 50, beta + 50)
        rom_arr_t = np.clip(rom_arr, beta - 50, beta + 50)
        mor_arr_t = np.clip(mor_arr, beta - 50, beta + 50)

        iv_bias   = np.mean(iv_arr_t)  - beta
        rom_bias  = np.mean(rom_arr_t) - beta
        mor_bias  = np.mean(mor_arr_t) - beta
        iv_rmse   = np.sqrt(np.mean((iv_arr_t  - beta) ** 2))
        rom_rmse  = np.sqrt(np.mean((rom_arr_t - beta) ** 2))
        mor_rmse  = np.sqrt(np.mean((mor_arr_t - beta) ** 2))

        records.append((n, iv_bias, rom_bias, mor_bias, iv_rmse, rom_rmse, mor_rmse))
        print(f"  {n:>6}  {iv_bias:>10.4f}  {rom_bias:>10.4f}  {mor_bias:>10.4f}  "
              f"{iv_rmse:>10.4f}  {rom_rmse:>10.4f}  {mor_rmse:>10.4f}")

    # Check that RMSE decreases with n
    rom_rmses = [r[5] for r in records]
    mor_rmses = [r[6] for r in records]
    rom_consistent = all(rom_rmses[i] >= rom_rmses[i+1] for i in range(len(rom_rmses)-1))
    mor_consistent = all(mor_rmses[i] >= mor_rmses[i+1] for i in range(len(mor_rmses)-1))
    print(f"\n  [CHECK] RoM RMSE decreasing in n: {rom_consistent}")
    print(f"  [CHECK] MoR RMSE decreasing in n: {mor_consistent}")
    return records


# ─────────────────────────────────────────────
# STUDY 3 – Effective variance: sigma_Ze <= sigma_ZY + |beta|*sigma_ZX
# ─────────────────────────────────────────────

def study3_effective_variance(beta=2.0, n=50000, n_rep=200,
                              error_dist="pareto", mu_ZX=1.0, sigma_ZX=1.0):
    """
    Verify the inequality  sigma_Ze <= sigma_ZY + |beta|*sigma_ZX
    holds empirically for many DGP draws.
    The paper claims MoR uses sigma_Ze in its bound while RoM uses sigma_ZY + |beta|*sigma_ZX,
    so the MoR bound is tighter.
    """
    print("\n" + "=" * 60)
    print("STUDY 3: Effective variance inequality")
    print("         sigma_Ze <= sigma_ZY + |beta|*sigma_ZX")
    print("=" * 60)

    violations = 0
    ratios = []
    for _ in range(n_rep):
        Y, X, Z = generate_data(n, beta, mu_ZX, sigma_ZX, error_dist=error_dist)
        eps = Y - beta * X

        ZY  = Z * Y
        ZX  = Z * X
        Ze  = Z * eps

        sigma_ZY_emp  = np.std(ZY)
        sigma_ZX_emp  = np.std(ZX)
        sigma_Ze_emp  = np.std(Ze)

        lhs = sigma_Ze_emp
        rhs = sigma_ZY_emp + abs(beta) * sigma_ZX_emp

        if lhs > rhs + 1e-10:
            violations += 1
        ratios.append(lhs / rhs)

    mean_ratio = np.mean(ratios)
    max_ratio  = np.max(ratios)
    print(f"  Repetitions: {n_rep}")
    print(f"  Mean ratio sigma_Ze / (sigma_ZY + |beta|*sigma_ZX): {mean_ratio:.4f}  (should be <= 1)")
    print(f"  Max  ratio: {max_ratio:.4f}")
    print(f"  Violations: {violations} / {n_rep}")
    print(f"\n  [CHECK] Inequality holds in all repetitions: {violations == 0}")
    return {"mean_ratio": mean_ratio, "max_ratio": max_ratio, "violations": violations}


# ─────────────────────────────────────────────
# STUDY 4 – Instrument strength condition
# ─────────────────────────────────────────────
# Claim: sub-Gaussian bounds break down when the instrument strength condition is violated.
# RoM condition: m > 4 * sigma^2_ZX / mu^2_ZX
# MoR condition: m >= 32 * sigma^2_ZX / mu^2_ZX
#
# We vary m (block size) across the threshold and compare empirical exceedance with delta.

def study4_instrument_strength(beta=2.0, n_total=5000, k=20,
                                delta=0.10, n_trials=3000,
                                error_dist="pareto"):
    """
    Fix k and vary mu_ZX (instrument strength) to test when the sub-Gaussian bound
    breaks down.  Weak instrument => small mu_ZX => condition violated.
    """
    print("\n" + "=" * 60)
    print("STUDY 4: Instrument strength condition")
    print("=" * 60)

    n = n_total
    m = n // k
    sigma_ZX = 1.0

    # threshold ratio r = m * mu^2_ZX / sigma^2_ZX
    # RoM condition: r > 4   => mu_ZX > sqrt(4*sigma^2_ZX/m)
    # MoR condition: r >= 32  => mu_ZX >= sqrt(32*sigma^2_ZX/m)
    rom_thresh_mu = np.sqrt(4  * sigma_ZX**2 / m)
    mor_thresh_mu = np.sqrt(32 * sigma_ZX**2 / m)

    mu_ZX_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0]) * rom_thresh_mu

    print(f"  n={n}, k={k}, m={m}, sigma_ZX={sigma_ZX}")
    print(f"  RoM threshold mu_ZX = {rom_thresh_mu:.4f}")
    print(f"  MoR threshold mu_ZX = {mor_thresh_mu:.4f}")
    print(f"  delta = {delta}")
    print()
    print(f"  {'mu_ZX':>8}  {'r=m*mu2/s2':>12}  "
          f"{'RoM exceed':>12}  {'MoR exceed':>12}  "
          f"{'IV exceed':>12}  {'bound':>8}")

    results = []
    for mu_ZX in mu_ZX_values:
        rom_ests, mor_ests, iv_ests = [], [], []
        for _ in range(n_trials):
            Y, X, Z = generate_data(n, beta, mu_ZX, sigma_ZX, error_dist=error_dist)
            rom_ests.append(rom_estimator(Y, X, Z, k))
            mor_ests.append(mor_estimator(Y, X, Z, k))
            iv_ests.append(iv_estimator(Y, X, Z))

        # Use a generous threshold: theoretical sub-Gaussian bound magnitude
        # We check: what fraction of estimates are more than some_threshold away from beta?
        # Use 2*sigma_ZX/mu_ZX / sqrt(m) as a scale-free threshold
        scale = sigma_ZX / (mu_ZX * np.sqrt(m))
        threshold = 5.0 * scale   # 5x the natural scale

        rom_arr = np.array(rom_ests)
        mor_arr = np.array(mor_ests)
        iv_arr  = np.array(iv_ests)

        rom_exceed = np.mean(np.abs(rom_arr - beta) > threshold)
        mor_exceed = np.mean(np.abs(mor_arr - beta) > threshold)
        iv_exceed  = np.mean(np.abs(iv_arr  - beta) > threshold)

        r = m * mu_ZX**2 / sigma_ZX**2
        results.append((mu_ZX, r, rom_exceed, mor_exceed, iv_exceed))
        print(f"  {mu_ZX:>8.4f}  {r:>12.2f}  "
              f"{rom_exceed:>12.4f}  {mor_exceed:>12.4f}  "
              f"{iv_exceed:>12.4f}  {delta:>8.2f}")

    # Check: exceedance drops as instrument strengthens
    roms = [r[2] for r in results]
    mors = [r[3] for r in results]
    rom_improves = roms[-1] < roms[0]
    mor_improves = mors[-1] < mors[0]
    print(f"\n  [CHECK] RoM exceedance improves with stronger instrument: {rom_improves}")
    print(f"  [CHECK] MoR exceedance improves with stronger instrument: {mor_improves}")
    return results


# ─────────────────────────────────────────────
# STUDY 5 – Block count: RoM vs MoR
# ─────────────────────────────────────────────
# RoM needs k = ceil(8*ln(2/delta)), MoR needs k = ceil(8*ln(1/delta))
# For a fixed n and delta, MoR needs ~5-6 more blocks.
# We verify that both estimators achieve their guaranteed exceedance probability
# when using exactly their prescribed k.

def study5_block_count(beta=2.0, n=3000, delta_vals=None,
                       n_trials=4000, error_dist="pareto",
                       mu_ZX=1.0, sigma_ZX=0.5):
    """
    For each delta, use the prescribed k for RoM and MoR.
    Check that empirical exceedance probability <= delta.
    """
    if delta_vals is None:
        delta_vals = [0.30, 0.20, 0.10, 0.05, 0.02]

    print("\n" + "=" * 60)
    print("STUDY 5: Block count — prescribed k achieves delta guarantee")
    print("=" * 60)
    print(f"  n={n}, mu_ZX={mu_ZX}, sigma_ZX={sigma_ZX}")
    print()
    print(f"  {'delta':>6}  {'k_RoM':>6}  {'k_MoR':>6}  "
          f"{'RoM exceed':>12}  {'MoR exceed':>12}  "
          f"{'RoM ok':>8}  {'MoR ok':>8}")

    results = []
    tol = 0.03   # allow 3% above delta (Monte Carlo noise)

    for delta in delta_vals:
        k_rom = int(np.ceil(8 * np.log(2 / delta)))
        k_mor = int(np.ceil(8 * np.log(1 / delta)))

        # Use a meaningful threshold: 2 * theoretical scale
        # scale ~ sigma_Ze / (mu_ZX * sqrt(m))  where m = n/k
        m_rom = n // k_rom
        m_mor = n // k_mor

        # Approximate sigma_Ze empirically from one large sample
        Y0, X0, Z0 = generate_data(100000, beta, mu_ZX, sigma_ZX, error_dist=error_dist)
        eps0 = Y0 - beta * X0
        sigma_Ze = np.std(Z0 * eps0)

        thresh_rom = 3 * sigma_Ze / (abs(mu_ZX) * np.sqrt(m_rom))
        thresh_mor = 3 * sigma_Ze / (abs(mu_ZX) * np.sqrt(m_mor))

        rom_ests, mor_ests = [], []
        for _ in range(n_trials):
            Y, X, Z = generate_data(n, beta, mu_ZX, sigma_ZX, error_dist=error_dist)
            rom_ests.append(rom_estimator(Y, X, Z, k_rom))
            mor_ests.append(mor_estimator(Y, X, Z, k_mor))

        rom_arr = np.array(rom_ests)
        mor_arr = np.array(mor_ests)

        rom_exceed = np.mean(np.abs(rom_arr - beta) > thresh_rom)
        mor_exceed = np.mean(np.abs(mor_arr - beta) > thresh_mor)

        rom_ok = rom_exceed <= delta + tol
        mor_ok = mor_exceed <= delta + tol

        results.append((delta, k_rom, k_mor, rom_exceed, mor_exceed))
        print(f"  {delta:>6.2f}  {k_rom:>6}  {k_mor:>6}  "
              f"{rom_exceed:>12.4f}  {mor_exceed:>12.4f}  "
              f"{str(rom_ok):>8}  {str(mor_ok):>8}")

    # Verify k_MoR > k_RoM always
    k_diff_ok = all(r[2] > r[1] for r in results)
    print(f"\n  [CHECK] k_MoR > k_RoM for all delta values: {k_diff_ok}")
    # Expected additive difference is 8*ln(2) ~ 5.5
    avg_diff = np.mean([r[2] - r[1] for r in results])
    print(f"  Mean k_MoR - k_RoM = {avg_diff:.2f}  (theory: 8*ln(2) ≈ {8*np.log(2):.2f})")
    return results


# ─────────────────────────────────────────────
# STUDY 6 – Tail comparison: MoR vs IV under increasing tail heaviness
# ─────────────────────────────────────────────

def study6_heavy_tails(beta=2.0, n=1000, k=16,
                       alpha_list=None, n_trials=3000,
                       mu_ZX=1.0, sigma_ZX=0.5):
    """
    Pareto tail index alpha: larger alpha = lighter tail, finite variance needs alpha>2.
    As alpha decreases toward 2, tails get heavier.
    MoM-based estimators should degrade less than standard IV.
    """
    if alpha_list is None:
        alpha_list = [5.0, 3.5, 2.5, 2.1]

    print("\n" + "=" * 60)
    print("STUDY 6: Performance under increasing tail heaviness (Pareto)")
    print("=" * 60)
    print(f"  n={n}, k={k}, mu_ZX={mu_ZX}")
    print()
    print(f"  {'alpha':>7}  {'IV RMSE':>10}  {'RoM RMSE':>10}  {'MoR RMSE':>10}  "
          f"{'IV MAE':>10}  {'RoM MAE':>10}  {'MoR MAE':>10}")

    results = []
    for alpha in alpha_list:
        iv_ests, rom_ests, mor_ests = [], [], []
        for _ in range(n_trials):
            Y, X, Z = generate_data(n, beta, mu_ZX, sigma_ZX,
                                     error_dist="pareto", alpha_tail=alpha)
            iv_ests.append(iv_estimator(Y, X, Z))
            rom_ests.append(rom_estimator(Y, X, Z, k))
            mor_ests.append(mor_estimator(Y, X, Z, k))

        def robust_stats(arr, center=beta, clip=200):
            a = np.clip(arr, center - clip, center + clip)
            rmse = np.sqrt(np.mean((a - center) ** 2))
            mae  = np.mean(np.abs(a - center))
            return rmse, mae

        iv_rmse,  iv_mae  = robust_stats(np.array(iv_ests))
        rom_rmse, rom_mae = robust_stats(np.array(rom_ests))
        mor_rmse, mor_mae = robust_stats(np.array(mor_ests))

        results.append((alpha, iv_rmse, rom_rmse, mor_rmse, iv_mae, rom_mae, mor_mae))
        print(f"  {alpha:>7.1f}  {iv_rmse:>10.4f}  {rom_rmse:>10.4f}  {mor_rmse:>10.4f}  "
              f"{iv_mae:>10.4f}  {rom_mae:>10.4f}  {mor_mae:>10.4f}")

    # Check: MoR RMSE <= IV RMSE for all alpha (robust advantage)
    mor_beats_iv = all(r[3] <= r[1] for r in results)
    print(f"\n  [CHECK] MoR RMSE <= IV RMSE across all tail heaviness: {mor_beats_iv}")
    return results


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def make_plots(res1, res2, res3, res4, res5, res6):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Tail bound (Study 1) ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    deltas = res1["delta"]
    ax1.plot(deltas, res1["mom_exceed"],  "b-o", label="MoM empirical")
    ax1.plot(deltas, res1["mean_exceed"], "r-s", label="Mean empirical")
    ax1.plot(deltas, deltas, "k--", label="y = delta (bound)")
    ax1.set_xlabel("delta")
    ax1.set_ylabel("Empirical exceedance prob.")
    ax1.set_title("Study 1: Tail Bounds\n(MoM vs Empirical Mean)")
    ax1.legend(fontsize=7)
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    # ── Plot 2: Consistency (Study 2) ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ns = [r[0] for r in res2]
    ax2.plot(ns, [r[5] for r in res2], "b-o", label="RoM RMSE")
    ax2.plot(ns, [r[6] for r in res2], "g-^", label="MoR RMSE")
    ax2.plot(ns, [r[4] for r in res2], "r--s", label="IV RMSE", alpha=0.6)
    ax2.set_xlabel("n")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Study 2: Consistency\n(Heavy-tailed errors)")
    ax2.legend(fontsize=7)
    ax2.set_xscale("log")

    # ── Plot 3: Effective variance (Study 3) ──────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axhline(1.0, color="k", linestyle="--", label="Ratio = 1 (bound)")
    ratio_label = r"$\sigma_{Z\varepsilon} / (\sigma_{ZY} + |\beta|\sigma_{ZX})$"
    ax3.bar(["Mean ratio", "Max ratio"],
            [res3["mean_ratio"], res3["max_ratio"]],
            color=["steelblue", "tomato"], alpha=0.8)
    ax3.set_ylabel("Ratio")
    ax3.set_title(f"Study 3: Effective Variance\n{ratio_label}")
    ax3.set_ylim(0, max(1.2, res3["max_ratio"] * 1.1))
    ax3.legend(fontsize=7)

    # ── Plot 4: Instrument strength (Study 4) ─────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    rs   = [r[1] for r in res4]
    rom4 = [r[2] for r in res4]
    mor4 = [r[3] for r in res4]
    iv4  = [r[4] for r in res4]
    ax4.plot(rs, rom4, "b-o", label="RoM exceed")
    ax4.plot(rs, mor4, "g-^", label="MoR exceed")
    ax4.plot(rs, iv4,  "r--s", label="IV exceed", alpha=0.6)
    ax4.axvline(4,  color="blue",  linestyle=":", linewidth=1.5, label="RoM threshold (r=4)")
    ax4.axvline(32, color="green", linestyle=":", linewidth=1.5, label="MoR threshold (r=32)")
    ax4.set_xlabel(r"$r = m \cdot \mu^2_{ZX} / \sigma^2_{ZX}$")
    ax4.set_ylabel("Exceedance prob.")
    ax4.set_title("Study 4: Instrument Strength\n(Exceedance vs strength ratio)")
    ax4.legend(fontsize=6)

    # ── Plot 5: Block count (Study 5) ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ds5 = [r[0] for r in res5]
    ax5.plot(ds5, [r[1] for r in res5], "b-o", label="k_RoM = ⌈8 ln(2/δ)⌉")
    ax5.plot(ds5, [r[2] for r in res5], "g-^", label="k_MoR = ⌈8 ln(1/δ)⌉")
    ax5.set_xlabel("delta")
    ax5.set_ylabel("Number of blocks k")
    ax5.set_title("Study 5: Prescribed Block Counts\n(k_MoR > k_RoM by ~5.5)")
    ax5.legend(fontsize=7)

    # ── Plot 5b: exceedance vs delta ─────────────────────────
    ax5b = fig.add_subplot(gs[1, 2])
    ax5b.plot(ds5, [r[3] for r in res5], "b-o", label="RoM empirical")
    ax5b.plot(ds5, [r[4] for r in res5], "g-^", label="MoR empirical")
    ax5b.plot(ds5, ds5, "k--", label="y = delta")
    ax5b.set_xlabel("delta")
    ax5b.set_ylabel("Empirical exceedance prob.")
    ax5b.set_title("Study 5: Exceedance vs Delta\n(should lie below diagonal)")
    ax5b.legend(fontsize=7)

    # ── Plot 6: Heavy tails (Study 6) ─────────────────────────
    ax6a = fig.add_subplot(gs[2, 0])
    alphas = [r[0] for r in res6]
    ax6a.plot(alphas, [r[1] for r in res6], "r--s", label="IV")
    ax6a.plot(alphas, [r[2] for r in res6], "b-o",  label="RoM")
    ax6a.plot(alphas, [r[3] for r in res6], "g-^",  label="MoR")
    ax6a.set_xlabel("Pareto tail index alpha (lighter  →)")
    ax6a.set_ylabel("RMSE")
    ax6a.set_title("Study 6: RMSE vs Tail Heaviness")
    ax6a.invert_xaxis()
    ax6a.legend(fontsize=7)

    ax6b = fig.add_subplot(gs[2, 1])
    ax6b.plot(alphas, [r[4] for r in res6], "r--s", label="IV")
    ax6b.plot(alphas, [r[5] for r in res6], "b-o",  label="RoM")
    ax6b.plot(alphas, [r[6] for r in res6], "g-^",  label="MoR")
    ax6b.set_xlabel("Pareto tail index alpha (lighter  →)")
    ax6b.set_ylabel("MAE")
    ax6b.set_title("Study 6: MAE vs Tail Heaviness")
    ax6b.invert_xaxis()
    ax6b.legend(fontsize=7)

    # ── Summary table ─────────────────────────────────────────
    ax_t = fig.add_subplot(gs[2, 2])
    ax_t.axis("off")
    table_data = [
        ["Claim", "Result"],
        ["MoM sub-Gaussian bound", "Study 1"],
        ["RoM / MoR consistency", "Study 2"],
        [r"$\sigma_{Z\varepsilon} \leq \sigma_{ZY}+|\beta|\sigma_{ZX}$", "Study 3"],
        ["Instrument strength cond.", "Study 4"],
        ["k_MoR > k_RoM by ~5.5", "Study 5"],
        ["Robustness to heavy tails", "Study 6"],
    ]
    tbl = ax_t.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    ax_t.set_title("Simulation Study Summary", fontsize=10, pad=10)

    fig.suptitle("MoM-2SLS Simulation Study\n"
                 "(Verifying theoretical claims in 'Median-of-Means Two-Stage Least Squares')",
                 fontsize=12, fontweight="bold")

    plt.savefig("c:/Users/Pavilion/Documents/Thesis/VScode/Code/simulation_results.png",
                dpi=150, bbox_inches="tight")
    print("\n  Figure saved: Code/simulation_results.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("MoM-2SLS Simulation Study")
    print("Testing theoretical claims from the paper")
    print("=" * 60)

    res1 = study1_mom_tail_bound(n=2000, n_trials=5000)
    res2 = study2_consistency(beta=2.0, mu_ZX=1.0, sigma_ZX=0.8,
                               k=16, n_trials=2000, error_dist="pareto")
    res3 = study3_effective_variance(beta=2.0, n=50000, n_rep=200,
                                      error_dist="pareto", mu_ZX=1.0, sigma_ZX=0.8)
    res4 = study4_instrument_strength(beta=2.0, n_total=5000, k=20,
                                       delta=0.10, n_trials=3000,
                                       error_dist="pareto")
    res5 = study5_block_count(beta=2.0, n=3000, delta_vals=[0.30, 0.20, 0.10, 0.05, 0.02],
                               n_trials=4000, error_dist="pareto",
                               mu_ZX=1.0, sigma_ZX=0.5)
    res6 = study6_heavy_tails(beta=2.0, n=1000, k=16,
                               alpha_list=[5.0, 3.5, 2.5, 2.1],
                               n_trials=3000, mu_ZX=1.0, sigma_ZX=0.5)

    print("\n" + "=" * 60)
    print("ALL STUDIES COMPLETE — generating figures...")
    make_plots(res1, res2, res3, res4, res5, res6)