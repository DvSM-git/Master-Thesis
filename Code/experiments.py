"""
Monte Carlo experiment suite for the simulation section of the thesis.

Each experiment maps to specific theoretical results in Paper/iteration4:

  E1   Gaussian, strong instrument: standard IV most efficient
       -> Thm thm:iv (the "price of robustness" of MoR/RoM/Catoni)
  E2   Heavy tails (t3, t2.1, Pareto 2.5, Tukey mixture): MoM estimators beat standard IV
       -> Thm thm:rom, thm:mor; Remarks rem:iv_polynomial, rem:mor_logarithmic
  E2b  Empirical deviation quantiles vs ln(1/delta): polynomial vs
       logarithmic delta-dependence -> rem:iv_polynomial vs rem:mor_logarithmic
  E3   Instrument-strength sweep: degradation when eq:rom_strength /
       eq:mor_strength fail
  E4   Moment existence vs kurtosis: Gaussian mixtures (huge kurtosis, all
       moments finite, sub-Gaussian tails) against t(2.01) at the edge of (A3)
       -> isolates what the MoM advantage actually requires
  I1   Size of MoM-AR (oracle & feasible), SN-AR, standard AR
       -> Thm thm:mom_ar_size, Prop prop:sn_pivotal
  I2   Power curves (rejection frequency vs beta0 - beta)
  I3   Confidence sets: coverage, length, components, unboundedness
       -> Thm thm:coverage, Cor cor:union, cor:sn_cs (exact Alg alg:mom_ar_cs)
  I4   Monotonicity / single-interval condition
       -> Prop prop:mono_det, prop:mono_cheby
  I5   R_k critical value table for the appendix (sec:artable)
       -> Prop prop:sn_pivotal
  AG   Size and power of the aggregated MoM-AR test across instrument
       strength and aggregation depth B
       -> Def def:randomised_test, Thm thm:agg_size, Prop prop:agg_median_form
  AB   Aggregated point estimator across instrument strength and error
       distribution, by aggregation depth B
       -> Def def:randomised, Prop prop:conditional, Thm thm:aggregation

Usage:
    python experiments.py --pilot            # reduced replication counts
    python experiments.py --full             # thesis-quality counts
    python experiments.py --full --only e1,i3
    python experiments.py --verify           # internal consistency checks
"""
from __future__ import annotations

import argparse
import textwrap
import time
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: avoid Tk in background/parallel runs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from simulation import (
    generate_data,
    iv_estimate,
    iv_estimate_catoni,
    iv_estimate_mr,
    iv_estimate_rm,
    iv_estimate_trimmed,
)
from simulation_study import ESTIMATOR_COLORS, GRAPHS_DIR, styled_boxplot  # noqa: F401 (rcParams side effect)
import inference as inf

DELTA = 0.05

# Raw per-replication draws are written here so figures can be redesigned
# without re-running the experiments. Gitignored: the contents are large and
# fully regenerable (every driver takes a fixed seed).
RAW_DIR = Path(__file__).resolve().parent / "output" / "raw"


def _stable_hash(*parts) -> int:
    """
    Deterministic stand-in for the builtin hash().

    Python salts string hashing per process (PYTHONHASHSEED), so seeds built
    from hash(<str>) differ between runs and the affected experiments were not
    reproducible. crc32 of the joined parts is stable across processes and
    machines, so re-running now reproduces results exactly.
    """
    return zlib.crc32("|".join(map(str, parts)).encode())


def _save_raw(name: str, **arrays) -> None:
    """Persist raw replication-level arrays for `name` to output/raw/<name>.npz."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW_DIR / f"{name}.npz",
                        **{k: np.asarray(v) for k, v in arrays.items()})

# Canonical DGP: calibrated so that at n = 2000 every finite-sample condition
# of the point-estimator theorems holds (strong-instrument regime):
#   IV  (eq:iv_strength):  n >= 8*2.5/0.05      = 400
#   RoM (eq:rom_strength): m > 4*2.5  = 10,  k = 30, n > 300
#   MoR (eq:mor_strength): m >= 32*2.5 = 80, k = 24, n >= 1920
BASE = dict(beta=1.0, mu_ZX=1.0, sigma2_ZX=2.5, sigma2_Ze=1.0, rho=0.5)

# Error families for eps_Y (eps_X stays Gaussian so heavy tails enter only
# through Z*eps, the channel the theory is about). All standardised to unit
# variance, so sigma2_Ze is exact for every family. Mix(0.1,3) is Tukey's
# (1960) contaminated normal, 0.9 N(0,1) + 0.1 N(0,9): finite variance in
# every moment but kurtosis 8.33 vs the Gaussian's 3.
DISTS = {
    "Gaussian": None,
    "t(3)": ("t", 3.0),
    "t(2.1)": ("t", 2.1),
    "Pareto(2.5)": ("pareto", 2.5),
    "Mix(0.1,3)": ("mixture", 0.1, 3.0),
    # E4 only: a far more violent contamination (0.99 N(0,1) + 0.01 N(0,900)),
    # and a t at the very edge of (A3). See exp_e4_moments.
    "Mix(0.01,30)": ("mixture", 0.01, 30.0),
    "t(2.01)": ("t", 2.01),
}

TEST_COLORS = {
    "MoM-AR (oracle)": "#4C72B0",
    "MoM-AR (feasible)": "#64B5CD",
    "SN-AR": "#55A868",
    "AR (standard)": "#8172B3",
}

TEST_STYLES = {
    "MoM-AR (oracle)": "-",
    "MoM-AR (feasible)": "--",
    "SN-AR": "-",
    "AR (standard)": "-",
}


def rep_counts(full: bool) -> dict[str, int]:
    if full:
        return dict(box=10_000, tail=50_000, size=10_000, power=5_000, cs=5_000, mono=2_000,
                    ps_D=100, ps_B=200, ps_Bspot=2_000, moments=50_000, ag_D=2_000,
                    ab_D=2_000)
    return dict(box=1_000, tail=5_000, size=1_000, power=500, cs=500, mono=300,
                ps_D=20, ps_B=40, ps_Bspot=400, moments=5_000, ag_D=100,
                ab_D=100)


def _note(fig, text: str) -> None:
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8, color="0.3")


def _robust_ylim(arrs: list[np.ndarray], lo_q=0.005, hi_q=0.995, pad=0.15) -> tuple[float, float, float]:
    """y-limits from pooled quantiles; returns (lo, hi, frac_clipped)."""
    pooled = np.concatenate(arrs)
    lo, hi = np.quantile(pooled, [lo_q, hi_q])
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    frac = float(np.mean((pooled < lo) | (pooled > hi)))
    return lo, hi, frac


def _save(fig, name: str) -> Path:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    out = GRAPHS_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ----------------------------------------------------------------------------
# Point estimator replications (E1, E2, E2b, E3)
# ----------------------------------------------------------------------------


def _point_rep(seed, n, dgp, dist, delta) -> tuple[float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    df = generate_data(n=n, eps_Y_dist=dist, rng=rng, **dgp)
    # The trimmed mean is appended last and draws from `rng` after every other
    # estimator, so adding it leaves the other four bit-identical to earlier runs.
    return (
        iv_estimate(df)["beta_hat"],
        iv_estimate_rm(df, delta=delta, rng=rng)["beta_hat"],
        iv_estimate_mr(df, delta=delta, rng=rng)["beta_hat"],
        iv_estimate_catoni(df, delta=delta)["beta_hat"],
        iv_estimate_trimmed(df, delta=delta, rng=rng)["beta_hat"],
    )


def run_point_estimators(n, dgp, dist, n_reps, seed, n_jobs=-1, delta=DELTA) -> dict[str, np.ndarray]:
    seeds = np.random.SeedSequence(seed).spawn(n_reps)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_point_rep)(s, n, dgp, dist, delta) for s in seeds
    )
    iv, rm, mr, cat, trim = (np.asarray(c) for c in zip(*res))
    return {"Mean IV": iv, "Ratio-of-Medians": rm, "Median-of-Ratios": mr,
            "Catoni": cat, "Trimmed Mean": trim}


def _summary_rows(estimates: dict[str, np.ndarray], beta: float, tag: str) -> list[dict]:
    rows = []
    for name, arr in estimates.items():
        err = arr - beta
        rows.append({
            "config": tag, "estimator": name,
            "mean_bias": float(err.mean()), "median_bias": float(np.median(err)),
            "sd": float(arr.std()), "iqr": float(np.subtract(*np.quantile(arr, [0.75, 0.25]))),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mad_err": float(np.median(np.abs(err))),
            "q99_abs_err": float(np.quantile(np.abs(err), 0.99)),
        })
    return rows


def exp_e1_e2(reps: int, n_jobs: int, seed: int = 101) -> None:
    """E1 (Gaussian) and E2 (heavy tails) boxplots at n=2000, plus summary CSV."""
    n = 2000
    beta = BASE["beta"]
    rows = []

    # --- E1: Gaussian ---
    t0 = time.perf_counter()
    est = run_point_estimators(n, BASE, DISTS["Gaussian"], reps, seed, n_jobs)
    _save_raw("e1_estimates_Gaussian", **est)
    rows += _summary_rows(est, beta, "Gaussian")
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    styled_boxplot(ax, est, true_beta=beta)
    lo, hi, frac = _robust_ylim(list(est.values()))
    ax.set_ylim(lo, hi)
    _note(fig, f"Note. {reps} replications; n = {n}, Gaussian errors, "
               f"$\\mu_{{ZX}}$ = {BASE['mu_ZX']:g}, $\\rho$ = {BASE['rho']:g}, "
               f"$\\delta$ = {DELTA:g}. Axis truncated at 0.5/99.5% quantiles "
               f"({100*frac:.2f}% of points outside).")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.18)
    _save(fig, "e1_boxplot_gaussian.png")

    # --- E2: heavy tails, four panels ---
    heavy = ["t(3)", "t(2.1)", "Pareto(2.5)", "Mix(0.1,3)"]
    fig, axes = plt.subplots(1, len(heavy), figsize=(17.5, 4.6))
    for ax, dname in zip(axes, heavy):
        est = run_point_estimators(n, BASE, DISTS[dname], reps, seed + _stable_hash(dname) % 1000, n_jobs)
        _save_raw(f"e2_estimates_{dname}", **est)
        rows += _summary_rows(est, beta, dname)
        styled_boxplot(ax, est, true_beta=beta, show_legend=(dname == heavy[0]))
        lo, hi, frac = _robust_ylim(list(est.values()))
        ax.set_ylim(lo, hi)
        ax.set_title(f"$\\varepsilon_Y \\sim$ {dname}  ({100*frac:.1f}% clipped)", fontsize=12)
        ax.tick_params(axis="x", labelrotation=20)
        if ax is not axes[0]:
            ax.set_ylabel("")
    _note(fig, f"Note. {reps} replications; n = {n}, "
               f"$\\mu_{{ZX}}$ = {BASE['mu_ZX']:g}, $\\rho$ = {BASE['rho']:g}, "
               f"$\\delta$ = {DELTA:g}. Axes truncated at 0.5/99.5% quantiles.")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.92, bottom=0.24, wspace=0.26)
    _save(fig, "e2_boxplot_heavytails.png")

    df = pd.DataFrame(rows)
    df.to_csv(GRAPHS_DIR / "e1_e2_summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[E1/E2] runtime {time.perf_counter()-t0:.1f}s")


def exp_e2b_tails(reps: int, n_jobs: int, seed: int = 202) -> None:
    """
    E2b: empirical (1-delta)-quantiles of |beta_hat - beta| against ln(1/delta).
    Validates the delta-dependence: IV grows polynomially in 1/delta
    (rem:iv_polynomial) while the MoM estimators grow like sqrt(ln(1/delta))
    (rem:mor_logarithmic).
    """
    n = 2000
    beta = BASE["beta"]
    deltas = np.geomspace(0.002, 0.5, 40)
    x = np.log(1.0 / deltas)
    panels = ["Gaussian", "t(2.1)"]

    t0 = time.perf_counter()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    qtab = []
    for ax, dname in zip(axes, panels):
        est = run_point_estimators(n, BASE, DISTS[dname], reps, seed + _stable_hash(dname) % 1000, n_jobs)
        _save_raw(f"e2b_estimates_{dname}", **est)
        for ename, arr in est.items():
            q = np.quantile(np.abs(arr - beta), 1.0 - deltas)
            ax.plot(x, q, color=ESTIMATOR_COLORS[ename], linewidth=1.8, label=ename)
            for d, v in zip(deltas, q):
                qtab.append({"dist": dname, "estimator": ename, "delta": d, "quantile_abs_err": v})
        ax.set_title(f"$\\varepsilon_Y \\sim$ {dname}", fontsize=12)
        ax.set_xlabel(r"$\ln(1/\delta)$")
        ax.set_yscale("log")
        ax.yaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"Empirical $(1-\delta)$-quantile of $|\hat{\beta}-\beta|$")
    axes[0].legend(frameon=False, fontsize=10)
    _note(fig, f"Note. {reps} replications; n = {n}. Log scale. A sub-Gaussian estimator "
               r"grows like $\sqrt{\ln(1/\delta)}$ (flat); the polynomial rate of the "
               r"empirical mean grows like $\delta^{-1/2}$ (steep at large $\ln(1/\delta)$).")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.2, wspace=0.16)
    _save(fig, "e2b_deviation_quantiles.png")
    pd.DataFrame(qtab).to_csv(GRAPHS_DIR / "e2b_quantiles.csv", index=False)
    print(f"[E2b] runtime {time.perf_counter()-t0:.1f}s")


def exp_e3_strength(reps: int, n_jobs: int, seed: int = 303) -> None:
    """E3: instrument-strength sweep (Gaussian errors, n fixed)."""
    n = 2000
    beta = BASE["beta"]
    mus = [1.0, 0.4, 0.2, 0.1]
    k_mor = int(np.ceil(8 * np.log(1 / DELTA)))
    k_rom = int(np.ceil(8 * np.log(2 / DELTA)))
    rows = []

    t0 = time.perf_counter()
    fig, axes = plt.subplots(1, len(mus), figsize=(17.5, 4.6))
    for ax, mu in zip(axes, mus):
        dgp = dict(BASE, mu_ZX=mu)
        est = run_point_estimators(n, dgp, None, reps, seed + int(mu * 1000), n_jobs)
        _save_raw(f"e3_estimates_mu{mu:g}", **est)
        rows += _summary_rows(est, beta, f"mu_ZX={mu:g}")
        styled_boxplot(ax, est, true_beta=beta, show_legend=(mu == mus[0]))
        lo, hi, frac = _robust_ylim(list(est.values()))
        ax.set_ylim(lo, hi)
        ratio = BASE["sigma2_ZX"] / mu ** 2
        cond_iv = n >= 8 * ratio / DELTA
        cond_rom = (n // k_rom) > 4 * ratio
        cond_mor = (n // k_mor) >= 32 * ratio
        marks = "".join(f"{name}{'+' if ok else '-'} " for name, ok in
                        [("IV", cond_iv), ("RoM", cond_rom), ("MoR", cond_mor)])
        ax.set_title(f"$\\mu_{{ZX}}$ = {mu:g}   [{marks.strip()}]"
                     f"  ({100*frac:.1f}% clipped)", fontsize=11)
        ax.tick_params(axis="x", labelrotation=25)
        if ax is not axes[0]:
            ax.set_ylabel("")
    _note(fig, f"Note. {reps} replications; n = {n}, Gaussian errors. In brackets: whether each "
               f"estimator's instrument-strength condition holds (+/-): IV eq:iv_strength, "
               f"RoM eq:rom_strength, MoR eq:mor_strength. Axes truncated at 0.5/99.5% quantiles.")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.26, wspace=0.26)
    _save(fig, "e3_boxplot_strength.png")
    pd.DataFrame(rows).to_csv(GRAPHS_DIR / "e3_summary.csv", index=False)
    print(f"[E3] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# E4: moment existence vs kurtosis (candidate experiment)
# ----------------------------------------------------------------------------

# Panels span a kurtosis gradient in which the FIRST THREE have every moment
# finite (they are Gaussian mixtures, hence sub-Gaussian: the tail of a finite
# normal mixture is dominated by its widest component, exp(-t^2/2s^2)) and only
# the LAST violates (A3) beyond the variance. If the MoM advantage tracked
# kurtosis, "GMM high" would already show it; if it tracks moment existence,
# only t(2.01) does.
E4_DISTS = ["Gaussian", "Mix(0.1,3)", "Mix(0.01,30)", "t(2.01)"]
E4_TITLES = {
    "Gaussian": "Gaussian",
    "Mix(0.1,3)": "GMM low: $0.9\\,N(0,1)+0.1\\,N(0,9)$",
    "Mix(0.01,30)": "GMM high: $0.99\\,N(0,1)+0.01\\,N(0,900)$",
    "t(2.01)": "$t(2.01)$",
}


def _kurt_Zeps(dist) -> float:
    """
    Population kurtosis of Z*eps for Z ~ N(0,1) independent of eps.

    E[(Z eps)^4] = E[Z^4] E[eps^4] = 3 E[eps^4] and Var(Z eps) = Var(eps),
    so kurt(Z eps) = 3 * kurt(eps). Returns inf when the fourth moment of eps
    does not exist (t with df <= 4).
    """
    if dist is None:
        return 9.0
    family, *params = dist
    if family == "mixture":
        e, s = params
        return 3.0 * 3.0 * ((1 - e) + e * s**4) / ((1 - e) + e * s**2) ** 2
    if family == "t":
        return np.inf if params[0] <= 4 else 3.0 * 3.0 * (params[0] - 2) / (params[0] - 4)
    return np.inf


def exp_e4_moments(reps: int, n_jobs: int, seed: int = 909) -> None:
    """
    E4: does the MoM advantage track kurtosis, or moment existence?

    Four error families at n = 2000 spanning kurt(Z*eps) from 9 to infinity.
    The two Gaussian-mixture panels ("GMM low"/"GMM high") have enormous
    kurtosis but every moment finite and sub-Gaussian tails; t(2.01) sits at
    the edge of (A3) with an infinite third moment.

    Row (a): sampling distributions (as E1/E2).
    Row (b): empirical (1-delta)-quantile of |beta_hat - beta| against
    ln(1/delta) (as E2b) -- the delta-dependence is where the distinction
    between the two kinds of tail actually lives.
    """
    n = 2000
    beta = BASE["beta"]
    deltas = np.geomspace(0.002, 0.5, 40)
    x = np.log(1.0 / deltas)
    rows, qtab = [], []

    t0 = time.perf_counter()
    fig, axes = plt.subplots(2, len(E4_DISTS), figsize=(17.5, 8.2))
    for col, dname in enumerate(E4_DISTS):
        est = run_point_estimators(n, BASE, DISTS[dname], reps,
                                   seed + _stable_hash(dname) % 1000, n_jobs)
        _save_raw(f"e4_estimates_{dname}", **est)
        rows += _summary_rows(est, beta, dname)

        ax = axes[0, col]
        styled_boxplot(ax, est, true_beta=beta, show_legend=(col == 0))
        lo, hi, frac = _robust_ylim(list(est.values()))
        ax.set_ylim(lo, hi)
        kt = _kurt_Zeps(DISTS[dname])
        kt_s = r"\infty" if not np.isfinite(kt) else f"{kt:.0f}"
        ax.set_title(f"{E4_TITLES[dname]}\n"
                     f"$\\mathrm{{kurt}}(Z\\varepsilon)$ = ${kt_s}$"
                     f"  ({100*frac:.1f}% clipped)", fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        if col:
            ax.set_ylabel("")

        axq = axes[1, col]
        for ename, arr in est.items():
            q = np.quantile(np.abs(arr - beta), 1.0 - deltas)
            axq.plot(x, q, color=ESTIMATOR_COLORS[ename], linewidth=1.8, label=ename)
            for d, v in zip(deltas, q):
                qtab.append({"dist": dname, "estimator": ename,
                             "delta": d, "quantile_abs_err": v})
        axq.set_yscale("log")
        axq.set_xlabel(r"$\ln(1/\delta)$")
        axq.yaxis.grid(True, color="0.88", linewidth=0.7)
        axq.set_axisbelow(True)
        if col == 0:
            axq.legend(frameon=False, fontsize=9)

    axes[0, 0].set_ylabel(r"Estimate $\hat{\beta}$")
    axes[1, 0].set_ylabel(r"$(1-\delta)$-quantile of $|\hat{\beta}-\beta|$")
    _note(fig, f"Note. {reps} replications; n = {n}, $\\mu_{{ZX}}$ = {BASE['mu_ZX']:g}, "
               f"$\\rho$ = {BASE['rho']:g}, $\\delta$ = {DELTA:g}. The first three families have "
               r"every moment finite (finite normal mixtures are sub-Gaussian); only $t(2.01)$ "
               r"violates (A3) beyond the variance. Row (b) on a log scale; a sub-Gaussian "
               r"estimator grows like $\sqrt{\ln(1/\delta)}$.")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.14,
                        wspace=0.28, hspace=0.46)
    _save(fig, "e4_moment_existence.png")

    df = pd.DataFrame(rows)
    df.to_csv(GRAPHS_DIR / "e4_moments_summary.csv", index=False)
    pd.DataFrame(qtab).to_csv(GRAPHS_DIR / "e4_moments_quantiles.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[E4] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# Inference replications (I1, I2): all tests on a common beta0 grid
# ----------------------------------------------------------------------------


def _tests_rep(seed, n, dgp, dist, beta0_grid, delta, c_sn) -> dict[str, np.ndarray]:
    """
    One replication: generate data, run all four tests on the beta0 grid.
    All block-based tests share the same block partition (fair comparison);
    the feasible threshold is computed once, free of beta0 (per the
    Feasibility remark). Returns dict of boolean rejection arrays.
    """
    rng = np.random.default_rng(seed)
    df = generate_data(n=n, eps_Y_dist=dist, rng=rng, **dgp)
    Y, X, Z = (df[c].to_numpy() for c in ("Y", "X", "Z"))
    k = inf.k_blocks(delta)
    a, b, _ = inf.block_means(Y, X, Z, k, rng=rng)

    W_tilde = inf.mom_ar_statistic(a, b, beta0_grid)
    tau_orac = inf.tau_n(np.sqrt(dgp["sigma2_Ze"]), n, delta)
    sig_hat = inf.robust_sigma_Ze(Y, X, Z, delta, rng=rng)
    tau_feas = inf.tau_n(sig_hat, n, delta)
    T_sn = inf.sn_statistic(a, b, beta0_grid)
    std = inf.standard_ar_test(df, beta0_grid, delta=delta)

    return {
        "MoM-AR (oracle)": np.abs(W_tilde) > tau_orac,
        "MoM-AR (feasible)": np.abs(W_tilde) > tau_feas,
        "SN-AR": T_sn > c_sn,
        "AR (standard)": np.asarray(std["reject"]),
        # Diagnostics, not tests: underscore-prefixed so _rejection_rates
        # skips them when averaging. Used to report how wide the finite-sample
        # band tau_n is relative to the null spread of the statistic.
        "_W_tilde": np.asarray(W_tilde),
        "_tau_oracle": np.asarray(tau_orac),
        "_tau_feasible": np.asarray(tau_feas),
    }


def _rejection_rates(n, dgp, dist, beta0_grid, n_reps, seed, n_jobs, delta=DELTA,
                     raw_out: dict | None = None) -> dict[str, np.ndarray]:
    c_sn = inf.rk_critical_value(inf.k_blocks(delta), delta)  # precompute (cache is not process-safe)
    seeds = np.random.SeedSequence(seed).spawn(n_reps)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_tests_rep)(s, n, dgp, dist, beta0_grid, delta, c_sn) for s in seeds
    )
    if raw_out is not None:
        # (n_reps, len(beta0_grid)) boolean rejection matrix per test
        for name in res[0]:
            raw_out[name] = np.stack([r[name] for r in res])
    return {name: np.mean([r[name] for r in res], axis=0)
            for name in res[0] if not name.startswith("_")}


def exp_i1_size(reps: int, n_jobs: int, seed: int = 404) -> None:
    """I1: empirical size at beta0 = beta across tails and sample sizes."""
    ns = [500, 2000, 8000]
    dists = ["Gaussian", "t(2.1)", "Pareto(2.5)", "Mix(0.1,3)"]
    beta = np.array([BASE["beta"]])
    rows = []
    tau_rows: list[dict] = []

    t0 = time.perf_counter()
    fig, axes = plt.subplots(1, len(dists), figsize=(17.0, 4.2), sharey=True)
    for ax, dname in zip(axes, dists):
        rates = {name: [] for name in TEST_COLORS}
        for n in ns:
            raw: dict = {}
            rr = _rejection_rates(n, BASE, DISTS[dname], beta, reps,
                                  seed + _stable_hash(dname, n) % 10_000, n_jobs, raw_out=raw)
            _save_raw(f"i1_reject_{dname}_n{n}", **raw)
            # How conservative is the finite-sample threshold? Compare tau_n
            # with the actual standard deviation of W_tilde under H0. The
            # oracle tau is fixed by sigma_Ze and cannot adapt to the tails;
            # the feasible one is rebuilt from a robust scale estimate.
            sd_W = float(raw["_W_tilde"][:, 0].std(ddof=1))
            tau_rows.append({
                "dist": dname, "n": n, "sd_W_null": sd_W,
                "tau_oracle": float(np.median(raw["_tau_oracle"])),
                "tau_feasible": float(np.median(raw["_tau_feasible"])),
                "ratio_oracle": float(np.median(raw["_tau_oracle"])) / sd_W,
                "ratio_feasible": float(np.median(raw["_tau_feasible"])) / sd_W,
            })
            for name in rates:
                rates[name].append(float(rr[name][0]))
                rows.append({"dist": dname, "n": n, "test": name, "size": float(rr[name][0])})
        for name, vals in rates.items():
            ax.plot(ns, vals, marker="o", markersize=5, linewidth=1.8,
                    linestyle=TEST_STYLES[name], color=TEST_COLORS[name], label=name)
        ax.axhline(DELTA, color="0.4", linestyle=":", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(v) for v in ns])
        ax.set_title(f"$\\varepsilon_Y \\sim$ {dname}", fontsize=12)
        ax.set_xlabel("n")
        ax.yaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Empirical rejection rate under $H_0$")
    axes[0].legend(frameon=False, fontsize=9)
    _note(fig, f"Note. {reps} replications per point; nominal level $\\delta$ = {DELTA:g} (dotted). "
               f"MoM-AR sizes are bounds (Thm thm:mom_ar_size guarantees size $\\leq\\delta$, not $=\\delta$).")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.2, wspace=0.1)
    _save(fig, "i1_size.png")
    pd.DataFrame(rows).to_csv(GRAPHS_DIR / "i1_size.csv", index=False)
    tau_tab = pd.DataFrame(tau_rows)
    tau_tab.to_csv(GRAPHS_DIR / "i1_tau_ratio.csv", index=False)
    print(tau_tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(rows).pivot_table(index=["dist", "n"], columns="test", values="size")
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"[I1] runtime {time.perf_counter()-t0:.1f}s")


def exp_i2_power(reps: int, n_jobs: int, seed: int = 505) -> None:
    """I2: power curves over beta0 for Gaussian and t(2.1) errors."""
    n = 2000
    beta = BASE["beta"]
    grid = beta + np.linspace(-0.6, 0.6, 49)
    dists = ["Gaussian", "t(2.1)"]
    rows = []

    t0 = time.perf_counter()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, dname in zip(axes, dists):
        raw: dict = {}
        rr = _rejection_rates(n, BASE, DISTS[dname], grid, reps,
                              seed + _stable_hash(dname) % 10_000, n_jobs, raw_out=raw)
        _save_raw(f"i2_reject_{dname}", beta0_grid=grid, **raw)
        for name, vals in rr.items():
            ax.plot(grid - beta, vals, linewidth=1.8, linestyle=TEST_STYLES[name],
                    color=TEST_COLORS[name], label=name)
            for b0, v in zip(grid, vals):
                rows.append({"dist": dname, "beta0_minus_beta": b0 - beta, "test": name,
                             "rejection_rate": float(v)})
        ax.axhline(DELTA, color="0.4", linestyle=":", linewidth=1.2)
        ax.set_title(f"$\\varepsilon_Y \\sim$ {dname}", fontsize=12)
        ax.set_xlabel(r"$\beta_0 - \beta$")
        ax.yaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Rejection frequency")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    _note(fig, f"Note. {reps} replications; n = {n}, $\\delta$ = {DELTA:g} (dotted).")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.2, wspace=0.08)
    _save(fig, "i2_power.png")
    pd.DataFrame(rows).to_csv(GRAPHS_DIR / "i2_power.csv", index=False)
    print(f"[I2] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# I3: confidence sets
# ----------------------------------------------------------------------------


def _cs_rep(seed, n, dgp, dist, delta, c_sn) -> dict[str, dict]:
    """One replication: all four confidence sets, their summaries + coverage."""
    rng = np.random.default_rng(seed)
    df = generate_data(n=n, eps_Y_dist=dist, rng=rng, **dgp)
    Y, X, Z = (df[c].to_numpy() for c in ("Y", "X", "Z"))
    beta = dgp["beta"]
    k = inf.k_blocks(delta)
    a, b, _ = inf.block_means(Y, X, Z, k, rng=rng)

    tau_orac = inf.tau_n(np.sqrt(dgp["sigma2_Ze"]), n, delta)
    sig_hat = inf.robust_sigma_Ze(Y, X, Z, delta, rng=rng)
    tau_feas = inf.tau_n(sig_hat, n, delta)

    sets = {
        "MoM-AR (oracle)": inf.mom_ar_cs_exact(a, b, tau_orac),
        "MoM-AR (feasible)": inf.mom_ar_cs_exact(a, b, tau_feas),
        "SN-AR": inf.sn_ar_cs(a, b, c_sn),
        "AR (standard)": inf.standard_ar_cs(df, delta=delta),
    }
    out = {}
    same_sign = bool(np.all(b > 0) or np.all(b < 0))
    for name, cs in sets.items():
        s = inf.cs_summary(cs)
        s["covers"] = inf.cs_contains(cs, beta)
        s["same_sign"] = same_sign
        out[name] = s
    return out


def exp_i3_cs(reps: int, n_jobs: int, seed: int = 606) -> None:
    """I3: CS coverage, length, components; strong and weak instruments."""
    n = 2000
    configs = [
        ("Gaussian, strong", "Gaussian", 1.0),
        ("t(2.1), strong", "t(2.1)", 1.0),
        ("Gaussian, weak", "Gaussian", 0.05),
        ("t(2.1), weak", "t(2.1)", 0.05),
    ]
    c_sn = inf.rk_critical_value(inf.k_blocks(DELTA), DELTA)
    rows = []
    lengths: dict[tuple[str, str], np.ndarray] = {}

    t0 = time.perf_counter()
    for tag, dname, mu in configs:
        dgp = dict(BASE, mu_ZX=mu)
        seeds = np.random.SeedSequence(seed + _stable_hash(tag) % 10_000).spawn(reps)
        res = Parallel(n_jobs=n_jobs)(
            delayed(_cs_rep)(s, n, dgp, DISTS[dname], DELTA, c_sn) for s in seeds
        )
        _save_raw("i3_cs_" + tag.replace(", ", "_").replace("(", "").replace(")", ""),
                  **{f"{name}__{field}": [r[name][field] for r in res]
                     for name in TEST_COLORS
                     for field in ("covers", "n_components", "unbounded", "length")})
        for name in TEST_COLORS:
            cov = np.mean([r[name]["covers"] for r in res])
            ncomp = np.array([r[name]["n_components"] for r in res])
            unb = np.array([r[name]["unbounded"] for r in res])
            lens = np.array([r[name]["length"] for r in res])
            bounded = lens[np.isfinite(lens)]
            empty = ncomp == 0
            lengths[(tag, name)] = bounded
            rows.append({
                "config": tag, "test": name, "coverage": float(cov),
                "median_length_bounded": float(np.median(bounded)) if bounded.size else np.nan,
                "pct_unbounded": float(unb.mean()),
                "pct_empty": float(empty.mean()),
                "mean_components": float(ncomp[~empty].mean()) if (~empty).any() else np.nan,
                "pct_single_interval": float(np.mean(ncomp <= 1)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(GRAPHS_DIR / "i3_cs_table.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Length boxplots (bounded sets only), one panel per config, log scale.
    fig, axes = plt.subplots(1, len(configs), figsize=(16.0, 4.6), sharey=False)
    for ax, (tag, dname, mu) in zip(axes, configs):
        data, labels, colors = [], [], []
        for name in TEST_COLORS:
            arr = lengths[(tag, name)]
            if arr.size:
                data.append(arr)
                labels.append(name.replace(" ", "\n", 1))
                colors.append(TEST_COLORS[name])
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.4))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor(color)
        ax.set_yscale("log")
        ax.set_title(tag, fontsize=12)
        ax.tick_params(axis="x", labelsize=8)
        ax.yaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("CS length (bounded sets only, log scale)")
    _note(fig, f"Note. {reps} replications; n = {n}, $\\delta$ = {DELTA:g}. Whiskers at 1.5 IQR, "
               f"outliers hidden. Unbounded sets excluded (see i3_cs_table.csv).")
    fig.subplots_adjust(left=0.05, right=0.995, top=0.92, bottom=0.18, wspace=0.25)
    _save(fig, "i3_cs_lengths.png")
    print(f"[I3] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# I4: monotonicity / single-interval condition
# ----------------------------------------------------------------------------


def _mono_rep(seed, n, dgp, delta) -> tuple[bool, bool, bool]:
    """Returns (all_same_sign, single_interval, deterministic_violation)."""
    rng = np.random.default_rng(seed)
    df = generate_data(n=n, rng=rng, **dgp)
    Y, X, Z = (df[c].to_numpy() for c in ("Y", "X", "Z"))
    k = inf.k_blocks(delta)
    a, b, _ = inf.block_means(Y, X, Z, k, rng=rng)
    tau = inf.tau_n(np.sqrt(dgp["sigma2_Ze"]), n, delta)
    cs = inf.mom_ar_cs_exact(a, b, tau)
    same_sign = bool(np.all(b > 0) or np.all(b < 0))
    single = len(cs) <= 1
    violation = same_sign and not single  # prop:mono_det says this must never occur
    return same_sign, single, violation


def exp_i4_mono(reps: int, n_jobs: int, seed: int = 707) -> None:
    """
    I4: single-interval diagnostics (prop:mono_det, prop:mono_cheby).

    Panel A: sweep instrument strength mu_ZX at n = 2000 — the empirical
    transition of "all block means share a sign" (the premise of
    prop:mono_det) and "CS is a single interval".

    Panel B: sweep n at mu_ZX = 0.75 through the Chebyshev threshold
    n* = k * ceil(k sigma2_ZX/(delta mu_ZX^2)) of prop:mono_cheby — the
    empirical fractions reach 1 far below n*, showing the sufficient
    condition is conservative.

    In both panels a replication with all block means of equal sign but a
    multi-interval CS would falsify prop:mono_det; the count is asserted zero.
    """
    k = inf.k_blocks(DELTA)
    t0 = time.perf_counter()

    def sweep(param_values, make_dgp, make_n) -> pd.DataFrame:
        rows = []
        for val in param_values:
            dgp, n = make_dgp(val), make_n(val)
            seeds = np.random.SeedSequence(seed + int(1e4 * val)).spawn(reps)
            res = Parallel(n_jobs=n_jobs)(
                delayed(_mono_rep)(s, n, dgp, DELTA) for s in seeds
            )
            same, single, viol = (np.asarray(c) for c in zip(*res))
            _save_raw(f"i4_mono_val{val:g}_n{n}",
                      same_sign=same, single_interval=single, violation=viol)
            rows.append({"param": val, "n": n,
                         "frac_same_sign": float(same.mean()),
                         "frac_single_interval": float(single.mean()),
                         "violations": int(viol.sum())})
        return pd.DataFrame(rows)

    # Panel A: strength sweep at fixed n.
    n_A = 2000
    mus = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
    dfA = sweep(mus, lambda mu: dict(BASE, mu_ZX=mu), lambda mu: n_A)
    m_A = n_A // k
    mu_star_A = np.sqrt(k * BASE["sigma2_ZX"] / (DELTA * m_A))  # strength needed by eq:mono_cheby

    # Panel B: n sweep at fixed moderate strength through the Chebyshev threshold.
    mu_B = 0.75
    m_star = k * BASE["sigma2_ZX"] / (DELTA * mu_B ** 2)
    n_star = k * int(np.ceil(m_star))
    ns = [250, 500, 1_000, 2_000, 8_000, n_star]
    dfB = sweep([float(n) for n in ns], lambda n: dict(BASE, mu_ZX=mu_B), lambda n: int(n))

    dfA.assign(panel="A: mu sweep (n=2000)").pipe(
        lambda d: pd.concat([d, dfB.assign(panel=f"B: n sweep (mu={mu_B})")])
    ).to_csv(GRAPHS_DIR / "i4_monotonicity.csv", index=False)
    print(dfA.to_string(index=False))
    print(dfB.to_string(index=False))
    n_viol = int(dfA["violations"].sum() + dfB["violations"].sum())
    assert n_viol == 0, "prop:mono_det violated — check implementation!"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, df, xcol, xlabel in [
        (axes[0], dfA, "param", r"$\mu_{ZX}$"),
        (axes[1], dfB, "n", "n"),
    ]:
        ax.plot(df[xcol], df["frac_same_sign"], marker="o", markersize=5, linewidth=1.8,
                color="#4C72B0", label="all block means same sign")
        ax.plot(df[xcol], df["frac_single_interval"], marker="s", markersize=5, linewidth=1.8,
                color="#55A868", label="CS is a single interval")
        ax.axhline(1 - DELTA, color="0.4", linestyle=":", linewidth=1.2, label=r"$1-\delta$")
        ax.set_xlabel(xlabel)
        ax.yaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_title(f"Strength sweep (n = {n_A})", fontsize=12)
    axes[1].set_title(f"Sample size sweep ($\\mu_{{ZX}}$ = {mu_B:g})", fontsize=12)
    axes[1].set_xscale("log")
    axes[1].axvline(n_star, color="0.4", linestyle="--", linewidth=1.2,
                    label=f"Chebyshev threshold $n^*$ = {n_star:,}")
    axes[0].set_ylabel("Fraction of replications")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    _note(fig, f"Note. {reps} replications per point; Gaussian errors, "
               f"$\\sigma^2_{{ZX}}$ = {BASE['sigma2_ZX']:g}, $\\delta$ = {DELTA:g}, k = {k}. "
               f"Left: eq:mono_cheby would require $\\mu_{{ZX}} \\geq$ {mu_star_A:.2f} at n = {n_A} "
               f"(outside the axis). Zero violations of prop:mono_det observed.")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.2, wspace=0.08)
    _save(fig, "i4_monotonicity.png")
    print(f"[I4] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# PS: partition randomness of the block estimators (nested design + spotlight)
# ----------------------------------------------------------------------------

# Strength grid: gamma = sigma2_ZX / mu_ZX^2 = 2.5 / mu^2, log-ish spaced from
# strong (gamma = 2.5) to very weak (gamma = 1000).
PS_MUS = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]
# Levels highlighted in the fixed-data spotlight figure (strong / transition / weak).
PS_SPOTLIGHT_MUS = [1.0, 0.3, 0.05]
# Levels drawn as boxplots in panel (a); chosen to match the E3 strength sweep
# so the two figures are directly comparable. All of PS_MUS is still computed
# and stored -- this only controls what is plotted.
PS_BOX_MUS = [1.0, 0.4, 0.2, 0.1]


def _ps_spotlight_level(seed, n, mu, delta, B) -> dict:
    """
    Fixed-data view: ONE dataset at strength mu, B random partitions, raw
    estimates returned. This is the purest form of the question 'same data,
    different seed -- different answer?'; the Mean IV estimate is the
    partition-free reference. Conditional on a single draw by design (the
    nested design in _ps_dataset is the draw-robust counterpart).
    """
    rng = np.random.default_rng(seed)
    dgp = dict(BASE, mu_ZX=mu)
    df = generate_data(n=n, eps_Y_dist=("t", 2.1), rng=rng, **dgp)
    Y, X, Z = (df[c].to_numpy() for c in ("Y", "X", "Z"))
    out = {"mu": mu, "gamma": BASE["sigma2_ZX"] / mu ** 2,
           "iv": iv_estimate(df)["beta_hat"], "mor": [], "rom": [], "cat": []}
    for _ in range(B):
        perm = rng.permutation(n)
        dfp = pd.DataFrame({"Y": Y[perm], "X": X[perm], "Z": Z[perm]})
        out["mor"].append(iv_estimate_mr(dfp, delta=delta, shuffle=False)["beta_hat"])
        out["rom"].append(iv_estimate_rm(dfp, delta=delta, shuffle=False)["beta_hat"])
        out["cat"].append(iv_estimate_catoni(dfp, delta=delta)["beta_hat"])
    for est in ("mor", "rom", "cat"):
        out[est] = np.asarray(out[est])
    return out


def _ps_dataset(seed, n, mu, delta, c_sn, B) -> dict:
    """
    One dataset of the nested design: draw the data once, then B independent
    random partitions. Mirroring replication_ak91_sensitivity.py, each
    partition is ONE permutation of the rows shared by every MoM-based
    procedure (methods differ only through their k); the standard AR test is
    partition-free and computed once. Returns within-dataset aggregates.
    """
    rng = np.random.default_rng(seed)
    dgp = dict(BASE, mu_ZX=mu)
    df = generate_data(n=n, eps_Y_dist=("t", 2.1), rng=rng, **dgp)
    Y, X, Z = (df[c].to_numpy() for c in ("Y", "X", "Z"))
    beta = dgp["beta"]
    ZX = Z * X
    gamma_hat = float(ZX.var(ddof=1) / ZX.mean() ** 2)
    k = inf.k_blocks(delta)
    beta0 = np.array([0.0, beta])

    ar = inf.standard_ar_test(df, beta0, delta=delta)
    ar_rej0, ar_rejb = bool(ar["reject"][0]), bool(ar["reject"][1])

    cols = {name: [] for name in
            ("mor", "rom", "cat", "cert", "mom_split", "sn_split",
             "mom_rej0", "mom_rejb", "sn_rej0", "sn_rejb")}
    for _ in range(B):
        perm = rng.permutation(n)
        dfp = pd.DataFrame({"Y": Y[perm], "X": X[perm], "Z": Z[perm]})
        cols["mor"].append(iv_estimate_mr(dfp, delta=delta, shuffle=False)["beta_hat"])
        cols["rom"].append(iv_estimate_rm(dfp, delta=delta, shuffle=False)["beta_hat"])
        cols["cat"].append(iv_estimate_catoni(dfp, delta=delta)["beta_hat"])

        a, b, _ = inf.block_means(Y[perm], X[perm], Z[perm], k, shuffle=False)
        cols["cert"].append(bool(np.all(b > 0) or np.all(b < 0)))
        sigma_hat = inf.robust_sigma_Ze(Y[perm], X[perm], Z[perm], delta, shuffle=False)
        tau = inf.tau_n(sigma_hat, n, delta)
        cols["mom_split"].append(len(inf.mom_ar_cs_exact(a, b, tau)) > 1)
        cols["sn_split"].append(len(inf.sn_ar_cs(a, b, c_sn)) > 1)
        W = np.abs(inf.mom_ar_statistic(a, b, beta0))
        T = inf.sn_statistic(a, b, beta0)
        cols["mom_rej0"].append(bool(W[0] > tau))
        cols["mom_rejb"].append(bool(W[1] > tau))
        cols["sn_rej0"].append(bool(T[0] > c_sn))
        cols["sn_rejb"].append(bool(T[1] > c_sn))

    out = {"mu": mu, "gamma_hat": gamma_hat,
           "ar_rej0": ar_rej0, "ar_rejb": ar_rejb}
    for est in ("mor", "rom", "cat"):
        v = np.asarray(cols[est])
        out[f"{est}_mean"] = float(v.mean())
        out[f"{est}_pvar"] = float(v.var(ddof=1))          # partition variance
        out[f"{est}_piqr"] = float(np.subtract(*np.quantile(v, [0.75, 0.25])))
        # Raw per-partition draws, kept so figures can be rebuilt without
        # re-running the nested loop (persisted to ps_partition_draws.npz).
        out[f"{est}_draws"] = v
    for key in ("cert", "mom_split", "sn_split",
                "mom_rej0", "mom_rejb", "sn_rej0", "sn_rejb"):
        out[key] = float(np.mean(cols[key]))
    return out


def exp_ps_partition(reps_D: int, B: int, n_jobs: int, seed: int = 808,
                     B_spot: int = 2000) -> None:
    """
    PS: how random is each block estimator, and how does that randomness
    decompose into partition choice vs the data themselves?

    Nested design (headline figure): per strength level gamma =
    sigma2_ZX/mu_ZX^2 (log axis, population value), D datasets x B random
    partitions per dataset (t(2.1) errors, n = 2000). Decomposition per
    estimator:
        V_part = mean over datasets of the within-dataset partition variance,
        V_samp = variance over datasets of the within-dataset mean,
        partition share = V_part / (V_part + V_samp).
    An IQR-based share (same construction with squared IQRs) is written to
    the CSV as an outlier-robust check, along with the median plug-in
    gamma_hat. Corroborating panels on the same axis: certificate frequency
    (all block means of ZX share a sign, the prop:mono_det diagnostic), CS
    split frequency, and rejection frequency at the false null beta0 = 0
    (power) and at the truth (size floor, <= delta everywhere).

    Spotlight figure: one FIXED dataset at three strength levels, B_spot
    partitions each -- the raw 'same data, different seed' distributions,
    with Mean IV as the partition-free reference.
    """
    n = 2000
    m_sim = n // inf.k_blocks(DELTA)
    c_sn = inf.rk_critical_value(inf.k_blocks(DELTA), DELTA)

    t0 = time.perf_counter()
    rows = []
    draws: dict[str, np.ndarray] = {}   # "mu<level>_<est>" -> (D, B) raw draws
    for mu in PS_MUS:
        seeds = np.random.SeedSequence(seed + int(1e4 * mu)).spawn(reps_D)
        res = Parallel(n_jobs=n_jobs)(
            delayed(_ps_dataset)(s, n, mu, DELTA, c_sn, B) for s in seeds
        )
        d = pd.DataFrame(res)
        # x-position: exact population gamma; the plug-in median is kept in
        # the CSV as the observable counterpart (right-skewed at weak strength)
        row = {"mu": mu, "gamma": BASE["sigma2_ZX"] / mu ** 2,
               "gamma_hat_median": float(d["gamma_hat"].median())}
        for est in ("mor", "rom", "cat"):
            v_part = float(d[f"{est}_pvar"].mean())
            v_samp = float(d[f"{est}_mean"].var(ddof=1))
            row[f"{est}_share"] = v_part / (v_part + v_samp)
            p_iqr = float((d[f"{est}_piqr"] ** 2).mean())
            s_iqr = float(np.subtract(*np.quantile(d[f"{est}_mean"], [0.75, 0.25])) ** 2)
            row[f"{est}_share_iqr"] = p_iqr / (p_iqr + s_iqr) if (p_iqr + s_iqr) > 0 else np.nan
            row[f"{est}_v_part"] = v_part
            row[f"{est}_v_samp"] = v_samp
        for key in ("cert", "mom_split", "sn_split",
                    "mom_rej0", "mom_rejb", "sn_rej0", "sn_rejb",
                    "ar_rej0", "ar_rejb"):
            row[key] = float(d[key].mean())
        # Decision instability: share of datasets whose reject/don't-reject
        # verdict at beta0 = 0 is NOT unanimous across the B partitions, i.e.
        # where the conclusion depends on the seed. The per-dataset entries
        # are rejection *rates* over partitions, so a value strictly inside
        # (0, 1) is a dataset on which the decision flipped.
        for pre in ("mom", "sn"):
            r = d[f"{pre}_rej0"].to_numpy()
            row[f"{pre}_unstable0"] = float(np.mean((r > 0.0) & (r < 1.0)))
        for est in ("mor", "rom", "cat"):
            draws[f"mu{mu:g}_{est}"] = np.stack(d[f"{est}_draws"].to_list())
        rows.append(row)
        print(f"  mu={mu:g}: gamma={row['gamma']:.1f}, "
              f"MoR share={row['mor_share']:.3f}, cert={row['cert']:.3f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(GRAPHS_DIR / "ps_partition_randomness.csv", index=False)
    # Raw draws are persisted so the figures can be redesigned without paying
    # for the nested loop again.
    _save_raw("ps_partition_draws", beta=np.array(BASE["beta"]), **draws)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- headline figure (F4): partition vs sampling noise, and decision
    # instability. Panel (a) is one small-multiple per strength level, each
    # pairing the within-dataset-centred partition deviations against the
    # dataset means centred at beta (so the second box carries bias as well
    # as sampling spread); panel (b) spans the full gamma sweep.
    beta_true = BASE["beta"]
    est_spec = [("mor", "Median-of-Ratios"), ("rom", "Ratio-of-Medians"),
                ("cat", "Catoni")]

    fig = plt.figure(figsize=(13.0, 7.1))
    gs = fig.add_gridspec(2, len(PS_BOX_MUS), height_ratios=[1.3, 1.0],
                          hspace=0.30, wspace=0.28)

    clipped = 0.0
    for col, mu in enumerate(PS_BOX_MUS):
        ax = fig.add_subplot(gs[0, col])
        data, colors, faces, centres = [], [], [], []
        for i_e, (est, label) in enumerate(est_spec):
            arr = draws[f"mu{mu:g}_{est}"]                 # (D, B)
            part = (arr - arr.mean(axis=1, keepdims=True)).ravel()
            samp = arr.mean(axis=1) - beta_true
            centre = 1.0 + 2.2 * i_e
            data += [part, samp]
            colors += [ESTIMATOR_COLORS[label]] * 2
            faces += [True, False]
            centres.append(centre)
        positions = [c + off for c in centres for off in (-0.42, 0.42)]

        # y-limits from the whisker extents rather than raw quantiles, so the
        # boxes and whiskers of every estimator fit inside the panel. Fliers
        # are not drawn (20,000 deviations per box merge into a solid bar).
        caps = []
        for v in data:
            q1, q3 = np.percentile(v, [25, 75])
            iqr = q3 - q1
            inner = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
            caps += [inner.min(), inner.max()] if inner.size else [q1, q3]
        lo, hi = min(caps), max(caps)
        pad = 0.10 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        clipped = max(clipped,
                      float(np.mean([np.mean((v < lo) | (v > hi)) for v in data])))
        ax.axhline(0.0, color="0.45", linestyle=(0, (5, 4)), linewidth=1.1, zorder=1)
        bp = ax.boxplot(data, positions=positions, widths=0.62,
                        patch_artist=True, showfliers=False, zorder=3,
                        medianprops=dict(color="black", linewidth=1.4),
                        whiskerprops=dict(color="0.3", linewidth=1.0),
                        capprops=dict(color="0.3", linewidth=1.0))
        for patch, color, filled in zip(bp["boxes"], colors, faces):
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)
            if filled:
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
            else:
                patch.set_facecolor("white")
        ax.set_xticks(centres)
        ax.set_xticklabels([lab.replace("-of-", "-of-\n") for _, lab in est_spec],
                           fontsize=9)
        ax.set_ylim(lo, hi)
        ax.yaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
        gamma = BASE["sigma2_ZX"] / mu ** 2
        ax.set_title(rf"$\gamma$ = {gamma:.3g}  ($\mu_{{ZX}}$ = {mu:g})", fontsize=11)
        if col == 0:
            ax.set_ylabel(r"Deviation from centre")
            handles = [Patch(facecolor="0.55", edgecolor="0.3", alpha=0.45,
                             label="Partition noise (within dataset)"),
                       Patch(facecolor="white", edgecolor="0.3",
                             label=r"Sampling noise (dataset means $-\ \beta$)")]
            ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")

    axI = fig.add_subplot(gs[1, :])
    g = tab["gamma"]
    for pre, label in (("mom", "MoM-AR (feasible)"), ("sn", "SN-AR")):
        axI.plot(g, tab[f"{pre}_unstable0"], marker="o", markersize=4.5,
                 linewidth=1.8, color=TEST_COLORS[label], label=label)
    axI.set_xscale("log")
    axI.set_xlabel(r"$\gamma = \sigma^2_{ZX}/\mu_{ZX}^2$ (log scale)")
    axI.set_ylabel("Share of datasets")
    axI.set_title(r"(b) Decision instability at $\beta_0 = 0$: verdict not unanimous "
                  r"across partitions", fontsize=11)
    axI.legend(frameon=False, fontsize=9)
    axI.yaxis.grid(True, color="0.88", linewidth=0.7)
    axI.set_axisbelow(True)

    fig.text(0.5, 0.955, "(a) Partition noise against sampling noise, by instrument strength",
             ha="center", fontsize=11)
    _note(fig, rf"Note. D = {reps_D} datasets $\times$ B = {B} partitions per level; n = {n}, "
               rf"t(2.1) errors, $\delta$ = {DELTA:g}, k = {inf.k_blocks(DELTA)}, m = {m_sim}. "
               rf"Panel (a): boxes and whiskers (1.5$\times$IQR); outliers beyond the "
               rf"whiskers are not drawn ({clipped:.1%} of points on average), and "
               rf"y-axes are per-level.")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.11)
    _save(fig, "ps_partition_randomness.png")

    # --- spotlight figure: one fixed dataset per level, seed-only randomness ---
    spot = Parallel(n_jobs=min(n_jobs if n_jobs > 0 else len(PS_SPOTLIGHT_MUS),
                               len(PS_SPOTLIGHT_MUS)))(
        delayed(_ps_spotlight_level)(np.random.SeedSequence(seed + 1 + i).spawn(1)[0],
                                     n, mu, DELTA, B_spot)
        for i, mu in enumerate(PS_SPOTLIGHT_MUS)
    )
    _save_raw("ps_spotlight",
              **{f"gamma{s['gamma']:g}_{est}": s[est]
                 for s in spot for est in ("mor", "rom", "cat")},
              **{f"gamma{s['gamma']:g}_iv": np.array(s["iv"]) for s in spot})
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(1, len(spot), figsize=(13.5, 4.4))
    for ax, s in zip(axes, spot):
        pooled = np.concatenate([s["mor"], s["rom"]])
        lo, hi = np.quantile(pooled, [0.005, 0.995])
        pad = 0.1 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 600)
        for est, label in (("mor", "Median-of-Ratios"), ("rom", "Ratio-of-Medians"),
                           ("cat", "Catoni")):
            v = s[est]
            color = ESTIMATOR_COLORS[label]
            if v.std() < 1e-3 * max(hi - lo, 1e-12):
                ax.axvline(v.mean(), color=color, linewidth=1.6,
                           label=f"{label} (near-constant)")
                continue
            dens = gaussian_kde(v)(grid)
            dens /= dens.max()
            ax.plot(grid, dens, color=color, linewidth=1.6, label=label)
            ax.fill_between(grid, dens, color=color, alpha=0.12)
        ax.axvline(BASE["beta"], color="0.25", linestyle=(0, (5, 4)), linewidth=1.2,
                   label=r"true $\beta$")
        ax.axvline(s["iv"], color="0.25", linestyle=(0, (1, 2)), linewidth=1.2,
                   label="Mean IV (partition-free)")
        ax.set_title(f"$\\gamma$ = {s['gamma']:.3g}", fontsize=12)
        ax.set_xlabel(r"$\hat\beta$")
        ax.yaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("density (scaled to max 1)")
    axes[0].legend(frameon=False, fontsize=8)
    _note(fig, f"Note. One fixed dataset per panel (n = {n}, t(2.1) errors), "
               f"B = {B_spot} random partitions: all variation within a panel is the "
               f"partition seed alone. Axes truncated at 0.5/99.5% quantiles.")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.19, wspace=0.18)
    _save(fig, "ps_partition_spotlight.png")
    print(f"[PS] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# AG: aggregated MoM-AR test (def:randomised_test, thm:agg_size)
# ----------------------------------------------------------------------------

# Same strength axis as the partition-randomness study, so the two figures are
# read on one scale: gamma = sigma2_ZX/mu_ZX^2 from 2.5 (strong) to 1000.
AG_MUS = PS_MUS
# Aggregation depths. All odd, so the majority vote of eq:agg_test coincides
# with the median-statistic form of prop:agg_median_form. B = 1 is the
# single-partition randomised test, i.e. the no-aggregation baseline.
AG_BS = [1, 5, 25, 101]
# Strength levels drawn in the power figure: strong, the transition where
# sss:ps finds the verdict seed-dependent, and weak.
AG_POWER_MUS = [1.0, 0.2, 0.1]
AG_B_COLORS = {1: "#8C8C8C", 5: "#9ECAE1", 25: "#4292C6", 101: "#08519C"}
AG_B_STYLES = {1: (0, (4, 3)), 5: "-", 25: "-", 101: "-"}
AG_TESTS = ["MoM-AR (oracle)", "MoM-AR (feasible)"]


def _ag_dataset(seed, n, mu, delta, grid, B_max):
    """
    One dataset of the aggregated design: draw the sample once, then B_max
    independent random partitions. Each partition contributes |W_tilde| over
    the whole beta0 grid and its own feasible threshold; the oracle threshold
    is partition-free.

    Returns (|W_tilde| as (B_max, len(grid)), feasible taus (B_max,), oracle tau).
    """
    rng = np.random.default_rng(seed)
    dgp = dict(BASE, mu_ZX=mu)
    df = generate_data(n=n, eps_Y_dist=("t", 2.1), rng=rng, **dgp)
    Y, X, Z = (df[c].to_numpy() for c in ("Y", "X", "Z"))
    k = inf.k_blocks(delta)

    W = np.empty((B_max, grid.size))
    tau_feas = np.empty(B_max)
    for b in range(B_max):
        perm = rng.permutation(n)
        Yp, Xp, Zp = Y[perm], X[perm], Z[perm]
        a, bb, _ = inf.block_means(Yp, Xp, Zp, k, shuffle=False)
        W[b] = np.abs(inf.mom_ar_statistic(a, bb, grid))
        tau_feas[b] = inf.tau_n(
            inf.robust_sigma_Ze(Yp, Xp, Zp, delta, shuffle=False), n, delta)
    return W, tau_feas, inf.tau_n(np.sqrt(dgp["sigma2_Ze"]), n, delta)


def _ag_reject(W, tau_feas, tau_orac):
    """
    Aggregated verdicts at every depth in AG_BS from one dataset's partitions.

    eq:agg_test rejects when strictly more than half of the replicates do. The
    first B of the B_max partitions are themselves i.i.d. uniform, so a single
    pass serves every B. Returns {test: {B: bool array over the grid}}.
    """
    rej = {
        "MoM-AR (oracle)": W > tau_orac,
        "MoM-AR (feasible)": W > tau_feas[:, None],
    }
    return {t: {B: r[:B].mean(axis=0) > 0.5 for B in AG_BS} for t, r in rej.items()}


def exp_ag_aggregated(reps_D: int, n_jobs: int, seed: int = 1010) -> None:
    """
    AG: size and power of the aggregated MoM-AR test (def:randomised_test).

    Follows on from exp_ps_partition. sss:ps shows the reject/don't-reject
    verdict depending on the seed on a majority of datasets once the instrument
    weakens; the aggregated test removes that dependence by majority vote over
    B partitions. Both figures sweep the same strength axis, with t(2.1) errors
    and n = 2000 as in exp_ps_partition, and one line per aggregation depth.

    Size is read at beta0 = beta off the same beta0 grid as the power curves,
    so one pass over the partitions serves both. The reference lines mark delta
    and the 2*delta ceiling of thm:agg_size, which is the level the aggregated
    test is guaranteed against -- not delta.
    """
    n = 2000
    beta = BASE["beta"]
    B_max = max(AG_BS)
    # W_tilde moves with beta0 at a rate set by the block means of Z*X, which
    # scale with mu_ZX, so a grid fixed in absolute terms saturates only under a
    # strong instrument. Widening it as 1/mu_ZX puts each strength level on its
    # own scale and lets the weak-instrument curves reach their plateau.
    n_grid = 31
    i0 = n_grid // 2            # symmetric odd grid: the midpoint is beta exactly

    t0 = time.perf_counter()
    size = {t: {B: [] for B in AG_BS} for t in AG_TESTS}
    power: dict = {}
    grids: dict = {}
    rows, prows = [], []
    for mu in AG_MUS:
        grid = beta + np.linspace(-0.6 / mu, 0.6 / mu, n_grid)
        grids[mu] = grid
        seeds = np.random.SeedSequence(seed + int(1e4 * mu)).spawn(reps_D)
        res = Parallel(n_jobs=n_jobs)(
            delayed(_ag_dataset)(s, n, mu, DELTA, grid, B_max) for s in seeds
        )
        agg = [_ag_reject(*r) for r in res]
        gamma = BASE["sigma2_ZX"] / mu ** 2
        for t in AG_TESTS:
            for B in AG_BS:
                rate = np.mean([a[t][B] for a in agg], axis=0)   # over datasets
                power[(t, mu, B)] = rate
                size[t][B].append(float(rate[i0]))
                rows.append({"mu": mu, "gamma": gamma, "test": t, "B": B,
                             "size": float(rate[i0])})
                for b0, v in zip(grid, rate):
                    prows.append({"mu": mu, "gamma": gamma, "test": t, "B": B,
                                  "beta0_minus_beta": b0 - beta,
                                  "rejection_rate": float(v)})
        feas = size["MoM-AR (feasible)"]
        print(f"  mu={mu:g}: gamma={gamma:.1f}, size(feasible) "
              f"B=1 {feas[1][-1]:.4f} -> B={B_max} {feas[B_max][-1]:.4f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(GRAPHS_DIR / "ag_size.csv", index=False)
    pd.DataFrame(prows).to_csv(GRAPHS_DIR / "ag_power.csv", index=False)
    print(tab.pivot_table(index=["test", "gamma"], columns="B", values="size")
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # --- size figure ---
    gammas = [BASE["sigma2_ZX"] / mu ** 2 for mu in AG_MUS]
    fig, axes = plt.subplots(1, len(AG_TESTS), figsize=(11.0, 4.6), sharey=True)
    for ax, t in zip(axes, AG_TESTS):
        for B in AG_BS:
            ax.plot(gammas, size[t][B], marker="o", markersize=4.5, linewidth=1.8,
                    linestyle=AG_B_STYLES[B], color=AG_B_COLORS[B],
                    label=f"B = {B}" + (" (single partition)" if B == 1 else ""))
        ax.axhline(DELTA, color="0.4", linestyle=":", linewidth=1.2)
        ax.axhline(2 * DELTA, color="0.4", linestyle="-.", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma = \sigma^2_{ZX}/\mu_{ZX}^2$ (log scale)")
        ax.set_title(t, fontsize=12)
        ax.yaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"Rejection rate at $\beta_0 = \beta$")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for lvl, lab in ((DELTA, r"$\delta$"), (2 * DELTA, r"$2\delta$")):
        axes[-1].annotate(lab, xy=(gammas[-1], lvl), fontsize=9, color="0.35",
                          ha="right", va="bottom")
    _note(fig, f"Note. D = {reps_D} datasets per point; n = {n}, t(2.1) errors, "
               f"$\\delta$ = {DELTA:g}, k = {inf.k_blocks(DELTA)}. Dotted line $\\delta$, "
               f"dash-dotted line 2$\\delta$: the aggregated test is guaranteed against "
               f"2$\\delta$, not $\\delta$. B = 1 is the single-partition randomised test.")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.2, wspace=0.08)
    _save(fig, "ag_size.png")

    # --- power figure ---
    fig, axes = plt.subplots(len(AG_TESTS), len(AG_POWER_MUS),
                             figsize=(13.5, 7.6), sharey=True)
    for i, t in enumerate(AG_TESTS):
        for j, mu in enumerate(AG_POWER_MUS):
            ax = axes[i, j]
            for B in AG_BS:
                ax.plot(grids[mu] - beta, power[(t, mu, B)], linewidth=1.8,
                        linestyle=AG_B_STYLES[B], color=AG_B_COLORS[B],
                        label=f"B = {B}" + (" (single partition)" if B == 1 else ""))
            ax.axhline(DELTA, color="0.4", linestyle=":", linewidth=1.2)
            ax.yaxis.grid(True, color="0.88", linewidth=0.7)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(rf"$\gamma$ = {BASE['sigma2_ZX'] / mu ** 2:.3g}"
                             rf"  ($\mu_{{ZX}}$ = {mu:g})", fontsize=11)
            if i == len(AG_TESTS) - 1:
                ax.set_xlabel(r"$\beta_0 - \beta$")
            if j == 0:
                ax.set_ylabel(f"{t}\nRejection frequency", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper center")
    _note(fig, f"Note. D = {reps_D} datasets per curve; n = {n}, t(2.1) errors, "
               f"$\\delta$ = {DELTA:g} (dotted). B = 1 is the single-partition "
               f"randomised test; larger B votes over more partitions.")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.12,
                        wspace=0.08, hspace=0.12)
    _save(fig, "ag_power.png")
    print(f"[AG] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# AB: aggregated point estimator (def:randomised, thm:aggregation)
# ----------------------------------------------------------------------------

# Same strength axis and the same aggregation ladder as the aggregated test, so
# the estimator and the test figures of sss:agg are read on one scale.
AB_MUS = PS_MUS
AB_BS = AG_BS
AB_B_COLORS, AB_B_STYLES = AG_B_COLORS, AG_B_STYLES
AB_ESTS = ["Median-of-Ratios", "Ratio-of-Medians"]
# Permutations drawn per dataset. A multiple of max(AB_BS), so that at every
# depth the P draws split into R = P // B DISJOINT groups of size B. Each group
# is then an independent aggregate on the same data, and their spread is the
# seed dispersion of rem:agg_not_sampling: computational randomness, not
# uncertainty about beta.
AB_P = 5 * max(AB_BS)
# Distribution panel: the E1/E2 families at the canonical (strong) DGP.
AB_DISTS = ["Gaussian", "t(3)", "t(2.1)", "Pareto(2.5)", "Mix(0.1,3)"]
# The strength sweep has to fix one family; t(2.1) keeps it on the DGP of
# sss:ps and sss:agg, where the seed dependence being aggregated away was
# measured in the first place.
AB_STRENGTH_DIST = "t(2.1)"


def _ab_dataset(seed, n, mu, dist, delta, P):
    """
    One dataset of the aggregated-estimator design: draw the sample once, then
    P independent random permutations of its rows. Each permutation contributes
    one Median-of-Ratios and one Ratio-of-Medians estimate, both cut from the
    SAME reordering, so the two differ only through their block count k (the
    convention of the empirical replications).

    The block means are formed inline rather than through iv_estimate_mr /
    iv_estimate_rm: those rebuild a DataFrame and revalidate delta on every
    call, which dominates the runtime at P = AB_P permutations per dataset.
    verify() checks that the two agree exactly.

    Returns (MoR draws (P,), RoM draws (P,), partition-free Mean IV).
    """
    rng = np.random.default_rng(seed)
    df = generate_data(n=n, eps_Y_dist=dist, rng=rng, **dict(BASE, mu_ZX=mu))
    Z = df["Z"].to_numpy()
    ZY, ZX = Z * df["Y"].to_numpy(), Z * df["X"].to_numpy()

    k_mor = inf.k_blocks(delta)                        # ceil(8 ln(1/delta))
    k_rom = int(np.ceil(8 * np.log(2 / delta)))        # ceil(8 ln(2/delta))
    m_mor, m_rom = n // k_mor, n // k_rom

    mor, rom = np.empty(P), np.empty(P)
    for p in range(P):
        perm = rng.permutation(n)
        y, x = ZY[perm], ZX[perm]
        by = y[: k_mor * m_mor].reshape(k_mor, m_mor).mean(axis=1)
        bx = x[: k_mor * m_mor].reshape(k_mor, m_mor).mean(axis=1)
        mor[p] = np.median(by / bx)
        by = y[: k_rom * m_rom].reshape(k_rom, m_rom).mean(axis=1)
        bx = x[: k_rom * m_rom].reshape(k_rom, m_rom).mean(axis=1)
        rom[p] = np.median(by) / np.median(bx)
    return mor, rom, float(ZY.mean() / ZX.mean())


def _ab_aggregate(v: np.ndarray, B: int) -> np.ndarray:
    """
    Aggregated estimates at depth B from the P single-partition draws.

    def:randomised takes the median over B independent permutations. Given the
    data the P draws are i.i.d., so cutting them into R = P // B disjoint
    groups yields R independent aggregates per dataset (leftover draws are
    dropped). Input (D, P), output (D, R); B = 1 returns the draws unchanged,
    the no-aggregation baseline.
    """
    D, P = v.shape
    R = P // B
    return np.median(v[:, : R * B].reshape(D, R, B), axis=2)


def _ab_metrics(v: np.ndarray, beta: float, delta: float) -> dict[int, dict]:
    """
    Per-depth summary of one (configuration, estimator) cell from its (D, P)
    draws.

    q_dev   empirical (1-delta)-quantile of |beta_tilde_B - beta|, pooled over
            datasets and permutation groups. This is the deviation t of
            thm:rom / thm:mor, and thm:aggregation bounds exactly this joint
            probability over the sample and the seeds together.
    sd_part root of the mean over datasets of the within-dataset variance of
            the aggregates: what survives of the seed dependence at depth B.
    share   partition variance as a fraction of the total, the decomposition
            exp_ps_partition uses, so the two sections report one quantity.
    fail_t1 fraction of aggregates further from beta than the SINGLE-partition
            deviation q_dev(B=1). thm:aggregation bounds it by
            delta/lambda + exp(-2B(1/2-lambda)^2); the empirical counterpart
            says how loose that is.
    """
    agg = {B: _ab_aggregate(v, B) for B in AB_BS}
    t1 = float(np.quantile(np.abs(agg[1] - beta), 1.0 - delta))
    out = {}
    for B, a in agg.items():
        err = a - beta
        v_part = float(np.mean(a.var(axis=1, ddof=1))) if a.shape[1] > 1 else 0.0
        v_samp = float(a.mean(axis=1).var(ddof=1))
        out[B] = {
            "R": a.shape[1],
            "q_dev": float(np.quantile(np.abs(err), 1.0 - delta)),
            "mad_err": float(np.median(np.abs(err))),
            "median_bias": float(np.median(err)),
            "sd_part": float(np.sqrt(v_part)),
            "sd_samp": float(np.sqrt(v_samp)),
            "share": v_part / (v_part + v_samp) if (v_part + v_samp) > 0 else np.nan,
            "fail_t1": float(np.mean(np.abs(err) > t1)),
        }
    return out


def _ab_run(reps_D, n, mu, dist, n_jobs, seed):
    """D datasets at one (mu, distribution) cell; returns the (D, P) draws."""
    seeds = np.random.SeedSequence(seed).spawn(reps_D)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_ab_dataset)(s, n, mu, DISTS[dist], DELTA, AB_P) for s in seeds
    )
    mor, rom, iv = (np.asarray(c) for c in zip(*res))
    return {"Median-of-Ratios": mor, "Ratio-of-Medians": rom}, iv


def _ab_note(fig, text: str, width: int) -> None:
    """_note for a note too long for one line; `width` is in characters."""
    _note(fig, "\n".join(textwrap.wrap(text, width)))


def _ab_bs_axis(ax) -> None:
    """Shared x-axis for the depth panels: log scale, ticks at the depths run."""
    ax.set_xscale("log")
    ax.set_xticks(AB_BS)
    ax.set_xticklabels([str(B) for B in AB_BS])
    ax.minorticks_off()


def exp_ab_aggregated(reps_D: int, n_jobs: int, seed: int = 1212) -> None:
    """
    AB: the aggregated point estimator of def:randomised, across instrument
    strength and across error distributions.

    Companion to exp_ag_aggregated, which does the same for the test. Both
    sweep gamma = sigma2_ZX/mu_ZX^2 on one axis and share the aggregation
    ladder B, and both ask two questions of every cell: what the aggregation
    costs in accuracy (the (1-delta)-deviation quantile, the t of thm:rom and
    thm:mor) and what it buys in stability (the dispersion of the estimate
    across seeds on one fixed dataset, the object measured in sss:ps).

    Figure ab_strength.png sweeps gamma at t(2.1) errors; the shading marks
    where eq:mor_strength and eq:rom_strength stop holding at this n. Figure
    ab_dists.png fixes the canonical strong DGP and sweeps the error family.
    The partition-free Mean IV is the reference in both.

    The raw (D, P) draws are not written to output/raw: at the full counts they
    are some 200 MB of incompressible floats, and the driver is reproducible
    from its fixed seed.
    """
    n = 2000
    beta = BASE["beta"]
    k_mor, k_rom = inf.k_blocks(DELTA), int(np.ceil(8 * np.log(2 / DELTA)))
    m_mor, m_rom = n // k_mor, n // k_rom
    # Largest gamma at which each estimator's finite-sample strength condition
    # still holds at this n: eq:mor_strength is m >= 32 gamma, eq:rom_strength
    # is m > 4 gamma.
    gamma_max = {"Median-of-Ratios": m_mor / 32.0, "Ratio-of-Medians": m_rom / 4.0}

    t0 = time.perf_counter()
    rows = []
    bmax = max(AB_BS)

    # --- (1) instrument strength sweep -------------------------------------
    strength: dict = {}
    iv_strength: dict = {}
    for mu in AB_MUS:
        gamma = BASE["sigma2_ZX"] / mu ** 2
        draws, iv = _ab_run(reps_D, n, mu, AB_STRENGTH_DIST, n_jobs,
                            seed + int(1e4 * mu))
        iv_strength[mu] = float(np.quantile(np.abs(iv - beta), 1.0 - DELTA))
        for est, v in draws.items():
            met = _ab_metrics(v, beta, DELTA)
            strength[(est, mu)] = met
            for B, r in met.items():
                rows.append(dict(figure="strength", dist=AB_STRENGTH_DIST, mu=mu,
                                 gamma=gamma, estimator=est, B=B, **r))
        mo = strength[("Median-of-Ratios", mu)]
        print(f"  mu={mu:g}: gamma={gamma:7.1f}  MoR q_dev "
              f"B=1 {mo[1]['q_dev']:.4f} -> B={bmax} {mo[bmax]['q_dev']:.4f}"
              f"   seed SD {mo[1]['sd_part']:.4f} -> {mo[bmax]['sd_part']:.4f}")

    # --- (2) error-distribution panel at the canonical DGP ------------------
    dists: dict = {}
    iv_dists: dict = {}
    for dname in AB_DISTS:
        draws, iv = _ab_run(reps_D, n, BASE["mu_ZX"], dname, n_jobs,
                            seed + _stable_hash(dname) % 1000)
        iv_dists[dname] = float(np.quantile(np.abs(iv - beta), 1.0 - DELTA))
        for est, v in draws.items():
            met = _ab_metrics(v, beta, DELTA)
            dists[(est, dname)] = met
            for B, r in met.items():
                rows.append(dict(figure="dists", dist=dname, mu=BASE["mu_ZX"],
                                 gamma=BASE["sigma2_ZX"] / BASE["mu_ZX"] ** 2,
                                 estimator=est, B=B, **r))
        mo = dists[("Median-of-Ratios", dname)]
        print(f"  {dname:<12}: MoR q_dev B=1 {mo[1]['q_dev']:.4f} -> "
              f"B={bmax} {mo[bmax]['q_dev']:.4f}   seed SD "
              f"{mo[1]['sd_part']:.4f} -> {mo[bmax]['sd_part']:.4f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(GRAPHS_DIR / "ab_aggregated.csv", index=False)
    for metric in ("q_dev", "sd_part", "fail_t1"):
        print(f"\n--- {metric} ---")
        print(tab[tab.figure == "strength"]
              .pivot_table(index=["estimator", "gamma"], columns="B", values=metric)
              .to_string(float_format=lambda v: f"{v:.4f}"))
        print(tab[tab.figure == "dists"]
              .pivot_table(index=["estimator", "dist"], columns="B", values=metric)
              .to_string(float_format=lambda v: f"{v:.4f}"))

    # --- figure 1: instrument strength -------------------------------------
    gammas = [BASE["sigma2_ZX"] / mu ** 2 for mu in AB_MUS]
    fig, axes = plt.subplots(2, len(AB_ESTS), figsize=(11.0, 7.4), sharex=True)
    for col, est in enumerate(AB_ESTS):
        for row, metric in enumerate(("q_dev", "sd_part")):
            ax = axes[row, col]
            # Right of this rule the estimator's strength condition fails.
            ax.axvspan(gamma_max[est], gammas[-1] * 1.6, color="0.94", zorder=0)
            ax.axvline(gamma_max[est], color="0.55", linestyle=(0, (5, 4)),
                       linewidth=1.1, zorder=1)
            for B in AB_BS:
                ax.plot(gammas, [strength[(est, mu)][B][metric] for mu in AB_MUS],
                        marker="o", markersize=4.0, linewidth=1.8,
                        linestyle=AB_B_STYLES[B], color=AB_B_COLORS[B], zorder=3,
                        label=f"B = {B}" + (" (single partition)" if B == 1 else ""))
            if metric == "q_dev":
                ax.plot(gammas, [iv_strength[mu] for mu in AB_MUS], linewidth=1.4,
                        linestyle=(0, (1, 2)), color="0.35", zorder=3,
                        label="Mean IV (partition-free)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(gammas[0] / 1.4, gammas[-1] * 1.6)
            ax.yaxis.grid(True, color="0.88", linewidth=0.7)
            ax.set_axisbelow(True)
            if row == 0:
                cond = (r"$m \geq 32\gamma$" if est == "Median-of-Ratios"
                        else r"$m > 4\gamma$")
                ax.set_title(f"{est}\n" + cond +
                             rf" holds for $\gamma \leq$ {gamma_max[est]:.3g}"
                             r" (unshaded)", fontsize=11)
            else:
                ax.set_xlabel(r"$\gamma = \sigma^2_{ZX}/\mu_{ZX}^2$ (log scale)")
    axes[0, 0].set_ylabel(r"(a) $(1-\delta)$-quantile of $|\tilde\beta_B-\beta|$")
    axes[1, 0].set_ylabel(r"(b) SD across seeds, one dataset")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    _ab_note(fig, f"Note. D = {reps_D} datasets $\\times$ P = {AB_P} permutations per "
                  f"point; n = {n}, t(2.1) errors, $\\delta$ = {DELTA:g}, k = {k_mor} "
                  f"(MoR) and {k_rom} (RoM). Panel (a) pools datasets and seeds, the "
                  f"joint probability the aggregation theorem bounds; panel (b) is the "
                  f"within-dataset dispersion of $\\tilde\\beta_B$ across independent "
                  f"groups of B permutations, what is left of the seed dependence. "
                  f"Shaded: the estimator's finite-sample strength condition fails. "
                  f"Both axes log.",
             width=145)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.90, bottom=0.175,
                        wspace=0.14, hspace=0.10)
    _save(fig, "ab_strength.png")

    # --- figure 2: error distributions at the canonical DGP -----------------
    # sharey per row: every panel is the same DGP under a different error
    # family, so one scale per row is what makes the families comparable.
    fig, axes = plt.subplots(2, len(AB_DISTS), figsize=(15.5, 6.8), sharex=True,
                             sharey="row")
    for col, dname in enumerate(AB_DISTS):
        for row, metric in enumerate(("q_dev", "sd_part")):
            ax = axes[row, col]
            for est in AB_ESTS:
                ax.plot(AB_BS, [dists[(est, dname)][B][metric] for B in AB_BS],
                        marker="o", markersize=4.5, linewidth=1.8,
                        color=ESTIMATOR_COLORS[est], label=est)
            if metric == "q_dev":
                ax.axhline(iv_dists[dname], color="0.35", linestyle=(0, (1, 2)),
                           linewidth=1.4, label="Mean IV (partition-free)")
            else:
                # 1/sqrt(B) guide anchored at the single-partition dispersion:
                # the slope a mean of B i.i.d. draws would follow exactly, and
                # a median of B up to the efficiency factor of the median.
                ref = dists[(AB_ESTS[0], dname)][1]["sd_part"]
                ax.plot(AB_BS, [ref / np.sqrt(B) for B in AB_BS], color="0.55",
                        linestyle=(0, (5, 4)), linewidth=1.2,
                        label=r"$1/\sqrt{B}$ reference")
            _ab_bs_axis(ax)
            # Row (a) spans less than one decade at this strength, so a linear
            # scale reads better; row (b) spans two, and log-log is where the
            # 1/sqrt(B) reference is a straight line.
            if row == 1:
                ax.set_yscale("log")
            ax.yaxis.grid(True, color="0.88", linewidth=0.7)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(f"$\\varepsilon_Y \\sim$ {dname}", fontsize=11)
            else:
                ax.set_xlabel("aggregation depth $B$")
    axes[0, 0].set_ylabel(r"(a) $(1-\delta)$-quantile of $|\tilde\beta_B-\beta|$",
                          fontsize=11)
    axes[1, 0].set_ylabel(r"(b) SD across seeds, one dataset", fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower left")
    axes[1, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
    _ab_note(fig, f"Note. D = {reps_D} datasets $\\times$ P = {AB_P} permutations per "
                  f"point; n = {n}, $\\mu_{{ZX}}$ = {BASE['mu_ZX']:g} ($\\gamma$ = "
                  f"{BASE['sigma2_ZX'] / BASE['mu_ZX'] ** 2:g}, every strength condition "
                  f"holds), $\\rho$ = {BASE['rho']:g}, $\\delta$ = {DELTA:g}. B = 1 is the "
                  f"single randomised partition, and the $1/\\sqrt{{B}}$ guide is anchored "
                  f"at it. Scales are shared within a row; row (b) is log-log.",
             width=195)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.175,
                        wspace=0.22, hspace=0.10)
    _save(fig, "ab_dists.png")
    print(f"[AB] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# I5: R_k critical value table (appendix sec:artable)
# ----------------------------------------------------------------------------


def exp_i5_rk_table(n_sims: int = 1_000_000) -> None:
    ks = [5, 10, 15, 19, 20, 24, 25, 30, 37, 40, 50]
    deltas = [0.10, 0.05, 0.025, 0.01]
    t0 = time.perf_counter()
    rows = []
    for k in ks:
        draws = inf.simulate_rk(k, n_sims=n_sims)
        row = {"k": k}
        for d in deltas:
            row[f"{1-d:.3f}"] = float(np.quantile(draws, 1 - d))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("k")
    df.to_csv(GRAPHS_DIR / "i5_rk_critical_values.csv")
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))

    # LaTeX table for the appendix (sec:artable)
    lines = [
        "% Auto-generated by Code/experiments.py (exp_i5_rk_table);",
        f"% {n_sims:,} simulations per k; R_k = |med(xi)|/MAD(xi), xi_j iid N(0,1),",
        "% med/MAD use the rank-ceil(k/2) convention of Theorem thm:piecewise.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Simulated critical values $c_{k,\\delta}$: the $(1-\\delta)$ quantiles of "
        "$R_k$ (Proposition~\\ref{prop:sn_pivotal}).}",
        "\\label{tab:rk_critical_values}",
        "\\begin{tabular}{r" + "r" * len(deltas) + "}",
        "\\toprule",
        "$k$ & " + " & ".join(f"$\\delta={d:g}$" for d in deltas) + " \\\\",
        "\\midrule",
    ]
    for k in ks:
        vals = " & ".join(f"{df.loc[k, f'{1-d:.3f}']:.3f}" for d in deltas)
        lines.append(f"{k} & {vals} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out = GRAPHS_DIR / "i5_rk_table.tex"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")
    print(f"[I5] runtime {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------


def verify(n_jobs: int) -> None:
    """Internal consistency checks; all must pass before trusting results."""
    # 1. Exact CS vs brute-force grid membership.
    inf.verify_cs_exactness(n_cases=200)

    # 2. SN-CS (grid+bisection) endpoints satisfy T(endpoint) ~= c.
    rng = np.random.default_rng(7)
    for _ in range(50):
        k = 24
        a, b = rng.standard_normal(k), rng.standard_normal(k) + 1.0
        c = 2.0
        cs = inf.sn_ar_cs(a, b, c)
        for lo, hi in cs:
            for e in (lo, hi):
                if np.isfinite(e):
                    T = float(inf.sn_statistic(a, b, e))
                    assert abs(T - c) < 1e-4, f"SN-CS endpoint mismatch: T={T}, c={c}"
    print("SN-CS endpoint check: OK (50 instances)")

    # 3. SN-CS membership agrees with the SN test on a grid.
    for _ in range(20):
        a, b = rng.standard_normal(24), rng.standard_normal(24) + 1.0
        c = 2.0
        cs = inf.sn_ar_cs(a, b, c)
        grid = np.linspace(-10, 10, 801)
        T = inf.sn_statistic(a, b, grid)
        for x, t in zip(grid, T):
            claimed = inf.cs_contains(cs, x)
            inside = t <= c
            if claimed != inside:
                near = any(np.isfinite(e) and abs(x - e) < 1e-6 for iv in cs for e in iv)
                assert near, f"SN-CS/test disagreement at beta0={x}"
    print("SN-CS membership check: OK (20 instances)")

    # 4. Catoni on Gaussian data ~ sample mean (psi nearly linear near 0).
    x = rng.standard_normal(5000) * 3.0 + 1.0
    from simulation import catoni_mean
    assert abs(catoni_mean(x, 0.05) - x.mean()) < 0.05
    print("Catoni sanity check: OK")

    # 5. Standard AR CS matches test inversion on a grid.
    df = generate_data(n=500, rng=rng, **BASE)
    cs = inf.standard_ar_cs(df, delta=0.05)
    grid = np.linspace(-3, 5, 1601)
    rej = inf.standard_ar_test(df, grid, delta=0.05)["reject"]
    for x, r in zip(grid, rej):
        claimed = inf.cs_contains(cs, x)
        if claimed != (not r):
            near = any(np.isfinite(e) and abs(x - e) < 1e-3 for iv in cs for e in iv)
            assert near, f"standard AR CS/test disagreement at beta0={x}"
    print("Standard AR CS check: OK")
    # 6. _ab_dataset's inline block estimators == iv_estimate_mr / _rm.
    mor, rom, iv = _ab_dataset(np.random.SeedSequence(7), 2000, BASE["mu_ZX"],
                               ("t", 2.1), DELTA, 3)
    # Re-draw the same dataset and permutations to compare like with like.
    r2 = np.random.default_rng(np.random.SeedSequence(7))
    d2 = generate_data(n=2000, eps_Y_dist=("t", 2.1), rng=r2, **BASE)
    for p_i in range(3):
        dp = d2.iloc[r2.permutation(len(d2))].reset_index(drop=True)
        assert np.isclose(iv_estimate_mr(dp, delta=DELTA, shuffle=False)["beta_hat"],
                          mor[p_i], rtol=0, atol=1e-12), (p_i, mor[p_i])
        assert np.isclose(iv_estimate_rm(dp, delta=DELTA, shuffle=False)["beta_hat"],
                          rom[p_i], rtol=0, atol=1e-12), (p_i, rom[p_i])
    assert np.isclose(iv_estimate(d2)["beta_hat"], iv)
    # Disjoint-group aggregation: at B = 1 the draws are returned unchanged,
    # and at B = P the single aggregate is the median of every draw.
    v = rng.standard_normal((4, 10))
    assert np.array_equal(_ab_aggregate(v, 1), v)
    assert np.allclose(_ab_aggregate(v, 10)[:, 0], np.median(v, axis=1))
    print("AB aggregated-estimator check: OK (3 permutations)")

    # 7. Aggregated CS (cor:agg_cs) against a dense grid, on random unions of
    #    closed intervals including unbounded ones.
    for _ in range(200):
        B = int(rng.integers(1, 8))
        cs_list = []
        for _ in range(B):
            n_iv = int(rng.integers(0, 4))
            pts = np.sort(rng.uniform(-5, 5, 2 * n_iv))
            cs = inf._merge_intervals([(pts[2 * i], pts[2 * i + 1]) for i in range(n_iv)])
            if n_iv and rng.random() < 0.25:
                cs = [(-np.inf, cs[0][1])] + cs[1:]
            if n_iv and rng.random() < 0.25:
                cs = cs[:-1] + [(cs[-1][0], np.inf)]
            cs_list.append(cs)
        agg = inf.aggregate_cs(cs_list)
        grid = np.linspace(-7, 7, 4001)
        for x in grid:
            claimed = inf.cs_contains(agg, x)
            inside = sum(inf.cs_contains(c, x) for c in cs_list) >= B / 2
            if claimed != inside:
                near = any(np.isfinite(e) and abs(x - e) < 1e-2
                           for iv in agg for e in iv)
                assert near, f"aggregate_cs disagreement at beta0={x}"
    print("Aggregated CS check: OK (200 instances)")

    print("All verification checks passed.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

ALL_EXPERIMENTS = ["e1e2", "e2b", "e3", "e4", "i1", "i2", "i3", "i4", "i5", "ps",
                   "ag", "ab"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true", help="thesis-quality replication counts")
    p.add_argument("--pilot", action="store_true", help="reduced counts (default)")
    p.add_argument("--only", type=str, default=None,
                   help=f"comma-separated subset of {ALL_EXPERIMENTS}")
    p.add_argument("--verify", action="store_true", help="run consistency checks only")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    if args.verify:
        verify(args.n_jobs)
        return

    full = args.full
    counts = rep_counts(full)
    todo = args.only.split(",") if args.only else ALL_EXPERIMENTS
    print(f"Mode: {'FULL' if full else 'PILOT'}; experiments: {todo}; counts: {counts}")

    t0 = time.perf_counter()
    if "e1e2" in todo:
        exp_e1_e2(counts["box"], args.n_jobs)
    if "e2b" in todo:
        exp_e2b_tails(counts["tail"], args.n_jobs)
    if "e3" in todo:
        exp_e3_strength(counts["box"], args.n_jobs)
    if "e4" in todo:
        exp_e4_moments(counts["moments"], args.n_jobs)
    if "i1" in todo:
        exp_i1_size(counts["size"], args.n_jobs)
    if "i2" in todo:
        exp_i2_power(counts["power"], args.n_jobs)
    if "i3" in todo:
        exp_i3_cs(counts["cs"], args.n_jobs)
    if "i4" in todo:
        exp_i4_mono(counts["mono"], args.n_jobs)
    if "i5" in todo:
        exp_i5_rk_table(n_sims=1_000_000 if full else 200_000)
    if "ps" in todo:
        exp_ps_partition(counts["ps_D"], counts["ps_B"], args.n_jobs,
                         B_spot=counts["ps_Bspot"])
    if "ag" in todo:
        exp_ag_aggregated(counts["ag_D"], args.n_jobs)
    if "ab" in todo:
        exp_ab_aggregated(counts["ab_D"], args.n_jobs)
    print(f"\nTotal runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
