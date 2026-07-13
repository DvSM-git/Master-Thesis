"""
Monte Carlo experiment suite for the simulation section of the thesis.

Each experiment maps to specific theoretical results in Paper/iteration4:

  E1   Gaussian, strong instrument: standard IV most efficient
       -> Thm thm:iv (the "price of robustness" of MoR/RoM/Catoni)
  E2   Heavy tails (t3, t2.1, Pareto 2.5): MoM estimators beat standard IV
       -> Thm thm:rom, thm:mor; Remarks rem:iv_polynomial, rem:mor_logarithmic
  E2b  Empirical deviation quantiles vs ln(1/delta): polynomial vs
       logarithmic delta-dependence -> rem:iv_polynomial vs rem:mor_logarithmic
  E3   Instrument-strength sweep: degradation when eq:rom_strength /
       eq:mor_strength fail
  I1   Size of MoM-AR (oracle & feasible), SN-AR, standard AR
       -> Thm thm:mom_ar_size, Prop prop:sn_pivotal
  I2   Power curves (rejection frequency vs beta0 - beta)
  I3   Confidence sets: coverage, length, components, unboundedness
       -> Thm thm:coverage, Cor cor:union, cor:sn_cs (exact Alg alg:mom_ar_cs)
  I4   Monotonicity / single-interval condition
       -> Prop prop:mono_det, prop:mono_cheby
  I5   R_k critical value table for the appendix (sec:artable)
       -> Prop prop:sn_pivotal

Usage:
    python experiments.py --pilot            # reduced replication counts
    python experiments.py --full             # thesis-quality counts
    python experiments.py --full --only e1,i3
    python experiments.py --verify           # internal consistency checks
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: avoid Tk in background/parallel runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from simulation import (
    generate_data,
    iv_estimate,
    iv_estimate_catoni,
    iv_estimate_mr,
    iv_estimate_rm,
)
from simulation_study import ESTIMATOR_COLORS, GRAPHS_DIR, styled_boxplot  # noqa: F401 (rcParams side effect)
import inference as inf

DELTA = 0.05

# Canonical DGP: calibrated so that at n = 2000 every finite-sample condition
# of the point-estimator theorems holds (strong-instrument regime):
#   IV  (eq:iv_strength):  n >= 8*2.5/0.05      = 400
#   RoM (eq:rom_strength): m > 4*2.5  = 10,  k = 30, n > 300
#   MoR (eq:mor_strength): m >= 32*2.5 = 80, k = 24, n >= 1920
BASE = dict(beta=1.0, mu_ZX=1.0, sigma2_ZX=2.5, sigma2_Ze=1.0, rho=0.5)

# Error families for eps_Y (eps_X stays Gaussian so heavy tails enter only
# through Z*eps, the channel the theory is about). All standardised to unit
# variance, so sigma2_Ze is exact for every family.
DISTS = {
    "Gaussian": None,
    "t(3)": ("t", 3.0),
    "t(2.1)": ("t", 2.1),
    "Pareto(2.5)": ("pareto", 2.5),
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
        return dict(box=10_000, tail=50_000, size=10_000, power=5_000, cs=5_000, mono=2_000)
    return dict(box=1_000, tail=5_000, size=1_000, power=500, cs=500, mono=300)


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


def _point_rep(seed, n, dgp, dist, delta) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    df = generate_data(n=n, eps_Y_dist=dist, rng=rng, **dgp)
    return (
        iv_estimate(df)["beta_hat"],
        iv_estimate_rm(df, delta=delta, rng=rng)["beta_hat"],
        iv_estimate_mr(df, delta=delta, rng=rng)["beta_hat"],
        iv_estimate_catoni(df, delta=delta)["beta_hat"],
    )


def run_point_estimators(n, dgp, dist, n_reps, seed, n_jobs=-1, delta=DELTA) -> dict[str, np.ndarray]:
    seeds = np.random.SeedSequence(seed).spawn(n_reps)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_point_rep)(s, n, dgp, dist, delta) for s in seeds
    )
    iv, rm, mr, cat = (np.asarray(c) for c in zip(*res))
    return {"Mean IV": iv, "Ratio-of-Medians": rm, "Median-of-Ratios": mr, "Catoni": cat}


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
    rows += _summary_rows(est, beta, "Gaussian")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    styled_boxplot(ax, est, true_beta=beta)
    lo, hi, frac = _robust_ylim(list(est.values()))
    ax.set_ylim(lo, hi)
    _note(fig, f"Note. {reps} replications; n = {n}, Gaussian errors, "
               f"$\\mu_{{ZX}}$ = {BASE['mu_ZX']:g}, $\\rho$ = {BASE['rho']:g}, "
               f"$\\delta$ = {DELTA:g}. Axis truncated at 0.5/99.5% quantiles "
               f"({100*frac:.2f}% of points outside).")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.18)
    _save(fig, "e1_boxplot_gaussian.png")

    # --- E2: heavy tails, three panels ---
    heavy = ["t(3)", "t(2.1)", "Pareto(2.5)"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, dname in zip(axes, heavy):
        est = run_point_estimators(n, BASE, DISTS[dname], reps, seed + hash(dname) % 1000, n_jobs)
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
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.22, wspace=0.22)
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
        est = run_point_estimators(n, BASE, DISTS[dname], reps, seed + hash(dname) % 1000, n_jobs)
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
    fig, axes = plt.subplots(1, len(mus), figsize=(16.0, 4.6))
    for ax, mu in zip(axes, mus):
        dgp = dict(BASE, mu_ZX=mu)
        est = run_point_estimators(n, dgp, None, reps, seed + int(mu * 1000), n_jobs)
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
    fig.subplots_adjust(left=0.05, right=0.995, top=0.9, bottom=0.24, wspace=0.24)
    _save(fig, "e3_boxplot_strength.png")
    pd.DataFrame(rows).to_csv(GRAPHS_DIR / "e3_summary.csv", index=False)
    print(f"[E3] runtime {time.perf_counter()-t0:.1f}s")


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
    }


def _rejection_rates(n, dgp, dist, beta0_grid, n_reps, seed, n_jobs, delta=DELTA) -> dict[str, np.ndarray]:
    c_sn = inf.rk_critical_value(inf.k_blocks(delta), delta)  # precompute (cache is not process-safe)
    seeds = np.random.SeedSequence(seed).spawn(n_reps)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_tests_rep)(s, n, dgp, dist, beta0_grid, delta, c_sn) for s in seeds
    )
    return {name: np.mean([r[name] for r in res], axis=0) for name in res[0]}


def exp_i1_size(reps: int, n_jobs: int, seed: int = 404) -> None:
    """I1: empirical size at beta0 = beta across tails and sample sizes."""
    ns = [500, 2000, 8000]
    dists = ["Gaussian", "t(2.1)", "Pareto(2.5)"]
    beta = np.array([BASE["beta"]])
    rows = []

    t0 = time.perf_counter()
    fig, axes = plt.subplots(1, len(dists), figsize=(13.5, 4.2), sharey=True)
    for ax, dname in zip(axes, dists):
        rates = {name: [] for name in TEST_COLORS}
        for n in ns:
            rr = _rejection_rates(n, BASE, DISTS[dname], beta, reps,
                                  seed + hash((dname, n)) % 10_000, n_jobs)
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
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.2, wspace=0.1)
    _save(fig, "i1_size.png")
    pd.DataFrame(rows).to_csv(GRAPHS_DIR / "i1_size.csv", index=False)
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
        rr = _rejection_rates(n, BASE, DISTS[dname], grid, reps,
                              seed + hash(dname) % 10_000, n_jobs)
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
        seeds = np.random.SeedSequence(seed + hash(tag) % 10_000).spawn(reps)
        res = Parallel(n_jobs=n_jobs)(
            delayed(_cs_rep)(s, n, dgp, DISTS[dname], DELTA, c_sn) for s in seeds
        )
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
    print("All verification checks passed.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

ALL_EXPERIMENTS = ["e1e2", "e2b", "e3", "i1", "i2", "i3", "i4", "i5"]


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
    print(f"\nTotal runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
