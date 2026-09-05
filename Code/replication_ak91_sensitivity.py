"""
Partition-sensitivity study for the AK91 replication (replication_ak91.py).

All MoM-based procedures partition the sample into blocks after a random
shuffle of the rows; on real data that shuffle is the only source of
randomness. Because the finite-sample instrument strength conditions fail
for the quarter-of-birth instrument, block-level quantities are noisy
relative to |mu_ZX| and results can move materially across partitions.
This script quantifies that: it draws B independent permutations of the
data and, for each one (the SAME permutation for every procedure, blocks
differing only through each method's k), recomputes

    - the MoR, RoM and Catoni point estimates,
    - the feasible MoM-AR and SN-AR confidence sets and their rejection
      decisions at beta0 = 0 and beta0 = beta_OLS.

The Wald estimator and the standard AR test are partition-free and serve
as fixed references.

Outputs:
    Paper/images/graphs/ak91_seed_sensitivity.png
        (a) overlapping scaled KDEs of the estimator distributions across
            partitions, Wald and OLS as vertical references;
        (b) scaled KDEs of the lower/upper (outer) confidence set endpoints.
    Paper/iteration4/ak91_sensitivity.tex
        estimator dispersion + rejection/split frequency table.
    Paper/iteration4/ak91_aggregated.tex
        the aggregated estimators and aggregated confidence sets that answer
        that sensitivity (def:randomised, cor:agg_cs), by aggregation depth B.
    Code/output/ak91_sensitivity_seeds.csv
        one row per permutation (all raw numbers, for reuse).

Usage:
    python replication_ak91_sensitivity.py [--reps 1000] [--seed 1991]
                                           [--delta 0.05] [--jobs -1]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde

import inference as inf
from replication_ak91 import (
    DATA_PATH,
    _fmt_cs,
    _fmt_cs_tex,
    _tex_num,
    load_data,
    table3_panel_b,
    to_iv_frame,
)
from simulation import iv_estimate_catoni, iv_estimate_mr, iv_estimate_rm
from simulation_study import ESTIMATOR_COLORS, GRAPHS_DIR  # noqa: F401 (rcParams side effect)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path(__file__).resolve().parent / "output" / "ak91_sensitivity_seeds.csv"
TEX_PATH = ROOT / "Paper" / "iteration4" / "ak91_sensitivity.tex"
AGG_TEX_PATH = ROOT / "Paper" / "iteration4" / "ak91_aggregated.tex"
AGG_CSV_PATH = Path(__file__).resolve().parent / "output" / "ak91_aggregated.csv"
FIG_PATH = GRAPHS_DIR / "ak91_seed_sensitivity.png"

TEST_COLORS = {"MoM-AR (feasible)": "#4C72B0", "SN-AR": "#55A868"}


def one_partition(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    seed_pair: tuple[int, int],
    delta: float,
    ols_beta: float,
    c_crit: float,
) -> dict:
    """
    Everything partition-dependent for one permutation of the rows. The
    permutation is applied once; every procedure then runs with
    shuffle=False on the identical row order (replication_ak91 convention).
    """
    perm = np.random.default_rng(list(seed_pair)).permutation(len(Y))
    Yp, Xp, Zp = Y[perm], X[perm], Z[perm]
    df = pd.DataFrame({"Y": Yp, "X": Xp, "Z": Zp})

    out: dict = {"seed": seed_pair[1]}
    out["beta_mor"] = iv_estimate_mr(df, delta=delta, shuffle=False)["beta_hat"]
    out["beta_rom"] = iv_estimate_rm(df, delta=delta, shuffle=False)["beta_hat"]
    out["beta_catoni"] = iv_estimate_catoni(df, delta=delta)["beta_hat"]

    k = inf.k_blocks(delta)
    a, b, _ = inf.block_means(Yp, Xp, Zp, k, shuffle=False)
    sigma_hat = inf.robust_sigma_Ze(Yp, Xp, Zp, delta, shuffle=False)
    tau = inf.tau_n(sigma_hat, len(Yp), delta)
    out["sigma_hat_Ze"] = sigma_hat
    out["all_same_sign"] = bool(np.all(b > 0) or np.all(b < 0))

    beta0 = np.array([0.0, ols_beta])
    W = np.abs(inf.mom_ar_statistic(a, b, beta0))
    T = inf.sn_statistic(a, b, beta0)

    for name, cs, rej in (
        ("mom", inf.mom_ar_cs_exact(a, b, tau), W > tau),
        ("sn", inf.sn_ar_cs(a, b, c_crit), T > c_crit),
    ):
        out[f"{name}_lo"] = cs[0][0] if cs else np.nan
        out[f"{name}_hi"] = cs[-1][1] if cs else np.nan
        out[f"{name}_ncomp"] = len(cs)
        out[f"{name}_unbounded"] = bool(
            cs and (np.isinf(cs[0][0]) or np.isinf(cs[-1][1]))
        )
        out[f"{name}_length"] = (
            np.inf
            if out[f"{name}_unbounded"]
            else float(sum(hi - lo for lo, hi in cs))
        )
        out[f"{name}_rej0"] = bool(rej[0])
        out[f"{name}_rejOLS"] = bool(rej[1])
        # The whole set, not only its outer endpoints: cor:agg_cs needs every
        # component of every permutation's set to take the majority vote.
        out[f"{name}_cs"] = cs
    return out


# ----------------------------------------------------------------------------
# Aggregation over partitions (def:randomised, thm:aggregation, cor:agg_cs)
# ----------------------------------------------------------------------------

AGG_ESTS = [("beta_mor", "Median-of-Ratios"), ("beta_rom", "Ratio-of-Medians")]
AGG_TESTS = [("mom", "MoM-AR (feasible)"), ("sn", "SN-AR")]


def aggregate_over_depths(rows: list[dict], ols_beta: float,
                          depths: list[int] | None = None) -> pd.DataFrame:
    """
    Aggregated estimators and aggregated confidence sets built from the
    permutations already drawn by one_partition.

    By default the single depth B = len(rows) is used, so every permutation
    drawn enters one aggregate: that is the estimate and the set the case
    studies report. Passing `depths` cuts the permutations into R = len(rows)
    // B disjoint groups instead, each an independent aggregate on the same
    data, which is how the residual seed dependence at a shallower depth can be
    measured (the empirical counterpart of panel (b) of
    Figure~fig:ab_strength). Remark~rem:agg_not_sampling forbids reading that
    spread as uncertainty about beta.

    Point estimates aggregate by the median of def:randomised. Confidence sets
    aggregate by cor:agg_cs: beta0 survives when it lies in at least half of
    the group's per-permutation sets. Both the Wald estimator and the standard
    AR test are partition-free, so aggregation leaves them unchanged.

    Returns one row per depth, carrying the first group's aggregate and, when
    more than one group is available, the dispersion across groups.
    """
    if depths is None:
        depths = [len(rows)]
    out = []
    for B in depths:
        R = len(rows) // B
        if R == 0:
            print(f"  skipping B = {B}: fewer than B permutations drawn")
            continue
        groups = [rows[g * B:(g + 1) * B] for g in range(R)]
        rec: dict = {"B": B, "R": R}
        for key, _ in AGG_ESTS:
            vals = np.array([np.median([r[key] for r in grp]) for grp in groups])
            rec[f"{key}_hat"] = float(vals[0])
            rec[f"{key}_sd"] = float(vals.std(ddof=1)) if R > 1 else np.nan
            rec[f"{key}_min"] = float(vals.min())
            rec[f"{key}_max"] = float(vals.max())
        for key, _ in AGG_TESTS:
            css = [inf.aggregate_cs([r[f"{key}_cs"] for r in grp]) for grp in groups]
            rec[f"{key}_cs"] = css[0]
            rec[f"{key}_ncomp"] = len(css[0])
            rec[f"{key}_excl0"] = not inf.cs_contains(css[0], 0.0)
            rec[f"{key}_exclOLS"] = not inf.cs_contains(css[0], ols_beta)
            rec[f"{key}_unbounded"] = bool(css[0]) and (
                np.isinf(css[0][0][0]) or np.isinf(css[0][-1][1]))
            rec[f"{key}_length"] = (
                np.inf if rec[f"{key}_unbounded"]
                else float(sum(hi - lo for lo, hi in css[0])))
            rec[f"{key}_pct_excl0"] = 100.0 * float(
                np.mean([not inf.cs_contains(cs, 0.0) for cs in css]))
        out.append(rec)
    return pd.DataFrame(out)


def print_aggregated(agg: pd.DataFrame, wald: float, dp: int = 4) -> None:
    """Console view of aggregate_over_depths."""
    print(f"\nAggregated over partitions (Wald reference = {wald:.{dp}f}):")
    for _, r in agg.iterrows():
        extra = "" if int(r["R"]) == 1 else f", {int(r['R'])} independent groups"
        print(f"  B = {int(r['B']):>4}{extra}")
        for key, label in AGG_ESTS:
            sd = (f"  (SD across groups {r[f'{key}_sd']:.{dp}f})"
                  if np.isfinite(r[f"{key}_sd"]) else "")
            print(f"    {label:<18} {r[f'{key}_hat']:.{dp}f}{sd}")
        for key, label in AGG_TESTS:
            print(f"    {label:<18} {_fmt_cs(r[f'{key}_cs'])}"
                  f"  [{'excludes' if r[f'{key}_excl0'] else 'contains'} 0]")


# ----------------------------------------------------------------------------
# Aggregation and outputs
# ----------------------------------------------------------------------------


def _scaled_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE evaluated on grid, scaled to unit maximum so estimators
    with very different dispersions remain readable on one axis."""
    dens = gaussian_kde(values)(grid)
    return dens / dens.max()


def make_figure(res: pd.DataFrame, wald: float, ols: float,
                out_path: Path = FIG_PATH) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # (a) estimator distributions across partitions
    est_specs = [
        ("beta_mor", "Median-of-Ratios", ESTIMATOR_COLORS["Median-of-Ratios"]),
        ("beta_rom", "Ratio-of-Medians", ESTIMATOR_COLORS["Ratio-of-Medians"]),
        ("beta_catoni", "Catoni", ESTIMATOR_COLORS["Catoni"]),
    ]
    all_vals = np.concatenate([res[c].to_numpy() for c, _, _ in est_specs])
    span = all_vals.max() - all_vals.min()
    pad = 0.05 * span
    grid = np.linspace(all_vals.min() - pad, all_vals.max() + pad, 800)
    for col, label, color in est_specs:
        vals = res[col].to_numpy()
        if vals.std() < 1e-3 * span:
            # Effectively partition-free (e.g. Catoni, which depends on the
            # partition only through its variance pre-estimate): a KDE would
            # be degenerate, so draw a spike at the mean instead.
            ax1.axvline(vals.mean(), color=color, linewidth=1.6,
                        label=f"{label} (near-constant)")
            continue
        dens = _scaled_kde(vals, grid)
        ax1.plot(grid, dens, color=color, linewidth=1.6, label=label)
        ax1.fill_between(grid, dens, color=color, alpha=0.12)
    ax1.axvline(wald, color="0.25", linestyle=(0, (5, 4)), linewidth=1.2,
                label="Wald (partition-free)")
    ax1.axvline(ols, color="0.25", linestyle=(0, (1, 2)), linewidth=1.2,
                label="OLS")
    ax1.set_xlabel(r"$\hat{\beta}$")
    ax1.set_ylabel("density (scaled to max 1)")
    ax1.set_title("(a) Point estimates across partitions", fontsize=12)
    ax1.legend(fontsize=9, frameon=False)

    # (b) confidence set outer endpoints across partitions
    ep_specs = [
        ("mom_lo", "MoM-AR (feasible)", "-"),
        ("mom_hi", "MoM-AR (feasible)", "--"),
        ("sn_lo", "SN-AR", "-"),
        ("sn_hi", "SN-AR", "--"),
    ]
    finite = np.concatenate(
        [res[c].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
         for c, _, _ in ep_specs]
    )
    pad = 0.05 * (finite.max() - finite.min())
    grid2 = np.linspace(finite.min() - pad, finite.max() + pad, 800)
    seen = set()
    for col, test, ls in ep_specs:
        vals = res[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(vals) < 2:
            continue
        label = test if test not in seen else None
        seen.add(test)
        ax2.plot(grid2, _scaled_kde(vals, grid2), color=TEST_COLORS[test],
                 linestyle=ls, linewidth=1.6, label=label)
    ax2.axvline(wald, color="0.25", linestyle=(0, (5, 4)), linewidth=1.2)
    ax2.set_xlabel(r"$\beta_0$")
    ax2.set_ylabel("density (scaled to max 1)")
    ax2.set_title("(b) CS endpoints (solid: lower, dashed: upper)", fontsize=12)
    ax2.legend(fontsize=9, frameon=False)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.13, wspace=0.22)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def summarise(res: pd.DataFrame, wald: float, ols: float, ar_ref: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    est_rows = []
    for col, label in (
        ("beta_mor", "Median-of-Ratios"),
        ("beta_rom", "Ratio-of-Medians"),
        ("beta_catoni", "Catoni ratio"),
    ):
        v = res[col].to_numpy()
        est_rows.append(
            dict(estimator=label, median=float(np.median(v)), sd=v.std(ddof=1),
                 q05=np.quantile(v, 0.05), q95=np.quantile(v, 0.95),
                 min=v.min(), max=v.max())
        )
    est_tab = pd.DataFrame(est_rows)

    freq_rows = []
    for pre, label in (("mom", "MoM-AR (feasible)"), ("sn", "SN-AR")):
        lengths = res[f"{pre}_length"].to_numpy()
        freq_rows.append(
            dict(
                test=label,
                pct_reject_zero=100.0 * res[f"{pre}_rej0"].mean(),
                pct_reject_ols=100.0 * res[f"{pre}_rejOLS"].mean(),
                pct_split=100.0 * (res[f"{pre}_ncomp"] > 1).mean(),
                pct_unbounded=100.0 * res[f"{pre}_unbounded"].mean(),
                median_length=float(np.median(lengths)),
            )
        )
    freq_rows.append(
        dict(test="AR (standard)",
             pct_reject_zero=100.0 * ar_ref["rej0"],
             pct_reject_ols=100.0 * ar_ref["rejOLS"],
             pct_split=100.0 * ar_ref["split"],
             pct_unbounded=100.0 * ar_ref["unbounded"],
             median_length=ar_ref["length"])
    )
    freq_tab = pd.DataFrame(freq_rows)
    return est_tab, freq_tab


def write_tex(est_tab: pd.DataFrame, freq_tab: pd.DataFrame, B: int, wald: float) -> None:
    lines = [
        "% Auto-generated by Code/replication_ak91_sensitivity.py -- do not edit by hand.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Partition sensitivity on the AK91 extract: " f"$B = {B:,}$".replace(",", r"{,}") +
        r" independent random partitions, each shared by all MoM-based "
        r"procedures ($\delta = 0.05$). The partition-free Wald estimate is "
        f"${wald:.4f}$"
        r". Top panel: dispersion of the point estimates across partitions. "
        r"Bottom panel: rejection and shape frequencies of the 95\% "
        r"confidence sets; the standard AR test does not depend on the "
        r"partition and is shown for reference.}",
        r"\label{tab:ak91_sensitivity}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Estimator & Median & SD & Q$_{0.05}$ & Q$_{0.95}$ & Range \\",
        r"\midrule",
    ]
    for _, r in est_tab.iterrows():
        lines.append(
            f"{r['estimator']} & {r['median']:.4f} & {r['sd']:.4f} & "
            f"{r['q05']:.4f} & {r['q95']:.4f} & "
            f"[{r['min']:.4f}, {r['max']:.4f}] \\\\"
        )
    lines += [
        r"\midrule",
        r"Test & \multicolumn{1}{c}{\% rej.\ $0$} & "
        r"\multicolumn{1}{c}{\% rej.\ $\beta_{\mathrm{OLS}}$} & "
        r"\multicolumn{1}{c}{\% split} & \multicolumn{1}{c}{\% unb.} & "
        r"\multicolumn{1}{c}{Med.\ len.} \\",
        r"\midrule",
    ]
    for _, r in freq_tab.iterrows():
        ln = "$\\infty$" if np.isinf(r["median_length"]) else f"{r['median_length']:.3f}"
        lines.append(
            f"{r['test']} & {r['pct_reject_zero']:.1f} & {r['pct_reject_ols']:.1f} & "
            f"{r['pct_split']:.1f} & {r['pct_unbounded']:.1f} & {ln} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TEX_PATH}")


def write_agg_tex(agg: pd.DataFrame, B_total: int, wald: float,
                  ar_cs: list[tuple[float, float]], k_mor: int, k_rom: int) -> None:
    """
    The AK91 aggregated table, laid out as Table~tab:ak91_robust so the two can
    be read side by side: that one is a single partition, this one is the
    aggregate over all B of them.
    """
    r = agg.iloc[-1]
    lines = [
        "% Auto-generated by Code/replication_ak91_sensitivity.py -- do not edit by hand.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Aggregated estimates and the aggregated 95\% confidence set "
        r"on the AK91 extract, over all $B = " + _tex_num(B_total) +
        r"$ random partitions of Table~\ref{tab:ak91_sensitivity} "
        r"(Definition~\ref{def:randomised} and Corollary~\ref{cor:agg_cs}, "
        r"$\delta = 0.05$; " f"$k = {k_mor}$ for MoR and the tests, $k = {k_rom}$ "
        r"for RoM). Table~\ref{tab:ak91_robust} is the same quantities on a "
        r"single partition. The Wald estimator and the standard AR test do not "
        r"depend on the partition and are shown for reference; by "
        r"Corollary~\ref{cor:agg_cs} the aggregated set is guaranteed at "
        r"$2\delta$, not $\delta$.}",
        r"\label{tab:ak91_aggregated}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Aggregated estimator & $\widetilde{\beta}_B$ \\",
        r"\midrule",
    ]
    for key, label in AGG_ESTS:
        lines.append(f"{label} & {r[f'{key}_hat']:.4f} \\\\")
    lines += [
        f"Wald / Mean IV (partition-free) & {wald:.4f} \\\\",
        r"\midrule",
        r"Aggregated test & 95\% confidence set \\",
        r"\midrule",
    ]
    for key, label in AGG_TESTS:
        lines.append(f"{label} & {_fmt_cs_tex(r[f'{key}_cs'])} \\\\")
    lines += [
        f"AR (standard, partition-free) & {_fmt_cs_tex(ar_cs)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    AGG_TEX_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {AGG_TEX_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1991)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--data", type=Path, default=DATA_PATH)
    args = p.parse_args()

    df = load_data(args.data)
    rep = table3_panel_b(df)
    iv_df = to_iv_frame(df)
    Y = iv_df["Y"].to_numpy()
    X = iv_df["X"].to_numpy()
    Z = iv_df["Z"].to_numpy()

    k = inf.k_blocks(args.delta)
    c_crit = inf.rk_critical_value(k, args.delta)  # simulate/cache before forking workers

    # Partition-free references: Wald, OLS, standard AR.
    ar_cs = inf.standard_ar_cs(iv_df, delta=args.delta)
    ar_test = inf.standard_ar_test(iv_df, np.array([0.0, rep["ols"]]), delta=args.delta)
    ar_ref = dict(
        rej0=bool(ar_test["reject"][0]), rejOLS=bool(ar_test["reject"][1]),
        split=len(ar_cs) > 1,
        unbounded=any(np.isinf(lo) or np.isinf(hi) for lo, hi in ar_cs),
        length=float(sum(hi - lo for lo, hi in ar_cs)),
    )

    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_partition)(Y, X, Z, (args.seed, i), args.delta, rep["ols"], c_crit)
        for i in range(args.reps)
    )
    res = pd.DataFrame(rows)
    print(f"{args.reps} partitions in {time.time() - t0:.1f}s")

    CSV_PATH.parent.mkdir(exist_ok=True)
    # The raw confidence sets are kept in memory for the aggregation below;
    # the per-seed CSV keeps the scalar columns it has always had.
    res.drop(columns=[c for c in res.columns if c.endswith("_cs")]).to_csv(
        CSV_PATH, index=False)
    print(f"wrote {CSV_PATH}")

    est_tab, freq_tab = summarise(res, rep["wald"], rep["ols"], ar_ref)
    print("\nEstimator dispersion across partitions "
          f"(Wald reference = {rep['wald']:.4f}, OLS = {rep['ols']:.4f}):")
    print(est_tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nTest frequencies across partitions (%):")
    print(freq_tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nblocks of ZX all same sign in {100.0 * res['all_same_sign'].mean():.1f}% "
          "of partitions")

    make_figure(res, rep["wald"], rep["ols"])
    write_tex(est_tab, freq_tab, args.reps, rep["wald"])

    agg = aggregate_over_depths(rows, rep["ols"])
    print_aggregated(agg, rep["wald"])
    agg_out = agg.copy()
    for key, _ in AGG_TESTS:
        agg_out[f"{key}_cs"] = agg_out[f"{key}_cs"].map(_fmt_cs)
    agg_out.to_csv(AGG_CSV_PATH, index=False)
    print(f"wrote {AGG_CSV_PATH}")
    write_agg_tex(agg, args.reps, rep["wald"], ar_cs, k,
                  int(np.ceil(8 * np.log(2 / args.delta))))


if __name__ == "__main__":
    main()
