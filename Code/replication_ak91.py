"""
Replication of Angrist & Krueger (1991), Table III, Panel B.

Data: the authors' posted 1980 Census extract (men born 1930-1939 with
positive earnings), data/AngristKreuger1991/asciiqob.txt, columns

    lwklywge educ yob qob pob

Note: the posted extract has 329,509 rows (the Table V sample) while the
Table III Panel B footnote reports 327,509; replicated numbers therefore
match the published ones closely but not necessarily to the last digit.
Panel A (1970 Census) is not in the archive and is not replicated.

Mapping into the thesis framework (Y = beta*X + eps, beta = mu_ZY/mu_ZX):
AK's instrument is the binary first-quarter dummy and their model has an
intercept, so all three variables are demeaned at the full sample level,

    Y = lwklywge - mean,  X = educ - mean,  Z = 1{qob=1} - mean.

With a binary instrument the full-sample ratio mean(ZY)/mean(ZX) is then
exactly the Wald estimator (covariance-ratio identity), and the AR moment
W_i(beta0) = Z_i(Y_i - beta0*X_i) has mean zero under H0 without the
intercept inflating sigma_Ze. Demeaning by full-sample means introduces
only O(1/n) cross-block dependence, negligible at n ~ 330k.

Estimators (delta = 0.05 throughout):
    Wald / Mean IV      iv_estimate            (simulation.py)
    Ratio-of-Medians    iv_estimate_rm         (alg:rom,  k = ceil(8 ln(2/delta)))
    Median-of-Ratios    iv_estimate_mr         (alg:mor,  k = ceil(8 ln(1/delta)))
    Catoni ratio        iv_estimate_catoni     (sec:sim_catoni, delta/2 per coordinate)

Tests / confidence sets (inference.py, all sharing one block assignment so
test decisions and confidence sets are mutually consistent):
    MoM-AR (feasible)   robust sigma_Ze at the MoR preliminary estimate,
                        exact CS via breakpoint enumeration (alg:mom_ar_cs)
    SN-AR               simulated R_k critical value (prop:sn_pivotal, cor:sn_cs)
    AR (standard)       chi2(1) inversion, Dufour geometry (baseline)

The oracle MoM-AR test is infeasible on real data (sigma_Ze unknown).

Usage:
    python replication_ak91.py [--delta 0.05] [--seed 1991] [--no-tex]

Outputs:
    console report
    Code/output/ak91_replication.csv
    Paper/iteration4/ak91_replication.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import inference as inf
from simulation import (
    iv_estimate,
    iv_estimate_catoni,
    iv_estimate_mr,
    iv_estimate_rm,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "AngristKreuger1991" / "asciiqob.txt"
CSV_PATH = Path(__file__).resolve().parent / "output" / "ak91_replication.csv"
TEX_PATH = ROOT / "Paper" / "iteration4" / "ak91_replication.tex"

# Published values, AK (1991) Table III, Panel B (1980 Census, born 1930-39).
PUBLISHED = {
    "lwklywge_q1": (5.8916, None),
    "lwklywge_q234": (5.9027, None),
    "lwklywge_diff": (-0.01110, 0.00274),
    "educ_q1": (12.6881, None),
    "educ_q234": (12.7969, None),
    "educ_diff": (-0.1088, 0.0132),
    "wald": (0.1020, 0.0239),
    "ols": (0.0709, 0.0003),
}

# Display strings exactly as printed in the paper (preserving trailing zeros).
PUBLISHED_STR = {
    "lwklywge_q1": ("5.8916", ""),
    "lwklywge_q234": ("5.9027", ""),
    "lwklywge_diff": ("-0.01110", "(0.00274)"),
    "educ_q1": ("12.6881", ""),
    "educ_q234": ("12.7969", ""),
    "educ_diff": ("-0.1088", "(0.0132)"),
    "wald": ("0.1020", "(0.0239)"),
    "ols": ("0.0709", "(0.0003)"),
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the posted AK91 extract (whitespace-delimited, no header)."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lwklywge", "educ", "yob", "qob", "pob"],
    )
    if not df["qob"].isin([1, 2, 3, 4]).all():
        raise ValueError("unexpected qob values in extract")
    return df


# ----------------------------------------------------------------------------
# Panel B replication (AK's own quantities)
# ----------------------------------------------------------------------------


def table3_panel_b(df: pd.DataFrame) -> dict:
    """
    Replicate the Table III Panel B quantities: group means of log weekly
    wage and education for Q1 vs Q2-4 births, their differences with
    standard errors, the Wald estimate with delta-method SE (including the
    within-group wage-education covariance), and the bivariate OLS return.
    """
    q1 = df["qob"] == 1
    out: dict[str, float] = {"n": len(df), "n_q1": int(q1.sum()), "n_q234": int((~q1).sum())}

    stats = {}
    for name, g in (("q1", df[q1]), ("q234", df[~q1])):
        y, x = g["lwklywge"].to_numpy(), g["educ"].to_numpy()
        stats[name] = {
            "n": len(g),
            "my": y.mean(), "mx": x.mean(),
            "vy": y.var(ddof=1), "vx": x.var(ddof=1),
            "cyx": np.cov(y, x, ddof=1)[0, 1],
        }
        out[f"lwklywge_{name}"] = stats[name]["my"]
        out[f"educ_{name}"] = stats[name]["mx"]

    s1, s2 = stats["q1"], stats["q234"]
    dy = s1["my"] - s2["my"]
    dx = s1["mx"] - s2["mx"]
    var_dy = s1["vy"] / s1["n"] + s2["vy"] / s2["n"]
    var_dx = s1["vx"] / s1["n"] + s2["vx"] / s2["n"]
    cov_d = s1["cyx"] / s1["n"] + s2["cyx"] / s2["n"]

    wald = dy / dx
    # Delta method for the ratio of two (correlated) mean differences.
    var_wald = (var_dy - 2.0 * wald * cov_d + wald**2 * var_dx) / dx**2

    out.update(
        lwklywge_diff=dy, lwklywge_diff_se=np.sqrt(var_dy),
        educ_diff=dx, educ_diff_se=np.sqrt(var_dx),
        wald=wald, wald_se=np.sqrt(var_wald),
    )

    # Bivariate OLS of log wage on education (with intercept), conventional SE.
    y = df["lwklywge"].to_numpy()
    x = df["educ"].to_numpy()
    n = len(y)
    xc = x - x.mean()
    yc = y - y.mean()
    sxx = np.sum(xc**2)
    b_ols = np.sum(xc * yc) / sxx
    resid = yc - b_ols * xc
    out["ols"] = b_ols
    out["ols_se"] = np.sqrt(np.sum(resid**2) / (n - 2) / sxx)
    out["ols_se_robust"] = np.sqrt(np.sum((xc * resid) ** 2)) / sxx
    return out


# ----------------------------------------------------------------------------
# Thesis estimators and tests
# ----------------------------------------------------------------------------


def to_iv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Demeaned (Y, X, Z) frame; see module docstring for the mapping."""
    z = (df["qob"] == 1).astype(float)
    return pd.DataFrame(
        {
            "Y": df["lwklywge"] - df["lwklywge"].mean(),
            "X": df["educ"] - df["educ"].mean(),
            "Z": z - z.mean(),
        }
    )


def permute_frame(iv_df: pd.DataFrame, seed) -> pd.DataFrame:
    """
    One random permutation of the rows, the ONLY source of randomness in the
    replication. Every partition-based procedure downstream is then called
    with shuffle=False, so for a given seed all MoM-based estimators and
    tests block the identical row ordering (they differ only in their k).
    The raw file is sorted by year of birth, so blocking without this
    permutation would confound blocks with cohorts.
    """
    perm = np.random.default_rng(seed).permutation(len(iv_df))
    return iv_df.iloc[perm].reset_index(drop=True)


def run_estimators(iv_df: pd.DataFrame, delta: float) -> dict:
    """
    All four point estimators on the (already permuted) frame. shuffle=False:
    the partition is fixed by the row order set in permute_frame. Catoni's
    internal MoM variance pre-estimate also blocks this same row order.
    """
    est = {
        "Wald / Mean IV": iv_estimate(iv_df),
        "Ratio-of-Medians": iv_estimate_rm(iv_df, delta=delta, shuffle=False),
        "Median-of-Ratios": iv_estimate_mr(iv_df, delta=delta, shuffle=False),
        "Catoni ratio": iv_estimate_catoni(iv_df, delta=delta),
    }
    return est


def run_inference(iv_df: pd.DataFrame, delta: float, nulls: dict[str, float]) -> dict:
    """
    All three feasible tests and their confidence sets on the (already
    permuted) frame; the block assignment is the same row partition used by
    the point estimators. MoM-AR and SN-AR share the block means (a, b), so
    the reported rejection decisions at the null values are exactly
    membership in the reported confidence set.
    """
    Y = iv_df["Y"].to_numpy()
    X = iv_df["X"].to_numpy()
    Z = iv_df["Z"].to_numpy()
    n = len(Y)
    k = inf.k_blocks(delta)

    a, b, m = inf.block_means(Y, X, Z, k, shuffle=False)
    sigma_hat = inf.robust_sigma_Ze(Y, X, Z, delta, shuffle=False)
    tau = inf.tau_n(sigma_hat, n, delta)
    c_crit = inf.rk_critical_value(k, delta)

    beta0 = np.array(list(nulls.values()))
    W_tilde = inf.mom_ar_statistic(a, b, beta0)
    T_sn = inf.sn_statistic(a, b, beta0)
    ar = inf.standard_ar_test(iv_df, beta0, delta=delta)

    results = {
        "MoM-AR (feasible)": {
            "cs": inf.mom_ar_cs_exact(a, b, tau),
            "reject": dict(zip(nulls, np.abs(W_tilde) > tau)),
            "detail": f"tau = {tau:.5f}, sigma_hat_Ze = {sigma_hat:.5f}",
        },
        "SN-AR": {
            "cs": inf.sn_ar_cs(a, b, c_crit),
            "reject": dict(zip(nulls, T_sn > c_crit)),
            "detail": f"c_(k,delta) = {c_crit:.4f}",
        },
        "AR (standard)": {
            "cs": inf.standard_ar_cs(iv_df, delta=delta),
            "reject": dict(zip(nulls, np.atleast_1d(ar["reject"]))),
            "detail": f"chi2 crit = {ar['chi2_crit']:.4f}",
        },
    }
    diagnostics = {
        "k": k,
        "m": m,
        "sigma_hat_Ze": sigma_hat,
        "tau": tau,
        "c_crit": c_crit,
        "all_same_sign": bool(np.all(b > 0) or np.all(b < 0)),
        "n_blocks_pos": int(np.sum(b > 0)),
    }
    return {"tests": results, "diag": diagnostics}


def strength_conditions(iv_df: pd.DataFrame, delta: float) -> dict:
    """
    Plug-in check of the finite-sample instrument strength conditions, using
    sample moments of ZX for (mu_ZX, sigma2_ZX). These are diagnostics, not
    oracle statements: with QOB the canonical weak instrument, the interest
    is precisely in whether the thesis conditions hold at this n.
    """
    ZX = (iv_df["Z"] * iv_df["X"]).to_numpy()
    n = len(ZX)
    mu = ZX.mean()
    s2 = ZX.var(ddof=1)
    ratio = s2 / mu**2
    k_mr = inf.k_blocks(delta)                      # ceil(8 ln(1/delta))
    k_rm = int(np.ceil(8 * np.log(2 / delta)))       # ceil(8 ln(2/delta))
    m_mr = n // k_mr
    m_rm = n // k_rm
    return {
        "mu_ZX_hat": mu,
        "sigma2_ZX_hat": s2,
        "conditions": {
            "Mean IV (eq:iv_strength): n >= 8 sigma2/(delta mu^2)": (n, 8 * ratio / delta),
            f"RoM (eq:rom_strength): m > 4 sigma2/mu^2 (m={m_rm})": (m_rm, 4 * ratio),
            f"MoR (eq:mor_strength): m >= 32 sigma2/mu^2 (m={m_mr})": (m_mr, 32 * ratio),
            f"Single interval (eq:mono_cheby): m >= k sigma2/(delta mu^2) (m={m_mr})": (
                m_mr,
                k_mr * ratio / delta,
            ),
        },
    }


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------


def _fmt_cs(intervals: list[tuple[float, float]]) -> str:
    if not intervals:
        return "empty"
    parts = []
    for lo, hi in intervals:
        lo_s = "-inf" if np.isinf(lo) else f"{lo:.4f}"
        hi_s = "+inf" if np.isinf(hi) else f"{hi:.4f}"
        parts.append(f"[{lo_s}, {hi_s}]")
    return " u ".join(parts)


def _fmt_cs_tex(intervals: list[tuple[float, float]]) -> str:
    if not intervals:
        return r"$\varnothing$"
    parts = []
    for lo, hi in intervals:
        lo_s = r"-\infty" if np.isinf(lo) else f"{lo:.4f}"
        hi_s = r"+\infty" if np.isinf(hi) else f"{hi:.4f}"
        parts.append(f"[{lo_s},\\, {hi_s}]")
    return "$" + r" \cup ".join(parts) + "$"


def print_report(rep: dict, est: dict, inference: dict, cond: dict, nulls: dict) -> None:
    print("=" * 78)
    print("Angrist & Krueger (1991), Table III, Panel B - replication")
    print(f"n = {rep['n']:,} (published: 327,509; posted extract is the Table V sample)")
    print("=" * 78)
    print(f"{'quantity':<28}{'replicated':>14}{'(se)':>10}{'published':>14}{'(se)':>10}")
    rows = [
        ("ln wage, Q1", "lwklywge_q1", None),
        ("ln wage, Q2-4", "lwklywge_q234", None),
        ("  difference", "lwklywge_diff", "lwklywge_diff_se"),
        ("education, Q1", "educ_q1", None),
        ("education, Q2-4", "educ_q234", None),
        ("  difference", "educ_diff", "educ_diff_se"),
        ("Wald return to educ", "wald", "wald_se"),
        ("OLS return to educ", "ols", "ols_se"),
    ]
    for label, key, se_key in rows:
        se = f"({rep[se_key]:.5f})" if se_key else ""
        pub_s, pub_se_s = PUBLISHED_STR[key]
        print(f"{label:<28}{rep[key]:>14.5f}{se:>10}{pub_s:>14}{pub_se_s:>10}")

    print("\n" + "-" * 78)
    print("Thesis estimators (demeaned Y, X, Z; delta = 0.05)")
    print("-" * 78)
    for name, r in est.items():
        extra = ""
        if "k" in r:
            extra = f"  (k={r['k']}, m={r['m']:,})"
        elif "se" in r:
            extra = f"  (robust se = {r['se']:.4f})"
        print(f"{name:<22}beta_hat = {r['beta_hat']:.4f}{extra}")

    print("\n" + "-" * 78)
    print("Tests and 95% confidence sets (feasible only)")
    print("-" * 78)
    d = inference["diag"]
    print(f"k = {d['k']}, m = {d['m']:,}, sigma_hat_Ze = {d['sigma_hat_Ze']:.5f}, "
          f"tau_n = {d['tau']:.5f}, c_(k,0.05) = {d['c_crit']:.4f}")
    print(f"block means of ZX all same sign: {d['all_same_sign']} "
          f"({d['n_blocks_pos']}/{d['k']} positive)")
    for name, r in inference["tests"].items():
        print(f"\n{name}  [{r['detail']}]")
        print(f"  CS_95 = {_fmt_cs(r['cs'])}")
        for null_name, rej in r["reject"].items():
            print(f"  H0: beta = {nulls[null_name]:.4f} ({null_name}): "
                  f"{'REJECT' if rej else 'fail to reject'}")

    print("\n" + "-" * 78)
    print("Finite-sample instrument strength conditions (plug-in)")
    print("-" * 78)
    print(f"mu_hat_ZX = {cond['mu_ZX_hat']:.5f}, sigma2_hat_ZX = {cond['sigma2_ZX_hat']:.4f}")
    for label, (have, need) in cond["conditions"].items():
        ok = "HOLDS" if have >= need else "FAILS"
        print(f"  {label:<68} need {need:,.0f}: {ok}")


def write_csv(rep: dict, est: dict, inference: dict, cond: dict, nulls: dict) -> None:
    rows = []
    for key, (pub, pub_se) in PUBLISHED.items():
        rows.append(
            dict(section="table3_panelB", quantity=key, value=rep[key],
                 se=rep.get(key + "_se"), published=pub, published_se=pub_se)
        )
    rows.append(dict(section="table3_panelB", quantity="ols_se_robust",
                     value=rep["ols_se_robust"], se=None, published=None, published_se=None))
    for name, r in est.items():
        rows.append(dict(section="estimator", quantity=name, value=r["beta_hat"],
                         se=r.get("se"), k=r.get("k"), m=r.get("m")))
    d = inference["diag"]
    for key in ("k", "m", "sigma_hat_Ze", "tau", "c_crit", "all_same_sign", "n_blocks_pos"):
        rows.append(dict(section="diagnostic", quantity=key, value=d[key]))
    for name, r in inference["tests"].items():
        rows.append(dict(section="confidence_set", quantity=name, value=_fmt_cs(r["cs"])))
        for null_name, rej in r["reject"].items():
            rows.append(dict(section="test", quantity=f"{name} @ {null_name}",
                             value=bool(rej), beta0=nulls[null_name]))
    rows.append(dict(section="strength", quantity="mu_ZX_hat", value=cond["mu_ZX_hat"]))
    rows.append(dict(section="strength", quantity="sigma2_ZX_hat", value=cond["sigma2_ZX_hat"]))
    for label, (have, need) in cond["conditions"].items():
        rows.append(dict(section="strength", quantity=label, value=have,
                         se=None, need=need, holds=have >= need))
    CSV_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"\nwrote {CSV_PATH}")


def _tex_num(x: int) -> str:
    """Integer with LaTeX grouped thousands separators, e.g. 329{,}509."""
    return f"{x:,}".replace(",", r"{,}")


def write_tex(rep: dict, est: dict, inference: dict, nulls: dict) -> None:
    d = inference["diag"]

    lines = [
        "% Auto-generated by Code/replication_ak91.py -- do not edit by hand.",
        "% Replication of Angrist & Krueger (1991), Table III, Panel B.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Replication of \citet{angrist1991does}, Table~III, Panel~B "
        r"(1980 Census, men born 1930--1939). The posted extract has "
        f"$n = {_tex_num(rep['n'])}$"
        r" observations against the published 327{,}509; standard errors in "
        r"parentheses.}",
        r"\label{tab:ak91_panelB}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Replicated} & \multicolumn{2}{c}{Published} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & Estimate & (s.e.) & Estimate & (s.e.) \\",
        r"\midrule",
    ]
    # (label, key, se_key, value decimals, se decimals)
    tab_rows = [
        (r"$\ln$(wkly.\ wage), born Q1", "lwklywge_q1", None, 4, 0),
        (r"$\ln$(wkly.\ wage), born Q2--4", "lwklywge_q234", None, 4, 0),
        (r"\quad difference", "lwklywge_diff", "lwklywge_diff_se", 5, 5),
        (r"Education, born Q1", "educ_q1", None, 4, 0),
        (r"Education, born Q2--4", "educ_q234", None, 4, 0),
        (r"\quad difference", "educ_diff", "educ_diff_se", 4, 4),
        (r"Wald return to education", "wald", "wald_se", 4, 4),
        (r"OLS return to education", "ols", "ols_se", 4, 5),
    ]
    for label, key, se_key, dp, se_dp in tab_rows:
        se_s = f"({rep[se_key]:.{se_dp}f})" if se_key else ""
        pv, pse = PUBLISHED_STR[key]
        lines.append(f"{label} & {rep[key]:.{dp}f} & {se_s} & {pv} & {pse} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robust IV estimates and 95\% confidence sets for the return "
        r"to education, AK91 extract ($\delta = 0.05$; demeaned $Y$, $X$, $Z$; "
        f"$k = {d['k']}$, $m = {_tex_num(d['m'])}$"
        r"). The MoM-AR test uses the feasible robust scale estimate "
        f"$\\hat\\sigma_{{Z\\varepsilon}} = {d['sigma_hat_Ze']:.4f}$"
        r"; the oracle version is infeasible on real data.}",
        r"\label{tab:ak91_robust}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Estimator & $\hat\beta$ \\",
        r"\midrule",
    ]
    for name, r in est.items():
        label = name if name != "Catoni ratio" else r"Catoni ratio ($\delta/2$ per coordinate)"
        if "k" in r:
            label += f" ($k={r['k']}$)"
        se_s = f" ({r['se']:.4f})" if "se" in r else ""
        lines.append(f"{label} & {r['beta_hat']:.4f}{se_s} \\\\")
    lines += [
        r"\midrule",
        r"Test & 95\% confidence set \\",
        r"\midrule",
    ]
    for name, r in inference["tests"].items():
        lines.append(f"{name} & {_fmt_cs_tex(r['cs'])} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TEX_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1991)
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--no-tex", action="store_true",
                   help="console report only: skip writing the CSV and LaTeX outputs")
    args = p.parse_args()

    df = load_data(args.data)
    rep = table3_panel_b(df)
    iv_df = permute_frame(to_iv_frame(df), args.seed)

    est = run_estimators(iv_df, args.delta)

    # Sanity: with demeaned variables the Mean IV ratio IS the Wald estimator.
    assert np.isclose(est["Wald / Mean IV"]["beta_hat"], rep["wald"], rtol=1e-10), (
        est["Wald / Mean IV"]["beta_hat"], rep["wald"],
    )

    nulls = {"no return": 0.0, "OLS": rep["ols"]}
    inference = run_inference(iv_df, args.delta, nulls)
    cond = strength_conditions(iv_df, args.delta)

    print_report(rep, est, inference, cond, nulls)
    if not args.no_tex:
        write_csv(rep, est, inference, cond, nulls)
        write_tex(rep, est, inference, nulls)


if __name__ == "__main__":
    main()
