"""
Replication of Angrist & Evans (1998), Table 5 (Wald estimates), 1980 PUMS,
all-women sample, Same sex instrument.

Data: the authors' posted raw extract data/AngristEvans1998/m_d_806.sas7bdat
(927,267 women aged 20-60 with 1+ children, 1980 Census 5% PUMS, husband
variables matched where present). The analysis sample is constructed with
the selection rules of Farbmacher, Guber & Vikstroem (JAE 2018), who work
from the identical file and document the reconstruction:

    21 <= AGEM <= 35, KIDCOUNT >= 2, AGEQ2ND > 4 (second child older than
    one year), age at first birth >= 15 (ageqm = 4*(80-YOBM)-QTRBTHM-1,
    agefstm = floor((ageqm-AGEQK)/4)), and no allocated values:
    AAGE = AQTRBRTH = AAGE2ND = ASEX = ASEX2ND = 0.

This yields n = 394,840 (FGV's count) against the published 394,835; the
five-observation gap is a known, unresolved discrepancy of the archive.
All Table 2 moments are reproduced at published precision.

Variables (AE98 conventions, validated against Table 2):
    boy1st = (SEXK == 0), boy2nd = (SEX2ND == 0), samesex = equal sexes
    morekids = (KIDCOUNT > 2)                       [endogenous regressor]
    workedm = (WEEKSM > 0), weeksm, hoursm
    incomem = (INCOME1M + max(INCOME2M, 0)) * 2.099173554   [1995 dollars]
    lfaminc = ln(max(FAMINC, 1) * 2.099173554)

For each outcome y the bivariate model y = alpha + beta*morekids + eps is
estimated with the binary samesex instrument; demeaning Y, X, Z maps it
into the thesis framework exactly as in replication_ak91.py, whose
estimator and inference machinery is reused unchanged (one random
permutation per seed shared by every MoM-based procedure).

Usage:
    python replication_ae98.py [--delta 0.05] [--seed 1998] [--no-tex]
                               [--rebuild-cache]

Outputs:
    console report
    Code/output/ae98_replication.csv
    Paper/iteration4/ae98_replication.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import inference as inf
from replication_ak91 import (
    _fmt_cs,
    _fmt_cs_tex,
    _tex_num,
    permute_frame,
    run_estimators,
    run_inference,
    strength_conditions,
)

ROOT = Path(__file__).resolve().parents[1]
SAS_PATH = ROOT / "data" / "AngristEvans1998" / "m_d_806.sas7bdat"
RAW_PKL = Path(__file__).resolve().parent / "output" / "ae98_raw_1980.pkl"
SAMPLE_PKL = Path(__file__).resolve().parent / "output" / "ae98_sample_1980.pkl"
CSV_PATH = Path(__file__).resolve().parent / "output" / "ae98_replication.csv"
TEX_PATH = ROOT / "Paper" / "iteration4" / "ae98_replication.tex"

CPI_1979_TO_1995 = 2.099173554

RAW_COLS = [
    "SEXK", "SEX2ND", "AGEQK", "AGEQ2ND", "AGEQ3RD", "KIDCOUNT", "AGEM",
    "QTRBTHM", "WEEKSM", "HOURSM", "INCOME1M", "INCOME2M", "FAMINC",
    "ASEX", "AAGE", "ASEX2ND", "AAGE2ND", "AQTRBRTH", "AQTR2ND",
    "MARITAL", "TIMESMAR", "YOBM", "FERT",
]

OUTCOMES = {
    "workedm": "Worked for pay",
    "weeksm": "Weeks worked",
    "hoursm": "Hours/week",
    "incomem": "Labor income",
    "lfaminc": "ln(Family income)",
}

# Published values, AE (1998) Table 5, columns (1)-(2): 1980 PUMS, all women,
# Same sex instrument, More than 2 children as endogenous regressor.
# outcome: (mean diff by samesex, se, Wald morekids, se)
PUBLISHED = {
    "workedm": (-0.0080, 0.0016, -0.133, 0.026),
    "weeksm": (-0.3826, 0.0709, -6.38, 1.17),
    "hoursm": (-0.3110, 0.0602, -5.18, 1.00),
    "incomem": (-132.5, 34.4, -2208.8, 569.2),
    "lfaminc": (-0.0018, 0.0041, -0.029, 0.068),
}
PUBLISHED_FS = {
    "morekids": (0.0600, 0.0016),
    "numkids": (0.0765, 0.0026),
}


def _num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return pd.to_numeric(s.str.decode("ascii"), errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def extract_raw(sas_path: Path = SAS_PATH, cache: Path = RAW_PKL) -> pd.DataFrame:
    """Decode the needed columns of the raw SAS extract, cached as pickle."""
    if cache.exists():
        return pd.read_pickle(cache)
    parts = []
    for chunk in pd.read_sas(sas_path, chunksize=200_000):
        parts.append(pd.DataFrame({c: _num(chunk[c]) for c in RAW_COLS}))
    df = pd.concat(parts, ignore_index=True)
    cache.parent.mkdir(exist_ok=True)
    df.to_pickle(cache)
    return df


def build_sample(rebuild: bool = False) -> pd.DataFrame:
    """Analysis sample (all women, 1980) with AE98 variables; cached."""
    if SAMPLE_PKL.exists() and not rebuild:
        return pd.read_pickle(SAMPLE_PKL)
    df = extract_raw()
    ageqm = 4 * (80 - df.YOBM) - df.QTRBTHM - 1
    agefstm = np.floor((ageqm - df.AGEQK) / 4)
    sel = (
        (df.AGEM >= 21) & (df.AGEM <= 35)
        & (df.KIDCOUNT >= 2) & df.AGEQ2ND.notna() & (df.AGEQ2ND > 4)
        & (agefstm >= 15)
        & (df.AAGE == 0) & (df.AQTRBRTH == 0) & (df.AAGE2ND == 0)
        & (df.ASEX == 0) & (df.ASEX2ND == 0)
    )
    d = df[sel]
    boy1st = (d.SEXK == 0)
    boy2nd = (d.SEX2ND == 0)
    out = pd.DataFrame(
        {
            "samesex": (boy1st == boy2nd).astype(float),
            "boy1st": boy1st.astype(float),
            "boy2nd": boy2nd.astype(float),
            "morekids": (d.KIDCOUNT > 2).astype(float),
            "numkids": d.KIDCOUNT.astype(float),
            "everborn": (d.FERT - 1).astype(float),
            "workedm": (d.WEEKSM > 0).astype(float),
            "weeksm": d.WEEKSM.astype(float),
            "hoursm": d.HOURSM.astype(float),
            "incomem": (d.INCOME1M + np.maximum(d.INCOME2M, 0)) * CPI_1979_TO_1995,
            "lfaminc": np.log(np.maximum(d.FAMINC, 1) * CPI_1979_TO_1995),
        }
    ).reset_index(drop=True)
    SAMPLE_PKL.parent.mkdir(exist_ok=True)
    out.to_pickle(SAMPLE_PKL)
    return out


# ----------------------------------------------------------------------------
# Table 5 replication (AE's own quantities)
# ----------------------------------------------------------------------------


def _mean_diff(y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Difference in means of y by binary z, with its standard error."""
    y1, y0 = y[z == 1], y[z == 0]
    diff = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return float(diff), float(se)


def _wald(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Wald estimate (dy/dx by binary z) with delta-method SE incl. covariance."""
    stats = {}
    for g in (1, 0):
        yy, xx = y[z == g], x[z == g]
        stats[g] = dict(
            n=len(yy), my=yy.mean(), mx=xx.mean(),
            vy=yy.var(ddof=1), vx=xx.var(ddof=1),
            cyx=np.cov(yy, xx, ddof=1)[0, 1],
        )
    dy = stats[1]["my"] - stats[0]["my"]
    dx = stats[1]["mx"] - stats[0]["mx"]
    var_dy = sum(s["vy"] / s["n"] for s in stats.values())
    var_dx = sum(s["vx"] / s["n"] for s in stats.values())
    cov_d = sum(s["cyx"] / s["n"] for s in stats.values())
    w = dy / dx
    var_w = (var_dy - 2.0 * w * cov_d + w**2 * var_dx) / dx**2
    return float(w), float(np.sqrt(var_w))


def replicate_table5(d: pd.DataFrame) -> dict:
    z = d["samesex"].to_numpy()
    x = d["morekids"].to_numpy()
    rep: dict = {"n": len(d)}
    rep["fs_morekids"] = _mean_diff(x, z)
    rep["fs_numkids"] = _mean_diff(d["numkids"].to_numpy(), z)
    for key in OUTCOMES:
        y = d[key].to_numpy()
        rep[f"{key}_diff"] = _mean_diff(y, z)
        rep[f"{key}_wald"] = _wald(y, x, z)
        # Bivariate OLS of y on morekids (beta0 candidate for the tests).
        xc = x - x.mean()
        rep[f"{key}_ols"] = float(np.sum(xc * (y - y.mean())) / np.sum(xc**2))
    return rep


def to_iv_frame(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Demeaned (Y, X, Z) frame for one outcome (thesis framework mapping)."""
    return pd.DataFrame(
        {
            "Y": d[outcome] - d[outcome].mean(),
            "X": d["morekids"] - d["morekids"].mean(),
            "Z": d["samesex"] - d["samesex"].mean(),
        }
    )


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------


def print_report(rep: dict, results: dict, cond: dict, delta: float) -> None:
    print("=" * 78)
    print("Angrist & Evans (1998), Table 5 (1980 PUMS, all women, Same sex) - replication")
    print(f"n = {rep['n']:,} (published: 394,835; FGV reconstruction: 394,840)")
    print("=" * 78)
    print(f"{'quantity':<26}{'replicated':>13}{'(se)':>10}{'published':>12}{'(se)':>9}")
    for key, label in (("fs_morekids", "More than 2 children"), ("fs_numkids", "Number of children")):
        v, se = rep[key]
        pv, pse = PUBLISHED_FS[key.replace("fs_", "")]
        print(f"{label:<26}{v:>13.4f}{f'({se:.4f})':>10}{pv:>12}{f'({pse})':>9}")
    for key, label in OUTCOMES.items():
        v, se = rep[f"{key}_diff"]
        pv, pse, pw, pwse = PUBLISHED[key]
        print(f"{label:<26}{v:>13.4f}{f'({se:.4f})':>10}{pv:>12}{f'({pse})':>9}")
        w, wse = rep[f"{key}_wald"]
        print(f"{'  Wald (morekids)':<26}{w:>13.4f}{f'({wse:.4f})':>10}{pw:>12}{f'({pwse})':>9}")

    for key, label in OUTCOMES.items():
        r = results[key]
        print("\n" + "-" * 78)
        print(f"Outcome: {label}   (OLS bivariate = {rep[f'{key}_ols']:.4f})")
        print("-" * 78)
        for name, e in r["est"].items():
            extra = f"  (k={e['k']}, m={e['m']:,})" if "k" in e else (
                f"  (robust se = {e['se']:.4f})" if "se" in e else "")
            print(f"  {name:<22}beta_hat = {e['beta_hat']:.4f}{extra}")
        d = r["inference"]["diag"]
        print(f"  [sigma_hat_Ze = {d['sigma_hat_Ze']:.4f}, tau = {d['tau']:.5f}, "
              f"ZX blocks same sign: {d['all_same_sign']}]")
        for name, t in r["inference"]["tests"].items():
            rejs = ", ".join(
                f"{nn}: {'REJECT' if rej else 'fail'}" for nn, rej in t["reject"].items()
            )
            print(f"  {name:<20} CS_95 = {_fmt_cs(t['cs'])}   [{rejs}]")

    print("\n" + "-" * 78)
    print("Finite-sample instrument strength conditions (plug-in, outcome-independent)")
    print("-" * 78)
    print(f"mu_hat_ZX = {cond['mu_ZX_hat']:.5f}, sigma2_hat_ZX = {cond['sigma2_ZX_hat']:.5f}")
    for label, (have, need) in cond["conditions"].items():
        ok = "HOLDS" if have >= need else "FAILS"
        print(f"  {label:<68} need {need:,.0f}: {ok}")


def write_csv(rep: dict, results: dict, cond: dict) -> None:
    rows = []
    for key in ("morekids", "numkids"):
        v, se = rep[f"fs_{key}"]
        pv, pse = PUBLISHED_FS[key]
        rows.append(dict(section="table5", quantity=f"first_stage_{key}", value=v, se=se,
                         published=pv, published_se=pse))
    for key in OUTCOMES:
        v, se = rep[f"{key}_diff"]
        pv, pse, pw, pwse = PUBLISHED[key]
        rows.append(dict(section="table5", quantity=f"{key}_diff", value=v, se=se,
                         published=pv, published_se=pse))
        w, wse = rep[f"{key}_wald"]
        rows.append(dict(section="table5", quantity=f"{key}_wald", value=w, se=wse,
                         published=pw, published_se=pwse))
        rows.append(dict(section="table5", quantity=f"{key}_ols_bivariate",
                         value=rep[f"{key}_ols"]))
    for key in OUTCOMES:
        r = results[key]
        for name, e in r["est"].items():
            rows.append(dict(section="estimator", outcome=key, quantity=name,
                             value=e["beta_hat"], se=e.get("se"), k=e.get("k"), m=e.get("m")))
        d = r["inference"]["diag"]
        for dk in ("sigma_hat_Ze", "tau", "all_same_sign", "n_blocks_pos"):
            rows.append(dict(section="diagnostic", outcome=key, quantity=dk, value=d[dk]))
        for name, t in r["inference"]["tests"].items():
            rows.append(dict(section="confidence_set", outcome=key, quantity=name,
                             value=_fmt_cs(t["cs"])))
            for nn, rej in t["reject"].items():
                rows.append(dict(section="test", outcome=key,
                                 quantity=f"{name} @ {nn}", value=bool(rej)))
    rows.append(dict(section="strength", quantity="mu_ZX_hat", value=cond["mu_ZX_hat"]))
    rows.append(dict(section="strength", quantity="sigma2_ZX_hat", value=cond["sigma2_ZX_hat"]))
    for label, (have, need) in cond["conditions"].items():
        rows.append(dict(section="strength", quantity=label, value=have, need=need,
                         holds=have >= need))
    CSV_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"\nwrote {CSV_PATH}")


def write_tex(rep: dict, results: dict, delta: float) -> None:
    k = inf.k_blocks(delta)
    any_diag = results[next(iter(OUTCOMES))]["inference"]["diag"]
    lines = [
        "% Auto-generated by Code/replication_ae98.py -- do not edit by hand.",
        "% Replication of Angrist & Evans (1998), Table 5, 1980 PUMS, all women.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Replication of \citet{angrist1998children}, Table~5, columns"
        r" (1)--(2): 1980 PUMS, all women aged 21--35 with two or more children,"
        r" \emph{Same sex} instrument, \emph{More than 2 children} as endogenous"
        r" regressor. Reconstructed sample $n = " + _tex_num(rep["n"]) +
        r"$ against the published 394{,}835 (five-observation archive discrepancy"
        r" documented by \citet{farbmacher2018twin}). Standard errors in"
        r" parentheses; Wald SEs by the delta method.}",
        r"\label{tab:ae98_table5}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Replicated} & \multicolumn{2}{c}{Published} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & Estimate & (s.e.) & Estimate & (s.e.) \\",
        r"\midrule",
    ]
    for key, label in (("morekids", r"\emph{More than 2 children}"),
                       ("numkids", r"\emph{Number of children}")):
        v, se = rep[f"fs_{key}"]
        pv, pse = PUBLISHED_FS[key]
        lines.append(f"{label} & {v:.4f} & ({se:.4f}) & {pv:.4f} & ({pse:.4f}) \\\\")
    lines.append(r"\midrule")
    for key, label in OUTCOMES.items():
        v, se = rep[f"{key}_diff"]
        pv, pse, pw, pwse = PUBLISHED[key]
        dp = 1 if key == "incomem" else 4
        lines.append(f"{label}, difference & {v:.{dp}f} & ({se:.{dp}f}) & {pv} & ({pse}) \\\\")
        w, wse = rep[f"{key}_wald"]
        dpw = 1 if key == "incomem" else 3
        lines.append(f"\\quad Wald & {w:.{dpw}f} & ({wse:.{dpw}f}) & {pw} & ({pwse}) \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robust IV estimates and 95\% confidence sets, AE98 1980"
        r" all-women sample ($\delta = 0.05$; demeaned $Y$, $X$, $Z$; one shared"
        f" random partition, $k = {k}$, $m = " + _tex_num(any_diag["m"]) +
        r"$ for MoR and the tests, $k = 30$ for RoM). The MoM-AR test uses the"
        r" feasible robust scale estimate; the oracle version is infeasible on"
        r" real data.}",
        r"\label{tab:ae98_robust}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Outcome & Wald/Mean IV & MoR & RoM & Catoni \\",
        r"\midrule",
    ]
    for key, label in OUTCOMES.items():
        e = results[key]["est"]
        dp = 1 if key == "incomem" else 4
        cells = [f"{e[n]['beta_hat']:.{dp}f}" for n in
                 ("Wald / Mean IV", "Median-of-Ratios", "Ratio-of-Medians", "Catoni ratio")]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        r"Outcome & \multicolumn{2}{c}{MoM-AR (feasible)} & SN-AR & AR (standard) \\",
        r"\midrule",
    ]
    for key, label in OUTCOMES.items():
        t = results[key]["inference"]["tests"]
        lines.append(
            f"{label} & \\multicolumn{{2}}{{c}}{{{_fmt_cs_tex(t['MoM-AR (feasible)']['cs'])}}} & "
            f"{_fmt_cs_tex(t['SN-AR']['cs'])} & {_fmt_cs_tex(t['AR (standard)']['cs'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TEX_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1998)
    p.add_argument("--no-tex", action="store_true",
                   help="console report only: skip writing the CSV and LaTeX outputs")
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()

    d = build_sample(rebuild=args.rebuild_cache)
    rep = replicate_table5(d)

    results: dict = {}
    cond = None
    for key in OUTCOMES:
        iv_df = permute_frame(to_iv_frame(d, key), args.seed)
        est = run_estimators(iv_df, args.delta)
        # Sanity: the Mean IV ratio on demeaned data IS the Wald estimator.
        assert np.isclose(est["Wald / Mean IV"]["beta_hat"], rep[f"{key}_wald"][0],
                          rtol=1e-9)
        nulls = {"no effect": 0.0, "OLS": rep[f"{key}_ols"]}
        inference = run_inference(iv_df, args.delta, nulls)
        results[key] = {"est": est, "inference": inference}
        if cond is None:
            cond = strength_conditions(iv_df, args.delta)

    print_report(rep, results, cond, args.delta)
    if not args.no_tex:
        write_csv(rep, results, cond)
        write_tex(rep, results, args.delta)


if __name__ == "__main__":
    main()
