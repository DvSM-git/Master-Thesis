"""
Monte Carlo simulation study comparing the three IV estimators
(Mean IV, Ratio-of-Medians, Median-of-Ratios) on data from generate_data.

Produces a boxplot of the estimates across many replications and saves it to
Paper/images/graphs. The filename encodes every parameter, so re-running with
the same parameters overwrites the image, while changing any parameter writes
a new file.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

# --- Journal-style figure defaults (applied to every figure this module makes) ---
# Clean econometrics-journal look: full axis box, readable sans-serif type,
# purposeful colour. Tuned to be publication-ready without manual editing.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "0.2",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    # NOTE: deliberately no savefig.bbox="tight" -- tight cropping makes each
    # image a different size. We use fixed figsize + subplots_adjust instead so
    # every saved plot has identical dimensions.
})

# One colour per estimator (colourblind-friendly), reused across the figure.
ESTIMATOR_COLORS = {
    "Mean IV": "#4C72B0",            # blue
    "Ratio-of-Medians": "#DD8452",   # orange
    "Median-of-Ratios": "#55A868",   # green
}

from simulation import (
    generate_data,
    iv_estimate,
    iv_estimate_mr,
    iv_estimate_rm,
)

# Directory where plots are saved (relative to the repo root, two levels up
# from this file: Code/ -> repo root -> Paper/images/graphs).
GRAPHS_DIR = Path(__file__).resolve().parents[1] / "Paper" / "images" / "graphs"


def _fmt(value) -> str:
    """Compact, filename-safe formatting of a parameter value."""
    if value is None:
        return "normal"
    if isinstance(value, float):
        # Trim trailing zeros: 1.50 -> 1.5, 0.625 -> 0.625, 2.0 -> 2
        return f"{value:g}".replace("-", "neg").replace(".", "p")
    return str(value).replace("-", "neg").replace(".", "p")


def build_filename(params: dict) -> str:
    """Build a parameter-encoding filename so identical params overwrite."""
    parts = [f"{key}-{_fmt(val)}" for key, val in params.items()]
    return "iv_boxplot__" + "__".join(parts) + ".png"


def min_sample_sizes(mu_ZX: float, sigma2_ZX: float, delta: float) -> dict[str, dict]:
    """
    Theoretical minimum sample size n for each estimator to satisfy its
    concentration guarantee at confidence level delta, given the instrument
    moments mu_ZX = E[ZX] and sigma2_ZX = Var(ZX).

    Standard IV (Thm 2.3):   n >= 8 sigma2_ZX / (delta mu_ZX^2)
    RoM        (Thm 3.1):    k = ceil(8 ln(2/delta)),  m > 4 sigma2_ZX / mu_ZX^2,
                             n >= k * m_min   (m_min = smallest int > the bound)
    MoR        (Thm 4.2):    k = ceil(8 ln(1/delta)),  m >= 32 sigma2_ZX / mu_ZX^2,
                             n >= k * m_min   (m_min = ceil of the bound)

    Returns a dict keyed by estimator name; each value has 'n_min' and, for the
    block estimators, 'k' and 'm_min'.
    """
    ratio = sigma2_ZX / mu_ZX**2  # sigma2_ZX / mu_ZX^2 appears in all three

    # Standard IV: direct n bound.
    n_min_iv = int(np.ceil(8 * ratio / delta))

    # RoM: strict inequality m > 4*ratio  -> smallest integer m is floor+1.
    k_rom = int(np.ceil(8 * np.log(2 / delta)))
    m_min_rom = int(np.floor(4 * ratio)) + 1
    n_min_rom = k_rom * m_min_rom

    # MoR: non-strict m >= 32*ratio  -> smallest integer m is ceil.
    k_mor = int(np.ceil(8 * np.log(1 / delta)))
    m_min_mor = int(np.ceil(3 * ratio))
    n_min_mor = k_mor * m_min_mor

    return {
        "Mean IV": {"n_min": n_min_iv},
        "Ratio-of-Medians": {"n_min": n_min_rom, "k": k_rom, "m_min": m_min_rom},
        "Median-of-Ratios": {"n_min": n_min_mor, "k": k_mor, "m_min": m_min_mor},
    }


def _one_replication(data_seed, n, beta, mu_ZX, sigma2_ZX, sigma2_Ze,
                     rho, eps_Y_df, eps_X_df, delta) -> tuple[float, float, float]:
    """
    Run a single Monte Carlo replication and return the three beta estimates.

    Module-level (not a closure) so joblib can pickle it for worker processes.
    The per-rep rng is built from `data_seed` (a spawned SeedSequence), making
    each replication independent and the whole study reproducible.
    """
    rep_rng = np.random.default_rng(data_seed)
    df = generate_data(
        n=n,
        beta=beta,
        mu_ZX=mu_ZX,
        sigma2_ZX=sigma2_ZX,
        sigma2_Ze=sigma2_Ze,
        rho=rho,
        eps_Y_df=eps_Y_df,
        eps_X_df=eps_X_df,
        rng=rep_rng,
    )
    return (
        iv_estimate(df)["beta_hat"],
        iv_estimate_rm(df, delta=delta, rng=rep_rng)["beta_hat"],
        iv_estimate_mr(df, delta=delta, rng=rep_rng)["beta_hat"],
    )


def run_simulation_study(
    n: int = 10_000,
    beta: float = 1.5,
    mu_ZX: float = 0.8,
    sigma2_ZX: float = 1.5,
    sigma2_Ze: float = 0.625,
    rho: float = 0.7,
    eps_Y_df: float | None = 5,
    eps_X_df: float | None = 5,
    delta: float = 0.05,
    n_reps: int = 500,
    seed: int = 0,
    n_jobs: int = -1,
    show: bool = False,
) -> Path:
    """
    Run n_reps Monte Carlo replications and save a boxplot of the estimates.

    Each replication draws a fresh sample (with an independent rng derived from
    `seed`) and applies all three estimators. The boxplot shows the sampling
    distribution of each estimator, with a dashed line at the true beta.

    Replications run in parallel across CPU cores via joblib. `n_jobs` controls
    the number of worker processes: -1 uses all cores, 1 runs sequentially.

    Returns the path of the saved image.
    """
    # Parameters that define this study, in the order they appear in the filename.
    params = {
        "n": n,
        "beta": beta,
        "muZX": mu_ZX,
        "s2ZX": sigma2_ZX,
        "s2Ze": sigma2_Ze,
        "rho": rho,
        "epsYdf": eps_Y_df,
        "epsXdf": eps_X_df,
        "delta": delta,
        "reps": n_reps,
        "seed": seed,
    }

    # One independent SeedSequence per replication -> reproducible regardless of
    # how joblib schedules the work across processes.
    data_seeds = np.random.SeedSequence(seed).spawn(n_reps)

    t_start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_replication)(
            data_seeds[rep], n, beta, mu_ZX, sigma2_ZX, sigma2_Ze,
            rho, eps_Y_df, eps_X_df, delta,
        )
        for rep in range(n_reps)
    )
    runtime = time.perf_counter() - t_start

    # results is a list of (mean_iv, rm, mr) tuples; unpack column-wise.
    mean_iv, rm, mr = (np.asarray(col) for col in zip(*results))
    estimates = {
        "Mean IV": mean_iv,
        "Ratio-of-Medians": rm,
        "Median-of-Ratios": mr,
    }

    # --- plot (econometrics-journal style) ---
    labels = list(estimates.keys())
    data = [estimates[name] for name in labels]
    colors = [ESTIMATOR_COLORS[name] for name in labels]
    positions = np.arange(1, len(labels) + 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Reference line at the true parameter, drawn first so boxes sit on top.
    ax.axhline(beta, color="0.45", linestyle=(0, (5, 4)), linewidth=1.1,
               zorder=1, label=r"True $\beta$")

    bp = ax.boxplot(
        data,
        positions=positions,
        tick_labels=labels,
        showmeans=True,
        showfliers=True,
        widths=0.55,
        patch_artist=True,
        zorder=3,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="0.3", linewidth=1.1),
        capprops=dict(color="0.3", linewidth=1.1),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=6, zorder=4),
    )
    # Colour each box (semi-transparent fill, solid coloured edge).
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.6)
    # Outliers as small open circles in the matching colour.
    for flier, color in zip(bp["fliers"], colors):
        flier.set(marker="o", markerfacecolor="none",
                  markeredgecolor=color, markersize=4, alpha=0.7)

    # Show the full range: let matplotlib autoscale so every point is visible
    # (no y-axis clipping, no hidden outliers).

    ax.set_ylabel(r"Estimate $\hat{\beta}$")
    ax.set_xlabel("")
    ax.margins(x=0.08)

    # Light horizontal grid; full boxed frame from rcParams.
    ax.yaxis.grid(True, color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=4, color="0.3")
    ax.legend(frameon=False, loc="upper right")

    # Parameter provenance as a small note in the reserved bottom margin, so the
    # saved file is self-documenting without cluttering the plot area.
    note = (
        f"Note. {n_reps} Monte Carlo replications; "
        f"n = {n}, ρ = {rho:g}, df$_Y$ = {eps_Y_df}, df$_X$ = {eps_X_df}, "
        f"δ = {delta:g}. Diamonds mark means; whiskers extend to 1.5×IQR."
    )
    fig.text(0.5, 0.035, note, ha="center", va="bottom", fontsize=8, color="0.3")

    # FIXED layout: identical margins (and therefore identical pixel size) for
    # every figure, so plots line up exactly when scrolled through. Do NOT use
    # tight_layout or bbox="tight" here -- those size the output to its content.
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.18)

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GRAPHS_DIR / build_filename(params)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)

    # --- console summary ---
    print(f"Saved: {out_path}")
    print(f"Runtime: {runtime:.2f}s for {n_reps} reps "
          f"(n_jobs={n_jobs}, {1000 * runtime / n_reps:.2f} ms/rep)")
    print(f"{'estimator':<20} {'mean':>10} {'median':>10} {'std':>10}")
    for name in labels:
        arr = np.asarray(estimates[name])
        print(f"{name:<20} {arr.mean():>10.4f} {np.median(arr):>10.4f} {arr.std():>10.4f}")

    # --- theoretical minimum sample sizes ---
    n_min = min_sample_sizes(mu_ZX, sigma2_ZX, delta)
    print(f"\nTheoretical minimum n (delta={delta:g}); current n={n}:")
    print(f"{'estimator':<20} {'n_min':>8} {'k':>6} {'m_min':>8} {'n>=n_min?':>10}")
    for name in labels:
        info = n_min[name]
        k_str = str(info.get("k", ""))
        m_str = str(info.get("m_min", ""))
        ok = "yes" if n >= info["n_min"] else "NO"
        print(f"{name:<20} {info['n_min']:>8} {k_str:>6} {m_str:>8} {ok:>10}")

    return out_path


if __name__ == "__main__":
    # Edit the parameters here to run a different study.
    run_simulation_study(
        n=100,
        beta=1.5,
        mu_ZX=0.8,
        sigma2_ZX=1.5,
        sigma2_Ze=0.625,
        rho=0.7,
        eps_Y_df=2.1,
        eps_X_df=2.1,
        delta=0.05,
        n_reps=10000,
        seed=0,
        n_jobs=-1,   # -1 = all cores, 1 = sequential
    )
