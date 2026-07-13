# Simulation code & experiment reference

Reference for what runs where and how to read the outputs. Interpretation of
the *results* (numbers, theorem mapping, judgment calls) lives in
[results_summary.md](results_summary.md); this file is the map.

## Code files (`Code/`)

| File | Contents |
|---|---|
| `simulation.py` | DGP + point estimators. `generate_data` draws `Z ~ N(0,1)`, `X = mu_ZX*Z + eps_X`, `Y = beta*X + eps_Y`, calibrated so `E[ZX]`, `Var(ZX)`, `Var(Z eps)` equal the requested values exactly (shock families: Gaussian, t(df), centered Pareto(alpha); `rho` = endogeneity). Estimators: `iv_estimate` (Mean IV, sec:standard_iv), `iv_estimate_rm` (RoM, alg:rom, k=⌈8 ln(2/δ)⌉), `iv_estimate_mr` (MoR, alg:mor, k=⌈8 ln(1/δ)⌉), `iv_estimate_catoni` (ratio of Catoni M-estimates, scale-adaptive tuning, δ/2 per coordinate). |
| `inference.py` | All tests and confidence sets. MoM-AR test (alg:mom_ar, threshold eq:tau; `sigma_Ze=None` switches oracle → feasible scale = 1.4826·MAD(block means)·√m at the MoR estimate); exact MoM-AR CS by breakpoint enumeration (`mom_ar_cs_exact`, alg:mom_ar_cs); self-normalised test (def:sn_stat) with simulated R_k critical values (cached in `Code/_rk_cache.json`); exact SN-CS; standard AR test/CS (χ²₁, quadratic Dufour inversion) as the non-robust baseline. All inference uses the rank-⌈k/2⌉ median of thm:piecewise. |
| `simulation_study.py` | Original single-config boxplot driver (kept working; all four estimators + oracle MoM-AR size check). Also hosts the shared plot style and `styled_boxplot`. |
| `experiments.py` | The experiment suite below. Every figure/CSV in this folder comes from here. |

## Running

```
cd Code
python experiments.py --full            # thesis-quality counts (~3 min, all cores)
python experiments.py --pilot           # ~10x fewer reps (~20 s), same outputs
python experiments.py --full --only e2b,i3   # any subset of: e1e2 e2b e3 i1 i2 i3 i4 i5
python experiments.py --verify          # consistency checks (exact CS vs brute force, etc.)
```

Same parameters overwrite the same files; seeds are fixed, so runs are
reproducible. Canonical DGP unless a sweep says otherwise: `beta=1, mu_ZX=1,
sigma2_ZX=2.5, sigma2_Ze=1, rho=0.5, delta=0.05, n=2000` (all
strong-instrument conditions hold at this n; MoR's m=83 is just above its
bound of 80).

## Experiments → outputs

| Exp | Outputs | Question it answers | Read it as |
|---|---|---|---|
| E1 | `e1_boxplot_gaussian.png`, `e1_e2_summary.csv` | Cost of robustness under Gaussian errors (regime where Mean IV wins) | Boxplot spread: IV = Catoni < MoR < RoM; all unbiased. Supports the efficiency contrast of thm:iv vs thm:rom/thm:mor. |
| E2 | `e2_boxplot_heavytails.png`, same CSV | Do MoR/RoM/Catoni beat Mean IV under heavy tails? (t(3), t(2.1), Pareto(2.5)) | IV's box stays similar but its outliers explode; MoR/Catoni win on RMSE and extreme quantiles. Skewed Pareto ⇒ no median bias: guarantees are moment-based. Supports thm:rom, thm:mor. |
| E2b | `e2b_deviation_quantiles.png`, `e2b_quantiles.csv` | The δ-dependence itself: empirical (1−δ)-quantile of \|β̂−β\| vs ln(1/δ) | Flat-ish curve = sub-Gaussian √ln(1/δ); steep upturn = polynomial δ^{−1/2}. Under t(2.1) IV crosses above MoR/Catoni ≈ ln(1/δ)=2. Supports rem:iv_polynomial vs rem:mor_logarithmic — the central claim. |
| E3 | `e3_boxplot_strength.png`, `e3_summary.csv` | What breaks when instrument-strength conditions fail (μ_ZX sweep 1→0.1)? | Panel brackets flag which of eq:iv_strength / eq:rom_strength / eq:mor_strength hold. MoR develops bias once eq:mor_strength fails; RoM's spread explodes; IV stays centered — the conditions bind. |
| I1 | `i1_size.png`, `i1_size.csv` | Empirical size of MoM-AR (oracle/feasible), SN-AR, standard AR across tails × n | MoM-AR ≈ 0: thm:mom_ar_size is a *bound*, holds with large slack. SN-AR → δ from below (prop:sn_pivotal). Standard AR near nominal even at t(2.1) — the robust advantage is the guarantee, not a baseline failure. |
| I2 | `i2_power.png`, `i2_power.csv` | Power (rejection rate vs β₀−β) of the same four tests | Acceptance-region width: AR ≈ SN-AR ≪ MoM-AR. The gap is the price of the finite-sample guarantee; SN-AR recovers near-standard power. Feasible > oracle power under t(2.1) = downward-biased scale estimate (flagged in results_summary §3). |
| I3 | `i3_cs_lengths.png`, `i3_cs_table.csv` | CS coverage, length, components, unboundedness; strong (μ=1) vs weak (μ=0.05) | Table: coverage (MoM-AR = 1.0, SN/AR ≈ 0.95), median bounded length, % unbounded, % single interval. Weak: AR/SN-AR often unbounded/multi-interval (Dufour); MoM-AR CS stays one bounded interval but ~21× longer. Supports thm:coverage, cor:union, cor:sn_cs. |
| I4 | `i4_monotonicity.png`, `i4_monotonicity.csv` | When is the CS a single interval? (prop:mono_det, prop:mono_cheby) | Left: μ_ZX sweep — "all block means same sign" transitions 0→1; single-interval fraction ≥ 0.996 throughout. Right: n sweep — empirical fraction hits 1 by n≈2000 vs Chebyshev threshold n*=51,216 (sufficient, ~25–50× conservative). Zero same-sign-but-multi-interval cases = deterministic confirmation of prop:mono_det. |
| I5 | `i5_rk_table.tex`, `i5_rk_critical_values.csv` | Critical values c_{k,δ} = (1−δ)-quantiles of R_k (prop:sn_pivotal) | 10⁶ sims per k; `\input` the .tex into appendix sec:artable. Even/odd k zigzag comes from the rank-median convention. |

## Reading conventions

- Boxplots: whiskers 1.5×IQR, diamonds = means, dashed line = true β; axes
  truncated at pooled 0.5/99.5% quantiles with the clipped fraction printed
  in the note (clipping only hides IV/RoM extremes, understating the MoM
  advantage).
- "Oracle" = true σ_Zε plugged into eq:tau; "feasible" = robust β₀-free
  estimate (see inference.py docstring). SN-AR needs no scale at all.
- CS lengths in I3 are over *bounded* sets only; unbounded/multi-interval
  rates are in the CSV columns `pct_unbounded`, `mean_components`,
  `pct_single_interval`.
