# Monte Carlo simulation results — summary and theorem mapping

All figures/tables in `Paper/images/graphs/`, produced by `Code/experiments.py`
(`python experiments.py --full`, ~3 min total; `--pilot` for a fast dry run;
`--verify` for the internal consistency checks). Every experiment is fully
reproducible: seeds are fixed per experiment and per replication.

## Common design

**DGP** (`Code/simulation.py::generate_data`): `Z ~ N(0,1)`,
`X = mu_ZX * Z + eps_X`, `Y = beta * X + eps_Y`, with `Corr(eps_Y, eps_X) = rho`
(endogeneity) and all shocks standardised to unit variance so the population
quantities the theorems depend on are **exact by construction**:
`E[ZX] = mu_ZX`, `Var(ZX) = sigma2_ZX`, `Var(Z eps) = sigma2_Ze`. This is what
lets each experiment check a finite-sample condition (e.g. eq:mor_strength)
against known truth rather than estimates.

**Canonical parameters**: `beta = 1, mu_ZX = 1, sigma2_ZX = 2.5, sigma2_Ze = 1,
rho = 0.5, delta = 0.05`. At `n = 2000` every strong-instrument condition of
the point-estimator theorems holds: IV needs `n >= 400` (eq:iv_strength), RoM
needs `m > 10` with `k = 30` (eq:rom_strength), MoR needs `m >= 80` with
`k = 24` (eq:mor_strength; here `m = 83`, deliberately just above the bound).

**Error families**: Gaussian; Student t(3) (moderate heavy tail); t(2.1)
(variance finite, third moment infinite — the boundary of (A3)); centered
Pareto(2.5) (asymmetric heavy tail, moments >= 2.5 infinite). Heavy tails
enter through `eps_Y` only, i.e. through `Z*eps` — the channel the theory is
about. `eps_X` stays Gaussian.

**Competitor**: Catoni's M-estimator (ratio of Catoni means of ZY and ZX,
confidence budget delta/2 per coordinate, tuning
`alpha = sqrt(2 ln(4/delta) / (n v (1 + ...)))` with `v` a robust MoM-based
variance pre-estimate) — the other canonical sub-Gaussian mean estimator, so
the natural robust benchmark for the MoM constructions.

---

## E1 — `e1_boxplot_gaussian.png` (+ `e1_e2_summary.csv`)

**Regime where standard IV wins.** Gaussian errors, strong instrument,
n = 2000, 10,000 replications. SDs: Mean IV 0.0224, Catoni 0.0224,
MoR 0.0274 (+22%), RoM 0.0396 (+77%). All essentially unbiased.

*Supports*: the "price of robustness" implicit in Theorem `thm:iv` vs
Theorems `thm:rom`/`thm:mor` — when the empirical mean is already efficient
(light tails), blocking + medians costs pure efficiency and buys nothing.
Catoni's psi is nearly linear on Gaussian data, so it tracks the mean exactly;
the MoM estimators pay the k-dependent efficiency factor. RoM pays more than
MoR, consistent with its wider bound (numerator `sigma_ZY + |beta| sigma_ZX`
in eq:rom_bound vs `sigma_Ze` in eq:mor_bound).

## E2 — `e2_boxplot_heavytails.png` (+ same CSV)

**Regime where MoR/RoM win.** t(3), t(2.1), Pareto(2.5) errors, n = 2000,
10,000 replications. At t(2.1): RMSE — MoR 0.0129, Catoni 0.0130 vs
Mean IV 0.0182 (+41%); 99% |error| quantile — MoR 0.0336 vs IV 0.0479.
At Pareto(2.5): RMSE — MoR 0.0176, Catoni 0.0180 vs IV 0.0227. The boxplots
show the mechanism: IV's box is comparable but its outliers are far more
extreme (the rare huge Z*eps draws propagate straight through the sample mean).
The asymmetric Pareto case shows the median-based estimators do **not**
develop bias under skewness — the guarantee really is moment-based, not
symmetry-based.

*Supports*: Theorems `thm:rom`, `thm:mor` (concentration under only (A3)),
and the contrast of Remark `rem:iv_polynomial` vs `rem:mor_logarithmic`.
*Honest caveat for the writeup*: with finite variance the CLT still holds for
the sample mean, so IV's **bulk** dispersion (IQR) is similar — the MoM
advantage lives in the deviation tails (RMSE, extreme quantiles, outliers),
which is exactly what the theory claims. E2b makes this precise.

## E2b — `e2b_deviation_quantiles.png` (+ `e2b_quantiles.csv`)

**The delta-dependence, directly.** Empirical (1-delta)-quantiles of
|beta_hat - beta| plotted against ln(1/delta), 50,000 replications, n = 2000,
log y-scale. Gaussian panel: IV/Catoni lowest at every delta (nothing to
robustify). t(2.1) panel: **the curves cross** — IV is marginally better at
moderate delta, then turns upward and diverges from MoR/Catoni around
ln(1/delta) ≈ 2 (delta ≈ 0.14), ending ~2x above them at delta = 0.002. This
is the polynomial `delta^{-1/2}` rate of eq:iv_bound against the
`sqrt(ln(1/delta))` rate of eq:mom_subgauss/eq:mor_bound, visible in raw data.

*Supports*: Remark `rem:iv_polynomial` (Devroye et al. lower bound —
polynomial dependence is unavoidable for the mean), Remark
`rem:mor_logarithmic`. This is the single most direct empirical validation of
the thesis's central point; I'd lead the section with it or E2.

## E3 — `e3_boxplot_strength.png` (+ `e3_summary.csv`)

**Instrument-strength sweep**, Gaussian errors, n = 2000,
mu_ZX ∈ {1, 0.4, 0.2, 0.1}; each panel's bracket reports which strength
conditions hold ("IV+ RoM+ MoR-" etc.). As mu_ZX falls below the point where
eq:mor_strength fails, MoR develops a visible *bias* (mean/median drift to
~1.10 at mu_ZX = 0.2, worse at 0.1) while Mean IV stays median-centered with
occasional wild draws — a second, different regime where standard IV
outperforms MoR. RoM keeps its center but its spread explodes (near-zero
median denominators).

*Supports*: the instrument-strength conditions eq:rom_strength and
eq:mor_strength are not decorative — the block-level ratio in Algorithm
`alg:mor` needs every block denominator bounded away from 0, and when
`m` is too small relative to `sigma2_ZX/mu_ZX^2` the block ratios become
skewed and the median inherits a bias. Mechanism: with rho > 0 the block
numerator and denominator are correlated, so near-zero denominators skew
the block-ratio distribution in one direction.

## I1 — `i1_size.png` (+ `i1_size.csv`)

**Size**, 10,000 replications per point, delta = 0.05, all four tests at
beta0 = beta, n ∈ {500, 2000, 8000} × {Gaussian, t(2.1), Pareto(2.5)}.

- MoM-AR (oracle and feasible): empirical size 0.0000–0.0001 everywhere.
  The finite-sample guarantee of Theorem `thm:mom_ar_size` holds with a lot
  of room: it is a *bound* (size <= delta), driven by Chebyshev + Hoeffding,
  and the boxplots of tau vs the statistic's actual spread imply rejection
  under H0 is essentially impossible. Report this as validity + conservatism,
  not as "good calibration".
- SN-AR: 0.037–0.051, approaching delta from below as n grows, in every
  tail regime — matching the asymptotic pivotality of Prop `prop:sn_pivotal`
  (k fixed, m -> infinity), with mild finite-m undersizing under heavy tails.
- Standard AR: 0.040–0.052; well calibrated under Gaussian, mildly
  undersized (not oversized) under heavy tails at these n.

*Supports*: Theorem `thm:mom_ar_size` (validity), Prop `prop:sn_pivotal`.
*Caveat to state*: the standard AR is not badly broken here — its statistic
is self-studentised, so its first-order validity survives heavy tails; the
robust tests' advantage is the *guarantee*, not a dramatic size failure of
the baseline.

## I2 — `i2_power.png` (+ `i2_power.csv`)

**Power curves**, n = 2000, 5,000 replications, beta0 - beta ∈ [-0.6, 0.6].
Ranking of acceptance-region width: AR (standard) ≈ SN-AR < MoM-AR
(feasible) < MoM-AR (oracle). MoM-AR's power is 0 until |beta0 - beta| ≈ 0.15
and reaches 1 by ≈ 0.3 (Gaussian) — the price of the conservative threshold
tau_n. SN-AR is nearly as powerful as the standard AR in both panels while
keeping (asymptotic) size control under heavy tails. Under t(2.1) the
feasible MoM-AR is *more* powerful than the oracle: the MAD-based scale
estimate sits below the true sigma_Ze when block means are heavy-tailed
(the MAD sees the central mass), shrinking tau_hat — size remains <= delta
in I1, but this is luck of conservatism, not an improved guarantee; flag it.

*Supports*: test-inversion logic of `ss:inversion`; quantifies the
power cost of the finite-sample guarantee vs the self-normalised route
(`sec:normalisedAR`), which recovers near-standard power.

## I3 — `i3_cs_lengths.png` + `i3_cs_table.csv`

**Confidence sets**, n = 2000, 5,000 replications; strong (mu_ZX = 1) and
genuinely weak (mu_ZX = 0.05) designs × {Gaussian, t(2.1)}.

Coverage: MoM-AR oracle 1.000 everywhere, feasible 0.9996–1.000 (>= 0.95,
Theorem `thm:coverage` validated, conservatively); SN-AR 0.947–0.952 and
AR 0.949–0.952 (nominal). Median bounded lengths, strong Gaussian:
AR 0.088 < SN 0.123 < MoM-AR feasible 0.398 < oracle 0.443 (the ~5x factor
is the same conservatism as I1/I2).

Weak instruments (the Dufour regime, `ss:inversion`): standard AR CS is
unbounded in 70%/67% of replications and SN-AR in 88%/85%, with SN-AR
multi-interval sets common (mean ~2.3 components, single-interval only
~39–41%) — the piecewise geometry of Cor `cor:sn_cs` in action. The exact
MoM-AR CS (Algorithm `alg:mom_ar_cs`) never returned more than ~1.05
components on average and pct_single_interval >= 0.96 even at mu_ZX = 0.05.

*Structural observation worth a remark in the thesis*: the MoM-AR CS was
**never unbounded** in 20,000 replications. That is geometry, not luck: as
beta0 -> ±infinity, W_tilde(beta0) ~ -beta0 * (median-rank block mean of ZX),
which is nonzero almost surely, so the sublevel set is a.s. bounded — the
weak-instrument problem shows up as *length* (median 9.45 vs 0.44, a 21x
inflation) rather than unboundedness. This does not contradict Dufour's
impossibility result: validity is only guaranteed at the structural beta
(where W_i has mean zero and variance sigma2_Ze); at mu_ZX = 0 other
observationally equivalent beta0 have W(beta0) with inflated variance not
covered by tau_n.

*Supports*: Theorem `thm:coverage`, Cor `cor:endpoints`/`cor:union` (the
exact enumeration is verified against brute force in `--verify`), Cor
`cor:sn_cs`, and the Dufour discussion in `ss:inversion`.

## I4 — `i4_monotonicity.png` (+ `i4_monotonicity.csv`)

**Single-interval condition.** Left: strength sweep at n = 2000 — the
premise of Prop `prop:mono_det` ("all block means share a sign") transitions
from 0 to 1 over mu_ZX ∈ [0.05, 0.8], while the fraction of single-interval
CSs never drops below 0.9955 (same-sign is sufficient, evidently far from
necessary). Right: n sweep at mu_ZX = 0.75 — empirically the same-sign
fraction hits 1.0 by n ≈ 1000–2000, while the Chebyshev condition
eq:mono_cheby requires n* = 51,216: the sufficient condition of Prop
`prop:mono_cheby` is valid but conservative by a factor ~25–50 (as expected
from Chebyshev + union bound; cf. Remark `rem:mono_cantelli`).

**Deterministic confirmation**: across all 28,000 replications with all block
means of one sign, the exact CS was a single interval in **every** case
(zero violations; asserted programmatically). Prop `prop:mono_det` confirmed.

## I5 — `i5_rk_table.tex` + `i5_rk_critical_values.csv`

Critical values c_{k,delta} = (1-delta)-quantiles of
R_k = |med(xi)|/MAD(xi), xi iid N(0,1) (Prop `prop:sn_pivotal`), 10^6
simulations per k, for k ∈ {5, 10, 15, 19, 20, 24, 25, 30, 37, 40, 50}
(covering k = ceil(8 ln(1/delta)) for delta = 0.10, 0.05, 0.01: k = 19, 24, 37).
The `.tex` file is ready to `\input` into appendix `sec:artable`. Note the
even/odd zigzag (e.g. c_{19,.05} = 0.996 < c_{20,.05} = 1.052): with the
rank-ceil(k/2) median convention, even k uses the lower-middle order
statistic, which shifts the null distribution — worth one sentence in the
appendix caption.

---

## Judgment calls and flags (things the tex leaves open)

1. **Median convention (affects even k, e.g. k = 24 at delta = 0.05).**
   Theorem `thm:piecewise` / Algorithm `alg:mom_ar_cs` define the median as
   the rank-ceil(k/2) order statistic (that's what makes the median a single
   block mean per piece). All inference code uses this convention, including
   the simulated R_k table (statistic and critical value always consistent).
   The point estimators keep numpy's midpoint convention. Consider stating
   the convention once in the tex.
2. **eq:tau rounding slippage.** tau_n = sigma sqrt(32 ln(1/delta)/n) equals
   the proof-exact 2 sigma/sqrt(m) only when k = 8 ln(1/delta) exactly and
   k | n; with the ceilings it is *slightly smaller* (anti-conservative in
   principle; numerically ~0.05% here). Implemented as written; one sentence
   in the thesis would inoculate it.
3. **Feasible scale estimator.** The tex asks for a beta0-free robust
   estimate but doesn't pick one. Implemented: residuals at the MoR estimate,
   then sigma_hat = 1.4826 * MAD(block means) * sqrt(m). A scalar-MoM
   estimate of E[(Z eps)^2] was tried and rejected: for tail index 2.1 the
   squared products have tail index ~1.05 and MoM sits far below the mean
   (sigma_hat ~ 0.43 instead of 1). The MAD version is also biased low under
   heavy tails (feasible CS length 0.19 vs oracle 0.44 at t(2.1)) but
   empirical size stayed <= delta everywhere. If you want a version with a
   provable guarantee, that requires a finite-fourth-moment assumption or a
   different argument — worth a remark, not a fix.
4. **Standard AR baseline.** Defined as T = sqrt(n) W_bar(beta0)/s(beta0)
   with s^2 the sample variance of W_i(beta0) (ddof = 1), chi2_1 critical
   value; CS by quadratic inversion (Dufour cases). Its variance is estimated
   at the tested beta0 — standard for AR, but it means the power comparison
   with the constant-band MoM-AR is not variance-protocol-matched.
5. **Boxplot axes** are truncated at pooled 0.5/99.5% quantiles (fraction
   clipped printed in each note); without truncation the heavy-tail panels
   are unreadable. The clipped points are IV/RoM extremes — i.e. truncation
   *understates* the MoM advantage.
6. **k for the SN test** is set to ceil(8 ln(1/delta)) = 24 to match the
   MoM-AR test (the tex only requires k fixed). The R_k table covers other k.
7. **Catoni tuning** follows Catoni (2012) with the variance replaced by a
   robust MoM pre-estimate; each coordinate gets budget delta/2. Catoni ≈
   Mean IV under Gaussian and ≈ MoR under heavy tails — i.e. MoR matches the
   canonical sub-Gaussian competitor while being simpler; that's a good
   sentence for the writeup.

## Suggested figure order for the section

1. E1 + E2 (boxplots: both directions of the comparison), 2. E2b (the
delta-dependence — the core claim), 3. E3 (strength conditions),
4. I1 size table/figure, 5. I2 power, 6. I3 CS table + lengths,
7. I4 monotonicity, 8. I5 table in the appendix.
