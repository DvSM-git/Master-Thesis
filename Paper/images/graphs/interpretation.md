# Interpretation of the simulation results

Plain-language reading of each experiment: what happened, why, and what it
means for the thesis. File map and run instructions: [README.md](README.md);
theorem labels and technical flags: [results_summary.md](results_summary.md).
All numbers below are from the full run (10,000 reps for boxplots and size,
50,000 for E2b, 5,000 for power/CS, 2,000 per point for I4).

## E1 — Gaussian, strong instrument: robustness is not free

All four estimators are unbiased, but their spreads differ: Mean IV and
Catoni have SD 0.0224, MoR 0.0275 (+23%), RoM 0.0394 (+76%). Under light
tails the sample mean is already the efficient estimator, so blocking and
taking medians can only throw information away. Catoni pays essentially
nothing because its influence function is nearly linear on well-behaved data
— it *is* the mean until an observation looks extreme.

**Takeaway:** the MoM guarantees are bought with a real but bounded
efficiency premium: ~20% extra SD for MoR, more for RoM. If you believe your
tails are Gaussian, use IV.

## E2 — Heavy tails: the premium pays out, and where it pays

Under t(2.1) errors the ranking flips: RMSE is 0.0129 for MoR and 0.0130 for
Catoni against 0.0182 for Mean IV (+41%), and the 99% error quantile is
0.0336 (MoR) vs 0.0479 (IV). Same picture under asymmetric Pareto(2.5)
tails: MoR 0.0176 vs IV 0.0227.

Two subtleties matter for an honest writeup:

1. **Where the gain lives.** IV's *box* (IQR) is similar to MoR's; what
   differs is the outliers. With finite variance the CLT still protects the
   bulk of IV's sampling distribution — the heavy tail leaks in through rare
   extreme draws of Z·ε that pass straight through the sample mean, while
   the block median simply outvotes them. So MoM improves the *deviations*,
   not the typical error. That is exactly what the theory promises (a tail
   bound, not a variance bound) and E2b shows it directly.
2. **Skewness is harmless.** The Pareto panel shows no bias for the
   median-based estimators despite strong asymmetry: the block *means* are
   what get medianed, and those are already nearly symmetric around the
   truth at m = 83. The guarantee is genuinely moment-based.

RoM improves on IV's tails too but is dominated by MoR everywhere — its
error compounds the deviations of two separate medians (numerator and
denominator), which is also why its theoretical bound carries the larger
constant.

**Takeaway:** MoR is the estimator the heavy-tailed regime rewards, it
matches Catoni (the canonical sub-Gaussian competitor) while being simpler,
and the improvement is in exactly the quantity the theorems bound.

## E2b — The δ-dependence: polynomial vs logarithmic, visible in raw data

This plot *is* the thesis's central claim, empirically. For each estimator
we plot the empirical (1−δ)-quantile of |β̂−β| against ln(1/δ).

- **Gaussian panel:** all four curves are parallel and gently concave — every
  estimator is effectively sub-Gaussian here, and IV/Catoni sit lowest at
  every δ (consistent with E1: nothing to robustify, constant premium for
  MoM).
- **t(2.1) panel:** at δ = 0.5 all estimators are equal (0.009). As δ
  shrinks, IV's curve bends *upward* and diverges: at δ = 0.002 IV's
  quantile is 0.0886 against MoR's 0.0411 — 2.2× worse, and the gap is still
  widening. MoR/Catoni stay on the same gentle √ln(1/δ) trajectory as in
  the Gaussian panel.

That upward bend is the δ^(−1/2) rate of the empirical mean (Remark
rem:iv_polynomial: unavoidable, by the Devroye et al. lower bound); the flat
trajectory is the √ln(1/δ) rate the MoM construction buys (Remark
rem:mor_logarithmic). The crossing point (≈ δ = 0.14 here) is the practical
message: *the more confidence you demand, the more MoM wins.*

**Takeaway:** point estimates of "spread" (SD, IQR) understate what MoM
does; the entire difference between the estimator classes is a statement
about this curve, and the data reproduce it.

## E3 — Weak instruments: two different failure modes

Sweeping μ_ZX from 1 down to 0.1 at n = 2000 (Gaussian errors):

- **μ_ZX = 0.4** (MoM conditions fail, IV's holds): everyone is still
  centered; IV/Catoni are tightest. Failure of the MoM conditions is
  graceful at first — just wider spread.
- **μ_ZX = 0.2:** MoR develops a real **bias** (+0.09, i.e. 9% of β) while
  IV remains median-unbiased. Mechanism: MoR's blocks have m = 83
  observations each, so block-level denominators μ̂_ZX⁽ʲ⁾ are frequently
  near zero; because ρ > 0 makes block numerators and denominators
  correlated, the resulting block ratios are *skewed* (not just fat-tailed),
  and the median inherits the skew. This is precisely what the
  instrument-strength condition eq:mor_strength exists to rule out.
- **μ_ZX = 0.1:** the failure modes trade places. IV and Catoni stay
  median-centered but produce catastrophic draws (SD 2.5, min/max in the
  hundreds); MoR is biased (+0.23) but *stable* (SD 0.17). On RMSE, MoR is
  actually best here (0.28 vs 2.5) — but for the wrong reason, and no theory
  covers either estimator this deep in the weak-instrument regime.

**Takeaway:** IV fails by variance explosion, MoR fails by bias, RoM fails
by both. This motivates the inference section: near weak identification, no
point estimator is trustworthy and test inversion is the right tool
(Dufour's argument in the tex).

## I1 — Size: a guarantee, not a calibration

At the true β, the MoM-AR test (oracle *and* feasible) rejects in ~0.00% of
10,000 replications in every design — versus the nominal δ = 5%. The
finite-sample size bound of thm:mom_ar_size holds, with enormous slack: the
threshold τ_n is about 4.9× the actual SD of the statistic under H₀, because
it is built from Chebyshev-plus-Hoeffding worst cases. Do not present these
zeros as "good size"; present them as *validity with conservatism* — the
test never lies about its level, at any n, under any finite-variance tail.

The comparison points:
- **SN-AR** sits at 0.037–0.051, drifting toward 0.05 from below as n grows,
  identically across Gaussian, t(2.1) and Pareto — its critical value is an
  exact functional of k standard normals, so only the within-block CLT (m →
  ∞) is asymptotic, and it evidently bites fast.
- **Standard AR** is well calibrated too (0.040–0.052), even under heavy
  tails: it studentises by a variance estimated from the same data, which
  self-corrects to first order. The honest conclusion is that the robust
  tests' advantage is the *finite-sample guarantee*, not a size disaster of
  the baseline at these sample sizes.

**Takeaway:** all tests are valid here; they differ in what they can
promise. MoM-AR promises size ≤ δ *for every n and every (A3) distribution*;
SN-AR and AR promise it only in the limit.

## I2 — Power: the price of the guarantee, and the SN way out

Power curves at n = 2000 make the trade concrete (rejection rates at
|β₀−β| = 0.05 / 0.15 / 0.30, Gaussian):

- Standard AR: 0.57–0.63 / ≈1.00 / 1.00
- SN-AR: 0.27–0.44 / ≈0.99 / 1.00
- MoM-AR feasible: ≈0.00 / ≈0.20 / ≈0.94
- MoM-AR oracle: 0.00 / ≈0.01 / ≈0.99

The MoM-AR acceptance region is roughly ±0.2 wide — the same 5× factor seen
in I1 translated into detectable effect sizes. The self-normalised test is
the pragmatic middle: nearly the standard AR's power (its handicap shrinks
further under t(2.1), where the two are indistinguishable by |β₀−β| = 0.05)
while remaining scale-free and robust in construction.

One artifact to flag rather than celebrate: under t(2.1) the *feasible*
MoM-AR is much more powerful than the oracle (0.94 vs 0.00 at 0.15). Its
MAD-based scale estimate sits below the true σ_Zε when block means are
heavy-tailed, so the band shrinks. Size stayed ≤ δ (I1), but that is the
conservatism slack absorbing an anti-conservative scale estimate — a lucky
cancellation, not a guarantee.

**Takeaway:** if you need the finite-sample guarantee, you pay in power; if
you need power with robustness, the SN test is the recommendation this
experiment supports.

## I3 — Confidence sets: what you pay for, in centimetres

Strong instrument (μ_ZX = 1): every CS is a single bounded interval;
coverage is 1.000 for MoM-AR (both variants), 0.947–0.952 for SN-AR and AR.
Median lengths under Gaussian: AR 0.088 < SN-AR 0.123 < MoM-AR feasible
0.398 < oracle 0.443. The 5× length ratio is the same conservatism seen in
I1/I2, now in interpretable units of β.

Weak instrument (μ_ZX = 0.05): the designs separate qualitatively.
- Standard AR: unbounded in ~67–70% of replications (the classic Dufour
  outcome — when the instrument can't be distinguished from irrelevant, an
  honest CS must be allowed to say "anything").
- SN-AR: unbounded in ~85–88% and frequently a union of 2–3 disjoint
  intervals — the piecewise-affine geometry is not a theoretical curiosity;
  it happens most of the time under weak identification.
- MoM-AR: **never unbounded** and almost always one interval — but its
  median length inflates from 0.44 to 9.45 (21×). Structurally, its
  statistic grows linearly in |β₀| with slope given by the median block mean
  of ZX, which is nonzero almost surely, so the set is always bounded; the
  weak-instrument problem is expressed entirely as length. (This is
  consistent with coverage: validity is only claimed at the structural β.)

**Takeaway:** all three procedures tell you identification is weak, in
different dialects: AR/SN-AR by going unbounded or disconnected, MoM-AR by
becoming very long. Coverage never fails anywhere — including 20,000
weak-instrument replications.

## I4 — Single-interval condition: confirmed, and conservative

Two clean facts:

1. **The deterministic proposition is airtight in practice.** In every one
   of the replications where all k block means of ZX shared a sign, the
   exact CS was a single interval — zero exceptions in 28,000 replications
   (checked programmatically). prop:mono_det is confirmed as stated.
2. **The probabilistic sufficient condition is very conservative.** The
   premise itself (all same sign) transitions from never-holding to
   always-holding as μ_ZX goes 0.05 → 0.6 at n = 2000, and as n goes 250 →
   2000 at μ_ZX = 0.75. But the Chebyshev bound eq:mono_cheby only certifies
   this at n* = 51,216 — roughly 25–50× after it is already empirically
   certain. Moreover the CS is a single interval ≥ 99.6% of the time even
   when the same-sign premise fails badly (at μ_ZX = 0.05 the premise never
   holds, yet 99.6% of CSs are single intervals): same-sign is sufficient
   but far from necessary, because the median line's sign behaviour, not
   every block's, is what matters in practice.

**Takeaway:** report prop:mono_det as exact and prop:mono_cheby as a valid
but loose certificate; in practice single-interval CSs are the norm far
beyond where the theory can promise them.

## I5 — The R_k table: what the critical values say

The (1−δ)-quantiles of R_k = |med(ξ)|/MAD(ξ) fall roughly like 1/√k (more
blocks → the median of k block means concentrates faster relative to their
MAD): c_{k,0.05} goes 4.24 (k=5) → 0.92 (k=24) → 0.57 (k=50). Two features
worth a sentence in the appendix:

- **Small k is expensive.** At k = 5 the 99% critical value is 10.3 — with
  so few blocks the MAD is a very noisy scale, and the test needs a huge
  margin. The choice k = ⌈8 ln(1/δ)⌉ (= 24 at δ = 0.05) is comfortably in
  the flat region.
- **The even/odd zigzag** (c_{19,0.05} = 0.996 < c_{20,0.05} = 1.052) is not
  simulation noise: with the rank-⌈k/2⌉ convention, even k uses the lower of
  the two middle order statistics, which shifts the null distribution of
  |med|/MAD slightly. Statistic and table use the same convention, so the
  test is exact either way — but the table should not be interpolated
  across the even/odd boundary.

## The one-paragraph story of the whole section

Under light tails the classical estimator and test are the efficient choice,
and robustness costs a visible but modest premium (E1, I1–I3 strong). Under
heavy tails with only finite variance, the classical point estimator loses
exactly where the theory says it must — in its deviation tails, at a
polynomial-in-1/δ rate — while the MoM estimators keep their sub-Gaussian
behaviour (E2, E2b), and the failure of the MoM instrument-strength
conditions is detectable as bias rather than silent (E3). On the inference
side, the MoM-AR test delivers a finite-sample, distribution-free-within-(A3)
guarantee whose price is a ~5× wider band (I1–I3), the self-normalised test
recovers nearly classical power while keeping robustness (I1–I2), the
confidence-set geometry behaves exactly as the theory describes — including
under weak instruments, where each procedure signals non-identification in
its own way (I3, I4) — and the critical values needed to use the SN test in
practice are tabulated once and for all (I5).
