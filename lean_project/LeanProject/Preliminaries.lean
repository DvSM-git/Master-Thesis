/-
# Preliminaries (`Paper/iteration4/preliminaries.tex`)

Machine-checked statements corresponding to Section "Preliminaries":

* `beta_identified` — identification of the structural parameter:
  under exogeneity (A1) and relevance (A2), `β = μ_{ZY} / μ_{ZX}`.
* `IsMedian` / `median_deviation_count` — Step 2 (counting argument) of
  Theorem `thm:mom_scalar`: if a median of the block means deviates from the
  target by more than `t`, at least half of the block means do.
* `variance_block_mean` — the variance of a block mean of `m` pairwise
  independent random variables with common variance `σ²` is `σ²/m`.
* `chebyshev_block_quarter` — Step 1 of Theorem `thm:mom_scalar`: each block
  mean deviates by more than `2σ/√m` with probability at most `1/4`.
* `hoeffding_count` — Step 3 of Theorem `thm:mom_scalar`: for independent
  `[0,1]`-valued indicators with mean at most `1/4`, the probability that
  their sum reaches `k/2` is at most `exp(-k/8)`.
* `exp_ceil_log_le` / `two_mul_exp_ceil_log_le` — the choice
  `k = ⌈8 ln(1/δ)⌉` (resp. `⌈8 ln(2/δ)⌉`) turns `e^{-k/8}` (resp. `2e^{-k/8}`)
  into a bound of at most `δ`.

Remark: the passage from `eq:mom_conc` to the sub-Gaussian form
`eq:mom_subgauss` in the paper compares the radius `2σ/√m` with
`σ√(32 ln(1/δ)/n)`. Since `k = ⌈8 ln(1/δ)⌉ ≥ 8 ln(1/δ)` and `m = ⌊n/k⌋ ≤ n/k`,
one has `2σ/√m ≥ σ√(32 ln(1/δ)/n)`, so the sub-Gaussian radius is *smaller*
than the proven radius and the implication requires `n = k·m` together with
`k = 8 ln(1/δ)` exactly (or an adjusted constant).  We therefore formalize the
constituent steps at radius `2σ/√m`.
-/
import Mathlib

open MeasureTheory ProbabilityTheory Real

namespace Thesis

/-! ## Identification (preliminaries.tex, after (A1)–(A3)) -/

/-- **Identification of the structural parameter.**
If `Y = β X + ε` (`eq:structural`), the moments `E[ZX]` and `E[Zε]` exist,
exogeneity (A1) `E[Zε] = 0` and relevance (A2) `E[ZX] ≠ 0` hold, then
`β = E[ZY] / E[ZX]`. -/
theorem beta_identified {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (Z X Y ε : Ω → ℝ) (β : ℝ)
    (hstruct : ∀ ω, Y ω = β * X ω + ε ω)
    (hZX : Integrable (fun ω => Z ω * X ω) μ)
    (hZε : Integrable (fun ω => Z ω * ε ω) μ)
    (hexog : ∫ ω, Z ω * ε ω ∂μ = 0)
    (hrelev : ∫ ω, Z ω * X ω ∂μ ≠ 0) :
    β = (∫ ω, Z ω * Y ω ∂μ) / (∫ ω, Z ω * X ω ∂μ) := by
  have hZY : (fun ω => Z ω * Y ω) = fun ω => β * (Z ω * X ω) + Z ω * ε ω := by
    funext ω; rw [hstruct ω]; ring
  have hint : ∫ ω, Z ω * Y ω ∂μ = β * ∫ ω, Z ω * X ω ∂μ := by
    rw [hZY, integral_add (hZX.const_mul β) hZε, integral_const_mul, hexog, add_zero]
  rw [hint]
  field_simp

/-! ## The median counting argument (Step 2 of Theorem `thm:mom_scalar`) -/

/-- `m` is a median of the values `w 0, …, w (k-1)`: at least half of them lie
at or below `m`, and at least half lie at or above `m`.  Any usual empirical
median (in particular the sorted middle order statistic) satisfies this. -/
def IsMedian {k : ℕ} (w : Fin k → ℝ) (m : ℝ) : Prop :=
  k ≤ 2 * (Finset.univ.filter fun j => w j ≤ m).card ∧
  k ≤ 2 * (Finset.univ.filter fun j => m ≤ w j).card

/-- **Step 2 (counting argument) of Theorem `thm:mom_scalar`**: if a median of
the block means `w j` deviates from `μX` by more than `t`, then at least `k/2`
of the block means deviate by more than `t`. -/
theorem median_deviation_count {k : ℕ} {w : Fin k → ℝ} {med μX t : ℝ}
    (hmed : IsMedian w med) (hdev : t < |med - μX|) :
    k ≤ 2 * (Finset.univ.filter fun j => t < |w j - μX|).card := by
  rcases lt_abs.mp hdev with h | h
  · -- the median exceeds `μX + t`: all blocks with `med ≤ w j` deviate upwards
    refine le_trans hmed.2 (Nat.mul_le_mul_left 2 (Finset.card_le_card ?_))
    intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    have hgt : t < w j - μX := by linarith
    exact hgt.trans_le (le_abs_self _)
  · -- the median falls below `μX - t`: all blocks with `w j ≤ med` deviate downwards
    refine le_trans hmed.1 (Nat.mul_le_mul_left 2 (Finset.card_le_card ?_))
    intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    have hgt : t < μX - w j := by linarith
    calc t < μX - w j := hgt
    _ ≤ |w j - μX| := by rw [abs_sub_comm]; exact le_abs_self _

/-! ## Step 1 of Theorem `thm:mom_scalar`: Chebyshev on each block -/

/-- The variance of a block mean of `m` pairwise independent random variables
with common variance `σ²` is `σ²/m` (used in Step 1 of `thm:mom_scalar` and in
Step 2 of `thm:iv`/`thm:mor`). -/
theorem variance_block_mean {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsFiniteMeasure μ] {m : ℕ} (hm : 0 < m)
    {X : Fin m → Ω → ℝ} (hL2 : ∀ i, MemLp (X i) 2 μ)
    (hindep : Set.Pairwise ↑(Finset.univ : Finset (Fin m))
      fun i j => IndepFun (X i) (X j) μ)
    {σ2 : ℝ} (hvar : ∀ i, Var[X i; μ] = σ2) :
    Var[fun ω => (∑ i, X i ω) / m; μ] = σ2 / m := by
  have hm' : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hm.ne'
  have h1 : (fun ω => (∑ i, X i ω) / m) = fun ω => (∑ i, X i ω) * (1 / m) := by
    funext ω; rw [mul_one_div]
  have h2 : (fun ω => ∑ i, X i ω) = ∑ i, X i := by
    funext ω; simp [Finset.sum_apply]
  rw [h1, variance_mul_const, h2,
    IndepFun.variance_sum (fun i _ => hL2 i) hindep]
  simp only [hvar, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- **Step 1 of Theorem `thm:mom_scalar`** (Chebyshev on each block): a random
variable `W` (e.g. a block mean) with variance at most `σ²/m` deviates from its
mean by at least `2σ/√m` with probability at most `1/4`. -/
theorem chebyshev_block_quarter {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsFiniteMeasure μ] {W : Ω → ℝ} (hW : MemLp W 2 μ)
    {σ m : ℝ} (hσ : 0 < σ) (hm : 0 < m)
    (hvar : Var[W; μ] ≤ σ ^ 2 / m) :
    μ {ω | 2 * σ / Real.sqrt m ≤ |W ω - μ[W]|} ≤ ENNReal.ofReal (1 / 4) := by
  have hsm : 0 < Real.sqrt m := Real.sqrt_pos.mpr hm
  have hc : 0 < 2 * σ / Real.sqrt m := by positivity
  refine le_trans (meas_ge_le_variance_div_sq hW hc) (ENNReal.ofReal_le_ofReal ?_)
  have hsq : (2 * σ / Real.sqrt m) ^ 2 = 4 * σ ^ 2 / m := by
    rw [div_pow, mul_pow, Real.sq_sqrt hm.le]; norm_num
  rw [hsq, div_le_div_iff₀ (by positivity) (by norm_num)]
  have h4 : 4 * σ ^ 2 / m = 4 * (σ ^ 2 / m) := by ring
  rw [h4]
  linarith

/-! ## Step 3 of Theorem `thm:mom_scalar`: Hoeffding's inequality -/

/-- **Step 3 of Theorem `thm:mom_scalar`** (Hoeffding): if `ζ 1, …, ζ k` are
independent, `[0,1]`-valued, and each has mean at most `1/4`, then
`P[∑ ζ j ≥ k/2] ≤ exp (-k/8)`.  Uses Hoeffding's lemma
(`hasSubgaussianMGF_of_mem_Icc`) and the sub-Gaussian Hoeffding inequality
(`measure_sum_ge_le_of_iIndepFun`) from Mathlib. -/
theorem hoeffding_count {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {k : ℕ} (hk : 0 < k)
    {ζ : Fin k → Ω → ℝ}
    (hmeas : ∀ i, AEMeasurable (ζ i) μ)
    (hindep : iIndepFun ζ μ)
    (hbdd : ∀ i, ∀ᵐ ω ∂μ, ζ i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : ∀ i, μ[ζ i] ≤ 1 / 4) :
    μ.real {ω | (k : ℝ) / 2 ≤ ∑ i, ζ i ω} ≤ Real.exp (-(k : ℝ) / 8) := by
  have hk' : (k : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hk.ne'
  -- each ζ i is integrable (bounded on a probability space)
  have hint : ∀ i, Integrable (ζ i) μ := fun i =>
    Integrable.of_mem_Icc 0 1 (hmeas i) (hbdd i)
  -- centred variables are sub-Gaussian with parameter 1/4 by Hoeffding's lemma
  have hsubG : ∀ i, HasSubgaussianMGF (fun ω => ζ i ω - μ[ζ i]) (1 / 4) μ := by
    intro i
    have h := hasSubgaussianMGF_of_mem_Icc (hmeas i) (hbdd i)
    have hc : ((‖(1 : ℝ) - 0‖₊ / 2) ^ 2 : NNReal) = 1 / 4 := by
      simp only [sub_zero, nnnorm_one]
      norm_num
    rwa [hc] at h
  -- centred variables remain independent
  have hindep' : iIndepFun (fun i ω => ζ i ω - μ[ζ i]) μ := by
    have := hindep.comp (g := fun i (x : ℝ) => x - μ[ζ i])
      (fun i => measurable_id.sub_const _)
    exact this
  -- Hoeffding's inequality at ε = k/4
  have hHoeff :=
    HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hindep'
      (c := fun _ => 1 / 4) (s := Finset.univ)
      (fun i _ => hsubG i)
      (ε := (k : ℝ) / 4) (by positivity)
  -- the event {∑ ζ ≥ k/2} entails {∑ (ζ - E ζ) ≥ k/4} since ∑ E ζ ≤ k/4
  have hsubset : {ω | (k : ℝ) / 2 ≤ ∑ i, ζ i ω}
      ⊆ {ω | (k : ℝ) / 4 ≤ ∑ i, (ζ i ω - μ[ζ i])} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have hsum : ∑ i, μ[ζ i] ≤ (k : ℝ) / 4 := by
      calc ∑ i, μ[ζ i] ≤ ∑ _i : Fin k, (1 / 4 : ℝ) :=
            Finset.sum_le_sum fun i _ => hmean i
      _ = (k : ℝ) / 4 := by
            simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
            ring
    rw [Finset.sum_sub_distrib]
    linarith
  have hmono : μ.real {ω | (k : ℝ) / 2 ≤ ∑ i, ζ i ω}
      ≤ μ.real {ω | (k : ℝ) / 4 ≤ ∑ i, (ζ i ω - μ[ζ i])} :=
    measureReal_mono hsubset
  refine le_trans (hmono.trans hHoeff) (le_of_eq ?_)
  congr 1
  -- -(k/4)² / (2 · (k · (1/4))) = -k/8
  have hsum : ((∑ _i : Fin k, (1 / 4 : NNReal) : NNReal) : ℝ) = (k : ℝ) / 4 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    push_cast
    ring
  rw [hsum]
  field_simp
  ring

/-! ## The choice `k = ⌈8 ln(1/δ)⌉` -/

/-- If `k ≥ 8 ln(1/δ)` then `e^{-k/8} ≤ δ` (used to pass from `eq:mom_conc` to
the confidence form in Theorems `thm:mom_scalar`, `thm:mor`, `thm:mom_ar_size`). -/
theorem exp_le_of_log_le {δ : ℝ} (hδ0 : 0 < δ) {k : ℝ}
    (hk : 8 * Real.log (1 / δ) ≤ k) : Real.exp (-k / 8) ≤ δ := by
  have h1 : Real.log (1 / δ) = -Real.log δ := by rw [one_div, Real.log_inv]
  have h2 : -k / 8 ≤ Real.log δ := by rw [h1] at hk; linarith
  calc Real.exp (-k / 8) ≤ Real.exp (Real.log δ) := Real.exp_le_exp.mpr h2
  _ = δ := Real.exp_log hδ0

/-- With `k = ⌈8 ln(1/δ)⌉` one has `e^{-k/8} ≤ δ`. -/
theorem exp_ceil_log_le {δ : ℝ} (hδ0 : 0 < δ) :
    Real.exp (-(⌈8 * Real.log (1 / δ)⌉₊ : ℝ) / 8) ≤ δ :=
  exp_le_of_log_le hδ0 (Nat.le_ceil _)

/-- With `k = ⌈8 ln(2/δ)⌉` one has `2 e^{-k/8} ≤ δ` (Step 2 of Theorem
`thm:rom`). -/
theorem two_mul_exp_ceil_log_le {δ : ℝ} (hδ0 : 0 < δ) :
    2 * Real.exp (-(⌈8 * Real.log (2 / δ)⌉₊ : ℝ) / 8) ≤ δ := by
  have h1 : (1 : ℝ) / (δ / 2) = 2 / δ := one_div_div δ 2
  have h : Real.exp (-(⌈8 * Real.log (2 / δ)⌉₊ : ℝ) / 8) ≤ δ / 2 := by
    refine exp_le_of_log_le (by positivity) ?_
    rw [h1]
    exact Nat.le_ceil _
  linarith

end Thesis
