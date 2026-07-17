/-
# The Standard Instrumental Variable Estimator (`Paper/iteration4/iv.tex`)

Machine-checked statements corresponding to Section "The Standard Instrumental
Variable Estimator":

* `error_decomposition` — Lemma `lem:iv_decomp` (and its block version
  Lemma `lem:mor_decomp`): the algebraic identity
  `μ̂_{ZY}/μ̂_{ZX} - β = μ̂_{Zε}/μ̂_{ZX}`, stated over an arbitrary finite index
  set so that it covers both the full sample and a single block.
* `sample_moment_mean_zero` — the second half of `lem:iv_decomp`:
  `E[μ̂_{Zε}] = 0` under exogeneity.
* `two_event_bound` — Step 1 of Theorem `thm:iv`/`thm:mor`: on
  `A(D) ∩ B(D,t)` the estimation error is at most `t`.
* `denominator_failure` / `prob_denominator_failure` — Step 2 of `thm:iv`:
  the reverse-triangle-inequality inclusion and the resulting Chebyshev bound
  `P[Aᶜ] ≤ 4σ²_{ZX}/(n μ²_{ZX})`.
* `prob_numerator_failure` — Step 2 of `thm:iv`: Chebyshev bound on `B(D,t)ᶜ`.
* `strength_budget_iff` / `numerator_threshold_eq` — Step 3 of `thm:iv`:
  the error-budget algebra and the explicit threshold
  `t = 2√2 σ_{Zε}/(|μ_{ZX}|√(δn))`.
* `opt_argmax_f`, `opt_f_le`, `opt_critical_mem_Ioo` — Step 3 of Theorem
  `thm:iv_opt`: with `ρ ∈ (0,1)`, the map `f(η) = η²(1 - ρ/(1-η)²)` is
  maximised over `(0,1)` at `η* = 1 - ρ^{1/3}` with value `(1-ρ^{1/3})³`.
  (The same optimisation is reused verbatim in Theorem `thm:mor_opt`.)
* `cantelli` — Lemma `lem:cantelli` (Cantelli's inequality), proved from
  Markov's inequality (it is not in Mathlib).
* `cantelli_lt_chebyshev` — the strict comparison stated in `lem:cantelli`.
* `h_iv_pos_iff` / `h_iv_feasible_iff` — the sign and feasibility analysis of
  `h(η)` in Theorem `thm:iv_cantelli` (`eq:h_eta_iv`, `eq:iv_cantelli_strength`).
-/
import Mathlib

open MeasureTheory ProbabilityTheory Real

namespace Thesis

/-! ## Error decomposition (Lemma `lem:iv_decomp`, Lemma `lem:mor_decomp`) -/

/-- **Error decomposition** (`eq:iv_error`, `eq:block_error`).  For any finite
index set `s` (the full sample in `lem:iv_decomp`, a block `B_j` in
`lem:mor_decomp`): if `Y_i = β X_i + ε_i` and the sample moment
`μ̂_{ZX} = (∑_{i∈s} Z_i X_i)/m` is nonzero, then
`μ̂_{ZY}/μ̂_{ZX} - β = μ̂_{Zε}/μ̂_{ZX}`. -/
theorem error_decomposition {ι : Type*} (s : Finset ι)
    (Z X Y ε : ι → ℝ) (β mm : ℝ) (hmm : mm ≠ 0)
    (hstruct : ∀ i ∈ s, Y i = β * X i + ε i)
    (hden : (∑ i ∈ s, Z i * X i) / mm ≠ 0) :
    ((∑ i ∈ s, Z i * Y i) / mm) / ((∑ i ∈ s, Z i * X i) / mm) - β
      = ((∑ i ∈ s, Z i * ε i) / mm) / ((∑ i ∈ s, Z i * X i) / mm) := by
  have hden' : (∑ i ∈ s, Z i * X i) ≠ 0 := by
    intro h; exact hden (by rw [h, zero_div])
  have h : ∑ i ∈ s, Z i * Y i
      = β * (∑ i ∈ s, Z i * X i) + ∑ i ∈ s, Z i * ε i := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i hi => by rw [hstruct i hi]; ring
  rw [h]
  field_simp
  ring

/-- The second half of Lemma `lem:iv_decomp` / `lem:mor_decomp`: the sample
moment `μ̂_{Zε} = (∑ W_i)/n` of mean-zero integrable variables has mean zero
(exogeneity (A1) plus linearity of expectation). -/
theorem sample_moment_mean_zero {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {n : ℕ} (W : Fin n → Ω → ℝ)
    (hint : ∀ i, Integrable (W i) μ)
    (hmean : ∀ i, ∫ ω, W i ω ∂μ = 0) :
    ∫ ω, (∑ i, W i ω) / n ∂μ = 0 := by
  simp_rw [div_eq_mul_inv]
  rw [integral_mul_const, integral_finsetSum _ fun i _ => hint i]
  simp [hmean]

/-! ## Step 1: the two-event decomposition -/

/-- **Step 1 of Theorems `thm:iv` and `thm:mor`** (deterministic core): on the
event `A(D) ∩ B(D,t)`, i.e. when `|den| ≥ D` and `|num| ≤ t·D` with `D > 0`,
the ratio satisfies `|num/den| ≤ t`. -/
theorem two_event_bound {num den D t : ℝ} (hD : 0 < D)
    (hA : D ≤ |den|) (hB : |num| ≤ t * D) : |num / den| ≤ t := by
  have hden_pos : 0 < |den| := lt_of_lt_of_le hD hA
  have ht : 0 ≤ t := by
    nlinarith [le_trans (abs_nonneg num) hB, hD]
  rw [abs_div, div_le_iff₀ hden_pos]
  calc |num| ≤ t * D := hB
  _ ≤ t * |den| := by nlinarith

/-! ## Step 2: Chebyshev bounds on the two failure events -/

/-- The reverse-triangle-inequality inclusion of Step 2 of `thm:iv`
(deterministic form): if `|w| < |μZX|/2` then `|w - μZX| ≥ |μZX|/2`. -/
theorem denominator_failure {w μZX : ℝ} (h : |w| < |μZX| / 2) :
    |μZX| / 2 ≤ |w - μZX| := by
  have h1 : |μZX| - |w| ≤ |w - μZX| := by
    rw [abs_sub_comm]
    linarith [abs_sub_abs_le_abs_sub μZX w]
  linarith

/-- **Step 2 of Theorem `thm:iv`, event `Aᶜ`**: for a random variable `W`
(the sample moment `μ̂_{ZX}`) with nonzero mean,
`P[|W| < |E W|/2] ≤ Var(W)/(|E W|/2)² = 4·Var(W)/(E W)²`.
With `Var(W) = σ²_{ZX}/n` this is the bound `4σ²_{ZX}/(n μ²_{ZX})`. -/
theorem prob_denominator_failure {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsFiniteMeasure μ]
    {W : Ω → ℝ} (hW : MemLp W 2 μ) (hμ : μ[W] ≠ 0) :
    μ {ω | |W ω| < |μ[W]| / 2} ≤ ENNReal.ofReal (4 * Var[W; μ] / μ[W] ^ 2) := by
  have hpos : 0 < |μ[W]| / 2 := by
    have := abs_pos.mpr hμ; linarith
  refine le_trans (measure_mono fun ω hω => denominator_failure hω)
    (le_trans (meas_ge_le_variance_div_sq hW hpos) (ENNReal.ofReal_le_ofReal ?_))
  have h1 : (|μ[W]| / 2) ^ 2 = μ[W] ^ 2 / 4 := by
    rw [div_pow, sq_abs]
    norm_num
  rw [h1, div_div_eq_mul_div, div_le_div_iff₀ (by positivity) (by positivity)]
  ring_nf
  nlinarith [variance_nonneg W μ, sq_nonneg (μ[W])]

/-- **Step 2 of Theorem `thm:iv`, event `Bᶜ`**: for a mean-zero random
variable `W` (the sample moment `μ̂_{Zε}`) and a threshold `c > 0`,
`P[c ≤ |W|] ≤ Var(W)/c²`.  Applied with `c = t·D` this is the numerator
failure bound. -/
theorem prob_numerator_failure {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsFiniteMeasure μ]
    {W : Ω → ℝ} (hW : MemLp W 2 μ) (hmean : μ[W] = 0)
    {c : ℝ} (hc : 0 < c) :
    μ {ω | c ≤ |W ω|} ≤ ENNReal.ofReal (Var[W; μ] / c ^ 2) := by
  have h := meas_ge_le_variance_div_sq hW hc
  simpa [hmean] using h

/-! ## Step 3: error-budget algebra -/

/-- **Step 3 of Theorem `thm:iv`, budget for `Aᶜ`** (`eq:iv_strength`):
`4σ²/(nμ²) ≤ δ/2` holds if and only if `n ≥ 8σ²/(δμ²)`. -/
theorem strength_budget_iff {σ2 μ2 δ n : ℝ} (_hσ : 0 ≤ σ2) (hμ : 0 < μ2)
    (hδ : 0 < δ) (hn : 0 < n) :
    4 * σ2 / (n * μ2) ≤ δ / 2 ↔ 8 * σ2 / (δ * μ2) ≤ n := by
  rw [div_le_div_iff₀ (by positivity) (by norm_num),
    div_le_iff₀ (by positivity)]
  constructor <;> intro h <;> nlinarith

/-- **Step 3 of Theorem `thm:iv`, threshold for `Bᶜ`**: the choice
`t = 2√2 σ_{Zε}/(|μ_{ZX}| √(δn))` makes the numerator failure budget exact:
`4σ²_{Zε}/(n t² μ²_{ZX}) = δ/2`. -/
theorem numerator_threshold_eq {σε μZX δ n : ℝ}
    (hσ : 0 < σε) (hμ : μZX ≠ 0) (hδ : 0 < δ) (hn : 0 < n) :
    4 * σε ^ 2 /
      (n * (2 * Real.sqrt 2 * σε / (|μZX| * Real.sqrt (δ * n))) ^ 2 * μZX ^ 2)
      = δ / 2 := by
  have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hδn : Real.sqrt (δ * n) ^ 2 = δ * n := Real.sq_sqrt (by positivity)
  have habs : |μZX| ^ 2 = μZX ^ 2 := sq_abs _
  have hμ2 : (0:ℝ) < μZX ^ 2 := by positivity
  have hsδn : 0 < Real.sqrt (δ * n) := Real.sqrt_pos.mpr (by positivity)
  have habs' : 0 < |μZX| := abs_pos.mpr hμ
  rw [div_pow, mul_pow, mul_pow, mul_pow, h2, hδn, habs]
  field_simp
  ring

/-! ## The optimised threshold (Theorem `thm:iv_opt`, Step 3) -/

/-- Key inequality behind Theorem `thm:iv_opt` (and Theorem `thm:mor_opt`):
for `c, η ∈ (0,1)`, `f(η) = η²(1 - c³/(1-η)²) ≤ (1-c)³`.
The proof uses the factorisation
`(1-c)³u² - (1-u)²(u² - c³) = (u - c)² (c + 2(1-c)u - u²)` with `u = 1-η`. -/
theorem opt_f_le {c η : ℝ} (hc0 : 0 < c) (hc1 : c < 1)
    (hη0 : 0 < η) (hη1 : η < 1) :
    η ^ 2 * (1 - c ^ 3 / (1 - η) ^ 2) ≤ (1 - c) ^ 3 := by
  have hu0 : 0 < 1 - η := by linarith
  have hu2 : (0:ℝ) < (1 - η) ^ 2 := by positivity
  -- the quadratic factor is positive on the relevant range
  have hfac : 0 < c + 2 * (1 - c) * (1 - η) - (1 - η) ^ 2 := by
    obtain h | h := le_total (1 - η) (1/2)
    · nlinarith
    · nlinarith
  have key : (1 - c) ^ 3 * (1 - η) ^ 2 - η ^ 2 * ((1 - η) ^ 2 - c ^ 3)
      = ((1 - η) - c) ^ 2 * (c + 2 * (1 - c) * (1 - η) - (1 - η) ^ 2) := by
    ring
  have h1 : η ^ 2 * ((1 - η) ^ 2 - c ^ 3) ≤ (1 - c) ^ 3 * (1 - η) ^ 2 := by
    nlinarith [mul_nonneg (sq_nonneg ((1 - η) - c)) hfac.le]
  have h2 : η ^ 2 * (1 - c ^ 3 / (1 - η) ^ 2)
      = η ^ 2 * ((1 - η) ^ 2 - c ^ 3) / (1 - η) ^ 2 := by
    field_simp
  rw [h2, div_le_iff₀ hu2]
  exact h1

/-- The critical point `η* = 1 - c` attains the value `(1-c)³`
(the computation `f(η*) = (1-ρ^{1/3})³` in Step 3 of `thm:iv_opt`). -/
theorem opt_f_eq {c : ℝ} (_hc0 : 0 < c) (_hc1 : c < 1) :
    (1 - c) ^ 2 * (1 - c ^ 3 / (1 - (1 - c)) ^ 2) = (1 - c) ^ 3 := by
  have h : (1 : ℝ) - (1 - c) = c := by ring
  rw [h]
  field_simp

/-- `η* = 1 - ρ^{1/3}` lies in `(0,1)` if and only if `ρ < 1`
(the feasibility condition of Theorem `thm:iv_opt`). -/
theorem opt_critical_mem_Ioo {ρ : ℝ} (hρ0 : 0 < ρ) :
    1 - ρ ^ ((1:ℝ)/3) ∈ Set.Ioo (0:ℝ) 1 ↔ ρ < 1 := by
  have hc0 : 0 < ρ ^ ((1:ℝ)/3) := Real.rpow_pos_of_pos hρ0 _
  have hc3 : (ρ ^ ((1:ℝ)/3)) ^ 3 = ρ := by
    rw [← Real.rpow_natCast (ρ ^ ((1:ℝ)/3)) 3, ← Real.rpow_mul hρ0.le]
    norm_num
  constructor
  · rintro ⟨h1, _⟩
    have hlt : ρ ^ ((1:ℝ)/3) < 1 := by linarith
    have hcube : (ρ ^ ((1:ℝ)/3)) ^ 3 < 1 := pow_lt_one₀ hc0.le hlt (by norm_num)
    rwa [hc3] at hcube
  · intro h
    have hlt : ρ ^ ((1:ℝ)/3) < 1 :=
      Real.rpow_lt_one hρ0.le h (by norm_num)
    exact ⟨by linarith, by linarith⟩

/-- **Step 3 of Theorem `thm:iv_opt`** (and of Theorem `thm:mor_opt`),
assembled: for `ρ ∈ (0,1)` and `c = ρ^{1/3}`, the function
`f(η) = η²(1 - ρ/(1-η)²)` satisfies `f(η) ≤ (1-c)³` for all `η ∈ (0,1)`,
with equality at the optimiser `η* = 1 - c`. -/
theorem opt_argmax_f {ρ : ℝ} (hρ0 : 0 < ρ) (hρ1 : ρ < 1) :
    (∀ η, 0 < η → η < 1 →
      η ^ 2 * (1 - ρ / (1 - η) ^ 2) ≤ (1 - ρ ^ ((1:ℝ)/3)) ^ 3) ∧
    (1 - ρ ^ ((1:ℝ)/3)) ^ 2 *
      (1 - ρ / (1 - (1 - ρ ^ ((1:ℝ)/3))) ^ 2) = (1 - ρ ^ ((1:ℝ)/3)) ^ 3 := by
  set c := ρ ^ ((1:ℝ)/3) with hc
  have hc0 : 0 < c := Real.rpow_pos_of_pos hρ0 _
  have hc1 : c < 1 := Real.rpow_lt_one hρ0.le hρ1 (by norm_num)
  have hc3 : c ^ 3 = ρ := by
    rw [hc, ← Real.rpow_natCast (ρ ^ ((1:ℝ)/3)) 3, ← Real.rpow_mul hρ0.le]
    norm_num
  constructor
  · intro η hη0 hη1
    have := opt_f_le hc0 hc1 hη0 hη1
    rwa [hc3] at this
  · have := opt_f_eq hc0 hc1
    rwa [hc3] at this

/-! ## Cantelli's inequality (Lemma `lem:cantelli`) -/

set_option maxHeartbeats 1000000 in
/-- **Cantelli's inequality** (Lemma `lem:cantelli`): for a square-integrable
random variable `W` on a probability space and `τ > 0`,
`P[W - E W ≤ -τ] ≤ Var(W)/(Var(W) + τ²)`.  (Not available in Mathlib; proved
here from Markov's inequality applied to `(u - (W - E W))²` with the optimal
shift `u = Var(W)/τ`.) -/
theorem cantelli {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {W : Ω → ℝ} (hW : MemLp W 2 μ)
    {τ : ℝ} (hτ : 0 < τ) :
    μ.real {ω | W ω - μ[W] ≤ -τ} ≤ Var[W; μ] / (Var[W; μ] + τ ^ 2) := by
  -- name the variance and the shift as opaque quantities
  obtain ⟨σ2, hσ2⟩ : ∃ V : ℝ, Var[W; μ] = V := ⟨_, rfl⟩
  have hσ2_nonneg : 0 ≤ σ2 := by rw [← hσ2]; exact variance_nonneg W μ
  obtain ⟨u, hu⟩ : ∃ x : ℝ, σ2 / τ = x := ⟨_, rfl⟩
  have hu_nonneg : 0 ≤ u := by rw [← hu]; positivity
  have hut : 0 < u + τ := by linarith
  rw [hσ2]
  -- the centred variable and its square
  have hWc : MemLp (fun ω => W ω - μ[W]) 2 μ := hW.sub (memLp_const _)
  have hWc_int : Integrable (fun ω => W ω - μ[W]) μ :=
    hWc.integrable one_le_two
  have hWc_sq_int : Integrable (fun ω => (W ω - μ[W]) ^ 2) μ := by
    simpa using hWc.integrable_sq
  have hWc_mean : ∫ ω, (W ω - μ[W]) ∂μ = 0 := by
    rw [integral_sub (hW.integrable one_le_two) (integrable_const _)]
    simp
  -- the auxiliary function V = u - (W - E W), and E[V²] = σ² + u²
  have hint2 : Integrable (fun ω => (2 * u) * (W ω - μ[W])) μ := by
    exact hWc_int.const_mul _
  have hint1 : Integrable
      (fun ω => (W ω - μ[W]) ^ 2 - (2 * u) * (W ω - μ[W])) μ := by
    exact hWc_sq_int.sub hint2
  have hV_sq_int : Integrable (fun ω => (u - (W ω - μ[W])) ^ 2) μ := by
    have h : (fun ω => (u - (W ω - μ[W])) ^ 2)
        = fun ω => ((W ω - μ[W]) ^ 2 - (2 * u) * (W ω - μ[W])) + u ^ 2 := by
      funext ω; ring
    rw [h]
    exact hint1.add (integrable_const _)
  have hV_sq_mean : ∫ ω, (u - (W ω - μ[W])) ^ 2 ∂μ = σ2 + u ^ 2 := by
    have h : ∀ ω, (u - (W ω - μ[W])) ^ 2
        = ((W ω - μ[W]) ^ 2 - (2 * u) * (W ω - μ[W])) + u ^ 2 := fun ω => by ring
    simp_rw [h]
    rw [integral_add hint1 (integrable_const _),
      integral_sub hWc_sq_int hint2, integral_const_mul, hWc_mean,
      ← variance_eq_integral hW.aemeasurable, hσ2]
    simp
  -- Markov's inequality at level (u + τ)²
  have hMarkov := mul_meas_ge_le_integral_of_nonneg
    (μ := μ) (f := fun ω => (u - (W ω - μ[W])) ^ 2)
    (Filter.Eventually.of_forall fun ω => sq_nonneg _) hV_sq_int ((u + τ) ^ 2)
  -- the target event is contained in the Markov event
  have hsubset : {ω | W ω - μ[W] ≤ -τ}
      ⊆ {ω | (u + τ) ^ 2 ≤ (u - (W ω - μ[W])) ^ 2} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have h1 : u + τ ≤ u - (W ω - μ[W]) := by linarith
    exact pow_le_pow_left₀ hut.le h1 2
  have hmono : μ.real {ω | W ω - μ[W] ≤ -τ}
      ≤ μ.real {ω | (u + τ) ^ 2 ≤ (u - (W ω - μ[W])) ^ 2} :=
    measureReal_mono hsubset
  have hstep : (u + τ) ^ 2 * μ.real {ω | W ω - μ[W] ≤ -τ} ≤ σ2 + u ^ 2 := by
    calc (u + τ) ^ 2 * μ.real {ω | W ω - μ[W] ≤ -τ}
        ≤ (u + τ) ^ 2 * μ.real {ω | (u + τ) ^ 2 ≤ (u - (W ω - μ[W])) ^ 2} :=
          mul_le_mul_of_nonneg_left hmono (by positivity)
      _ ≤ ∫ ω, (u - (W ω - μ[W])) ^ 2 ∂μ := hMarkov
      _ = σ2 + u ^ 2 := hV_sq_mean
  have hfinal : μ.real {ω | W ω - μ[W] ≤ -τ} ≤ (σ2 + u ^ 2) / (u + τ) ^ 2 := by
    rw [le_div_iff₀ (by positivity)]
    linarith [hstep]
  refine hfinal.trans (le_of_eq ?_)
  rw [← hu]
  have hτ' : τ ≠ 0 := hτ.ne'
  have hsum_ne : σ2 + τ ^ 2 ≠ 0 := by positivity
  field_simp
  ring

/-- The comparison stated in Lemma `lem:cantelli`: Cantelli's bound is strictly
smaller than Chebyshev's bound `σ²/τ²` for every `τ > 0` (when `σ² > 0`). -/
theorem cantelli_lt_chebyshev {σ2 τ : ℝ} (hσ : 0 < σ2) (hτ : 0 < τ) :
    σ2 / (σ2 + τ ^ 2) < σ2 / τ ^ 2 :=
  div_lt_div_of_pos_left hσ (by positivity) (by nlinarith)

/-! ## The Cantelli improvement (Theorem `thm:iv_cantelli`) -/

/-- Sign analysis of `h(η)` in Theorem `thm:iv_cantelli` (`eq:h_eta_iv`):
for `η ∈ (0,1)`, `h(η) > 0` iff `(1-η)² > γ(1-δ)/δ`. -/
theorem h_iv_pos_iff {γ δ η : ℝ} (hγ : 0 < γ) (hδ0 : 0 < δ)
    (hη0 : 0 < η) (hη1 : η < 1) :
    0 < η ^ 2 * ((δ * (γ + (1 - η) ^ 2) - γ) / (γ + (1 - η) ^ 2))
      ↔ γ * (1 - δ) / δ < (1 - η) ^ 2 := by
  have hden : 0 < γ + (1 - η) ^ 2 := by positivity
  have hη2 : 0 < η ^ 2 := by positivity
  rw [mul_pos_iff_of_pos_left hη2, div_pos_iff_of_pos_right hden,
    sub_pos, div_lt_iff₀ hδ0]
  constructor <;> intro h <;> nlinarith

/-- Feasibility in Theorem `thm:iv_cantelli` (`eq:iv_cantelli_strength`):
there exists `η ∈ (0,1)` with `h(η) > 0` iff `γ < δ/(1-δ)`. -/
theorem h_iv_feasible_iff {γ δ : ℝ} (hγ : 0 < γ) (hδ0 : 0 < δ) (hδ1 : δ < 1) :
    (∃ η, 0 < η ∧ η < 1 ∧ γ * (1 - δ) / δ < (1 - η) ^ 2)
      ↔ γ < δ / (1 - δ) := by
  have h1δ : 0 < 1 - δ := by linarith
  constructor
  · rintro ⟨η, hη0, hη1, hη⟩
    have hlt1 : (1 - η) ^ 2 < 1 := by nlinarith
    have : γ * (1 - δ) / δ < 1 := hη.trans hlt1
    rw [div_lt_one hδ0] at this
    rw [lt_div_iff₀ h1δ]
    linarith
  · intro h
    set r := γ * (1 - δ) / δ with hr
    have hr0 : 0 < r := by positivity
    have hr1 : r < 1 := by
      rw [hr, div_lt_one hδ0]
      rw [lt_div_iff₀ h1δ] at h
      linarith
    -- choose η with (1-η)² = (r+1)/2 ∈ (r, 1)
    have hmid0 : 0 < (r + 1) / 2 := by linarith
    have hmid1 : (r + 1) / 2 < 1 := by linarith
    set s := Real.sqrt ((r + 1) / 2) with hs
    have hs0 : 0 < s := Real.sqrt_pos.mpr hmid0
    have hs2 : s ^ 2 = (r + 1) / 2 := Real.sq_sqrt hmid0.le
    have hs1 : s < 1 := by
      nlinarith [hs2, hs0]
    refine ⟨1 - s, by linarith, by linarith, ?_⟩
    have h1s : (1 : ℝ) - (1 - s) = s := by ring
    rw [h1s, hs2]
    linarith

end Thesis
