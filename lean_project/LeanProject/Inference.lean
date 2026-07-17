/-
# Inference (`Paper/iteration4/inference.tex`)

Machine-checked statements corresponding to Section "Inference":

* `moment_at_true_beta` — the observation underlying `eq:Wdef` and Step 1 of
  Theorem `thm:mom_ar_size`: at the true parameter, the AR moment reduces to
  `Z ε`, i.e. `Z (Y - β X) = Z ε` under the structural equation.
* `Wbar_affine` — `eq:Wbar`: the block mean of the AR moment is the affine
  function `μ̂_{ZY} - β₀ μ̂_{ZX}` of the null value `β₀`.
* `coverage_of_size` — Theorem `thm:coverage`: the test-inversion duality;
  a size-`δ` bound at the true parameter yields coverage `≥ 1 - δ`.
* `crossing_unique` / `parallel_no_crossing` — Step 1 of Theorem
  `thm:piecewise`: two block-mean lines with distinct slopes cross exactly
  once; with equal slopes and distinct intercepts they never cross.
* `boundary_candidate_eq` — Corollary `cor:endpoints`: for a block with
  `μ̂_{ZX} ≠ 0`, the equation `μ̂_{ZY} - β₀ μ̂_{ZX} = c` has the unique
  solution `β₀ = (μ̂_{ZY} - c)/μ̂_{ZX}`; hence each block contributes the two
  candidate boundary points `(μ̂_{ZY} ∓ τ)/μ̂_{ZX}` of `eq:candidates`.

The size bound (Theorem `thm:mom_ar_size`) itself is the scalar MoM
concentration applied to the sequence `W_i(β₀) = Z_i ε_i`; its constituent
steps are formalized in `LeanProject.Preliminaries`.
-/
import Mathlib

open MeasureTheory

namespace Thesis

/-! ## The AR moment at the true parameter -/

/-- Under the structural equation `Y = β X + ε` (`eq:structural`), the AR
moment (`eq:Wdef`) evaluated at the true `β` reduces to `Z ε` (Step 1 of
Theorem `thm:mom_ar_size` and the key identity in Theorem `thm:coverage`). -/
theorem moment_at_true_beta {Z X Y ε β : ℝ} (h : Y = β * X + ε) :
    Z * (Y - β * X) = Z * ε := by
  rw [h]; ring

/-- `eq:Wbar`: because the moment is linear in `β₀`, its block mean is the
affine function `β₀ ↦ μ̂_{ZY} - β₀ μ̂_{ZX}` of the null value. -/
theorem Wbar_affine {ι : Type*} (s : Finset ι) (Z X Y : ι → ℝ)
    (β₀ mm : ℝ) :
    (∑ i ∈ s, Z i * (Y i - β₀ * X i)) / mm
      = (∑ i ∈ s, Z i * Y i) / mm - β₀ * ((∑ i ∈ s, Z i * X i) / mm) := by
  have h : ∑ i ∈ s, Z i * (Y i - β₀ * X i)
      = ∑ i ∈ s, Z i * Y i - β₀ * ∑ i ∈ s, Z i * X i := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [h]
  ring

/-! ## Coverage by test inversion (Theorem `thm:coverage`) -/

/-- **Theorem `thm:coverage`** (test-inversion duality): if the statistic `T`
evaluated at the true parameter exceeds the threshold `τ` with probability at
most `δ` (the size bound of Theorem `thm:mom_ar_size`), then the event
`|T| ≤ τ` — i.e. the true parameter belongs to the confidence set `eq:cs` —
has probability at least `1 - δ`. -/
theorem coverage_of_size {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {T : Ω → ℝ} (hT : Measurable T) {τ δ : ℝ}
    (hsize : μ {ω | τ < |T ω|} ≤ ENNReal.ofReal δ) :
    1 - ENNReal.ofReal δ ≤ μ {ω | |T ω| ≤ τ} := by
  have hset : {ω | |T ω| ≤ τ} = {ω | τ < |T ω|}ᶜ := by
    ext ω; simp [not_lt]
  have hmeas : MeasurableSet {ω | τ < |T ω|} :=
    measurableSet_lt measurable_const hT.abs
  rw [hset, measure_compl hmeas (measure_ne_top μ _), measure_univ]
  exact tsub_le_tsub_left hsize 1

/-! ## Geometry of the confidence set (Theorem `thm:piecewise`,
Corollary `cor:endpoints`) -/

/-- **Step 1 of Theorem `thm:piecewise`**: two block-mean lines
`β₀ ↦ aY - β₀ aX` and `β₀ ↦ bY - β₀ bX` with distinct slopes cross at exactly
one point, namely `β₀ = (bY - aY)/(bX - aX)`. -/
theorem crossing_unique {aY aX bY bX : ℝ} (h : aX ≠ bX) :
    ∃! β₀ : ℝ, aY - β₀ * aX = bY - β₀ * bX := by
  have hne : bX - aX ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  refine ⟨(bY - aY) / (bX - aX), ?_, ?_⟩
  · field_simp
    ring
  · intro y hy
    field_simp
    nlinarith [hy]

/-- **Step 1 of Theorem `thm:piecewise`**, parallel case: two block-mean
lines with equal slopes and distinct intercepts never cross. -/
theorem parallel_no_crossing {aY aX bY bX : ℝ} (hX : aX = bX) (hY : aY ≠ bY)
    (β₀ : ℝ) : aY - β₀ * aX ≠ bY - β₀ * bX := by
  subst hX
  intro h
  exact hY (by linarith)

/-- **Corollary `cor:endpoints`** (algebraic core): for a block with slope
`μ̂_{ZX} ≠ 0`, the boundary equation `μ̂_{ZY} - β₀ μ̂_{ZX} = c` has the unique
solution `β₀ = (μ̂_{ZY} - c)/μ̂_{ZX}`.  With `c = ±τ_n(δ)` this yields the two
candidate boundary points of `eq:candidates`, so each block contributes at
most two and `|B_cand| ≤ 2k`. -/
theorem boundary_candidate_eq {μZY μZX c β₀ : ℝ} (h : μZX ≠ 0) :
    μZY - β₀ * μZX = c ↔ β₀ = (μZY - c) / μZX := by
  rw [eq_div_iff h]
  constructor <;> intro h' <;> linarith

end Thesis
