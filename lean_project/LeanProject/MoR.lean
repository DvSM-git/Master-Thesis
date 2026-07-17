/-
# The Median-of-Ratios (MoR) Estimator (`Paper/iteration4/mor.tex`)

Machine-checked statements corresponding to Section "The Median-of-Ratios
(MoR) Estimator":

* Lemma `lem:mor_decomp` (block-level error decomposition) is the general
  `Thesis.error_decomposition` in `LeanProject.IV`, applied to a block `B_j`;
  its mean-zero part is `Thesis.sample_moment_mean_zero`.
* `mor_strength_budget_iff` — Step 3 of Theorem `thm:mor`: the per-block
  budget `4σ²_{ZX}/(m μ²_{ZX}) ≤ 1/8` holds iff `m ≥ 32σ²_{ZX}/μ²_{ZX}`
  (`eq:mor_strength`).
* `mor_numerator_threshold_eq` — Step 3 of Theorem `thm:mor`: the threshold
  `t = √32 σ_{Zε}/(|μ_{ZX}|√m)` makes the numerator budget exact:
  `4σ²_{Zε}/(m t² μ²_{ZX}) = 1/8`.
* Steps 4–5 (counting argument and Hoeffding) are
  `Thesis.median_deviation_count` and `Thesis.hoeffding_count` in
  `LeanProject.Preliminaries`; the choice `k = ⌈8 ln(1/δ)⌉` is
  `Thesis.exp_ceil_log_le` there.
* Step 3 of Theorem `thm:mor_opt` is *identical* to that of `thm:iv_opt`
  (with `ρ_MR` in place of `ρ_IV`); it is `Thesis.opt_argmax_f` in
  `LeanProject.IV`, as the paper notes.
* `h_mor_pos_iff` / `h_mor_feasible_iff` — the sign and feasibility analysis
  of `h(η)` in Theorem `thm:mor_cantelli` (`eq:h_eta_mor`,
  `eq:mor_cantelli_strength`): `h(η) > 0` iff `(1-η)² > 3γ_MR`, and the
  feasible region is non-empty iff `γ_MR < 1/3`.
-/
import Mathlib
import LeanProject.IV

open MeasureTheory

namespace Thesis

/-! ## Step 3 of Theorem `thm:mor`: the per-block error budget -/

/-- **Step 3 of Theorem `thm:mor`, budget for `A_jᶜ`** (`eq:mor_strength`):
the per-block denominator budget `4σ²/(mμ²) ≤ 1/8` holds if and only if
`m ≥ 32σ²/μ²`. -/
theorem mor_strength_budget_iff {σ2 μ2 m : ℝ} (hμ : 0 < μ2) (hm : 0 < m) :
    4 * σ2 / (m * μ2) ≤ 1 / 8 ↔ 32 * σ2 / μ2 ≤ m := by
  rw [div_le_div_iff₀ (by positivity) (by norm_num),
    div_le_iff₀ (by positivity)]
  constructor <;> intro h <;> nlinarith

/-- **Step 3 of Theorem `thm:mor`, threshold for `B_jᶜ`**: the choice
`t = √32 σ_{Zε}/(|μ_{ZX}|√m)` makes the per-block numerator budget exact:
`4σ²_{Zε}/(m t² μ²_{ZX}) = 1/8`. -/
theorem mor_numerator_threshold_eq {σε μZX m : ℝ}
    (hσ : 0 < σε) (hμ : μZX ≠ 0) (hm : 0 < m) :
    4 * σε ^ 2 /
      (m * (Real.sqrt 32 * σε / (|μZX| * Real.sqrt m)) ^ 2 * μZX ^ 2)
      = 1 / 8 := by
  have h32 : Real.sqrt 32 ^ 2 = 32 := Real.sq_sqrt (by norm_num)
  have hsm : Real.sqrt m ^ 2 = m := Real.sq_sqrt hm.le
  have habs : |μZX| ^ 2 = μZX ^ 2 := sq_abs _
  have hμ2 : (0:ℝ) < μZX ^ 2 := by positivity
  have hsm' : 0 < Real.sqrt m := Real.sqrt_pos.mpr hm
  have habs' : 0 < |μZX| := abs_pos.mpr hμ
  rw [div_pow, mul_pow, mul_pow, h32, hsm, habs]
  field_simp
  ring

/-! ## The Cantelli improvement (Theorem `thm:mor_cantelli`) -/

/-- Sign analysis of `h(η)` in Theorem `thm:mor_cantelli` (`eq:h_eta_mor`):
for `η ∈ (0,1)`, `h(η) = η²((1-η)² - 3γ)/(γ + (1-η)²) > 0` iff
`(1-η)² > 3γ`. -/
theorem h_mor_pos_iff {γ η : ℝ} (hγ : 0 < γ) (hη0 : 0 < η) (hη1 : η < 1) :
    0 < η ^ 2 * (((1 - η) ^ 2 - 3 * γ) / (γ + (1 - η) ^ 2))
      ↔ 3 * γ < (1 - η) ^ 2 := by
  have hden : 0 < γ + (1 - η) ^ 2 := by positivity
  have hη2 : 0 < η ^ 2 := by positivity
  rw [mul_pos_iff_of_pos_left hη2, div_pos_iff_of_pos_right hden, sub_pos]

/-- Feasibility in Theorem `thm:mor_cantelli` (`eq:mor_cantelli_strength`):
there exists `η ∈ (0,1)` with `h(η) > 0` iff `γ_MR < 1/3` (equivalently,
`m > 3σ²_{ZX}/μ²_{ZX}`). -/
theorem h_mor_feasible_iff {γ : ℝ} (hγ : 0 < γ) :
    (∃ η, 0 < η ∧ η < 1 ∧ 3 * γ < (1 - η) ^ 2) ↔ γ < 1 / 3 := by
  constructor
  · rintro ⟨η, hη0, hη1, hη⟩
    have hlt1 : (1 - η) ^ 2 < 1 := by nlinarith
    linarith
  · intro h
    set r := 3 * γ with hr
    have hr0 : 0 < r := by positivity
    have hr1 : r < 1 := by rw [hr]; linarith
    have hmid0 : 0 < (r + 1) / 2 := by linarith
    set s := Real.sqrt ((r + 1) / 2) with hs
    have hs0 : 0 < s := Real.sqrt_pos.mpr hmid0
    have hs2 : s ^ 2 = (r + 1) / 2 := Real.sq_sqrt hmid0.le
    have hs1 : s < 1 := by nlinarith [hs2, hs0]
    refine ⟨1 - s, by linarith, by linarith, ?_⟩
    have h1s : (1 : ℝ) - (1 - s) = s := by ring
    rw [h1s, hs2]
    linarith

/-- The equivalence of the two forms of `eq:mor_cantelli_strength`:
`m > 3σ²/μ²` iff `γ_MR = σ²/(mμ²) < 1/3`. -/
theorem mor_cantelli_strength_iff {σ2 μ2 m : ℝ} (_hσ : 0 < σ2) (hμ : 0 < μ2)
    (hm : 0 < m) :
    3 * σ2 / μ2 < m ↔ σ2 / (m * μ2) < 1 / 3 := by
  rw [div_lt_iff₀ hμ, div_lt_div_iff₀ (by positivity) (by norm_num)]
  constructor <;> intro h <;> nlinarith

end Thesis
