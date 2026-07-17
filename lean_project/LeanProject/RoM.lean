/-
# The Ratio-of-Medians (RoM) Estimator (`Paper/iteration4/rom.tex`)

Machine-checked statements corresponding to Section "The Ratio-of-Medians
(RoM) Estimator" (Theorem `thm:rom`):

* `union_bound_compl` — Step 2: `P[(A ∩ B)ᶜ] ≤ P[Aᶜ] + P[Bᶜ]`.
* `ratio_error_decomposition` — Step 3: the exact algebraic decomposition
  `μ̃_{ZY}/μ̃_{ZX} - β = ((μ̃_{ZY} - μ_{ZY}) - β(μ̃_{ZX} - μ_{ZX}))/μ̃_{ZX}`.
* `rom_error_bound` — Steps 3–4 combined (deterministic core): on the event
  `A ∩ B` the estimation error obeys the triangle-inequality bound with the
  reverse-triangle-inequality control of the denominator.
* `rom_bound_simplification` — the final algebraic simplification to
  `2(σ_{ZY} + |β|σ_{ZX}) / (|μ_{ZX}|√m - 2σ_{ZX})` (`eq:rom_bound`).
* `rom_deterministic_core` — the assembled deterministic statement: on
  `A ∩ B`, under the instrument strength condition `eq:rom_strength`, the RoM
  error is bounded by the right-hand side of `eq:rom_bound`.

Step 1 (MoM on each coordinate) is Theorem `thm:mom_scalar`, whose
constituents are formalized in `LeanProject.Preliminaries`; the choice
`k = ⌈8 ln(2/δ)⌉` is `two_mul_exp_ceil_log_le` there.
-/
import Mathlib

open MeasureTheory

namespace Thesis

/-! ## Step 2: union bound -/

/-- **Step 2 of Theorem `thm:rom`** (union bound): the events `A`, `B` need
not be independent; still `P[(A ∩ B)ᶜ] ≤ P[Aᶜ] + P[Bᶜ]`. -/
theorem union_bound_compl {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (A B : Set Ω) : μ (A ∩ B)ᶜ ≤ μ Aᶜ + μ Bᶜ := by
  rw [Set.compl_inter]
  exact measure_union_le _ _

/-! ## Step 3: algebraic decomposition of the ratio error -/

/-- **Step 3 of Theorem `thm:rom`**: if `μ_{ZY} = β μ_{ZX}` (identification)
and the median denominators are nonzero, then
`μ̃_{ZY}/μ̃_{ZX} - β = ((μ̃_{ZY} - μ_{ZY}) - β(μ̃_{ZX} - μ_{ZX}))/μ̃_{ZX}`. -/
theorem ratio_error_decomposition {tY tX μY μX β : ℝ}
    (hid : μY = β * μX) (htX : tX ≠ 0) :
    tY / tX - β = ((tY - μY) - β * (tX - μX)) / tX := by
  subst hid
  field_simp
  ring

/-! ## Steps 3–4: the deterministic error bound on `A ∩ B` -/

/-- **Steps 3–4 of Theorem `thm:rom`** (deterministic core): suppose
identification `μ_{ZY} = β μ_{ZX}` holds, the coordinate-wise medians satisfy
the event bounds `|μ̃_{ZY} - μ_{ZY}| ≤ e_Y` (event `A`) and
`|μ̃_{ZX} - μ_{ZX}| ≤ e_X` (event `B`), and the instrument strength condition
`e_X < |μ_{ZX}|` holds.  Then
`|μ̃_{ZY}/μ̃_{ZX} - β| ≤ (e_Y + |β| e_X)/(|μ_{ZX}| - e_X)`. -/
theorem rom_error_bound {tY tX μY μX β eY eX : ℝ}
    (hid : μY = β * μX)
    (hA : |tY - μY| ≤ eY) (hB : |tX - μX| ≤ eX)
    (hstrength : eX < |μX|) :
    |tY / tX - β| ≤ (eY + |β| * eX) / (|μX| - eX) := by
  have heX : 0 ≤ eX := le_trans (abs_nonneg _) hB
  have heY : 0 ≤ eY := le_trans (abs_nonneg _) hA
  -- reverse triangle inequality: |μ̃_{ZX}| ≥ |μ_{ZX}| - e_X > 0
  have hden : |μX| - eX ≤ |tX| := by
    have h1 : |μX| - |tX| ≤ |tX - μX| := by
      rw [abs_sub_comm]
      exact abs_sub_abs_le_abs_sub μX tX
    linarith
  have hden_pos : 0 < |tX| := lt_of_lt_of_le (by linarith) hden
  have htX : tX ≠ 0 := abs_pos.mp hden_pos
  -- numerator bound via the triangle inequality
  have hnum : |(tY - μY) - β * (tX - μX)| ≤ eY + |β| * eX := by
    calc |(tY - μY) - β * (tX - μX)|
        ≤ |tY - μY| + |β * (tX - μX)| := abs_sub _ _
    _ = |tY - μY| + |β| * |tX - μX| := by rw [abs_mul]
    _ ≤ eY + |β| * eX := by
        have := mul_le_mul_of_nonneg_left hB (abs_nonneg β)
        linarith
  rw [ratio_error_decomposition hid htX, abs_div]
  exact div_le_div₀ (by positivity) hnum (by linarith) hden

/-- The final simplification in Step 4 of Theorem `thm:rom`: with
`e_Y = 2σ_{ZY}/√m` and `e_X = 2σ_{ZX}/√m`,
`(e_Y + |β| e_X)/(|μ_{ZX}| - e_X) = 2(σ_{ZY} + |β|σ_{ZX})/(|μ_{ZX}|√m - 2σ_{ZX})`. -/
theorem rom_bound_simplification {σY σX μX β m : ℝ} (hm : 0 < m)
    (hstrength : 2 * σX / Real.sqrt m < |μX|) :
    (2 * σY / Real.sqrt m + |β| * (2 * σX / Real.sqrt m))
        / (|μX| - 2 * σX / Real.sqrt m)
      = 2 * (σY + |β| * σX) / (|μX| * Real.sqrt m - 2 * σX) := by
  have hs : 0 < Real.sqrt m := Real.sqrt_pos.mpr hm
  have hden1 : 0 < |μX| - 2 * σX / Real.sqrt m := by linarith
  have hden2 : 0 < |μX| * Real.sqrt m - 2 * σX := by
    have := mul_lt_mul_of_pos_right hstrength hs
    rw [div_mul_cancel₀ _ hs.ne'] at this
    linarith
  rw [div_eq_div_iff hden1.ne' hden2.ne']
  field_simp

/-- **Theorem `thm:rom`, deterministic core assembled**: under identification,
on the event `A ∩ B` (both coordinate-wise medians within `2σ/√m` of their
targets), with the instrument strength condition `eq:rom_strength` in the form
`2σ_{ZX}/√m < |μ_{ZX}|`, the RoM error satisfies `eq:rom_bound`:
`|μ̃_{ZY}/μ̃_{ZX} - β| ≤ 2(σ_{ZY} + |β|σ_{ZX})/(|μ_{ZX}|√m - 2σ_{ZX})`. -/
theorem rom_deterministic_core {tY tX μY μX β σY σX m : ℝ}
    (hid : μY = β * μX) (hm : 0 < m)
    (hA : |tY - μY| ≤ 2 * σY / Real.sqrt m)
    (hB : |tX - μX| ≤ 2 * σX / Real.sqrt m)
    (hstrength : 2 * σX / Real.sqrt m < |μX|) :
    |tY / tX - β| ≤ 2 * (σY + |β| * σX) / (|μX| * Real.sqrt m - 2 * σX) := by
  rw [← rom_bound_simplification hm hstrength]
  exact rom_error_bound hid hA hB hstrength

/-- The instrument strength condition of Theorem `thm:rom`
(`eq:rom_strength`): `m > 4σ²_{ZX}/μ²_{ZX}` is equivalent to
`2σ_{ZX}/√m < |μ_{ZX}|` (the form used in Step 4). -/
theorem rom_strength_iff {σX μX m : ℝ} (hσ : 0 < σX) (hμ : μX ≠ 0) (hm : 0 < m) :
    4 * σX ^ 2 / μX ^ 2 < m ↔ 2 * σX / Real.sqrt m < |μX| := by
  have hs : 0 < Real.sqrt m := Real.sqrt_pos.mpr hm
  have habs : 0 < |μX| := abs_pos.mpr hμ
  have hμ2 : 0 < μX ^ 2 := by positivity
  have key : (|μX| * Real.sqrt m) ^ 2 = m * μX ^ 2 := by
    rw [mul_pow, sq_abs, Real.sq_sqrt hm.le]; ring
  rw [div_lt_iff₀ hμ2, div_lt_iff₀ hs]
  constructor
  · intro h
    have hpow : (2 * σX) ^ 2 < (|μX| * Real.sqrt m) ^ 2 := by
      rw [key]; nlinarith
    exact lt_of_pow_lt_pow_left₀ 2 (by positivity) hpow
  · intro h
    have hpow : (2 * σX) ^ 2 < (|μX| * Real.sqrt m) ^ 2 := by
      nlinarith [mul_self_lt_mul_self (by positivity : (0:ℝ) ≤ 2 * σX) h]
    rw [key] at hpow
    nlinarith

end Thesis
