/-
# Verification report

Run with:

    lake env lean LeanProject/Verify.lean

(or `lake build LeanProject.Verify`).

For every theorem in the development this file prints

    'Thesis.<name>' depends on axioms: [propext, Classical.choice, Quot.sound]

which certifies that the theorem is *fully proved* (type-checked by the Lean
kernel) using only the three standard axioms of classical mathematics.
If any proof were incomplete or used `sorry`, the axiom `sorryAx` would appear
in its list; if a theorem failed to compile, this file would not build at all.
-/
import LeanProject.Basic

/-! ## Preliminaries (`preliminaries.tex`) -/

#print axioms Thesis.beta_identified
#print axioms Thesis.median_deviation_count
#print axioms Thesis.variance_block_mean
#print axioms Thesis.chebyshev_block_quarter
#print axioms Thesis.hoeffding_count
#print axioms Thesis.exp_le_of_log_le
#print axioms Thesis.exp_ceil_log_le
#print axioms Thesis.two_mul_exp_ceil_log_le

/-! ## Standard IV estimator (`iv.tex`) -/

#print axioms Thesis.error_decomposition
#print axioms Thesis.sample_moment_mean_zero
#print axioms Thesis.two_event_bound
#print axioms Thesis.denominator_failure
#print axioms Thesis.prob_denominator_failure
#print axioms Thesis.prob_numerator_failure
#print axioms Thesis.strength_budget_iff
#print axioms Thesis.numerator_threshold_eq
#print axioms Thesis.opt_f_le
#print axioms Thesis.opt_f_eq
#print axioms Thesis.opt_critical_mem_Ioo
#print axioms Thesis.opt_argmax_f
#print axioms Thesis.cantelli
#print axioms Thesis.cantelli_lt_chebyshev
#print axioms Thesis.h_iv_pos_iff
#print axioms Thesis.h_iv_feasible_iff

/-! ## Ratio-of-Medians estimator (`rom.tex`) -/

#print axioms Thesis.union_bound_compl
#print axioms Thesis.ratio_error_decomposition
#print axioms Thesis.rom_error_bound
#print axioms Thesis.rom_bound_simplification
#print axioms Thesis.rom_deterministic_core
#print axioms Thesis.rom_strength_iff

/-! ## Median-of-Ratios estimator (`mor.tex`) -/

#print axioms Thesis.mor_strength_budget_iff
#print axioms Thesis.mor_numerator_threshold_eq
#print axioms Thesis.h_mor_pos_iff
#print axioms Thesis.h_mor_feasible_iff
#print axioms Thesis.mor_cantelli_strength_iff

/-! ## Inference (`inference.tex`) -/

#print axioms Thesis.moment_at_true_beta
#print axioms Thesis.Wbar_affine
#print axioms Thesis.coverage_of_size
#print axioms Thesis.crossing_unique
#print axioms Thesis.parallel_no_crossing
#print axioms Thesis.boundary_candidate_eq
