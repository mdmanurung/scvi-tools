# Topic: Continual Update and λ Sweep (B4/B6)

**Question**: Does the EWC continual update preserve prior knowledge while integrating case-control queries? What is the right λ?
**Dataset**: Roider BNHL (pseudo batch split — not a real case/control axis; plumbing evidence only)
**Source docs**: `.scratch/cytoanvi-benchmark/issues/10-roider-b4-b6-continual.md`

---

## F-008 — B4/B6: replay latent drift tied at 0.0 for all λ values (plumbing only)
**Date**: 2026-06-28
**Status**: NOT publication evidence — pseudo batch split, not biological case/control
**Validity caveat**: `roider-smoke`, pseudo-split (B4 used an arbitrary batch split as case/control proxy)
**Claim**: Plain surgery: query macro-F1 **0.859**, replay drift **0.0**. Continual update (λ=1.0): query macro-F1 **0.862**, replay drift **0.0**. λ sweep over {0, 1, 10, 100, 1000}: all drifts tied at **0.0**; highest macro-F1 at λ=**1.0** (**0.888**), but drift signal is uninformative.
**Interpretation**: The pseudo case/control split produces no measurable anchor drift regardless of λ. EWC penalty is non-zero in the loss but the anchor and query are similar enough that the constraint doesn't bind. This is a plumbing proof-of-concept only.
**Meets target**: not evaluable (drift metric not meaningful on pseudo split)
**Source**: `.scratch/cytoanvi-benchmark/issues/10-roider-b4-b6-continual.md`
**Implications**: B4/B6 require a real rLN reference + FL/MCL query axis before λ selection is biologically informative. The current sweep cannot provide a publication-ready CytoVI-specific λ default.

---

## F-009 — B8 synthetic: flat CE 0.295, HCE 0.166 on tiny data (expected, not informative)
**Date**: 2026-06-29
**Status**: synthetic smoke only — not publication-grade
**Validity caveat**: `b8-synthetic-smoke` — tiny synthetic dataset; scHPL sibling-only leaves produce identity reachability, making HCE≈flat CE at scale
**Claim**: Flat CE macro_f1 **0.295**, HCE **0.166** (Δ **−0.129**). HCE underperforms on tiny synthetic data.
**Interpretation**: Expected on tiny data + sibling-only hierarchy (identity reachability → HCE≈flat CE). Not evidence against HCE. Full B8 results pending (job 25108052, resubmitted after leaf_held bias fix).
**Source**: `.scratch/cytoanvi-benchmark/results/b8_synthetic_s0.json`
**Note**: Two correctness bugs in the prior harness (L-012 leaf_held bias, L-013 HCE routing) mean any B8 results before job 25108052 are invalid.
