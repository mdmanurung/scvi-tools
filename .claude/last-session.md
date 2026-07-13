# Session 67 — MrTotalVI default reverted to MoG; clustering-compactness rationale

**Date**: 2026-07-13
**Branch**: main
**Trigger**: T/NK UMAP comparison showed VampPrior disperses 6 clusters (all worse); user decision to keep MoG as default and expose VampPrior as explicit recipe.

---

## 1. Work completed

### MrTotalVI default reverted to MoG (`_model.py`)

Three defaults reverted:
- `u_prior`: `"vamp"` → `"mog"`
- `init_prior_from_data`: `True` → `False`
- `freeze_prior_after_init`: `True` → `False`

Architecture defaults (`use_batch_norm="none"`, `use_layer_norm="both"`) retained — these were already the validated architecture and are independent of the prior choice.

### Two tests updated (`tests/external/mrtotalvi/test_mrtotalvi.py`)

| Test | Change |
|------|--------|
| `test_vamprior_default_unchanged` | Renamed to `test_mog_default_unchanged`; asserts `u_prior_means` present, `u_vamp_pseudo` absent |
| `test_freeze_prior_false_default` | Updated to check MoG attrs (`u_prior_means.requires_grad`, `u_prior_logits.requires_grad`) |

**Result**: 54/54 passed.

### Rationale recorded

T/NK comparison (57k cells, 21 L3 labels) showed 6 clusters more dispersed under VampPrior (all increases, 0 improvements): Proliferating NK +90%, CM CD4 T TSHZ2+ +64%, SOX4+ Naïve CD4 T +57%, CM CD4 T +54%, KLRB1+ CM CD4 T Th17-like +46%, Treg +32%. VampPrior remains the recommended explicit recipe for DA analysis (D-041 std=0.192), but MoG gives better cell-type compactness for the more common clustering-first workflow.

---

## 2. Files modified

| File | Change |
|------|--------|
| `src/scvi/external/mrtotalvi/_model.py` | Revert u_prior/init_prior_from_data/freeze_prior_after_init defaults to MoG |
| `tests/external/mrtotalvi/test_mrtotalvi.py` | 2 test updates to match MoG default |

---

## 3. Outstanding items

| Item | Status |
|------|--------|
| Leiden calibration (job 25211796) | ⏳ Still pending (QOSMaxCpuPerUserLimit) |
| P1-006: MrTotalVI DA instability root-cause | GPU + real data required |
| P2-005: Macaque CITE-seq validation | Dataset + training compute required |

---

## 4. Next session priorities

1. Check Leiden calibration job 25211796 status.
2. Update `mr_multimodal.md` user guide with explicit VampPrior recipe + D-041 std=0.192 as recommended DA config.

---

# Session 66 — MrTotalVI defaults changed to VampPrior+LN; 7 tests fixed; float32 softplus-inverse bug patched

**Date**: 2026-07-13  
**Branch**: main  
**Trigger**: Continuation from session 65; D-041 confirmed (std=0.192); adjust MrTotalVI defaults to validated VampPrior+LN recipe.

---

## 1. Work completed

### MrTotalVI defaults updated (`_model.py`)

Three defaults flipped + two architecture params surfaced explicitly:
- `u_prior`: `"mog"` → `"vamp"`
- `init_prior_from_data`: `False` → `True`
- `freeze_prior_after_init`: `False` → `True`
- `use_batch_norm="none"` (explicit, matching MULTIVI)
- `use_layer_norm="both"` (explicit, matching MULTIVI)

### Float32 softplus-inverse bug fixed (`_model.py`)

`init_prior_from_data=True` k-means centroid softplus-inverse used `log(expm1(c))`, which overflows float32 for c > ~88.7 (exp overflow → inf pseudo-inputs → NaN in encoder). Fixed with numerically stable identity approximation for c > 20: `torch.where(c > 20.0, c, torch.log(torch.expm1(safe_c)))`. Logged as L-093.

### 7 failing tests fixed (`tests/external/mrtotalvi/test_mrtotalvi.py`)

| Test | Fix |
|------|-----|
| `test_vamprior_default_unchanged` | Update assertions to reflect `"vamp"` as new default |
| `test_freeze_prior_false_default` | Rewrite to pass `freeze_prior_after_init=False` explicitly; check `u_vamp_pseudo.requires_grad` |
| `test_freeze_prior_after_init_mog` | Add `u_prior="mog"` (tests MoG-specific behavior) |
| `test_mrtotalvi_gaussian_u_prior_and_z_u_prior_off` | Add `u_prior="mog"` |
| `test_mrtotalvi_save_load_preserves_latent_hierarchy` | Add `u_prior="mog"` |
| `test_mrtotalvi_label_conditioned_mog_prior` | Add `u_prior="mog"` |
| `test_mrtotalvi_lfc_sign_known_positive_control` | Add `u_prior="mog"` (LFC sign test; VampPrior needs >30 epochs to converge on synthetic signal) |

**Result**: 54/54 passed.

---

## 2. Files modified

| File | Change |
|------|--------|
| `src/scvi/external/mrtotalvi/_model.py` | VampPrior+LN defaults; float32 softplus-inverse fix |
| `tests/external/mrtotalvi/test_mrtotalvi.py` | 7 test fixes to reflect new defaults |
| `.living/learnings.md` | L-093 (float32 softplus-inverse overflow) |

---

## 3. Outstanding items

| Item | Status |
|------|--------|
| Leiden calibration (job 25211796) | ⏳ Still pending (QOSMaxCpuPerUserLimit) |
| P1-006: MrTotalVI DA instability root-cause | GPU + real data required |
| P2-005: Macaque CITE-seq validation | Dataset + training compute required |

---

## 4. Next session priorities

1. Consider adding a VampPrior-default smoke test that validates the full default config trains without NaN.
2. Check Leiden calibration job status.

---

# Session 65 — Skills installation complete; VampPrior jobs still pending

**Date**: 2026-07-13  
**Branch**: main  
**Trigger**: Continuation from session 64 after context compaction; skills install + plan verification.

---

## 1. Work completed

### ericmjl/skills installation
Installed 4 skills from `https://github.com/ericmjl/skills` to `~/.claude/skills/`:

| Skill | Files |
|-------|-------|
| `write-like-eric` | SKILL.md |
| `atomic-commits` | SKILL.md |
| `coherent-writing` | SKILL.md + references/coherence-patterns.md + references/subagent-prompts.md |
| `skill-creator` | SKILL.md + 3 reference files + 3 scripts (init_skill.py, package_skill.py, quick_validate.py) |

### scvi skills package
Verified complete: bundled `src/scvi/_skills/data/` with SKILL.md + 6 reference files; `cytoanvi-install-skills` console script in `pyproject.toml`.

### P1/P2 backlog
Verified: P1-001 through P1-005 and P2-001 through P2-004 all `done` in the TODO registry (done in sessions 60–63).

---

# Session 64 — VampPrior+freeze D-041 revalidation launched

**Date**: 2026-07-13  
**Branch**: main  
**Trigger**: User request to run empirical VampPrior validation (D-041 refuted in session 62; no artifact existed).

---

## 1. Work completed

### D-041 revalidation: LN+VampPrior+freeze+DTP training submitted

Added `train_mrtotalvi_ln_vamp()` to `train_multiseed.py` and `mrtotalvi_ln_vamp` to `run_dtp_da.py`. CPU smoke test passed (2-epoch, save/load verified). Three SLURM training jobs submitted:

| Seed | Job ID | Status |
|------|--------|--------|
| s0 | 25214226 | PENDING |
| s1 | 25214227 | PENDING |
| s2 | 25214228 | PENDING |

Config: `use_batch_norm="none"`, `use_layer_norm="both"`, `u_prior="vamp"`, `u_prior_mixture_k=20`, `init_prior_from_data=True`, `freeze_prior_after_init=True`, `sample_key="donor_timepoint"`.

Models will save to: `outputs/models/mrtotalvi_10k_human_ln_vamp_dtp_s{0,1,2}/`

DA script ready: `slurm/submit_da_dtp_mrtotalvi_ln_vamp.sh` — submit after all 3 training jobs complete.

### Pre-registered success criterion

**Success**: W22-enrichment std ≤ 0.30 (>65% reduction from LN-MoG baseline std=0.875) AND all 3 seeds positive.  
**Failure**: VampPrior does not rescue MrTotalVI-LN DA; use MrMultiVI for publication DA.

---

## 2. Files modified

| File | Change |
|------|--------|
| `schisto_citeseq/analysis/integration/mr_multimodal_publication/train_multiseed.py` | Added `train_mrtotalvi_ln_vamp()` + argparse choice |
| `.scratch/mr-schisto-benchmark/run_dtp_da.py` | Added `mrtotalvi_ln_vamp` entry + argparse choice |
| `.scratch/mr-schisto-benchmark/slurm/submit_train_mrtotalvi_dtp_ln_vamp_s{0,1,2}.sh` | New SLURM training scripts |
| `.scratch/mr-schisto-benchmark/slurm/submit_da_dtp_mrtotalvi_ln_vamp.sh` | New SLURM DA script |
| `.living/decisions.md` (D-041) | Added revalidation launch note + pre-registered criterion |

---

## 3. Outstanding items

| Item | Status |
|------|--------|
| D-041 LN+VampPrior | ✅ CONFIRMED — std=0.192, mean=+0.445 (artifact: `results/da_mrtotalvi_ln_vamp_dtp_summary.json`) |
| Leiden calibration (job 25211796) | ⏳ Still pending (QOSMaxCpuPerUserLimit) |
| P1-006: MrTotalVI DA instability root-cause | GPU + real data required |
| P2-005: Macaque CITE-seq validation | Dataset + training compute required |

---

## 4. Next session priorities

1. Check Leiden calibration job 25211796 status.
2. Consider updating `mr_multimodal.md` user guide with D-041 confirmed VampPrior recipe and std=0.192 number.
