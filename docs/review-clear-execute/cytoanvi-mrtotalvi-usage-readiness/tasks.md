# Execution checklist: CytoANVI and MrTotalVI usage readiness

## Baseline and governance

- [x] Re-read `plan.md`, `handoff.md`, the source plan, source review, AGENTS.md, and relevant ADRs before editing.
- [x] Record `HEAD`, branch, tracked diff, target-file hashes, Python/package identities, and stale-wheel SHA without modifying existing evidence.
- [x] Confirm no tracked target source/test/doc/workflow file differs from baseline for reasons outside this execution; stop on overlap.
- [x] Add ADR-0010 recording version, fail-closed APIs, compatibility migrations, artifact authority, 19 capability IDs, and P2/P3 gates.
- [x] Add a `0.2.0` migration table and accurate changelog entry.
- [x] Add strict machine-readable schemas for the artifact manifest, installed-acceptance receipt, scientific protocol, terminal run manifest, and capability decision matrix.
- [x] Add a tracked logical-quarantine record for the unchanged `0.1.0` wheel and verify SHA `340dfbd2d571e44cf5e8b6d1bc8a62798ce9753abc5df099654a026095f19c8d`.
- [x] Add the build and installed-wheel acceptance harnesses without building a final candidate yet.
- [x] Add or revise an exact artifact-acceptance dependency-authority contract; mark external resolution separately blocked if it requires network access.
- [x] Fix installed-doc validation so `docs/conf.py` uses `cytoanvi` metadata and cannot shadow the tested wheel with `src`.

## Primary guidance and tutorials

- [x] Create one authoritative 19-row evidence-status page and machine-readable table.
- [x] Link README, installation, CytoANVI/MrTotalVI model guides, parameter guides, API docstrings, and tutorials to that authority.
- [x] Remove supported TTA novelty guidance and identify the existing negative result plus latent-distance replacement requirement.
- [x] Mark `control_adata` required and repair all continual examples/signatures.
- [x] Correct fork installation, raw RNA/protein counts, legacy `u`, prior, label supervision, DA, DE, streaming, and new-sample guidance.
- [x] Repair direct same-panel treeArches surgery and the one-shot learn/update/predict path.
- [x] Add a reusable synthetic tutorial module and execute-test both treeArches paths.
- [x] Add link/content tests that reject contradictory readiness recommendations.

## CytoANVI contracts

- [x] Make every non-null `adversarial_classifier` fail before trainer construction; update signature/negative tests.
- [x] Make continual training with active continual state and no replay fail before trainer construction; replace warning-only tests.
- [x] Keep no public EWC-only mode in this packet and document explicit replay reconstruction after load.
- [x] Move/rename retained TTA functionality to a clearly experimental surface and make stable/indirect legacy novelty entry points fail closed.
- [x] Draw independent per-cell TTA masks and add deterministic seed support.
- [x] Add exact fixed-seed full-batch/chunk invariance tests and independent-row-mask tests.
- [x] Reject empty, NaN, positive-infinity, and negative-infinity TTA calibration arrays.
- [x] Remove the TTA threshold helper from stable top-level exports and update public API locks.
- [x] Resolve empirical prior/class weights from actual training indices before optimization and persist the boundary/resolved values.
- [x] Prove held-out-label perturbation cannot alter empirical prior/class weights.
- [x] Pin and runtime-check mapQC `0.1.1` while the private monkeypatch remains.
- [x] Run the complete CytoANVI source and tutorial test set to terminal exit and record count/runtime: 178 passed, 3 skipped in 882.40s.

## MrTotalVI contracts

- [x] Implement the exact prior enum/legacy-flag migration table and reject unknown/contradictory states before module construction.
- [x] Add consistent old-checkpoint load/save/reload/resave tests and explicit contradictory-checkpoint refusal tests.
- [x] Add `u_prior_supervision` with new-call default `none`, weight zero, explicit labels opt-in, and old-checkpoint migration.
- [x] Record hierarchy, encoder, resolved prior, supervision, and weight in summary, `init_params_`, save/load, and tests.
- [x] Restrict Vamp data initialization to frozen training indices and persist seed/index digest; prove held-out perturbation invariance.
- [x] Add MrTotalVI-specific exhaustive finite/non-negative/integer-like RNA and protein validation before any AnnData mutation.
- [x] Test hidden NaN, infinity, negative, and fractional values outside sampled/early rows for dense, sparse, DataFrame, and backed/chunked inputs where supported.
- [x] Add one authoritative sample/donor/covariate/subset validator used before DA and retained internal DE statistics.
- [x] Test missing columns, null values, conflicting mappings, unknown subsets, duplicate subsets, empty subsets, and declared-order preservation.
- [x] Make public legacy and centered-v2 `differential_expression()` fail closed with donor-pseudobulk guidance.
- [x] Keep DA explicitly descriptive and keep centered-v2 registered-sample boundaries unchanged.
- [x] Make `use_vmap=True` raise before inference/statistics and preserve `False` loop behavior.
- [x] Require exact authoritative multi-file protein names/order and reject empty, duplicate, renamed, or reordered axes.
- [x] Normalize DataFrame `.iloc`, sparse, and ndarray row access and add parity tests.
- [x] Remove `MrTotalVIBatchDataModule` from stable exports; retain only an unmistakably private registry adapter if needed.
- [x] Correct conditional `u`/`z` shape documentation and test non-isomorphic shapes.
- [x] Amend/supersede affected ADR-0005/ADR-0007 text without changing ADR-0008/ADR-0009 scientific policy.
- [x] Run the complete MrTotalVI source inventory as non-overlapping terminal partitions; the original 195 nodes passed, then the added Vamp checkpoint regression passed both alone and in the now-56-node affected partition (current total 196).
- [x] Run the smallest shared-helper MrMultiVI engineering regression; the labelled-Vamp compatibility node passed unchanged.

## Source freeze and candidate artifact

- [x] Run benchmark/governance contract tests listed in the plan to terminal exit and record count/runtime: benchmark 224 passed/2 skipped in 341.12s; final governance rerun 68 passed in 6.69s and 12 repository contracts validated.
- [x] Run `git diff --check` and targeted lint/compile checks available in the recorded environment.
- [x] Review the full task diff for scope, test defects, compatibility migrations, and accidental unrelated changes; independent final manual audit reported no concrete pre-commit blocker.
- [x] Stage only the 80 explicit task-owned paths; cached diff validation confirmed no `.living`, `.mycelium`, `.scratch`, source-plan/review, gitlink, wheel, cache, or unrelated artifact was staged.
- [ ] Create a scoped local source commit and record its full SHA; do not push.
- [ ] Reconstruct a clean source tree from that exact commit in a fresh `/tmp` path and verify version `0.2.0`.
- [ ] **BLOCKED — `blocked_build_backend_authority`:** refuse any pre-existing `0.2.0` output, then build exactly one wheel into an empty candidate directory. The recorded source environment has `build` but no local `hatchling`; no claim, candidate directory, backend command, or `0.2.0` wheel was created.
- [ ] **BLOCKED — no candidate exists:** generate sealed manifest/inventory evidence with commit/tree, build environment, dependency authority, metadata, SHA/size, RECORD, complete file inventory, and namespace ownership expectations.
- [x] Verify the old `0.1.0` wheel is unchanged at its original path and still has its recorded SHA.

## Installed-artifact acceptance

- [ ] **BLOCKED — no candidate/exact dependency authority:** run the standalone harness outside the checkout with `PYTHONPATH` unset and no editable install if an exact local dependency authority exists.
- [ ] **BLOCKED — no candidate/exact dependency authority:** verify `pip check`, `cytoanvi==0.2.0`, absence of `scvi-tools`, ownership of both namespaces, import paths, source identity, and installed inventory.
- [ ] **BLOCKED — no candidate/exact dependency authority:** run installed-wheel CytoANVI train/predict/latent/save-load/same-panel query and both treeArches paths.
- [ ] **BLOCKED — no candidate/exact dependency authority:** run installed-wheel MrTotalVI raw-count setup/train/`u`/`z`/summary/save-load.
- [ ] **BLOCKED — no candidate/exact dependency authority:** seal a terminal installed-acceptance receipt with exact command, exit status, runtime, and hashes.
- [x] Record `blocked_dependency_authority`, leave engineering acceptance false, and provide the exact later command without performing any network, dependency installation, or external write.

## P2 protocols and P3 matrix

- [x] Create `docs/protocols/cytoanvi-p2a-usage-readiness-v1.md` and `benchmarks/cytoanvi/usage_readiness_contract_v1.json`.
- [x] Create `docs/protocols/mrtotalvi-p2b-usage-readiness-v1.md` and `benchmarks/mrtotalvi/usage_readiness_contract_v1.json`.
- [x] Freeze required fields only where evidence is pre-existing and independent; all unresolved fields remain explicit blockers.
- [x] Reuse only the sealed 46,817-cell MrTotalVI lineage and B0-B3/D0-D5 authority; apply ADR-0009 rank policy and exclude the 42/48 preserved run.
- [x] Mark missing second cohorts, independent labels, algorithms, thresholds, or controls `draft_unfrozen`/`blocked` instead of inventing values.
- [x] Validate both machine contracts against the protocol schema.
- [x] Materialize all 19 mandatory capability rows with separate engineering, scientific, and promotion fields.
- [x] Prove the matrix validator rejects omitted rows, active/partial/self-declared grids, artifact mismatches, missing negative results, and unsigned promotion.
- [x] Leave independent protocol review, scheduler submission, result adjudication, and named human promotion pending.

## Final verification and reporting

- [x] Re-run all affected source tests after final edits and require terminal zero exits.
- [ ] **BLOCKED — no candidate/exact dependency authority:** re-run installed-artifact acceptance after any source/artifact change; the blocked receipt remains non-passing and no stale pass is reused.
- [x] Confirm no scheduler/GPU/network/credential/CI/push/tag/release action occurred.
- [x] Confirm all unrelated dirty files remain present and unstaged.
- [x] Update this checklist only with verified completions or explicit blocker annotations.
- [ ] Report changed files, local commits, exact test results, artifact identity/acceptance state, protocol freeze states, matrix state, blockers, residual risk, and one next approval/action.
