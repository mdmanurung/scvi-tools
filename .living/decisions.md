# Decisions Log

Record non-obvious choices made during development. One entry per decision. Reference IDs in commit messages and learnings where applicable.

## Format

```
### D-NNN — [YYYY-MM-DD] Title
**Context**: What situation prompted this decision.
**Decision**: What was chosen.
**Rationale**: Why, including alternatives considered.
**Consequences**: What this commits us to or rules out.
**Status**: active | superseded | revisited
```

---

### D-001 — [2026-06-01] Use M1+M2 hierarchy (scANVI-style), not GMM prior
**Source**: `docs/adr/0001-cytoanvi-m2-over-gmm-prior.md`
**Context**: CytoVI uses a GMM prior for label shaping. CytoANVI needed a classification objective.
**Decision**: Use scANVI's M1+M2 two-level latent (z1 classifier input, z2 per-label) instead of GMM. `CytoANVAE` mirrors `TOTALANVAE(SupervisedModuleClass, CytoVAE)` and forces `prior_mixture=False` in the semi-supervised path.
**Rationale**: GMM prior is unsupervised shaping; M1+M2 adds a discriminative loss directly. GMM alone gives no label-transfer accuracy guarantee. The M2 loss ignores `generative_outputs["pz"]`, so an active GMM prior is "dead weight at best and double-counts label shaping at worst."
**Rejected alternatives**: (a) GMM + classifier head only, no M2 — diverges from scANVI/totalANVI, won't align with Phase-2 csCANVI port; (b) keep GMM *and* M2 — double-counts label shaping, novel/unvalidated.
**Consequences**: CytoANVI is architecturally closer to scANVI than CytoVI. Label-conditioned GMM off by default. **Gotcha**: `prior_mixture=False` forced in `CytoANVAE.__init__` — the apparently-unused GMM code path is deliberate, not an oversight.
**Status**: active

### D-002 — [2026-06-10] Benchmark at max_epochs=1000, full cohorts only
**Context**: Prior results used vignette subsamples (100 epochs) — not publication-grade.
**Decision**: All Track-B benchmarks run at max_epochs=1000 on full Nuñez and Roider cohorts.
**Rationale**: Convergence at 100 epochs is insufficient; scib metrics need full-cohort cell counts.
**Consequences**: Each benchmark job takes ~hours on GPU. Gate publication on these results.
**Status**: active

### D-003 — [2026-06-17] Continual update via EWC + replay; follow the paper, not the released code
**Source**: `docs/adr/0002-cytoanvi-continual-follows-paper-not-code.md`
**Context**: B4/B6 tasks require integrating case-control query without catastrophic forgetting. The csCANVI released code and the bioRxiv Methods disagree on implementation details.
**Decision**: `ContinualUpdate` module combines Fisher-weighted EWC + experience replay. When paper and code disagree, **follow the paper**. Specifics: (1) experience replay rehearses ~20% reference cells in the ELBO (released code only uses replay for Fisher, never rehearses); (2) EWC weight = Hadamard product of reference-replay Fisher and query-control Fisher, `combine_type="product"` default; (3) query controls are **required** (must exist in both reference and query); (4) TTA masks **50%** of features for uncertainty (not 15%).
**Rationale**: EWC is theoretically grounded; replay alone risks mode collapse on small query sizes. Paper is the primary source; repo/README are not substitutes (Mikhael's instruction).
**Consequences**: λ (`ewc_importance`) is a new hyperparameter; B6 sweeps it. Fisher computation costs one extra forward pass. Default λ≠100 (paper used 100 for RNA/scANVI, but CytoVI intensity-likelihood Fisher scale differs — must retune; see B6).
**Status**: active

### D-004 — [2026-06-18] `encoder_marker_mask_` persisted on save/load
**Context**: `prepare_query_anndata` was failing when called from a saved model path (mask not reconstructable without original training data).
**Decision**: Persist the encoder marker mask as a model attribute alongside other state dicts.
**Rationale**: Without persistence, any surgery from disk was broken. Simple fix with no downsides.
**Consequences**: Old model checkpoints (pre-fix) need re-training if using path-based surgery.
**Status**: active

### D-005 — [2026-06-29] HCE and scHPL hierarchy are opt-in, matrix-gated; fail-fast on misuse
**Source**: `docs/adr/0003-cytoanvi-hce-schpl-hierarchy.md`
**Context**: CytoANVI needed to support cell-type hierarchies for structured label transfer. Two mechanisms were considered: HCE (hierarchical cross-entropy) and scHPL treeArches workflows.
**Decision**: (1) **HCE is core** but matrix-gated — `CytoANVAE.classification_loss` uses flat `F.cross_entropy` when `reachability_matrix_ is None`; activated explicitly via `set_hierarchy()` / `hierarchy_edges` / `reachability_matrix`. Hierarchy excludes the unlabeled category. (2) **scHPL treeArches** is an optional extra: lazy-import, behind `pip install scvi-tools[cytoanvi-hierarchy]`. **Fail-fast policy**: raise `ValueError` on `predict_hierarchical()` without hierarchy, `set_hierarchy` label mismatch, unmapped scHPL leaves, both `hierarchy_edges`+`reachability_matrix` at init, invalid shape/non-DAG; `ImportError` (with pip cmd) when extra missing. **Never** silently fall back to flat CE or skip scHPL.
**Rejected alternatives**: HCE always-on; both behind one extra; stateful `CytoANVIHierarchyAtlas` orchestrator.
**Consequences**: Users must opt in explicitly. Any code path that expects flat CE by default still works; HCE only activates when a reachability matrix is set.
**Status**: active

### D-006 — [2026-06-29] CytoANVI promoted to top-level package; `scvi.external.cytoanvi` removed
**Source**: `docs/adr/0004-cytoanvi-top-level-package.md`
**Context**: CytoANVI started as `scvi.external.cytoanvi` during prototyping. As it matured, the external namespace became misleading.
**Decision**: CytoANVI is a **top-level package** — `cytoanvi.CytoANVI`, `cytoanvi.CytoANVAE`, `cytoanvi.hierarchy`, `cytoanvi.mapping_qc`. The old `scvi.external.cytoanvi` / `scvi.external.CytoANVI` export and package path are **removed**.
**Consequences**: **Breaking change** — unpickling/model-loading of old artifacts that reference the previous class module path will fail. Tests, tutorials, benchmarks, and API docs import from `cytoanvi`. Benchmark harness stays under `benchmarks/cytoanvi` (not part of the public package). Extras keep names `scvi-tools[cytoanvi-hierarchy]`, `scvi-tools[cytoanvi-mapping-qc]`.
**Status**: active

### D-007 — [2026-06-29] Manifest-mode aggregation for publication; recursive --input is exploratory only
**Source**: `.scratch/cytoanvi-benchmark/PRD.md` (2026-06-28 decisions table)
**Context**: `aggregate_results.py` supports both recursive `--input` (gathers all JSONs under a path) and manifest mode (explicit `publication_manifest.json` allowlist). Recursive mode accidentally picked up smoke/synthetic/stale-roider results alongside e1000 full-cohort results.
**Decision**: Publication aggregation uses `publication_manifest.json` exclusively. Recursive `--input` is for exploratory inspection only and must not be used to generate publication claims.
**Consequences**: Any result JSON not listed in `publication_manifest.json` is silently excluded from publication aggregation. This is a feature, not a bug.
**Status**: active

### D-XXX — [2026-06-29] Cancel B8 job 25107490; resubmit as 25108052 after second review fixes
**Context**: Second code review found two B8 correctness issues after job 25107490 had been running for 2+ hours.
**Decision**: Cancel and resubmit rather than let invalid results land as publication artifacts.
- **#3 leaf_held bias**: `delta_hierarchical_vs_flat_macro_f1` was evaluated over all held cells; cells with internal-node true labels always score wrong under `predict_hierarchical(leaf_only=True)`, biasing delta against HCE. Fix: exclude internal-label cells from evaluation (see L-012).
- **#8 HCE routing**: Hand-rolled HCE setup/train can drift from flat arm's config; both arms now route through `train_cytoanvi(hierarchy_edges=...)` (see L-013).
**Consequences**: No result files were written before cancellation. Resubmitted as job 25108052.
**Status**: active

### D-008 — [2026-07-03] Distribute CytoANVI as a standalone package named `cytoanvi` (not an upstream scvi-tools PR)
**Source**: Publication-readiness review (session 13); maintainer decision.
**Context**: The packaging review found the fork still published under `name = "scvi-tools"` with scverse URLs/maintainers, which would collide with the upstream PyPI package. Two paths existed: upstream PR to scverse, or standalone release.
**Decision**: Release standalone under PyPI name `cytoanvi`. `pyproject.toml` identity (name, description, authors/maintainers, Source/Home-page URLs) now points to the fork (github.com/mdmanurung/scvi-tools, Mikhael Manurung). All self-referencing optional-dep extras rewritten `scvi-tools[...]` → `cytoanvi[...]`. Install docs/README updated accordingly. LICENSE now carries dual BSD-3 attribution (upstream scvi-tools dev team + CytoANVI contributors) to satisfy BSD-3 clause 1.
**Consequences**: The wheel still ships both `scvi` and `cytoanvi` packages. A new `[tool.hatch.build.targets.sdist]` exclude list + `.gitattributes` export-ignore keep `.scratch/`, `.living/`, `todo/`, logs, and PDFs out of published artifacts.
**Status**: active

### D-009 — [2026-07-03] B5 novelty headline metric is mean AUROC, not best AUROC
**Source**: Benchmark-rigor + methods reviews (session 13).
**Context**: B5 results led with `best_auroc` (max over ~47 held-out cell types ≈ 0.83) while `mean_auroc` (≈ 0.46, below chance) was buried. `best_auroc` is a maximum, not a summary statistic — reporting it as headline overstates the uncertainty module's utility.
**Decision**: `mean_auroc` (and `mean_auroc_fdr_sig`) is the PRIMARY reported field everywhere (task return dict, aggregation, manifest). `best_auroc` retained only as an explicitly-labeled secondary/alias (kept because `test_aggregate_results.py` asserts on it). Underlying computation unchanged — only presentation and labels.
**Consequences**: The user guide and manifests now report B5 honestly as near-chance on most T-cell subtypes; a genuine improvement requires either a better uncertainty formulation or an external novel-cell-type dataset.
**Status**: active

### D-010 — [2026-07-03] B3 cross-panel metric is inter-method concordance, labeled as NOT accuracy
**Source**: Benchmark-rigor + methods reviews (session 13).
**Context**: `task_b3_panel_divergent` reports `p2_inter_method_agreement_vs_knn` — agreement between CytoANVI predict() and CytoVI-kNN on panel-2, where both share the CytoVI encoder backbone and there is NO panel-2 ground truth. Two methods that are both wrong would score high.
**Decision**: Keep the metric (no independent labels available) but relabel it unmistakably as inter-method concordance, NOT ground-truth accuracy, in the metric docstring, task note, and ANALYSIS_MANIFEST. A definitive cross-panel claim requires independent manually-gated panel-2 labels or an architecturally-independent baseline (FlowSOM/XGBoost).
**Consequences**: B3 cannot be presented as validating cross-panel correctness until independent labels are obtained. Tracked as an open blocker.
**Status**: active

### D-012 — [2026-07-05] Honest full-cohort numbers reconciliation: p2 concordance gate NOT met; B5 is a robust negative result
**Context**: Full-cohort Roider B3/B5 results (jobs 25145151/52/53) arrived and substantially differed from roider-e1000 subset numbers. Every prose surface (ANALYSIS_MANIFEST, FINDINGS_REGISTRY, cross-panel-mapping, CHANGELOG, notes, plans) still showed the stale subset values.
**Decision**: (1) Update ALL prose surfaces to the honest full-cohort numbers (B3 p1 0.828±0.015, B3 p2 concordance 0.671±0.008, B5 mean_auroc 0.484±0.019 NEGATIVE). (2) The p2 concordance gate (≥0.80) is NOT met and must not be claimed as such — p1 macro-F1 (0.828) is the defensible supervised headline. (3) B5 TTA-uncertainty is a robust negative result — properly framed, not buried. (4) Historical docs annotated as SUPERSEDED rather than rewritten (they record what we believed at the time).
**Rationale**: Presenting stale numbers as current is an honesty liability regardless of whether the gate was met. The full-cohort result is the ground truth; all other numbers are exploratory history.
**Consequences**: B3 p2 concordance gate "≥0.80" will not be met unless independent panel-2 ground-truth labels are obtained (see Idea 4 in analysis/ideas/). This is explicitly tracked as an open blocker, not ignored.
**Status**: active

### D-013 — [2026-07-05] Version bump feat/cytoanvi to 0.1.0; push branch to GitHub origin
**Context**: `pyproject.toml` still carried the inherited upstream version `1.5.0rc1`. The feat/cytoanvi branch had never been pushed to origin (github.com/mdmanurung/scvi-tools).
**Decision**: Bump to `version = "0.1.0"` — signals this is an independent standalone release that predates any PyPI upload and does not collide semantically with upstream scvi-tools versions. Push `feat/cytoanvi` to origin.
**Rationale**: `0.1.0` is honest: the package is usable but not yet publication-released. `1.5.0rc1` implies we're shipping an scvi-tools release candidate, which is misleading.
**Consequences**: `importlib.metadata.version("cytoanvi")` returns `"0.1.0"` after `pip install -e .`. No PyPI upload yet (deferred, see D-011).
**Status**: active

### D-011 — [2026-07-03] Defer public packaging; ship standalone cytoanvi for internal/clean-env use only
**Source**: Engineering-maturity review (session 13); maintainer decision on the scvi/ namespace collision (L-035).
**Context**: dist `cytoanvi` ships both `scvi/` and `cytoanvi/` import packages (the fork needs a modified scvi for CytoVI at `scvi.external.cytovi`), which collides with upstream `scvi-tools` on the `scvi/` import name. Options were Replace / Coexist (vendor CytoVI) / upstream-PR / defer.
**Decision**: DEFER public/PyPI packaging. Keep the standalone dual-package wheel for research/internal/HPC use, installed alone in a clean environment. Document the "install from source, do not co-install with scvi-tools, not on PyPI" constraint in README + installation docs. Revisit the Replace-vs-Coexist-vs-upstream-PR decision before any PyPI upload.
**Consequences**: `pip install cytoanvi` from PyPI is intentionally NOT offered; docs use `pip install .` from a checkout. The dual-package collision is acceptable under the clean-env constraint. Version 1.5.0rc1 (inherited from scvi-tools) is left as-is until a real release.
**Status**: active

### D-014 — [2026-07-08] MrMultiVI: EncoderXU_MultiVI takes MULTIVAE mixed latent u0 (no log1p)
**Source**: MrMultiVI implementation session (this session)
**Context**: EncoderXU_TotalVI takes raw RNA+protein counts concatenated and applies `log1p` before the first linear layer. MrMultiVI's equivalent encoder needs a different input because MULTIVAE's output is already a continuous latent `z` (the mixed multimodal embedding from `mix_modalities`), not raw counts.
**Decision**: `EncoderXU_MultiVI.forward(u0, sample_covariate)` takes `u0 = MULTIVAE's inference["z"]` directly with NO `log1p` preprocessing. The same ConditionalNormalization → GELU → NormalDistOutputNN architecture is used, but the input semantics differ. Do not add log1p to this path — it would distort negative latent values (log1p of negative = NaN or bad gradients) and is meaningless for a latent representation.
**Consequences**: `EncoderXU_TotalVI` and `EncoderXU_MultiVI` share architectural structure (both live in `mrtotalvi/_components.py`) but differ in input preprocessing. Future MultiVI-style models (ATAC-only, etc.) should follow the same pattern: no log1p when input is a latent, log1p when input is raw counts.
**Status**: active

### D-017 — [2026-07-11] MrVI-style DE: sample u once per MC draw, hold fixed across donors
**Context**: MrVI's original DE implementation uses `torch.vmap(inference_fn, randomness="different")` which draws a fresh `u ~ q(u)` for *each* counterfactual donor substitution. This means `eps_d` for donor d uses a different u draw than `eps_{d'}`.
**Decision**: In `_differential_expression`, sample `u` once per MC draw and call `module.qz(u, cf_d)` for each kept donor with that *same* u held fixed. This is the strict counterfactual semantics: "given this cell's latent state u, what would its donor-specific residual be in each donor?" For large mc_samples both approaches converge to the same marginal, but our approach is computationally cheaper (n_donor qz calls per mc draw, not n_donor×mc_samples inference calls).
**Consequences**: `eps_d` and `eps_{d'}` are not iid across d — they share the same u draw. Effect on inference is negligible at mc_samples≥50 by CLT.
**Status**: active

### D-018 — [2026-07-11] donor_key in DA: within-donor centering of log_probs
**Context**: For multi-donor multi-condition experiments, the aggregated-posterior log_prob DA test conflates donor effects with condition effects. Donors vary in baseline cell composition.
**Decision**: When `donor_key` is provided to `differential_abundance`, subtract the per-donor mean log_prob across that donor's samples (within-donor centering) before the covariate aggregation step. This blocks out donor as a fixed effect and leaves only within-donor condition contrasts.
**Consequences**: Only valid when each donor spans ≥2 samples (e.g., multiple conditions or time points per donor). Warns if single-sample donors are present (skip centering for them). DA log_probs after centering have mean ≈ 0 per donor, and the condition_log_probs reflect condition contrasts within donors.
**Status**: active

### D-019 — [2026-07-11] Wald stat df = n_admissible_samples_per_cell (not n_latent)
**Context**: Old `differential_expression` in MrTotalVI used `df=n_latent` (number of latent dimensions) for the chi2 test. MrVI torch uses the number of samples as the degrees of freedom.
**Decision**: `ts[n, k]` is the sum over latent dims of `betas_norm[mc, n, k, d]^2` averaged over mc draws. Under H0 this is a weighted sum of squared standard normals. The effective df is `n_admissible_samples_per_cell` (the rank of the WLS system), not n_latent. We use `Chi2(df=admiss.sum(1).clamp(min=1))` per cell. This matches MrVI's convention.
**Consequences**: Cells with fewer admissible samples get lower df, more conservative p-values. Correct for sparse layouts.
**Status**: active

### D-016 — [2026-07-10] pz_scale: clamp lower bound at min=-4.0 to prevent kl_z→−∞
**Context**: When `learn_z_u_prior_scale=True` (non-default), `pz_scale` is an unclamped `nn.Parameter` that feeds `peps = Normal(0, exp(pz_scale))` in both `_module.py` loss paths. Under `use_map=True`, `kl_z = -Normal(0, exp(pz_scale)).log_prob(eps)`. If both flags are non-default together, the optimizer can drive `pz_scale→−∞` (σ→0) jointly with `eps→0`, sending `kl_z→−∞` and the ELBO unbounded above — a degeneracy collapse.
**Decision**: Apply `.clamp(min=-4.0)` at the point of use: `peps = Normal(0.0, torch.exp(self.pz_scale.clamp(min=-4.0)))` in both `mrtotalvi/_module.py` and `mrmultivi/_module.py`. The floor `exp(-4) ≈ 0.018` is narrow enough to allow learning without permitting the degenerate σ→0 collapse. The clamp is NOT stored back to the parameter (non-projected); it is applied only during the forward pass.
**Rationale**: Alternatives: (a) raise/warn when `learn_z_u_prior_scale=True and use_map=True` — too restrictive, both flags together is a legitimate non-default config; (b) use a softplus instead of exp for the scale — would require changing the parameter initialization and affects the default path (not worth the blast radius). The clamp is the minimal, targeted fix.
**Consequences**: `pz_scale` remains unclamped as a stored parameter; users inspecting `module.pz_scale.min()` can observe values below -4.0 in theory but the effective prior σ is floored. Added regression test `test_learnable_prior_scale_clamp` (both suites) that asserts `kl_z` is finite and `> -1e6` after training with both flags. Commit e0ff9255.
**Status**: active

### D-015 — [2026-07-08] use_map=False: use analytic KL(q(eps) || p(eps)) in loss
**Context**: When eps is stochastic (use_map=False), eps is sampled via reparameterisation from N(eps_mean, exp(eps_log_scale)). Original implementation used kl_z = -log p(eps), which is only the cross-entropy term and misses the entropy H[q(eps)].
**Decision**: `EncoderUZ.forward()` now returns a 3-tuple `(z_base, eps, eps_dist)` where `eps_dist = Normal(eps_mean, eps_scale)` when `use_map=False`, else `None`. Both `loss()` methods use `kl_divergence(eps_dist, peps).sum(dim=-1)` when `eps_dist is not None`, and fall back to `-peps.log_prob(eps)` when `use_map=True` (deterministic eps, entropy=0, so KL=cross-entropy).
**Rationale**: Correct ELBO for stochastic eps requires `KL(q||p) = H[q(eps)] + cross-entropy(q||p)`. Omitting H[q] causes the model to over-penalize eps variance → `eps_log_scale → -∞` → posterior collapse on the second-level latent. The analytic KL between two Gaussians is cheap and exact.
**Consequences**: `use_map=False` now trains correctly. All callers of `qz()` unpack 3-tuple. The `eps_dist` key is stored in inference outputs alongside `eps`. Commit 059fe952.
**Status**: active (supersedes the [2026-07-08 revisit] note)

---

### D-020 — [2026-07-11] Protein contrast quantity for MrTotalVI LFC: `py_["scale"]`
**Context**: When implementing decoded protein LFC for `MrTotalVI.differential_expression`,
the question arose: which decoded quantity should serve as the protein "expression" in
the fold-change contrast? MRVI uses `px.mean / library` (RNA only; no protein reference).
**Decision**: Use `py_["scale"]` (background-adjusted, L1-normalized foreground mean;
`_base_components.py:962`). RNA contrast uses `px_["scale"]` (softmax-normalized rate,
MRVI-`h` analog).
**Rationale**: `py_["scale"]` is TotalVI's own documented DE convention
(`DecoderTOTALVI` docstring `:874-881`: "foreground mean adjusted for background
probability and scaled to reside in simplex"). Using a different quantity would
diverge from the conventions users of TotalVI already know.
**Consequences**: `compute_h_from_x_eps` returns `concat(px_scale, py_scale)`;
feature axis is split into `gene`/`protein` coords at output. For MrMultiVI with no
protein layer, only `px_scale` is returned.
**Status**: active

---

### D-021 — [2026-07-11] Deterministic protein background on the LFC contrast path
**Context**: `DecoderTOTALVI.forward` computes `rate_back = exp(Normal(back_alpha,
back_beta).rsample())` (`_base_components.py:~942`), which feeds `rate_fore → py_["scale"]`.
With two separate forward passes (x_1 vs x_0 contrast), independent background samples
would add noise that does NOT cancel → protein LFC = biological signal + background noise.
**Decision**: On the contrast path inside `compute_h_from_x_eps`, reconstruct `py_["scale"]`
using the deterministic `back_alpha` mean (`rate_back = exp(back_alpha)`) instead of
sampling. Leave the training/ELBO `generative` path unchanged (stochastic background is
correct for the NB likelihood).
**Rationale**: The contrast should differ only via `extra_eps`; otherwise two
counterfactual states differ in background noise, not in biology. `rsample()` is inside
the inherited `DecoderTOTALVI.forward` — this is not a flag but requires a decoder
override or a post-forward reconstruction.
**Consequences**: `compute_h_from_x_eps` overrides the generative path on the contrast
path only. Must verify with the D2 determinism test: null-covariate protein `lfc_std ≈ 0`.
**Status**: active

---

### D-022 — [2026-07-11] Feature layout: concat(px_scale, py_scale) with coord split
**Context**: `compute_h_from_x_eps` needs to return a single tensor for the LFC
contrast; downstream code splits into `gene`/`protein` xarray coordinates.
**Decision**: Return `concat(px_scale, py_scale, dim=-1)` from the hook. In `_stats.py`
(B2) and the model wrappers (B3), split at `n_genes` into `gene`/`protein` feature coords.
MrMultiVI with no protein layer returns only `px_scale`; ATAC features are excluded.
**Rationale**: Mirrors MRVI's single-tensor return (`px.mean/library`) while extending
to multimodal output. Keeping a single tensor simplifies the einsum-based LFC computation.
**Consequences**: MrTotalVI output xarray gains `feature` dim with `gene_*`/`protein_*`
coord values. MrMultiVI output is `gene_*` only (RNA) unless protein layer present.
**Status**: active

---

### D-023 — [2026-07-11] vmap default: use_vmap=False for both models; opt-in for MrMultiVI
**Context**: MRVI uses `torch.vmap` to parallelize per-cell decoder calls in the LFC
path. MrTotalVI's TOTALVAE decoder defaults `use_batch_norm="both"` (`_totalvae.py:146`),
which is vmap-incompatible (BatchNorm uses running stats). MrMultiVI's MULTIVAE decoder
defaults `use_layer_norm="both"` (`_multivae.py:296-297`), which is vmap-safe.
**Decision**: Default `use_vmap=False` (explicit per-cell loop) for **both** models on
the LFC path, matching the existing eps-loop style (`_stats.py:460-471`). Expose
`use_vmap=True` as opt-in for MrMultiVI only.
**Rationale**: Defaulting both to False avoids BatchNorm errors on a first landing and
matches the existing loop style. MrMultiVI users can opt in once the path is validated.
**Consequences**: LFC computation is O(n_cells) per donor step but avoids BatchNorm
vmap errors. MrTotalVI users should not pass `use_vmap=True`.
**Status**: active

---

### D-024 — [2026-07-11] encode_covariates: pass batch/cont/cat kwargs to qu() unconditionally
**Context**: `_setup_hierarchy()` in both MrTotalVAE and MrMultiVAE was hardcoded to
`encode_covariates=False`, so covariate information never reached the u-encoder even
when users configured `encode_covariates=True` at model construction.
**Decision**: Change `_setup_hierarchy()` to use `self.encode_covariates`. Pass
`batch_index, cont_covs, cat_covs` as kwargs to `qu(...)` unconditionally — the encoder
ignores them when `encode_covariates=False`; they're used only when True. This avoids an
extra conditional branch in inference.
**Consequences**: `RecordingQu` test mock needed `**kwargs` to absorb these kw args
(else `unexpected keyword argument 'batch_index'`). Future test mocks for qu() should
accept **kwargs.
**Status**: active

### D-025 — [2026-07-11] VampPrior reference-prior design: sample_idx=0, Softplus-constrained pseudoinputs
**Context**: Adding a learnable VampPrior option to MrTotalVI/MrMultiVI. Prior pseudoinputs must pass through the actual `qu` encoder. Two design choices: (1) what sample index to use for the prior, and (2) whether to constrain pseudoinputs.
**Decision**: Use `sample_idx=zeros(K,1)` (reference donor 0) and Softplus-constrain TotalVI pseudoinputs. MrMultiVI pseudoinputs (in latent space) are unconstrained.
**Rationale**: The prior should be batch/sample-invariant; fixing to donor 0 excludes donor identity from the prior while letting the K learnable pseudoinput vectors absorb full manifold variation. Softplus ensures log1p inside EncoderXU_TotalVI stays well-defined (raw counts are ≥ 0). MrMultiVI's `qu` takes continuous MULTIVAE latent as input (no log1p), so no constraint is needed.
**Consequences**: VampPrior prior is weakly conditioned on the reference batch embedding when `encode_covariates=True`; this is intentional (reference prior, not a batch-free prior). Future work: could marginalize over batches.
**Status**: active

### D-026 — [2026-07-11] MrMultiVI protein_in_encoder: log1p(y) cat'd to u0 before qu, protein pseudo Softplus-constrained
**Context**: MrMultiVI's `qu` previously received only `u0` (MULTIVAE mixed latent), so protein never reached the sample-conditioned encoder. MrTotalVI feeds raw protein directly via `log1p` inside `EncoderXU_TotalVI`. Closing the parity gap.
**Decision**: Add `protein_in_encoder: bool = False` toggle. When True, `EncoderXU_MultiVI.forward` cats `log1p(y_protein)` to `u0` before `fc1`. In `_vamp_component_dist`, the protein slice of `u_vamp_pseudo` is Softplus-constrained before being passed (so `log1p(Softplus(x)) ≥ 0`). The u0 slice is unconstrained.
**Rationale**: Mirrors MrTotalVI's exact pattern. Using `log1p` is numerically stable for protein counts. Softplus-constraining only the protein pseudo-columns keeps the u0 pseudo-columns free to roam latent space. Default off so no existing behaviour changes.
**Consequences**: When `protein_in_encoder=True`, `fc1.in_features = n_latent + n_input_proteins` (wider). Save/load is safe because `protein_in_encoder` is in `init_params_`. The vamp + protein combo is covered by test_protein_in_encoder_with_vamprior.
**Status**: active
