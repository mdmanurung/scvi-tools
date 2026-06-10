# CytoANVI — Investigation

Investigation-first notes grounding the CytoANVI design in source actually read in this repo.
Every claim cites a file:line. Anything not confirmable from source is marked **unconfirmed**.

## Environment / version pin

- **Source under development:** this repo checkout, `scvi-tools` **1.4.3**
  (`pyproject.toml:7`, `requires-python >= 3.12` at `pyproject.toml:10`), git `6fe4bd88`.
- **Test/runtime env used:** conda env `scvi-test` (Python 3.13, `scvi-tools` 1.4.2 dependency
  set), invoked as `PYTHONPATH=src LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python` so the **repo 1.4.3
  source** is imported while dependencies (torch, lightning, anndata, …) come from the env. No
  editable install was performed, to avoid mutating any of the user's environments.
- `scvi.__version__` reports the *installed* dist metadata (1.4.2 in that env), **not** 1.4.3,
  because the source is run via `PYTHONPATH` rather than installed. When packaged/installed
  normally it will report 1.4.3. This is the only version caveat; all API decisions below match
  the 1.4.3 source files in `src/scvi/`.

## The two architectures, side by side

### CYTOVI (`scvi.external.cytovi`)
- Model `CYTOVI(RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass)`
  (`cytovi/_model.py:66-72`); `_module_cls = CytoVAE`, `_training_plan_cls = AdversarialTrainingPlan`
  (`:137-138`).
- Module `CytoVAE(BaseModuleClass)` (`cytovi/_module.py:19`): `Encoder` → `z`
  (`:139-153`); `DecoderCytoVI` → **protein likelihood Normal** (loc, softplus scale) **or Beta**
  (`:284-294`, `:446-571`). **Intensity-valued, not count-based** — confirmed: no NB/ZINB anywhere.
- **Label-conditioned mixture-of-Gaussians prior** (`:172-179`, `:296-318`): when
  `n_labels > 1`, mixture components = labels and the prior logits are boosted by
  `prior_label_weight * one_hot(y)`; cells with `y >= n_labels` (out-of-range/unlabeled) get the
  base prior only (`:301-307`).
- Reconstruction masks unobserved markers via `nan_layer` (`:335-357`); `encoder_marker_mask`
  selects backbone markers for the encoder (`:131-153`, applied in `_get_inference_input` at
  `:195-198`).
- `setup_anndata` registers labels with a **plain `CategoricalObsField`** — no unlabeled handling
  (`cytovi/_model.py:321`). Registry keys (`cytovi/_constants.py`) use the same string values as
  the global `REGISTRY_KEYS` (`X`, `batch`, `labels`, `extra_categorical_covs`,
  `extra_continuous_covs`) plus extras (`sample_id`, `nan_layer`).
- Existing **non-parametric label transfer**: `impute_categories_from_reference`
  (`cytovi/_model.py:1046`) does k-NN voting in latent space (`cytovi/_utils.py`), **not** a
  trained classifier. Also `impute_rna_from_reference` (`:1137`), `get_aggregated_posterior`
  (`:846`), `differential_abundance` (`:957`), `differential_expression` (`:699`).
- scArches surgery: inherited from `ArchesMixin`, no overrides (only `register_manager` appears).

### SCANVI / SCANVAE (`scvi.model`)
- `SCANVI(RNASeqMixin, SemisupervisedTrainingMixin, VAEMixin, ArchesMixin, BaseMinifiedModeModelClass)`,
  `_module_cls = SCANVAE`, `_training_plan_cls = SemiSupervisedTrainingPlan` (`model/_scanvi.py:51-113`).
- `SCANVAE(SupervisedModuleClass, VAE)` (`module/_scanvae.py:26`): **M1+M2** — `classifier` on z1,
  `encoder_z2_z1`/`decoder_z1_z2` (`:142-170`), `broadcast_labels` label marginalization, ELBO
  with z2 ~ N(0,I) and `classification_loss` on labeled cells (`:178-278`).
- `from_scvi_model(unlabeled_category, labels_key, ...)` transfers SCVI weights via
  `load_state_dict(strict=False)` (`model/_scanvi.py:211-295`); `setup_anndata` uses
  `LabelsWithUnlabeledObsField` (`:331`), `n_labels` adjusted for the unlabeled category
  (`:135-142`).

### TOTALANVI — the precedent we mirror (`scvi.external.totalanvi`)
- `TOTALANVI(SemisupervisedTrainingMixin, TOTALVI)`, `_module_cls = TOTALANVAE`
  (`totalanvi/_model.py:30,93`); `__init__` calls `super().__init__`, then
  `_set_indices_and_labels()`, then **rebuilds** `self.module` with `n_labels =
  summary_stats.n_labels - 1` (`:108-218`). `from_totalvi_model` mirrors `from_scvi_model`
  (`:220-297`); `setup_anndata` uses `LabelsWithUnlabeledObsField` (`:348`).
- `TOTALANVAE(SupervisedModuleClass, TOTALVAE)` — a literal SCANVAE M1+M2 port reusing totalVI's
  likelihood for reconstruction (`totalanvi/_module.py:22`, `broadcast_labels`/`encoder_z2_z1`/
  `decoder_z1_z2` at `:318-320`, `classification_loss` at `:436`).

## The semi-supervised contract (what CytoANVI must satisfy)

- **Module** must subclass `SupervisedModuleClass` (`module/base/_base_module.py:776`), which
  provides `classify` (`:784-830`), `classify_helper` (`:780`), and `classification_loss`
  (`:832-859`). Its `loss(..., kl_weight, labelled_tensors=None, classification_ratio=None)` must
  return a `LossOutput` populating `classification_loss/true_labels/logits` when
  `labelled_tensors` is given.
- The base `classify` (`:815-830`) uses `self.z_encoder`, `self.log_variational`,
  `self.encode_covariates` — all present on `CytoVAE` — and applies `log1p`, matching CytoVAE's
  `torch.log(1 + x)` (`cytovi/_module.py:235`). **Key fact:** both `predict()`
  (`model/base/_training_mixin.py:293`) and `classification_loss` (`:836`) build `data_inputs`
  via `module._get_inference_input(...)`, and CytoVAE's `_get_inference_input` **already applies
  `encoder_marker_mask`** (`cytovi/_module.py:195-198`). So `x` arrives pre-masked and the
  inherited `classify` works unchanged — no override needed (verified against missing-marker path).
- **Model** uses `SemisupervisedTrainingMixin` which sets `_training_plan_cls =
  SemiSupervisedTrainingPlan`, `_data_splitter_cls = SemiSupervisedDataSplitter`
  (`model/base/_training_mixin.py:178-180`). `_set_indices_and_labels()` (`:182-205`) populates
  `_labeled_indices/_unlabeled_indices/_label_mapping/_code_to_label/unlabeled_category_`;
  `train()` builds `self._training_plan_cls(self.module, self.n_labels, **plan_kwargs)` (requires
  `self.n_labels`); `predict()` returns labels via `module.classify` (`:207-356`).
- `LabelsWithUnlabeledObsField` remaps the unlabeled category to the **last** integer code
  (`data/fields/_scanvi.py:41-67`), so observed-label count = registry `n_labels − 1`.
- Because CYTOVI's registry-key strings equal the global `REGISTRY_KEYS` values, the
  global-keyed mixin / training-plan / dataloader interoperate with CYTOVI's registry unchanged.

## The genuine conflict and its resolution

CYTOVI injects label structure through a **label-conditioned GMM prior on z1**; SCANVAE/TOTALANVAE
inject it through the **M1+M2 hierarchy** assuming a plain N(0,I) prior, recomputing the z1 prior
via `decoder_z1_z2` inside the loss. The two overlap: an M2 loss does not read
`generative_outputs["pz"]`, so keeping CYTOVI's GMM prior active would either be dead weight or
double-count label shaping.

**Resolution (see ADR 0001):** CytoANVAE mirrors TOTALANVAE — M1+M2 port on CYTOVI's protein
likelihood — and **forces `prior_mixture=False`** in the semi-supervised path. CYTOVI's GMM prior
remains available in plain CYTOVI. This is also required for the Phase 2 `cscanvi` continual-update
port, whose replay/Fisher/freezing machinery assumes the scANVI M2 structure.

## Phase 2 — implemented (continual case-control update, paper-faithful)

Grounded in the **paper** (bioRxiv 10.64898/2026.03.03.708171, Methods pp.18-19), not the released
code. **The released `cscanvi` code diverges from the paper** in one key respect: its
`_replay_forward`/`loss_with_replay` only use the replay buffer to estimate Fisher importances and
never rehearse it in the ELBO. The **paper's loss** is

``L(θ_query) = ELBO(x_query, x_replay) + (λ/2) (F_X_Reference ∘ F_X_QueryCtrl)(θ_query − θ_ref)²``

so CytoANVI follows the paper (see ADR 0002):
- **Experience Replay**: the training step rehearses a replay-buffer minibatch (≈20% reference
  cells) by adding its plain ELBO to the query loss.
- **EWC weight** = Hadamard **product** of the reference-replay Fisher (`F_X_Reference`) and the
  query-control Fisher (`F_X_QueryCtrl`); `combine_type="product"` is the default.
- **Query controls required** (the term is `F_reference ∘ F_query_ctrl`); controls must exist in
  both reference and query.
- **TTA masks 50%** of features for the Bregman-Information uncertainty (`get_uncertainty`).

CytoANVI adaptations to modern scvi 1.4.3: `LossOutput` (not `LossRecorder`), `qz` distribution
(not `qz_m/qz_v`), reuse of `ArchesMixin.load_query_data` for surgery/freezing. Surface:
`CytoANVI.load_query_data_with_replay`, `get_uncertainty`,
`CytoANVAE.loss_with_replay`/`_replay_forward`, `CytoANVIContinualTrainingPlan` (adds the
replay-buffer ELBO + threads `ewc_importance` = λ). The configured update is owned by one module —
see below — and is absent (`None`) on the base path, so the base model is unchanged.

### ContinualUpdate — one module owns the configured update

The Phase-2 state used to be five loose attributes on `CytoANVAE` (`old_params`, `importances`,
`ctrl_importances`, `combine_type`, `_replay_batches`), set in `load_query_data_with_replay` and
read far away in `_ewc_penalty` and the training plan, each guarded by silent `if ... is None`
toggles. They are now one deep module, `ContinualUpdate` (`_continual.py`), holding the reference
anchor, both Fisher importances, the combine rule, and the replay buffer behind a small interface:
`configure(reference_model, query_model, replay_adata, control_adata, combine_type)`,
`penalty(module)` (unscaled — λ stays a train-time `ewc_importance`), `next_replay_batch`, and
`persistable_state`/`from_persistable_state`. `CytoANVAE` holds one `self.continual` (present =
active, absent = base path), so the silent toggles are gone by construction. The Fisher loop lives
in `fisher_importances` (moved from the old `_model._compute_importances`).

**Persistence (the former save/load gap).** The EWC anchor + both Fishers + combine rule are
persisted across `save`/`load`: `load_query_data_with_replay` stores
`model.continual_update_state_` (a `_`-suffixed model attribute, so it rides scvi's pickled attr
dict), and `CytoANVAE.on_load` rebuilds `self.continual` from it. The replay buffer is **not**
persisted (session-scoped); after a reload `predict`/`get_latent_representation`/`get_uncertainty`
work immediately, while resuming continual *training* requires re-supplying `replay_adata`. Tested
by `test_cytoanvi_continual_save_load` and the penalty math by `test_continual_update_penalty_math`.

## Divergence notes (CytoANVI vs cscanvi / scArches)

- **Freezing during surgery.** CytoANVI uses scvi's stock `ArchesMixin._set_params_online_update`
  (`model/base/_archesmixin.py:396`), which is the same code cscanvi vendored — including
  `mod_inference_mode = {"encoder_z2_z1","decoder_z1_z2"}` and `freeze_classifier`. cscanvi's only
  extras (`l_encoder`, `background_pro_*`) are RNA/totalVI params absent in CytoVI, so they are
  inert. Empirically, after CytoANVI surgery (default `freeze_classifier=True`) the trainable set
  is the encoder/decoder/classifier *first* layers, with `encoder_z2_z1` fully frozen. Note:
  `requires_grad=True` on a first layer does **not** mean it updates — scArches gradient hooks
  (`set_online_update_hooks`) mask gradients to the *new input columns* only (new-batch one-hot).
  The classifier's input is `z` (no new columns) so `freeze_classifier=True` is *effectively*
  frozen via the hook despite `requires_grad=True`; the encoder first layer does gain new-batch
  columns and adapts. This matches cscanvi and the canonical scArches behavior.

## Divergence verdicts (grilled vs the paper)

- **EWC penalty alignment** — CytoANVI aligns the reference `old_params`, replay Fisher, and
  query-control Fisher **by parameter name with a size guard** (skipping params resized by surgery),
  rather than the released code's positional `zip`. Strictly more robust; required for the
  new-batch case (resized batch params). Kept.
- **Fisher coverage** — importances are estimated over all trainable params (encoder, decoder,
  classifier, z2), a superset of the paper's "encoder and decoder weights"; frozen/size-mismatched
  params contribute ~0 to the penalty. Kept.
- **`ewc_importance` (λ)** — paper used `replay=0.2, EWC=100` (scANVI/RNA). Not adopted as the
  default: CytoVI's intensity likelihood has different Fisher magnitudes, so λ must be retuned;
  documented rather than hard-coded.
- **`get_uncertainty` `tta_rep`** — default 50 (paper example used ~200); a balance of BI stability
  vs cost, documented. Unlike the released code (which ignored the argument), CytoANVI respects it.
- **CytoVI intensity base, `y_prior="empirical"`, `latent_distribution="normal"` guard** — base is
  intrinsic to CytoANVI; the latter two are CytoANVI enhancements absent from the paper. Kept.

## Phase 2 reference — `theislab/comparative_atlas` (`cscanvi`)

A continual-learning extension of scANVI for case–control atlas building (read from the public
GitHub raw source; treat signatures as **unconfirmed** until re-read at implementation time):
- `load_query_data_with_replay(adata, reference_model, control_uns_key=None, replay_uns_key=None,
  freeze_* flags...)` — scArches surgery + a Bregman-Information-selected replay buffer of
  reference cells + optional healthy-control anchoring.
- `_compute_importances(model, dataloader)` — Fisher-information parameter importances (EWC-style).
- `CLSemiSupervisedTrainingPlan` calling `module._replay_forward(...)` — regularized fine-tuning
  that mixes replay-buffer cells back into minibatches.
- `get_uncertainty(...)` — test-time-augmentation predictive uncertainty for the query.
The module layout mirrors scvi (`_scanvae.py`, `_scanvi.py`, `_trainingplans.py`, `_utils.py`,
`_vae.py`), i.e. a fork of scANVI internals plus replay/Fisher utilities.

## Panel-aware scArches query prep (`prepare_query_anndata`)

The inherited `ArchesMixin.prepare_query_anndata` zero-fills markers absent from the query panel
(`_archesmixin.py:_pad_and_sort_query_anndata`, pads with `csr_matrix(zeros)`). For cytometry
intensities zero is a real measurement, so those padded markers would be read as observed-zero
signal. `CytoANVI.prepare_query_anndata` overrides this: it pads to the reference panel (reusing the
base pad/sort) and writes CytoVI's `nan_layer` so the absent markers are masked out of the
likelihood (`cytovi/_module.py:353-354`, `reconst_loss * nan_mask`), mirroring `register_nan_layer`
(`cytovi/_preprocessing.py:194-198`: `1` = observed, `0` = missing).

**Backbone constraint (grounded in source).** CytoVI derives the encoder backbone from the nan mask
— `backbone = markers with no zeros in any cell` (`cytovi/_model.py:176`) — and `encode_backbone_only`
defaults `True` (`:183-184,210-211`), so the encoder reads *only* the backbone. scArches
`load_query_data` rebuilds the query module and re-derives the backbone from the query's own nan
mask (the derived mask is not in `init_params_`), then transfers reference weights with a positional
resize that only *grows* dimensions (`_archesmixin.py:206-218`, `narrow` asserts non-negative
length). So the query backbone must equal the reference backbone exactly. The override therefore (a)
**raises** if the query is missing any backbone marker, and (b) force-masks every reference
non-backbone marker in the query so it re-derives the reference backbone (warning if the query had
measured any). Requires the reference to have been set up with a `nan_layer` (a genuine
backbone/panel-specific split); else it raises. Tests: `test_cytoanvi_prepare_query_panel_aware`,
`_rejects_missing_backbone`, `_requires_nan_layer`.

## What CytoANVI adds (implemented in Phase 1)

`scvi.external.cytoanvi`:
- `CytoANVAE(SupervisedModuleClass, CytoVAE)` — classifier on z1 + M1+M2 + label-marginalized ELBO
  with CytoVI Normal/Beta reconstruction and `nan_layer` masking; `prior_mixture` forced off.
- `CytoANVI(SemisupervisedTrainingMixin, CYTOVI)` — `setup_anndata(labels_key, unlabeled_category)`
  via `LabelsWithUnlabeledObsField`; `from_cytovi_model(...)`; inherits `train`/`predict`/
  `get_latent_representation` from mixins and `differential_expression`/`impute_*`/
  `differential_abundance`/scArches surgery from CYTOVI.
