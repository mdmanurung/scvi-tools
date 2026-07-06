# CytoANVI Biological Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve CytoANVI's biological-performance robustness without changing default model behavior, then validate whether the opt-in balanced recipe should be recommended for readiness runs.

**Architecture:** Keep the current CytoANVI API and benchmark defaults stable. Add opt-in class weighting in the core classifier loss, expose benchmark recipe flags, and add diagnostics/evaluation modes that distinguish true biological gains from short-run collapse, rare-class failure, or transductive validation.

**Tech Stack:** PyTorch, scvi-tools semi-supervised training plans, AnnData/CytoVI/CytoANVI, pytest, benchmark CLI JSON, optional AnnBatch/mapQC/scHPL extras.

---

## Plan Validation Summary

This plan was validated against the live checkout on 2026-07-02.

- Targeted baseline tests passed:
  - Command: `PYTHONPATH=src:. LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/cytoanvi/test_hce.py::test_hierarchical_cross_entropy_loss_finite tests/cytoanvi/test_cytoanvi.py::test_cytoanvi_y_prior_empirical tests/benchmarks/test_cytoanvi_smoke.py::test_cytoanvi_train_routes_semisupervised_options tests/benchmarks/test_cytoanvi_smoke.py::test_b1_optional_baseline_metric_conversion_errors_do_not_abort tests/benchmarks/test_json_serialization.py -q`
  - Result: `5 passed, 8 warnings in 0.51s`
- Core code check:
  - `src/cytoanvi/_module.py` has unweighted flat CE/HCE in `CytoANVAE.classification_loss`.
  - `src/cytoanvi/_hce.py` already accepts `weight`, so weighted HCE is a small patch.
  - `src/cytoanvi/_model.py` already supports `y_prior="empirical"`, hierarchy persistence, query surgery, replay, and uncertainty.
  - `benchmarks/common/training.py` exposes `n_samples_per_label`, `reduce_lr_on_plateau`, `batch_size`, and AnnBatch, but not `y_prior`, class weighting, or `classification_ratio`.
  - `benchmarks/cytoanvi/tasks.py` B1 lacks prediction-coverage and rare-class diagnostics.
  - B5 is explicitly transductive today; `benchmarks/cytoanvi/metrics.py::precision_at_specificity` can support an inductive calibration mode.
- Result artifact check:
  - Full/vignette e1000 Roider B1 is strong: CytoANVI macro-F1 `0.9079 +/- 0.0084` vs CytoVI kNN `0.7866 +/- 0.0393`.
  - Nuñez inductive B1 is strong but saturated: CytoANVI `0.9751 +/- 0.0003`, CytoVI kNN `0.9581 +/- 0.0007`; delta `+0.0170`, below the older `+0.03` target because the task is near ceiling.
  - Short Roider smoke runs show rare-class/early-training weakness: Roider3 at 2 epochs collapsed; Roider12 at 20 epochs had CytoANVI macro-F1 `0.4314` vs CytoVI kNN `0.4967` despite high accuracy.
  - B2 is the current biological tradeoff: CytoANVI improves bio conservation on Roider e1000 but batch correction is slightly below CytoVI.
  - Full-cohort B5 has a documented stability risk around repeated holdout-sweep model construction and `nan_layer` handling.

## Refined Scope

Do not treat the 2-epoch Roider3 collapse as proof that the core model is broken. The full-run artifacts show B1 can already be strong. The implementation should instead make the strong path more robust and diagnosable:

1. Add opt-in class weighting for rare-label robustness.
2. Add B1 diagnostics so collapse and rare-class failure are visible in JSON.
3. Add benchmark recipe flags that combine existing and new knobs without changing defaults.
4. Add B5 inductive calibration and protect full-cohort holdout sweeps from `nan_layer` regressions.
5. Add real-split hooks for B4/B6 and mapQC status clarity, but do not claim biological case-control performance until real case/control fields are supplied.
6. Keep AnnBatch experimental and performance-oriented; cache reuse is useful but lower priority than biological evaluation validity.

## File Map

- Modify `src/cytoanvi/_module.py`: class-weight storage, validation, and flat/HCE loss usage.
- Modify `src/cytoanvi/_model.py`: public constructor options, class-weight resolution, save/load/query synchronization.
- Modify `src/cytoanvi/_hce.py`: only tests should change if `weight` already works.
- Modify `benchmarks/common/training.py`: pass `y_prior`, class weighting, class-weight clip, and `classification_ratio` to training.
- Modify `benchmarks/cytoanvi/run.py`: CLI flags and recipe resolution.
- Modify `benchmarks/cytoanvi/tasks.py`: B1 diagnostics, B5 inductive mode, real split metadata, recipe pass-through.
- Modify `benchmarks/cytoanvi/metrics.py`: label-count, prediction-coverage, rare-class macro-F1 helper.
- Modify `benchmarks/common/annbatch.py`: optional cache-key reuse only after core performance work.
- Modify tests under `tests/cytoanvi/` and `tests/benchmarks/`.
- Update docs after validation: `docs/user_guide/models/cytoanvi.md` and `benchmarks/cytoanvi/README.md`.

---

### Task 1: Red Tests for Class Weighting

**Files:**
- Modify: `tests/cytoanvi/test_cytoanvi.py`
- Modify: `tests/cytoanvi/test_hce.py`

- [ ] **Step 1: Add class-weight construction tests**

Add tests that use an imbalanced labeled dataset and assert default behavior remains unweighted.

```python
def test_cytoanvi_class_weighting_default_none(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    assert model.class_weighting_ == "none"
    assert model.class_weights_ is None
    assert model.module.class_weights is None
```

```python
def test_cytoanvi_sqrt_inverse_frequency_class_weights(adata):
    labels = adata.obs[LABELS_KEY].astype(str).to_numpy()
    labels[:] = UNLABELED
    labels[:30] = "label_1"
    labels[30:40] = "label_2"
    labels[40:45] = "label_3"
    labels[45:47] = "label_4"
    adata.obs[LABELS_KEY] = labels

    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(
        adata,
        n_latent=10,
        class_weighting="sqrt_inverse_frequency",
        class_weight_clip=10.0,
    )

    weights = model.module.class_weights.detach().cpu().numpy()
    assert weights.shape == (model.n_labels,)
    assert np.isclose(weights.mean(), 1.0, atol=1e-6)
    assert weights[0] < weights[1] < weights[2] < weights[3]
```

- [ ] **Step 2: Add weighted-loss tests**

Add flat CE and HCE tests that compare weighted vs unweighted loss on the same logits/targets.

```python
def test_hierarchical_cross_entropy_accepts_class_weights():
    label_names = ["A", "B", "C", "D", "E"]
    reachability = torch.tensor(
        build_reachability_matrix(label_names, _toy_dag_edges()), dtype=torch.float32
    )
    logits = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))
    unweighted = hierarchical_cross_entropy_loss(logits, targets, reachability)
    weighted = hierarchical_cross_entropy_loss(
        logits, targets, reachability, weight=torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0])
    )
    assert torch.isfinite(weighted)
    assert not torch.allclose(unweighted, weighted)
```

- [ ] **Step 3: Add save/load/query-preservation tests**

Add a test that trains a tiny weighted reference, saves/loads it, and asserts `class_weights_` and `module.class_weights` are preserved. Add a separate query surgery test where the query labels are all unlabeled and the query model still receives the reference weights.

```python
def test_cytoanvi_class_weights_preserved_in_query_surgery(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    ref = CytoANVI(
        adata,
        n_latent=10,
        class_weighting="sqrt_inverse_frequency",
        class_weight_clip=10.0,
    )
    ref_weights = ref.module.class_weights.detach().cpu().clone()

    query = adata.copy()
    query.obs[LABELS_KEY] = UNLABELED
    q = CytoANVI.load_query_data(query, ref)

    torch.testing.assert_close(q.module.class_weights.detach().cpu(), ref_weights)
```

- [ ] **Step 4: Run tests and confirm failure before implementation**

Run:

```bash
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache \
$ENV/bin/python -m pytest \
  tests/cytoanvi/test_cytoanvi.py::test_cytoanvi_class_weighting_default_none \
  tests/cytoanvi/test_cytoanvi.py::test_cytoanvi_sqrt_inverse_frequency_class_weights \
  tests/cytoanvi/test_hce.py::test_hierarchical_cross_entropy_accepts_class_weights \
  -q
```

Expected before implementation: failures on missing `class_weighting_`, `class_weights_`, constructor argument, or weighted-loss plumbing.

---

### Task 2: Core Opt-In Class Weighting

**Files:**
- Modify: `src/cytoanvi/_module.py`
- Modify: `src/cytoanvi/_model.py`

- [ ] **Step 1: Add module class-weight support**

In `CytoANVAE.__init__`, add `class_weights: torch.Tensor | None = None` and register a non-persistent buffer to avoid strict state-dict incompatibility with older saved models.

```python
self.register_buffer("class_weights", None, persistent=False)
self.set_class_weights(class_weights)
```

Add:

```python
def set_class_weights(self, class_weights: torch.Tensor | None) -> None:
    if class_weights is None:
        self.class_weights = None
        return
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
    if weights.ndim != 1 or weights.shape[0] != self.n_labels:
        raise ValueError(
            f"class_weights must have shape ({self.n_labels},); got {tuple(weights.shape)}."
        )
    if not torch.isfinite(weights).all():
        raise ValueError("class_weights must contain only finite values.")
    if (weights <= 0).any():
        raise ValueError("class_weights must be strictly positive.")
    self.class_weights = weights
```

- [ ] **Step 2: Use class weights in flat CE and HCE**

In `classification_loss`:

```python
weight = self.class_weights
if self.reachability_matrix_ is None:
    ce_loss = F.cross_entropy(logits, y_long, weight=weight)
else:
    ce_loss = hierarchical_cross_entropy_loss(
        logits, y_long, self.reachability_matrix_, weight=weight
    )
```

- [ ] **Step 3: Add model constructor options and weight resolution**

In `CytoANVI.__init__`, add:

```python
class_weighting: Literal["none", "inverse_frequency", "sqrt_inverse_frequency"] | torch.Tensor | None = "none",
class_weight_clip: float = 10.0,
```

Implement `_resolve_class_weights`:

```python
def _resolve_class_weights(self, class_weighting, class_weight_clip: float, n_labels: int):
    if class_weighting is None or (isinstance(class_weighting, str) and class_weighting == "none"):
        return None
    if class_weight_clip <= 0 or not np.isfinite(class_weight_clip):
        raise ValueError("class_weight_clip must be finite and > 0.")
    if isinstance(class_weighting, torch.Tensor):
        weights = class_weighting.detach().clone().to(dtype=torch.float32)
    elif class_weighting in {"inverse_frequency", "sqrt_inverse_frequency"}:
        if len(self._labeled_indices) == 0:
            return None
        labeled_vals = self.labels_[self._labeled_indices]
        counts = np.array(
            [(labeled_vals == self._label_mapping[c]).sum() for c in range(n_labels)],
            dtype=np.float64,
        )
        if (counts <= 0).any():
            raise ValueError(
                "Cannot compute class weights because at least one observed label has zero labeled cells."
            )
        inv = counts.sum() / (n_labels * counts)
        weights_np = np.sqrt(inv) if class_weighting == "sqrt_inverse_frequency" else inv
        weights_np = np.minimum(weights_np, class_weight_clip)
        weights_np = weights_np / weights_np.mean()
        weights = torch.tensor(weights_np, dtype=torch.float32)
    else:
        raise ValueError(
            "class_weighting must be 'none', 'inverse_frequency', "
            "'sqrt_inverse_frequency', None, or a tensor."
        )
    if tuple(weights.shape) != (n_labels,):
        raise ValueError(f"class weights must have shape ({n_labels},); got {tuple(weights.shape)}.")
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("class weights must be finite and strictly positive.")
    return weights
```

- [ ] **Step 4: Persist as model attributes, not strict state**

After resolving weights:

```python
class_weights = self._resolve_class_weights(class_weighting, class_weight_clip, n_labels)
self.class_weighting_ = "tensor" if isinstance(class_weighting, torch.Tensor) else (class_weighting or "none")
self.class_weight_clip_ = float(class_weight_clip)
self.class_weights_ = None if class_weights is None else class_weights.detach().cpu().numpy()
```

Pass `class_weights=class_weights` into `self._module_cls(...)`.

Add:

```python
def _sync_class_weights_to_module(self) -> None:
    weights = getattr(self, "class_weights_", None)
    tensor = None if weights is None else torch.as_tensor(weights, dtype=torch.float32)
    self.module.set_class_weights(tensor)
```

Call `_sync_class_weights_to_module()` after load and after query surgery.

- [ ] **Step 5: Preserve weights through normal load and query surgery**

In `CytoANVAE.on_load`, after hierarchy reattachment:

```python
weights = getattr(model, "class_weights_", None)
if hasattr(self, "set_class_weights"):
    self.set_class_weights(None if weights is None else torch.as_tensor(weights, dtype=torch.float32))
```

Override `CytoANVI.load_query_data`:

```python
@classmethod
def load_query_data(cls, *args, **kwargs):
    model = super().load_query_data(*args, **kwargs)
    if hasattr(model, "_sync_class_weights_to_module"):
        model._sync_class_weights_to_module()
    return model
```

- [ ] **Step 6: Run class-weight tests**

Run:

```bash
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache \
$ENV/bin/python -m pytest tests/cytoanvi/test_cytoanvi.py tests/cytoanvi/test_hce.py -q
```

Expected: all tests pass; older saved-model strict-state compatibility is not affected because the buffer is non-persistent.

---

### Task 3: B1 Diagnostics for Rare-Class and Collapse Detection

**Files:**
- Modify: `benchmarks/cytoanvi/metrics.py`
- Modify: `benchmarks/cytoanvi/tasks.py`
- Modify: `tests/benchmarks/test_cytoanvi_smoke.py`

- [ ] **Step 1: Add metric helper tests**

Add tests for a collapsed predictor and a rare-class predictor.

```python
def test_b1_diagnostics_detect_prediction_collapse():
    y_true = np.asarray(["A"] * 8 + ["B"] * 2 + ["C"] * 1)
    y_train = np.asarray(["A"] * 10 + ["B"] * 3 + ["C"] * 1)
    y_pred = np.asarray(["A"] * len(y_true))

    out = metrics.label_transfer_diagnostics(y_train, y_true, y_pred, rare_max_count=2)

    assert out["n_predicted_labels"] == 1
    assert out["predicted_label_coverage"] == 1 / 3
    assert out["majority_prediction_fraction"] == 1.0
    assert out["collapse_warning"] is True
    assert "rare_macro_f1" in out
```

- [ ] **Step 2: Implement diagnostics helper**

Add to `benchmarks/cytoanvi/metrics.py`:

```python
def _counts(values) -> dict[str, int]:
    arr = np.asarray(values).astype(str)
    labels, counts = np.unique(arr, return_counts=True)
    return {str(k): int(v) for k, v in zip(labels, counts, strict=True)}


def label_transfer_diagnostics(y_train, y_true, y_pred, *, rare_max_count: int = 25) -> dict:
    y_train = np.asarray(y_train).astype(str)
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    observed = sorted(set(y_true))
    predicted = sorted(set(y_pred))
    train_counts = _counts(y_train)
    true_counts = _counts(y_true)
    pred_counts = _counts(y_pred)
    rare_labels = [lab for lab, count in true_counts.items() if count <= rare_max_count]
    rare_mask = np.isin(y_true, rare_labels)
    return {
        "train_label_counts": train_counts,
        "heldout_label_counts": true_counts,
        "predicted_label_counts": pred_counts,
        "n_true_labels": len(observed),
        "n_predicted_labels": len(predicted),
        "predicted_label_coverage": float(len(set(predicted) & set(observed)) / max(len(observed), 1)),
        "majority_prediction_fraction": float(max(pred_counts.values()) / max(len(y_pred), 1)) if pred_counts else 0.0,
        "collapse_warning": bool(len(predicted) <= 1 or (pred_counts and max(pred_counts.values()) / max(len(y_pred), 1) >= 0.95)),
        "rare_labels": rare_labels,
        "rare_macro_f1": float(f1_score(y_true[rare_mask], y_pred[rare_mask], average="macro", zero_division=0)) if rare_mask.any() else None,
    }
```

- [ ] **Step 3: Add diagnostics to B1 JSON**

In `task_b1_label_transfer`, compute:

```python
labeled_train_mask = masked != unlabeled_category
diagnostics = metrics.label_transfer_diagnostics(
    masked[labeled_train_mask],
    true[held],
    cytoanvi_pred[held],
)
```

Return:

```python
"cytoanvi_diagnostics": diagnostics,
```

- [ ] **Step 4: Run benchmark tests**

Run:

```bash
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache \
$ENV/bin/python -m pytest tests/benchmarks/test_cytoanvi_smoke.py -q
```

Expected: B1 tests pass and JSON remains strict through `to_jsonable`.

---

### Task 4: Opt-In Balanced Recipe and CLI Pass-Through

**Files:**
- Modify: `benchmarks/common/training.py`
- Modify: `benchmarks/cytoanvi/run.py`
- Modify: `benchmarks/cytoanvi/tasks.py`
- Modify: `benchmarks/cytoanvi/paired_rna_cytof.py`
- Modify: `tests/benchmarks/test_cytoanvi_smoke.py`
- Modify: `tests/benchmarks/test_annbatch_backend.py`

- [ ] **Step 1: Add pass-through tests**

Add a CLI-level test that verifies default runs pass no recipe changes, and balanced runs pass explicit values to every CytoANVI training task.

```python
def test_cli_balanced_recipe_passes_training_config(monkeypatch):
    captured = {}

    def fake_task(name):
        def _run(*args, **kwargs):
            captured[name] = kwargs
            return {"task": name}
        return _run

    monkeypatch.setattr(cyto_run.task_mod, "task_b1_label_transfer", fake_task("b1"))
    monkeypatch.setattr(cyto_run.task_mod, "task_b2_integration", fake_task("b2"))
    monkeypatch.setattr(cyto_run.task_mod, "task_b4_continual", fake_task("b4"))
    monkeypatch.setattr(cyto_run.task_mod, "task_b5_novelty", fake_task("b5"))
    monkeypatch.setattr(cyto_run.task_mod, "task_b8_hce_label_transfer", fake_task("b8"))
    monkeypatch.setattr(cyto_run.task_mod, "task_b9_mapqc", fake_task("b9"))

    args = types.SimpleNamespace(
        dataset="synthetic",
        task="all",
        labels_key="labels",
        unlabeled="Unknown",
        batch_key="batch",
        sample_key=None,
        seed=0,
        max_epochs=1,
        batch_size=64,
        n_samples_per_label=None,
        reduce_lr_on_plateau=False,
        subsample_per_batch=10,
        holdout_sweep=False,
        holdout_type=None,
        b5_mode="transductive",
        ewc_lambdas=None,
        hierarchy_edges=None,
        mapqc_run=False,
        mapqc_n_nhoods=1,
        mapqc_k_min=2,
        mapqc_k_max=5,
        annbatch=False,
        annbatch_cache_dir=None,
        annbatch_chunk_size=8,
        annbatch_preload_nchunks=2,
        cytoanvi_recipe="balanced",
        class_weighting=None,
        class_weight_clip=10.0,
        y_prior=None,
        classification_ratio=None,
        warm_start_cytovi=False,
        case_control_key=None,
        control_values=None,
        case_values=None,
    )

    p1 = ad.AnnData(
        X=np.ones((20, 4), dtype=np.float32),
        obs=pd.DataFrame({"labels": ["A", "B"] * 10, "batch": ["b0"] * 20}),
    )
    cyto_run._run_tasks(args, p1, None, "Unknown", seed=0)

    for kwargs in captured.values():
        assert kwargs["cytoanvi_training_config"]["y_prior"] == "empirical"
        assert kwargs["cytoanvi_training_config"]["class_weighting"] == "sqrt_inverse_frequency"
        assert kwargs["cytoanvi_training_config"]["class_weight_clip"] == 10.0
```

- [ ] **Step 2: Add recipe resolver**

In `benchmarks/cytoanvi/run.py`, add:

```python
def _cytoanvi_training_config_from_args(args) -> dict:
    config = {
        "y_prior": "uniform",
        "class_weighting": "none",
        "class_weight_clip": args.class_weight_clip,
        "classification_ratio": None,
        "warm_start_cytovi": False,
        "reduce_lr_on_plateau": False,
    }
    if args.cytoanvi_recipe == "balanced":
        config.update(
            {
                "y_prior": "empirical",
                "class_weighting": "sqrt_inverse_frequency",
                "reduce_lr_on_plateau": True,
            }
        )
    if args.y_prior is not None:
        config["y_prior"] = args.y_prior
    if args.class_weighting is not None:
        config["class_weighting"] = args.class_weighting
    if args.classification_ratio is not None:
        config["classification_ratio"] = args.classification_ratio
    if args.warm_start_cytovi:
        config["warm_start_cytovi"] = True
    return config
```

Add CLI flags:

```python
ap.add_argument("--cytoanvi-recipe", choices=["default", "balanced"], default="default")
ap.add_argument("--class-weighting", choices=["none", "inverse_frequency", "sqrt_inverse_frequency"], default=None)
ap.add_argument("--class-weight-clip", type=float, default=10.0)
ap.add_argument("--y-prior", choices=["uniform", "empirical"], default=None)
ap.add_argument("--classification-ratio", type=float, default=None)
ap.add_argument("--warm-start-cytovi", action="store_true")
```

- [ ] **Step 3: Thread config through training helpers**

In `train_cytoanvi`, add keyword args:

```python
y_prior="uniform",
class_weighting="none",
class_weight_clip=10.0,
classification_ratio=None,
```

Construct the model:

```python
model = CytoANVI(
    a,
    n_latent=n_latent,
    y_prior=y_prior,
    class_weighting=class_weighting,
    class_weight_clip=class_weight_clip,
)
```

Add to `plan_kw` only when provided:

```python
if classification_ratio is not None:
    plan_kw["classification_ratio"] = classification_ratio
```

- [ ] **Step 4: Route config into B1/B2/B4/B5/B7/B8/B9**

Add `cytoanvi_training_config=None` to task signatures and pass:

```python
training_config = cytoanvi_training_config or {}
...
model, a = train_cytoanvi(..., **training_config)
```

Keep `n_samples_per_label` as an explicit task kwarg because it is splitter behavior. Route `reduce_lr_on_plateau` from either the existing CLI flag or the resolved recipe config into `train_cytoanvi`.

- [ ] **Step 5: Defer warm-start reuse until tests pass**

Implement `--warm-start-cytovi` only for B1 after the recipe pass-through is stable:

1. Train CytoVI once for the baseline.
2. Initialize CytoANVI via `CytoANVI.from_cytovi_model(cytovi_model, unlabeled_category=..., labels_key=...)`.
3. Train the CytoANVI model.
4. Reuse the same CytoVI latent for the B1 kNN baseline.

Do not enable warm start in `balanced` by default until a Roider/Nuñez validation run shows it improves macro-F1 or convergence without worsening B2.

---

### Task 5: B5 Inductive Novelty Evaluation and Holdout Stability

**Files:**
- Modify: `benchmarks/cytoanvi/tasks.py`
- Modify: `benchmarks/cytoanvi/run.py`
- Modify: `tests/benchmarks/test_cytoanvi_smoke.py`

- [ ] **Step 1: Add B5 mode CLI and tests**

Add:

```python
ap.add_argument(
    "--b5-mode",
    choices=["transductive", "inductive"],
    default="transductive",
    help="B5 novelty evaluation mode; transductive preserves legacy JSON behavior.",
)
```

Test that `_run_tasks` passes `b5_mode` into `task_b5_novelty` and `task_b5_holdout_sweep`.

- [ ] **Step 2: Implement inductive calibration**

In `task_b5_novelty`, add `b5_mode="transductive"` and `specificity=0.95`.

For inductive mode:

```python
seen_idx = np.flatnonzero(~is_novel)
rng = np.random.default_rng(seed)
train_seen_parts = []
calib_seen_parts = []
for lab in sorted(set(labels[seen_idx])):
    lab_idx = seen_idx[labels[seen_idx] == lab]
    rng.shuffle(lab_idx)
    if len(lab_idx) < 5:
        raise ValueError(f"B5 inductive mode needs at least 5 cells for seen label {lab!r}.")
    cut = max(1, int(0.8 * len(lab_idx)))
    cut = min(cut, len(lab_idx) - 1)
    train_seen_parts.append(lab_idx[:cut])
    calib_seen_parts.append(lab_idx[cut:])
train_seen = np.concatenate(train_seen_parts)
calib_seen = np.concatenate(calib_seen_parts)
eval_idx = np.concatenate([calib_seen, np.flatnonzero(is_novel)])

work = adata[train_seen].copy()
eval_adata = adata[eval_idx].copy()
eval_labels = labels[eval_idx]
eval_adata.obs[labels_key] = unlabeled_category
eval_is_novel = eval_labels == holdout_type

masked = np.asarray(work.obs[labels_key].astype(str))
work.obs[labels_key] = masked

model, a = train_cytoanvi(...)
eval_unc = model.get_uncertainty(eval_adata, mode="latent")
calib_unc = eval_unc[~eval_is_novel]
```

Return both AUROC and thresholded metrics:

```python
"b5_evaluation_mode": "inductive_calibrated",
"latent": {
    **metrics.novelty_auroc(eval_unc, eval_is_novel),
    **metrics.precision_at_specificity(eval_unc, eval_is_novel, specificity=specificity, uncertainty_ref=calib_unc),
},
```

- [ ] **Step 3: Add a regression test for explicit `nan_layer` in holdout sweeps**

Test that `task_b5_holdout_sweep(..., nan_layer=NAN_LAYER)` passes the same `nan_layer` to every per-type `task_b5_novelty` call. This protects the full-cohort sequential sweep from reverting to fragile auto-detection.

---

### Task 6: Real Case-Control Hooks for B4/B6 and B9 Status Clarity

**Files:**
- Modify: `benchmarks/cytoanvi/run.py`
- Modify: `benchmarks/cytoanvi/tasks.py`
- Modify: `tests/benchmarks/test_cytoanvi_smoke.py`

- [ ] **Step 1: Add split flags**

Add CLI flags:

```python
ap.add_argument("--case-control-key", default=None)
ap.add_argument("--control-values", default=None, help="Comma-separated values in --case-control-key")
ap.add_argument("--case-values", default=None, help="Comma-separated values in --case-control-key")
```

Parse comma lists into lists of strings.

- [ ] **Step 2: Add real split helper**

In `tasks.py`, add:

```python
def _split_reference_query_by_case_control(
    adata,
    *,
    case_control_key: str,
    control_values: list[str],
    case_values: list[str],
):
    status = adata.obs[case_control_key].astype(str)
    is_control = status.isin(control_values).to_numpy()
    is_case = status.isin(case_values).to_numpy()
    if is_control.sum() < 64 or is_case.sum() < 64:
        raise ValueError(
            f"case/control split too small: controls={int(is_control.sum())}, cases={int(is_case.sum())}."
        )
    return adata[is_control].copy(), adata[is_case].copy(), False
```

Use this path in `_b4_setup` when all three real split fields are supplied. Otherwise keep the existing pseudo split and preserve the `"Pseudo case/control via batch split"` note.

- [ ] **Step 3: Reflect evaluation mode in JSON**

B4/B6 output should include:

```python
"case_control_mode": "real"  # or "pseudo_batch"
"case_control_key": case_control_key
```

B9 should keep `"status": "plumbing_only"` on synthetic and `"status": "blocked"` when mapQC is unavailable.

---

### Task 7: AnnBatch Cache Reuse Ergonomics

**Files:**
- Modify: `benchmarks/common/annbatch.py`
- Modify: `benchmarks/cytoanvi/run.py`
- Modify: `tests/benchmarks/test_annbatch_backend.py`

- [ ] **Step 1: Add config fields**

Extend `AnnBatchConfig`:

```python
cache_mode: Literal["temporary", "reuse"] = "temporary"
cache_key: str | None = None
```

Add CLI:

```python
ap.add_argument("--annbatch-cache-mode", choices=["temporary", "reuse"], default="temporary")
ap.add_argument("--annbatch-cache-key", default=None)
```

- [ ] **Step 2: Implement explicit reuse only**

In `AnnBatchSemiSupervisedDataModule.__init__`:

```python
if config.cache_mode == "reuse":
    if not config.cache_key:
        raise ValueError("--annbatch-cache-key is required when --annbatch-cache-mode=reuse")
    safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", config.cache_key)
    self._cache_root = cache_root / safe_key
else:
    self._cache_root = cache_root / f"run-{uuid.uuid4().hex}"
```

Do not implement implicit fingerprint reuse in this patch.

- [ ] **Step 3: Report cache mode in benchmark JSON**

Add an `annbatch` metadata block to the top-level payload when AnnBatch is enabled:

```python
"annbatch": {
    "enabled": True,
    "cache_dir": str(config.cache_dir),
    "cache_mode": config.cache_mode,
    "cache_key": config.cache_key,
}
```

---

## Validation Commands

Use the existing environment:

```bash
ENV=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test
export PYTHONPATH=src:.
export LD_LIBRARY_PATH=$ENV/lib
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp/numba-cache
OUT=.scratch/cytoanvi-benchmark/results/bioperf_20260702
mkdir -p "$OUT"
```

Run unit and benchmark tests:

```bash
$ENV/bin/python -m pytest tests/cytoanvi -q
$ENV/bin/python -m pytest \
  tests/benchmarks/test_cytoanvi_smoke.py \
  tests/benchmarks/test_common_scib.py \
  tests/benchmarks/test_json_serialization.py \
  tests/benchmarks/test_annbatch_backend.py \
  -q
```

Run default synthetic smoke:

```bash
$ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task all --max-epochs 2 \
  --subsample-per-batch 200 --batch-size 128 --seed 0 \
  --out "$OUT/synthetic_all_default_s0.json"
```

Run balanced synthetic smoke:

```bash
$ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task b1 --max-epochs 2 \
  --batch-size 128 --seed 0 --cytoanvi-recipe balanced \
  --out "$OUT/synthetic_b1_balanced_s0.json"
```

Run short Roider readiness comparison when the local full cache is available:

```bash
$ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset roider-full --roider-max-patients 12 \
  --task b1 --labels-key cell_type --batch-key batch --sample-key PatientID \
  --max-epochs 20 --batch-size 8192 --seed 0 \
  --out "$OUT/roider12_b1_default_s0.json"
```

```bash
$ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset roider-full --roider-max-patients 12 \
  --task b1 --labels-key cell_type --batch-key batch --sample-key PatientID \
  --max-epochs 20 --batch-size 8192 --seed 0 --cytoanvi-recipe balanced \
  --out "$OUT/roider12_b1_balanced_s0.json"
```

Run B5 inductive smoke:

```bash
$ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task b5 --max-epochs 2 \
  --batch-size 128 --seed 0 --b5-mode inductive \
  --out "$OUT/synthetic_b5_inductive_s0.json"
```

## Acceptance Gates

- Default behavior is unchanged:
  - Existing CytoANVI tests pass.
  - Default benchmark JSON schema remains compatible.
  - `class_weighting="none"` yields no class-weight tensor and unchanged loss path.
- Opt-in class weighting works:
  - Computed weights are finite, positive, mean-normalized, clipped, saved as model attributes, and reattached after `load()` and `load_query_data()`.
  - Flat CE and HCE both use weights.
- B1 is diagnosable:
  - JSON reports train/held/predicted counts, prediction coverage, majority prediction fraction, rare labels, rare macro-F1, and a collapse warning.
- Balanced recipe is not automatically declared superior:
  - It may be documented as experimental until Roider/Nuñez validation shows improved rare-class macro-F1 or equal/better macro-F1 without unacceptable B2 batch loss.
  - If balanced lowers global macro-F1 while improving rare-class F1, report the tradeoff explicitly.
- B5 biological claims require inductive mode:
  - Transductive B5 remains for backward compatibility and is labeled as such.
  - Inductive B5 reports threshold, specificity, precision, recall, AUROC, and evaluation mode.
- B4/B6 and B9 remain clearly labeled:
  - Pseudo batch split means plumbing-only, not biological case-control evidence.
  - mapQC unavailable produces a blocked JSON result rather than aborting unrelated tasks.

## Suggested Execution Order

1. Implement Tasks 1-3 first; they are narrow and directly address rare-class robustness.
2. Implement Task 4 recipe pass-through without warm-start reuse; validate default and balanced smokes.
3. Implement B5 inductive mode and `nan_layer` regression test.
4. Run Roider12 default vs balanced and inspect diagnostics.
5. Only then decide whether to implement warm-start reuse and AnnBatch cache reuse.
6. Update docs with measured recommendations, not aspirational defaults.
