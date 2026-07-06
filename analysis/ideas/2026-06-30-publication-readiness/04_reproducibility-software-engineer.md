# Reproducibility / Software Engineer — Publication Readiness Ideas (Refined)

**Persona**: Scientific software engineer and reproducibility advocate
**Session**: 2026-06-30 publication readiness ideation (refined 2026-06-30)
**Focus**: Software / reproducibility gaps preventing external verification

---

## Idea 10 — Archive `nunez_annotated.h5ad` to Figshare and wire auto-download in `load_nunez()`

**Reproducibility gap**

Reproducing B1, B2, B5, and B8 Nuñez results requires first running `benchmarks/cytoanvi/annotate_nunez.py`, a stochastic CytoVI + Leiden script with no archived checkpoint and no Figshare entry. The file is absent from the `FIGSHARE` dict in `benchmarks/cytoanvi/data.py`, so `load_nunez()` never auto-downloads it — it silently falls back to on-the-fly Leiden labeling, which produces different cell-type labels across GPU and PyTorch versions. The `require_annotated_nunez=True` guard raises `FileNotFoundError` with no path forward for external reproducers.

**What to do**

*File: `benchmarks/cytoanvi/data.py`*

1. `FIGSHARE` dict (existing, lines 23–28): add one entry `"nunez_annotated.h5ad": "<figshare_file_id>"` where the ID is a string matching the format of existing entries.

2. `load_nunez()` (lines 193–208): after the `h5ad = _resolve_file(data_dir, annotated_h5ad)` check, when the file is absent and `auto_download=True`, call `download(annotated_h5ad, data_dir)` before falling through to the FCS path.

3. Add `verify_nunez_checksum(path: str, *, expected_md5: str) -> None` that computes MD5 and raises `RuntimeError` if mismatched. Call it inside `load_nunez()` immediately after auto-download and immediately after loading an already-present file.

*File: `benchmarks/cytoanvi/data_checksums.json`* (new)

4. Create this file with one dict entry: `{"nunez_annotated.h5ad": "<md5hex>"}`. Read by `verify_nunez_checksum()` relative to `__file__` so it works regardless of CWD.

*File: `benchmarks/cytoanvi/run.py`*

5. Delete the `FileNotFoundError` raise block (lines 50–58 in `_require_annotated_nunez` guard). Replace with a comment explaining that `load_nunez()` now auto-downloads via the FIGSHARE entry. The `require_annotated_nunez` flag becomes a no-op and should be deprecated in the same PR.

**Implementation steps**

1. Run `PYTHONPATH=src:. python -m benchmarks.cytoanvi.annotate_nunez --data-dir data --out data/nunez_annotated.h5ad --max-epochs 100 --seed 0` to produce the canonical file. Record `md5sum data/nunez_annotated.h5ad`.
2. Upload `data/nunez_annotated.h5ad` to the project's Figshare item. Record the numeric file ID from the download URL (`https://figshare.com/ndownloader/files/<id>`).
3. Add `"nunez_annotated.h5ad": "<id>"` to the `FIGSHARE` dict. Add the MD5 hex to `benchmarks/cytoanvi/data_checksums.json`.
4. Add `verify_nunez_checksum()` to `data.py`; load `data_checksums.json` relative to that file's own `__file__`.
5. In `load_nunez()`: add `if not (os.path.exists(h5ad) and os.path.getsize(h5ad) > 0) and auto_download and annotated_h5ad in FIGSHARE: download(annotated_h5ad, data_dir); verify_nunez_checksum(h5ad, expected_md5=_CHECKSUMS["nunez_annotated.h5ad"])`.
6. Delete the `FileNotFoundError` block in `run.py`.
7. Commit. Add checksum guard to `annotate_nunez.py` docstring naming the Figshare DOI.

**Done criteria**

1. `python -c "from benchmarks.cytoanvi.data import load_nunez; import os; os.remove('data/nunez_annotated.h5ad'); ad = load_nunez('data'); assert ad.obs['cell_type'].nunique() == 11"` succeeds after removing the local copy (proving the download fires).
2. Loading a corrupted file raises `RuntimeError` (checksum guard blocks silently corrupt files).

**Dependencies**: None. Enables external reproducers for B1/B2/B5/B8/B9 without running `annotate_nunez.py`. Does not depend on Ideas 11 or 12.

**Effort**: Low (~3–4h, mostly the Figshare upload and checksum wiring) | **Priority**: Critical

---

## Idea 11 — Pin the execution environment with a lock file and Singularity container

**Reproducibility gap**

Every SLURM script in `.scratch/cytoanvi-benchmark/slurm/` sources `_env.sh`, which sets `PY=/exports/.../conda/envs/scvi-test/bin/python` — a path-specific conda environment with no lock file. A future user or reviewer cannot reconstruct the exact environment from `pyproject.toml` alone because pip solver output is non-deterministic across PyPI mirror states and platform library versions. Even a minor PyTorch or Lightning version change can shift macro-F1 by 1–3 points — within the range of the claimed effect sizes.

**What to do**

*File: `benchmarks/cytoanvi/environment-lock.yml`* (new)

1. Output of `conda env export --no-builds -n scvi-test` from the active env. Captures all conda-managed packages without build strings (for cross-platform readability).

*File: `benchmarks/cytoanvi/requirements-freeze.txt`* (new)

2. Output of `$ENV/bin/pip freeze` from the same env. Captures pip-installed packages (including the editable `scvi-tools` entry, which `conda env export` omits).

*File: `benchmarks/cytoanvi/Singularity.def`* (new)

3. A minimal definition that bootstraps a Miniconda layer, runs `conda env create -f environment-lock.yml -n bench`, and `pip install -r requirements-freeze.txt`. Sets `%environment` to activate the env.

*File: `.scratch/cytoanvi-benchmark/slurm/_env.sh`*

4. Add two variables: `SIF=$ROOT/cytoanvi-bench.sif` and `USE_SIF=${USE_SIF:-0}`. Add a conditional block:
   ```bash
   if [[ $USE_SIF -eq 1 ]]; then PY="singularity exec --nv $SIF python"; fi
   ```
   The existing `PY=$ENV/bin/python` line is kept as the default so existing jobs are unaffected.

**Implementation steps**

1. From the active `scvi-test` env: `conda env export --no-builds -n scvi-test > benchmarks/cytoanvi/environment-lock.yml && /exports/.../scvi-test/bin/pip freeze > benchmarks/cytoanvi/requirements-freeze.txt`.
2. Inspect `requirements-freeze.txt`; replace the `-e git+...` line for scvi-tools with the pinned release or leave it as a comment — the benchmark harness sets `PYTHONPATH` from `_env.sh` so the editable install is not needed inside the container.
3. Write `benchmarks/cytoanvi/Singularity.def`; submit a short test build job or build interactively on the login node if privileged mode is available.
4. Copy the resulting `.sif` to `$ROOT/cytoanvi-bench.sif`. Add `cytoanvi-bench.sif` to `.gitignore` (file is too large for git).
5. Patch `_env.sh` with the `SIF` and `USE_SIF` variables.
6. Run `USE_SIF=1 bash .scratch/cytoanvi-benchmark/slurm/phase0_pytest.slurm` as a smoke check.

**Done criteria**

1. `singularity exec --nv cytoanvi-bench.sif python -c "import torch, cytoanvi; print(torch.__version__)"` prints the same PyTorch version string as `$ENV/bin/python -c "import torch; print(torch.__version__)"`.
2. `conda env create --file benchmarks/cytoanvi/environment-lock.yml --name scvi-repro --dry-run` exits 0 (conda can resolve the pinned packages without errors).

**Dependencies**: None. Mutually independent of Ideas 10 and 12.

**Effort**: Medium (~4–6h; Singularity build iteration on HPC is the slow step) | **Priority**: Critical

---

## Idea 12 — Register `slow` pytest marker and add real-mapqc optional integration test

**Reproducibility gap**

Two distinct gaps:

**Gap 1**: `@pytest.mark.slow` is applied in `tests/benchmarks/test_cytoanvi_smoke.py` (line 20) but the marker is not listed in `[tool.pytest.ini_options].markers` in `pyproject.toml` (lines 143–153). This produces a `PytestUnknownMarkWarning` on every CI run and means `pytest -m slow` silently selects zero tests.

**Gap 2**: The optional CI job (`.github/workflows/test_linux_optional.yml`) installs `scvi-tools[tests,cytoanvi-hierarchy,cytoanvi-mapping-qc]` and runs `pytest --optional`. It collects exactly **one** `@pytest.mark.optional` test across all cytoanvi test files: `test_learn_hierarchy_on_synthetic_latent` in `tests/cytoanvi/test_hierarchy.py`. The `mapqc` package has zero integration coverage — every function in `tests/cytoanvi/test_mapping_qc_mock.py` monkeypatches `_require_mapqc` or `run_mapqc_on_joint` away. The optional job installs the real `mapqc` package and immediately discards it.

Note: the mock tests (which test validation logic, error handling, and delegation chain) correctly run in main CI without the optional extras — no changes are needed for them. `set_hierarchy_from_schpl()` does not call `_require_schpl()`, so `test_set_hierarchy_from_schpl_allows_internal_nodes` stays unmarked. The pipeline mock test patches `_require_schpl` to a noop, so it also stays unmarked.

**What to do**

*File: `pyproject.toml`*

1. Find `[tool.pytest.ini_options].markers` list (around line 143). Append:
   ```
   "slow: mark slow benchmark smoke tests that require a GPU and full max_epochs runs"
   ```

*File: `tests/cytoanvi/test_mapping_qc_mock.py`*

2. Add one new function at the bottom — a real-mapqc integration test that exercises `run_mapqc_on_cytoanvi` without monkeypatching:
   ```python
   @pytest.mark.optional
   def test_run_mapqc_on_cytoanvi_real():
       """Integration test: real mapqc package, no monkeypatching."""
       pytest.importorskip("mapqc")
       adata = _make_adata(n_batches=2, n_labels=5)
       work, is_ref = _assign_mapqc_samples(adata)
       ref = work[is_ref].copy()
       query = work[~is_ref].copy()
       model = _setup_and_train(work.copy())

       joint = mapping_qc.run_mapqc_on_cytoanvi(
           model, ref, query,
           sample_key="mapqc_sample",
           n_nhoods=3,
           k_min=5,
           k_max=15,
       )

       query_mask = joint.obs[mapping_qc.DEFAULT_REF_Q_KEY] == mapping_qc.QUERY_CAT
       assert "mapqc_score" in joint.obs.columns
       assert joint.obs.loc[query_mask, "mapqc_score"].notna().any()
       assert "mapqc_params" in joint.uns
   ```
   This reuses the private helpers `_make_adata`, `_assign_mapqc_samples`, and `_setup_and_train` already defined in that file.

**Implementation steps**

1. Edit `pyproject.toml`; find the `markers = [` block; append the `slow` marker entry inside the list.
2. Add the `test_run_mapqc_on_cytoanvi_real` function to the bottom of `tests/cytoanvi/test_mapping_qc_mock.py`.
3. Verify locally (no extras): `pytest tests/cytoanvi/test_mapping_qc_mock.py -W error::pytest.PytestUnknownMarkWarning --collect-only` — the new test must appear as collected.
4. Verify locally (with extras, `--optional`): `pytest tests/cytoanvi/ -v --optional` — both `test_run_mapqc_on_cytoanvi_real` and `test_learn_hierarchy_on_synthetic_latent` must appear in the pass list.
5. Commit both changes in one PR; CI validates in `test_linux_optional.yml` on next trigger.

**Done criteria**

1. `pytest tests/ --collect-only -W error::pytest.PytestUnknownMarkWarning` exits 0 with no warnings (proves `slow` is registered).
2. `pytest tests/cytoanvi/ -v --optional` reports ≥2 passed (both hierarchy and mapqc real-integration tests execute, confirming the optional CI job exercises actual package code rather than mocks).

**Dependencies**: None. Mutually independent of Ideas 10 and 11.

**Effort**: Low (~1–2h; two small edits plus verifying the mapqc test passes in the optional CI environment) | **Priority**: High
