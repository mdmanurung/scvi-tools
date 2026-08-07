# External tutorial repair handoff

Status: `blocked_external_submodule`

The root repository records `docs/tutorials/notebooks` as gitlink
`e36f55865a4d5f197045a62a6edd227e67ade843`, and the submodule is uninitialized (`git submodule
status` begins with `-`). This executor is not authorized to initialize, fetch, commit, or push that
external repository. Consequently no file below the gitlink can be part of the root `cytoanvi
0.2.0` source commit or wheel lineage.

The executable engineering fixture was moved to the root-owned tracked path
`vignettes/cytoanvi_treearches_synthetic.py`; root tests and installed-wheel acceptance use that
copy. The following upstream changes remain for an authorized submodule maintainer.

## Preservation audit

| External path | Pre-edit SHA-256 | Temporary task-edit SHA-256 | Restored SHA-256 |
| --- | --- | --- | --- |
| `cytometry/CytoANVI_treeArches_tutorial.md` | `8ea180e2d944d0059fc6a126546e9f8e4533d32bcec229b3dffd03f9b36eeadc` | `3bdd54dddd08fb75ec9a5c34ccb8ac133435feea92a75cd9139cdd0405801dde` | `8ea180e2d944d0059fc6a126546e9f8e4533d32bcec229b3dffd03f9b36eeadc` |
| `cytometry/cytoanvi_example_reference_query.py` | `242d08751f429c80cfc887c4d965763c5a08a9002e8c20dbc1a4f48eac0501af` | `051e71c824196a2419a3878f2189fc2ef722e3043d62024ae798c7e54fbdd405` | `242d08751f429c80cfc887c4d965763c5a08a9002e8c20dbc1a4f48eac0501af` |

The task-created external `cytometry/cytoanvi_treearches_synthetic.py` and its 2026-08-07
CPython-3.13 cache were removed after the tracked copy was created (tracked copy SHA-256
`4139c82de572721329cc139d7af11f2930e59136048f661ecf56836695602492`). The pre-existing
`cytoanvi_example_reference_query.cpython-313.pyc` dated 2026-06-28 was preserved.

## `cytometry/CytoANVI_treeArches_tutorial.md`

1. Replace `pip install scvi-tools[cytoanvi-hierarchy]` with
   `pip install "cytoanvi[cytoanvi-hierarchy]"` and link the root usage-readiness authority.
2. For direct same-panel surgery, assert identical marker order and call `load_query_data` directly.
   Call `prepare_query_anndata` only for a genuinely panel-divergent query whose reference has an
   explicit backbone/panel-specific `nan_layer` contract.
3. Import `anndata`. In the labeled-query update, create `query_update_latent`, derive
   `celltype_batch`, then use
   `anndata.concat([ref_latent, query_update_latent], join="inner", index_unique="-")`; do not pass
   query-only latent data to `update_hierarchy`.
4. Before the one-shot learn/update calls, derive `celltype_batch` on both source AnnData objects.
   Pass `obs_cols=[BATCH_KEY, "celltype_batch"]` to learn and pass the explicit reference-plus-query
   `combined_latent` to update.
5. Point readers to the root-owned `vignettes/cytoanvi_treearches_synthetic.py` engineering fixture
   and state that it is not scientific validation.

## `cytometry/cytoanvi_example_reference_query.py`

1. State that the example requires the `cytoanvi 0.2.0` distribution and link
   `docs/usage_readiness.md`.
2. Remove stable `get_uncertainty()` and `select_replay_by_uncertainty()` calls.
3. Rename `uncertainty_and_continual` to `continual_update`; select replay by a rule declared before
   query construction (the root-owned example uses `ref[: max(1, ref.n_obs // 5)].copy()`), retain
   the external matched control, and label continual use as a scientific no-go pending P2.

No external submodule edit is counted as complete until an authorized maintainer applies and tests
these changes in that repository and advances the root gitlink through its own review process.
