# Frozen execution plan: CytoANVI and MrTotalVI usage-readiness remediation

Frozen: 2026-08-07

Source plan: `docs/plans/2026-08-07-cytoanvi-mrtotalvi-usage-readiness-remediation.md`

Source plan SHA-256: `095bef05d3ff6f2c2a080482de89903190de5ca1d5873304b2d73ae7ef1fae8e`

Source review: `.living/outputs/reviews/2026-08-07-cytoanvi-mrtotalvi-usage-readiness.md`

Source review SHA-256: `6709acafbc79152e891db659f543fd429147c110e68e3ab39bdee4be10b3f9db`

Reviewed baseline commit: `297769d3c62b9228244a05469dc8349a55e4174c`

## Objective

Complete every locally authorized engineering and governance prerequisite for defensible expert
research use of CytoANVI and MrTotalVI. Produce fail-closed source contracts, consistent primary
guidance, a uniquely versioned candidate artifact and installed-artifact acceptance receipt when a
clean dependency authority is available, machine-readable scientific protocols, and a mandatory
capability matrix. Do not claim P2 scientific acceptance or P3 promotion without the separately
required scheduler authorization, terminal evidence, independent review, and named human approval.

Full end-state completion has five distinct states and they must never be conflated:

1. source changes present;
2. source engineering tests terminal;
3. exact installed artifact engineering-accepted;
4. capability-specific scientific evidence terminal;
5. capability row independently signed and promoted.

## Baseline and ownership boundary

- The worktree is intentionally dirty and user-owned. At freeze time only `.living/*` tracked files
  were modified; the source, tests, tracked docs, `pyproject.toml`, README, changelog, and workflows
  had no tracked diff. There are many unrelated untracked scientific artifacts.
- Never reset, clean, stash, discard, rewrite, or stage unrelated changes.
- Stop if a target source/test/doc file acquires an overlapping change not made by this execution.
- Preserve `dist/cytoanvi-0.1.0-py3-none-any.whl` byte-for-byte at its current path because another
  governed workflow records SHA-256
  `340dfbd2d571e44cf5e8b6d1bc8a62798ce9753abc5df099654a026095f19c8d` as an engineering
  baseline. Quarantine it logically with a tracked no-go record and primary-doc warnings; do not
  move, delete, overwrite, or rebuild it.
- MrMultiVI scientific inspection, remediation, recommendation, and promotion remain out of scope.
  Engineering-only MrMultiVI compatibility tests are allowed only when a shared helper cannot be
  changed safely without them. Do not change MrMultiVI behavior or source.

## Frozen compatibility decisions

### Distribution and artifact

- The next candidate version is `0.2.0`. The prior/schema, TTA, DE, continual, and streaming changes
  are intentionally breaking and do not qualify for a patch bump.
- Build no `0.2.0` candidate until all P1 source work and source-level validation are complete and
  committed locally. Build exactly once from a clean archive/worktree of that recorded commit.
- No materially different rebuild may reuse `0.2.0`. A failed or superseded candidate requires a
  new version and a new plan decision.
- `dist/` remains an ignored delivery directory. Tracked authority lives under `docs/artifacts/`:
  manifest schema, candidate manifest, complete wheel inventory, dependency-authority hash, and
  installed-acceptance receipt.
- The acceptance harness must run outside the checkout with `PYTHONPATH` unset, no editable install,
  and no installed `scvi-tools` distribution. It must prove that both `scvi` and `cytoanvi` load
  from the candidate wheel and that distribution/file inventory matches the recorded commit.
- Fix `docs/conf.py` so normal installed-doc validation uses `cytoanvi` metadata and does not
  silently shadow an installed wheel with `src`.

### CytoANVI

- `train(adversarial_classifier=<non-null>)` raises before trainer construction. `None` remains the
  only accepted compatibility value until a real adversarial objective exists.
- Continual state without replay raises before training. Do not add a new EWC-only public mode in
  this packet. A reloaded model must reconstruct replay through the existing explicit query/replay
  construction path before it can train.
- The existing TTA estimator is not a supported novelty API. The stable legacy method and indirect
  replay selector fail closed and point to a clearly named experimental TTA surface. If the
  experimental implementation is retained, it must use independent per-cell masks, expose a
  deterministic seed, be exactly invariant to batch/chunk partitioning for a fixed seed and cell
  order, and reject empty or non-finite calibration arrays. Remove the threshold helper from the
  stable top-level export.
- Empirical `y_prior` and data-derived class weights are resolved from the actual training indices
  after the split is fixed and before the first optimization step. Perturbing held-out labels must
  not change them. Persist the resolved values and split boundary. Do not label an early-stopping
  validation set independent if it contributed to these quantities.
- Pin mapQC to the exact tested compatibility version `0.1.1` while the private patch exists and
  fail clearly outside that version. A future range/upstream fix is a separate decision.
- `control_adata` is required for the implemented continual objective in every signature, example,
  tutorial, and error message.

### MrTotalVI prior and supervision migration

The new resolved prior enum is exactly `standard`, `mog`, or `vamp`, with new-call default `mog`.
The deprecated `u_prior_mixture: bool | None` exists only as a migration input:

| `u_prior` | legacy flag | result |
| --- | --- | --- |
| `standard` | `None` | `standard` |
| `mog` | `None` | `mog` |
| `vamp` | `None` | `vamp` |
| `standard` | `False` | `standard` plus deprecation warning |
| `mog` | `True` | `mog` plus deprecation warning |
| `vamp` | `True` | `vamp` plus deprecation warning |

Unknown enums and every other enum/legacy-flag combination raise before module construction.
Historical checkpoints with a contradictory combination are explicitly unsupported rather than
silently reinterpreted. Consistent historical checkpoints must load, preserve their objective,
save, reload, and resave with explicit resolved metadata.

Label supervision is separately named `u_prior_supervision` with resolved values `none` or
`labels`. New calls default to `none` and `u_prior_label_weight=0.0`; merely registering
`labels_key` never changes the objective. Explicit `labels` requires registered labels and a finite
positive weight. For old checkpoints only, missing supervision metadata plus a positive saved
weight resolves to `labels` with a deprecation warning; resave writes the explicit resolved value.
Explicit `none` with nonzero weight, explicit `labels` with zero/non-finite weight, or supervision
without labels raises. Model summaries, `init_params_`, saves, loads, and acceptance manifests must
record hierarchy mode, encoder mode, resolved prior, supervision mode, and weight.

Vamp data initialization must use training indices only. Store the initialization seed and the
ordered training-index digest; held-out-data perturbation must not affect centroids. Do not use the
internal validation loss for selection if the initialization boundary cannot be proven.

### MrTotalVI fail-closed data and statistics contracts

- Before mutating `adata`, `setup_anndata` deterministically validates every RNA and protein value
  as finite, non-negative, and integer-like. Support dense arrays, DataFrames, SciPy sparse data,
  and backed/chunked inputs without relying on the existing sampled warning helper. A hidden bad
  value outside early rows must fail.
- Use one pre-inference sample validator for DA and any retained internal legacy-DE implementation:
  sample key and requested metadata columns must exist; sample/donor/covariate values must be
  non-null; every selected sample maps to exactly one donor and covariate value; subsets must be
  non-empty, unique, known, and returned in declared order.
- `differential_expression()` fails closed for both legacy and centered-v2 models and directs
  biological inference to donor-pseudobulk PyDESeq2/edgeR/dreamlet. Keep historical internals only
  as private reproducibility code; do not return public p-values or decoded LFC through an escape
  flag. `differential_abundance()` remains explicitly descriptive and non-inferential.
- `use_vmap=True` raises before statistics or inference; `False` preserves the loop path.
- Multi-file protein axes require non-empty, unique, authoritative names in exact identical order.
  Equal width is insufficient. DataFrame rows use `.iloc`; sparse and ndarray paths have parity.
- Remove `MrTotalVIBatchDataModule` from the stable public export. Retain/rename it only as a private
  registry adapter whose name and docs cannot imply end-to-end training. Its data-contract fixes
  remain tested.
- Preserve centered-v2 registered-sample refusal boundaries and its existing engineering semantics.
- Amend/supersede ADR-0005/ADR-0007 only where these new fail-closed contracts change accepted
  behavior. ADR-0008/ADR-0009 scientific selection rules remain intact.

## Execution sequence

### 1. Record baseline and create governance contracts

- Capture baseline commit, tracked diff, target-file hashes, environment identities, and the stale
  wheel hash without altering existing evidence.
- Add ADR-0010 for this packet, a `0.2.0` migration table/changelog entry, the artifact manifest
  schema, installed-acceptance receipt schema, and strict capability-decision schema.
- Define all 19 mandatory capability IDs. CytoANVI: core train/predict/latent/save-load, same-panel
  mapping, panel-divergent mapping, hierarchy, integration/clustering, TTA OOD, continual, mapQC,
  artifact. MrTotalVI: in-memory core, `z`/`u` embeddings, prior choice, label supervision, DA,
  legacy DE, centered-v2, streaming, new-sample inference, artifact. No-go/experimental rows are
  mandatory and may not be omitted.
- Add the logical quarantine record for the stale wheel and build/acceptance tooling, but do not
  build the candidate yet.

### 2. Repair primary guidance and executable tutorials

- Create one authoritative evidence-status page/table and link every shorter README, installation,
  model guide, parameter guide, API docstring, and tutorial summary to it.
- Apply all P0.2 guidance corrections from the source plan, including the `u` semantics, raw-count
  boundary, prior uncertainty, DA/DE no-go, continual controls, and TTA/mapQC limitations.
- Repair both treeArches paths and back them with one reusable synthetic tutorial module tested as
  direct same-panel surgery and one-shot learn/update/predict. Use the `cytoanvi` distribution name.

### 3. Implement and test CytoANVI fail-closed contracts

- Implement the frozen CytoANVI decisions above with negative tests that prove refusal occurs before
  trainer/inference entry.
- Replace tests that currently pin shared TTA masks, warning-only replay continuation, or silent
  adversarial compatibility.
- Run the complete source CytoANVI suite and targeted tutorial tests.

### 4. Implement and test MrTotalVI fail-closed contracts

- Implement the prior/supervision migration matrix, deterministic counts validation, authoritative
  sample metadata/subset validation, exact protein-axis validation, streaming export removal,
  conditional latent-shape docs, `use_vmap` refusal, Vamp training-boundary initialization, and
  public DE refusal.
- Replace tests that currently pin implicit label supervision, ignored `use_vmap`, DataFrame failure,
  sampled count warnings, or inferential-looking DE.
- Confine behavior changes to MrTotalVI. If a shared helper must change, run the smallest relevant
  MrMultiVI engineering regression and prove behavior is unchanged; otherwise do not inspect it.

### 5. Validate source, record a scoped local commit, and build once

- Run all listed source-level tests to terminal exit. Passing dots, active processes, or partial
  logs do not count.
- Stage only task-owned code/docs/tests/workflows/scripts and the frozen packet. Never stage the
  pre-existing `.living` changes or unrelated artifacts. Create a scoped local commit because the
  artifact contract requires a recorded source commit. Do not push.
- Produce a clean source tree from that exact commit in a fresh `/tmp` directory or sibling
  worktree. Confirm it is clean, declares `0.2.0`, and contains no uncommitted task files.
- Build exactly one wheel into an empty candidate directory. Do not overwrite any existing wheel.
- Seal wheel SHA/size, source commit/tree, build interpreter/environment, dependency authority,
  metadata, complete RECORD/file inventory, and source-versus-installed inventory.

### 6. Run installed-artifact acceptance if its authority is locally available

- Create a fresh environment from the recorded dependency authority, install the wheel (not the
  checkout), run `pip check`, and execute from `/tmp` with `PYTHONPATH` unset.
- Assert `distribution("scvi-tools")` is absent; both namespaces are owned by `cytoanvi`; imports,
  versions, source paths, and wheel inventory match the manifest.
- Run CytoANVI train, deterministic predict, latent, save/load, same-panel query, and both tutorial
  paths. Run MrTotalVI setup, train, `u`/`z`, summary metadata, and save/load.
- If an exact dependency environment requires network/dependency installation or external writes,
  do not request or perform them from the executor. Record `blocked_dependency_authority`; keep the
  artifact engineering-unaccepted and provide the exact one-command harness for later authorized
  execution. A local contaminated-environment smoke is supplemental only.

### 7. Author P2 protocols without launching them

- Create human-readable and machine-readable CytoANVI and MrTotalVI v1 protocols. Every capability
  entry must freeze artifact identity, cohorts/hashes/roles, split and leakage boundaries, separate
  RNG streams with exact seeds (minimum `[0, 1, 2]`), equal compute budgets, representation
  semantics, primary endpoint and numeric margin, donor-level uncertainty, multiplicity, controls,
  no-call policy, immutable outputs, terminal manifests, and independent pre/post-run review.
- Reuse only verified MrTotalVI authorities: the sealed 46,817-cell/10-donor/20-sample human lineage,
  37,447/9,370 split, 5,000 training-only HVGs, 130 proteins, and B0-B3/D0-D5 grid. Amend effective
  rank to ADR-0009 alert-only semantics. Never splice the preserved 42/48 run.
- Mark missing second cohorts, independent CytoANVI target labels, algorithms, numerical thresholds,
  or controls explicitly `draft_unfrozen`/`blocked`; do not invent them from existing outcomes.
- Do not submit scheduler/GPU jobs, read new restricted datasets, install dependencies, or access
  credentials.

### 8. Seal the non-promoting capability matrix and stop at authority gates

- Materialize all 19 rows with separate engineering execution status, scientific result, decision,
  limitations, artifact identity, evidence links, approver, and date.
- Reject partial/active grids, artifact mismatch, omitted negative results, or promoted rows without
  terminal P2 evidence. Preserve negative/inconclusive results as valid outcomes.
- No agent may self-sign independent protocol approval, result adjudication, or P3 promotion. Leave
  those fields pending and report the exact next approval/action required.

## Validation commands

Use the recorded Python 3.13 environment for source-only tests:

```bash
env LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib \
  MPLCONFIGDIR=/tmp/cytoanvi-mpl NUMBA_CACHE_DIR=/tmp/cytoanvi-numba \
  PYTHONPATH=src \
  /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python \
  -m pytest -q -p no:cacheprovider tests/cytoanvi

env LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib \
  MPLCONFIGDIR=/tmp/cytoanvi-mpl NUMBA_CACHE_DIR=/tmp/cytoanvi-numba \
  PYTHONPATH=src \
  /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python \
  -m pytest -q -p no:cacheprovider tests/external/mrtotalvi

env LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib \
  MPLCONFIGDIR=/tmp/cytoanvi-mpl NUMBA_CACHE_DIR=/tmp/cytoanvi-numba \
  PYTHONPATH=src \
  /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python \
  -m pytest -q -p no:cacheprovider tests/benchmarks/mrtotalvi \
  tests/benchmarks/test_cytoanvi_smoke.py tests/benchmarks/test_cytoanvi_baselines.py \
  tests/benchmarks/test_aggregate_results.py

git diff --check
```

The implementation must add one standalone installed-artifact command with this stable interface:

```bash
scripts/accept_usage_readiness_wheel \
  --wheel dist/cytoanvi-0.2.0-py3-none-any.whl \
  --manifest docs/artifacts/cytoanvi-0.2.0/manifest.json \
  --dependency-authority <exact-local-lock-or-wheelhouse>
```

## Stop conditions

- Stop on target-file drift or overlap with user-owned changes.
- Stop before reset/clean/stash, overwriting an existing artifact, or changing the stale wheel.
- Stop before any network, dependency installation, external-environment write, credential access,
  scheduler/GPU submission, CI trigger, push, tag, release, deployment, or publication action.
- Stop rather than weaken installed-artifact isolation, raw-count exhaustiveness, independent label
  boundaries, terminal evidence, or human approval requirements.
- Stop before broadening MrMultiVI scope or changing its behavior.
- Stop if the exact candidate source commit cannot be cleanly reconstructed.
- Stop at `blocked_dependency_authority` if clean wheel acceptance cannot run without new authority.
- Stop with P2 protocols unfrozen rather than selecting thresholds/cohorts from outcome data.

## Required artifacts and final report

- Checked task list with every completed item backed by a file, command receipt, or explicit blocker.
- ADR/migration/changelog and one authoritative evidence-status page.
- CytoANVI and MrTotalVI source/tests satisfying the frozen P1 contracts.
- Tracked artifact schemas, stale-wheel quarantine record, build/acceptance scripts, and CI definition.
- At most one `0.2.0` candidate wheel plus manifest/inventory/acceptance receipt, or an explicit
  pre-seal blocker that does not overstate acceptance.
- Human- and machine-readable P2 protocol artifacts with freeze status.
- Strict 19-row capability matrix with no agent-issued promotion signature.
- Final summary listing changed files, scoped commits, exact test counts/exits, artifact SHA/commit,
  blockers, residual risks, and the next approval/action.
