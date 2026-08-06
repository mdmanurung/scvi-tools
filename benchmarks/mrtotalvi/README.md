# MrTotalVI controlled benchmark pilot

This directory contains a deterministic, CPU-scale mechanism benchmark for the opt-in MrTotalVI
v2 implementation. It is deliberately narrower than the scientific benchmark in
`docs/review-clear-execute/mrtotalvi-v2/plan.md`.

The pilot can establish software identities and test whether a mechanism moves an outcome in the
predicted direction on exogenous truth. It cannot select a model, validate biology, estimate
publication FDR, change the default, or resolve the canonical data-lineage gate.

## Frozen candidates

| Candidate | Prior | Hierarchy | `u` encoder | Observation weighting |
|---|---|---|---|---|
| C0 | MoG | legacy | sample-conditioned | cell-equal |
| C1 | initialized frozen VampPrior | legacy | sample-conditioned | cell-equal |
| C2 | C1 | centered-v2 | sample-conditioned | cell-equal |
| C3 | C1 | centered-v2 | sample-conditioned | sample-equal |
| C4 | C1 | centered-v2 | sample-blind | cell-equal |

All nonlisted model and training axes are held fixed by `FixtureRunConfig`. C1 remains an
unvalidated baseline candidate; running it in this simulator does not reproduce historical
real-data evidence.

## Scenario family

`simulation.py` defines `null`, `da_only`, `de_only`, `mixed`, `rare_state`, `unequal_cells`,
`continuous`, and `batch_confounded`. Raw RNA and protein observations are integer counts.
Exogenous truth is returned in a separate object and is never registered with the model. Truth,
training, and evaluation random-number streams are independently spawned from the requested seed.

The current runner scores latent and decoder recovery. It does not implement Milo inference,
calibrated DA FDR/power, LEMUR, miloDE, or pseudobulk DE; scenario labels must not be mistaken for
validation of those unimplemented endpoints.

## Metrics

`metric_dictionary.json` is the source contract for estimands, direction, units, splits,
aggregation, and missing/no-call behavior. Its pilot selection rule and tie-break are both
`none`. The only pilot threshold is the analytic centering tolerance of `1e-6`.

In particular:

- state prediction and within-state sample prediction are separate outcomes;
- lower within-state sample predictability is evidence about direct leakage, not proof that `u`
  is biologically sample-neutral;
- target-distance Spearman compares same-cell registered-target geometry to exogenous truth;
- RNA composition RMSE applies only to the explicit centered-v2 decoder;
- per-sample ELBO dispersion is an engineering proxy, not a DA-calibration endpoint.

## Run and verify

Use the repository's Python 3.13 test environment with its `lib` directory prepended to
`LD_LIBRARY_PATH`, `PYTHONPATH=.` or `PYTHONPATH=src:.`, CPU-only CUDA masking, and writable
temporary caches.

Example:

```bash
python -m benchmarks.mrtotalvi.run_pilot \
  --scenario mixed \
  --candidates C0,C1,C2,C3,C4 \
  --seeds 0,1,2 \
  --max-epochs 10
```

Each run is written to a temporary sibling directory and atomically renamed only after its exact
artifact inventory is sealed. The run identifier contains UTC timestamp plus code, configuration,
and data digests. `verify_run_manifest()` rejects missing, extra, size-mismatched, hash-mismatched,
or identity-mismatched artifacts. Publication-tier manifests additionally require a durable URI
for every artifact; `.scratch` runs are only pilot-cache evidence.

Run the contract suite with:

```bash
python -m pytest tests/benchmarks/mrtotalvi -q -p no:cacheprovider
```

The bounded amendment, consolidated roadmap, and dated validation report live under
`.scratch/mrtotalvi-v2/`.

## Historical-human sensitivity

`run_historical_comparator.py` reuses the exact 500-cell, 1,000-gene, 130-protein fixture sealed by
the package engineering run. It reads only one explicitly allowed state annotation from the
historical source (`cell_label_l1`, `cell_label_l1p5`, `cell_label_l2`, or `cell_label_l3`);
`pass_qc` is not an allowed field. The reader, ordered annotation digest, exact candidate-by-seed
grid, representations, aggregate, and atomic manifest are tested and recorded.

Example:

```bash
python -m benchmarks.mrtotalvi.run_historical_comparator \
  --candidates C0,C1,C2,C3,C4 \
  --seeds 0,1,2 \
  --max-epochs 3
```

This sensitivity reports held-out ELBO, state accuracy, within-state sample predictability,
15-neighbor state accuracy, cross-seed 15-neighbor Jaccard, and centered identities. Its fixed
label is:

`historical comparator; not canonical; not QC-pass; not biological validation; not promotion evidence`

The three-epoch fit is intentionally bounded and is not a convergence study. Absolute scores and
fast-screen thresholds cannot select a candidate on this fixture.
