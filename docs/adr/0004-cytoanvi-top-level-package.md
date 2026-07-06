# CytoANVI top-level package ownership

## Status

Accepted.

## Context

CytoANVI started as an implementation under `scvi.external.cytoanvi`, with
`scvi.external.CytoANVI` as its public import. The code now has its own model, module,
continual-learning helpers, hierarchy helpers, mapping-QC helpers, tests, tutorials, and benchmark
harness. Keeping it as an embedded `scvi.external` addition made the package boundary unclear.

## Decision

CytoANVI is a top-level package distributed by this scvi-tools checkout:

- `cytoanvi.CytoANVI`
- `cytoanvi.CytoANVAE`
- `cytoanvi.hierarchy`
- `cytoanvi.mapping_qc`

Internal implementation modules remain underscored. CytoANVI continues to depend on scvi-tools
internals and `scvi.external.cytovi`; CytoVI preprocessing, read/write, and plotting helpers are
not copied into CytoANVI.

The previous `scvi.external.CytoANVI` export and `scvi.external.cytoanvi` package path are removed
intentionally. This is a breaking import-path change, including for old artifacts that require the
previous Python class module path during unpickling or model loading.

## Consequences

Tests, tutorials, benchmark code, and API docs should import CytoANVI from `cytoanvi`. The benchmark
harness remains under `benchmarks/cytoanvi` and is not part of the public `cytoanvi` package.
Optional extras keep the existing distribution names:
`scvi-tools[cytoanvi-hierarchy]` and `scvi-tools[cytoanvi-mapping-qc]`.
