# ADR-0009: Prospective RDX-03 latent-integrity policy v2

## Status

Accepted prospectively on 2026-07-31.

## Context

ADR-0008 treated effective rank below half the configured latent dimension as
a terminal `latent_collapse` failure. Review of the preserved incomplete
RDX-03 run showed that effective rank is a variance-spectrum diagnostic, can
change under invertible anisotropic rescaling, and does not by itself establish
loss of usable representation information. The same threshold also coupled
shared `u` and factual `z` despite their distinct downstream roles.

Changing historical classifications would destroy replayability. Adding a
result-derived threshold or kNN rescue would introduce outcome-guided tuning.
The decision must therefore be versioned and prospective, with explicit
representation-specific terminal evidence.

## Decision

Adopt policy `mrtotalvi-rdx03-latent-integrity-v2`, whose canonical payload is
`benchmarks/mrtotalvi/latent_integrity_policy_v2.json` with canonical SHA-256
`d8e8513768514aa598f7881b3ff6012b06b62b467af3214d2c6c44eb162af2a7`.

For prospective v2 runs:

- effective rank below `0.5 * configured_latent_dimension` is
  `low_rank_alert` only;
- nonfinite representation/integrity inputs are terminal for the affected
  representation;
- exact zero centered variation is terminal, using exact row equality with no
  epsilon threshold;
- every required posterior-scale element must be present, finite, and
  strictly positive;
- MrTotalVI registered-residual gradient coverage must equal exactly `1.0`
  and is terminal for affected factual `z`, where that embedding contributes;
- terminal integrity is representation-specific;
- non-convergence remains a separate fit-wide hard scientific failure;
- invalid diagnostics serialize explicit reason-coded no-calls and the frozen
  grid continues; and
- rank alerts never suppress paired or cross-seed geometry.

Prospective factual-`z` posterior scale uses the analytic `qz.scale` for stock
scVI/TotalVI. MrTotalVI uses the evaluation-seed stream for 32 posterior draws
and their unbiased sample standard deviation (`ddof=1`). Representation export
restores Python, NumPy, torch CPU, and active torch device RNG states before
unchanged stochastic diagnostics run.

Historical v1 contract factories, assessments, artifacts, and verdicts retain
the ADR-0008 effective-rank terminal rule and must replay exactly.

All state-recovery, prediction, leakage, stability, Milo, direct-metric
preprocessing, fixture, split, seed, and downstream selection gates remain
unchanged. There is no result-derived threshold and no kNN rescue rule.
Factual human W22-versus-W00 differential abundance remains locked.

## Consequences

RDX-03 separates integrity, convergence, and rank diagnostics instead of
allowing one spectrum statistic to stand in for all three. A technically valid
negative RDX-03 result is handed to RDX-04; RDX-03 itself cannot issue a
`candidate` or terminal `stop`.

This ADR does not promote a model, change a package default or public API,
alter checkpoint semantics, authorize factual-human DA, or establish a
scientific claim.
