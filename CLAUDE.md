# CLAUDE.md — scvi-tools (CytoANVI branch)

## Project overview

This is a fork of [scvi-tools](https://scvi-tools.org/) extended with **CytoANVI** — a semi-supervised, annotation-aware variational autoencoder for antibody-based single-cell cytometry (mass cytometry, flow cytometry, CITE-seq protein).

**Active branch**: `feat/cytoanvi`
**Status**: Implementation complete; pending publication-grade benchmarks at max_epochs=1000.

## Domain language

See `CONTEXT.md` for the canonical CytoANVI glossary. Key terms:
- **Panel / backbone markers / missing markers** — antibody marker sets; backbone = shared across panels
- **Unlabeled category** — cells without ground-truth annotation (remapped to last integer code)
- **M1+M2 hierarchy** — two-level latent (z1 → classifier, z2 → per-label prior); scANVI-style
- **Reference / query / surgery** — trained model / new dataset / scArches-style parameter update
- **Continual update (Phase 2)** — EWC + replay for case-control atlas updates

## Key files

| Path | Description |
|------|-------------|
| `src/cytoanvi/` | CytoANVI implementation |
| `benchmarks/cytoanvi/` | Benchmark harness (B1–B9 tasks) |
| `benchmarks/common/` | Shared training loop, result aggregation |
| `data/` | Nuñez and Roider cytometry datasets |
| `tests/cytoanvi/` | Unit tests |
| `tests/benchmarks/test_cytoanvi_smoke.py` | Benchmark smoke tests |
| `vignettes/cytoanvi_showcase.py` | End-to-end showcase vignette |
| `.scratch/cytoanvi-benchmark/` | Benchmark planning, issues, results |

## Running tests

```bash
# Unit tests
pytest tests/cytoanvi/ -v

# Benchmark smoke tests
pytest tests/benchmarks/test_cytoanvi_smoke.py -v

# Full benchmark (GPU required)
python benchmarks/cytoanvi/run.py --task b1 --seed 0
```

## Benchmark tasks

See `benchmarks/ANALYSIS_MANIFEST.md` for full task table and status.

Publication gate: all B1–B9 tasks at max_epochs=1000, ≥3 seeds, full Nuñez and Roider cohorts.

## Living repository protocol

This repo uses [mycelium](https://github.com/K-Dense-AI/mycelium) for session continuity.

### On session start

1. Read `.claude/last-session.md` for context from the prior session.
2. Read `.living/INDEX.md` for the tag/cluster index.
3. Check `todo/TODO_REGISTRY.md` for open items.

### After significant work

Run the post-action protocol:
1. Update the relevant manifest (`benchmarks/ANALYSIS_MANIFEST.md`, `data/DATA_MANIFEST.md`, etc.)
2. Append decisions to `.living/decisions.md` (non-obvious choices)
3. Append learnings to `.living/learnings.md` (bugs, gotchas, edge cases)
4. Update `todo/TODO_REGISTRY.md` if new work items identified
5. Write/update `.claude/last-session.md` with a 5-section session summary

### Living directory structure

```
.living/
├── decisions.md          # D-NNN architecture/design decisions
├── learnings.md          # L-NNN bugs, gotchas, edge cases
├── conventions.md        # C-NNN crystallized conventions
├── INDEX.md              # Tag index + knowledge clusters
├── findings/             # Empirical results by topic
│   └── FINDINGS_REGISTRY.md
├── log/
│   └── LOG_REGISTRY.md   # Session log
└── outputs/
    └── knowledge-transfers/
```

## Active conventions

See `.living/conventions.md`. Key ones:
- **C-001**: Mask per-cell (not per-panel) when aggregating over marker dimensions with `nan_layer`
- **C-002**: Guard all classifier forward passes against `n_labels == 0`
- **C-003**: Subsample to ≤10k cells for Fisher/EWC importance computation

## Bioinformatics conventions

See `.living/conventions/bioinformatics/analysis-conventions.md` for installed domain conventions.
