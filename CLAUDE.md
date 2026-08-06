# CLAUDE.md — scvi-tools (MrTotalVI/MrMultiVI fork)

## Project overview

This is a fork of [scvi-tools](https://scvi-tools.org/) developed around two tracks:

- **MrTotalVI / MrMultiVI** (active) — semi-supervised, multi-resolution VAEs extending scvi-tools'
  TOTALVI/MultiVI with an MrVI-style two-level (u/z) donor latent space, for CITE-seq and multimodal
  data. See `docs/adr/0006-mrmultivi.md` and `docs/user_guide/models/mr_multimodal.md` for the model
  design and public API.
- **CytoANVI** (shelved 2026-07-12) — a semi-supervised, annotation-aware variational autoencoder for
  antibody-based single-cell cytometry (mass cytometry, flow cytometry, CITE-seq protein). Code,
  benchmarks, and tests remain in the tree for when it is revived; its `todo/TODO_REGISTRY.md` items
  are marked `shelved` with revival conditions rather than deleted.

**Active branch**: `main`

**Status**: current status lives in the living trackers below, not in this file — a status line
hardcoded here goes stale the moment work continues:
- `todo/TODO_REGISTRY.md` — full work-item registry, tagged by track (`mrtotalvi` / `cytoanvi` / `infra`), priority, and status
- `todo/2026-08-06-remaining-work.md` — one-page orientation brief, read this first
- `todo/2026-08-06-next-steps-plan.md` — phased sequencing plan (what to do in what order, which gates are HITL)

## Domain language

**MrTotalVI / MrMultiVI** — see `docs/adr/0006-mrmultivi.md` (architecture decision record) and
`docs/user_guide/models/mr_multimodal.md` (user guide: setup, prior options, batch representation,
statistical APIs) for canonical terminology and the current public API.

**CytoANVI** (shelved) — see `CONTEXT.md` for the canonical glossary. Key terms:
- **Panel / backbone markers / missing markers** — antibody marker sets; backbone = shared across panels
- **Unlabeled category** — cells without ground-truth annotation (remapped to last integer code)
- **M1+M2 hierarchy** — two-level latent (z1 → classifier, z2 → per-label prior); scANVI-style
- **Reference / query / surgery** — trained model / new dataset / scArches-style parameter update
- **Continual update (Phase 2)** — EWC + replay for case-control atlas updates

## Key files

| Path | Description |
|------|-------------|
| `src/scvi/external/mrtotalvi/` | MrTotalVI implementation (active) |
| `src/scvi/external/mrmultivi/` | MrMultiVI implementation (active) |
| `benchmarks/mrtotalvi/` | MrTotalVI/MrMultiVI benchmark & convergence-diagnosis harness (active) |
| `tests/external/mrtotalvi/` | MrTotalVI unit tests (active) |
| `tests/external/mrmultivi/` | MrMultiVI unit tests (active) |
| `docs/adr/0006-mrmultivi.md` | MrMultiVI architecture decision record |
| `docs/user_guide/models/mr_multimodal.md` | MrTotalVI/MrMultiVI user guide |
| `src/cytoanvi/` | CytoANVI implementation (shelved 2026-07-12) |
| `benchmarks/cytoanvi/` | CytoANVI benchmark harness — B1–B9 tasks (shelved) |
| `benchmarks/common/` | Shared training loop, result aggregation (used by both tracks) |
| `data/` | Nuñez, Roider, and other cytometry/CITE-seq datasets — see `data/DATA_MANIFEST.md` |
| `tests/cytoanvi/` | CytoANVI unit tests (shelved) |
| `tests/benchmarks/test_cytoanvi_smoke.py` | CytoANVI benchmark smoke tests (shelved) |
| `vignettes/cytoanvi_showcase.py` | CytoANVI end-to-end showcase vignette (shelved) |
| `.scratch/cytoanvi-benchmark/` | CytoANVI benchmark planning, issues, results (shelved) |

## Running tests

```bash
# MrTotalVI / MrMultiVI unit tests (active track)
pytest tests/external/mrtotalvi/ tests/external/mrmultivi/ -v

# CytoANVI unit tests (shelved track)
pytest tests/cytoanvi/ -v

# CytoANVI benchmark smoke tests (shelved track)
pytest tests/benchmarks/test_cytoanvi_smoke.py -v

# CytoANVI full benchmark (GPU required, shelved track)
python benchmarks/cytoanvi/run.py --task b1 --seed 0
```

## Benchmark tasks

- **MrTotalVI/MrMultiVI**: see `benchmarks/mrtotalvi/README.md` for what the harness currently
  measures, and `todo/TODO_REGISTRY.md` (track `mrtotalvi`) for open work and status — this changes
  often enough that it is not repeated here.
- **CytoANVI** (shelved): see `benchmarks/ANALYSIS_MANIFEST.md` for the B1–B9 task table as of shelving.

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

See `.living/conventions.md` for the full list. Key ones:
- **C-001** (CytoANVI): Mask per-cell (not per-panel) when aggregating over marker dimensions with `nan_layer`
- **C-002** (CytoANVI): Guard all classifier forward passes against `n_labels == 0`
- **C-003** (CytoANVI): Subsample to ≤10k cells for Fisher/EWC importance computation
- **C-004** (general, added 2026-07-28): Never compare embeddings with a statistic whose denominator or operating point encodes the effect under test — pair every variance-share/fixed-setting statistic with a denominator-free counterpart before drawing a conclusion. See `.living/conventions.md` for the full checklist.

## Bioinformatics conventions

See `.living/conventions/bioinformatics/analysis-conventions.md` for installed domain conventions.
