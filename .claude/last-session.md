# Last Session Summary — 2026-07-01 (autonomous continuation, session 9)

## 1. What was accomplished

**B1 Nuñez inductive complete (PID 3186154 v3, 05:58 CEST Jul 1):**
- All 3 seeds (0/1/2) finished at max_epochs=1000
- JSON: `.scratch/cytoanvi-benchmark/results/e1000/nunez_inductive_b1_multiseed.json`
- CytoANVI 0.9751 ± 0.0003 vs CytoVI kNN 0.9581 ± 0.0007 (Δ +0.017)
- XGBoost 0.9722 ± 0.0008; Raw marker kNN 0.9010 ± 0.0028; Harmony kNN 0.9111 ± 0.0026
- Gate (≥+0.03) formally misses — ceiling effect on clean 11-type PBMC; CytoANVI wins all comparisons
- Prior reversal Δ−0.013 fully explained by transductive leakage (L-022); no longer relevant
- Roider Δ+0.121 (F-002) remains primary B1 result

**Smoke test B3 Roider-full PASSED (job 25129287, completed 22:14:05 CEST Jun 30):**
- Elapsed 1:32:07 (88 min data prep + 3.4 min training — output buffering explained)
- CytoANVI: 3.49 s/epoch steady state; CytoVI: 6.55 s/epoch steady state
- No NaN divergence; 20-epoch concordance 0.641 (at 20 epochs, expect higher at 1000)

**Phase 3 SLURM jobs submitted (2026-07-01):**
- Job **25132400** — B3 full-cohort, 3 seeds, `--time=14:00:00`; ETA ~13:00 CEST Jul 1
  - Outputs: `roider_full_b3_s{0,1,2}.json`
- Job **25132401** — B5 holdout sweep, seed 0, 47 clusters, `--time=48:00:00`; ETA ~06:00 CEST Jul 2
  - Output: `roider_full_b5_sweep_s0.json`

**Living docs updated:**
- `benchmarks/ANALYSIS_MANIFEST.md` — B1, B3, B5 rows updated
- `.living/findings/FINDINGS_REGISTRY.md` — F-003 updated to inductive result
- `.living/findings/label-transfer-accuracy.md` — F-003 section fully rewritten
- `todo/TODO_REGISTRY.md` — B1 inductive marked done; B3/B5 as in_progress with job IDs; reversal monitor marked done
- `.scratch/cytoanvi-benchmark/issues/08-nunez-b1-b2-full.md` — B1 completion section added
- `.scratch/cytoanvi-benchmark/issues/09-roider-b3-b5-full.md` — smoke gate + B3/B5 submission section added
- `.living/log/LOG_REGISTRY.md` — session 9 row added

## 2. Cumulative benchmark status

| Task | Status | Result |
|------|--------|--------|
| B1 Roider | ✓ pub-grade | Δ+0.121±0.040 ✅ gate |
| B1 Nuñez inductive | ✓ complete (2026-07-01 05:58) | CytoANVI 0.9751±0.0003, Δ+0.017 (ceiling) |
| B2 Roider | ✓ | bio +0.108, batch Δ−0.006 |
| B2 Nuñez r0.05 | ✓ | batch Δ−0.005 |
| B3 roider-full | RUNNING (job 25132400, 3 seeds; ETA ~13:00 CEST Jul 1) | smoke gate ✅ (no NaN, 3.49s/epoch) |
| B4 | Blocked (pseudo split) | plumbing only |
| B5 roider-e1000 | ✓ multiseed | best 0.833±0.122, mean 0.462±0.075 |
| B5 roider-full sweep | RUNNING (job 25132401, seed 0, 47 clusters; ETA ~06:00 CEST Jul 2) | — |
| B6 | Blocked | plumbing only |
| B8 | ✅ pub-gate | Δ_hier +0.0862±0.0027 (3-seed, F-011) |
| B9 | Blocked (mapqc) | — |

## 3. SLURM state

| Job ID | Name | Status | Purpose |
|--------|------|--------|---------|
| 25132400 | cytoanvi_p3a_roider_b3 | RUNNING | B3 full-cohort 3 seeds, 1000 epochs; ETA ~13:00 Jul 1 |
| 25132401 | cytoanvi_p3b_roider_b5sweep | RUNNING | B5 sweep seed 0, 47 clusters; ETA ~06:00 Jul 2 |
| 25129287 | cytoanvi_smoke_b3_roider | COMPLETED | Smoke gate passed |
| 25089685 | vs_val | PENDING DependencyNeverSatisfied | Stale; USER should cancel |
| 25102610 | cytoanvi_p7_aggregate | PENDING DependencyNeverSatisfied | Stale; USER should cancel |
| 25102547 | cytoanvi_p6_b9_mapqc | PENDING DependencyNeverSatisfied | Stale; blocked on mapqc |

## 4. Open items checklist

| ID | Item | Status |
|----|------|--------|
| B3-monitor | Check job 25132400 logs for NaN + concordance ≥ 0.70 | RUNNING |
| B5-monitor | Check job 25132401 logs for AUROC > 0.70 | RUNNING |
| aggregate | Run manifest-mode aggregation after B3/B5 land | Pending B3+B5 |
| scancel | `scancel 25089685 25102610 25102547` | USER action required |
| mapqc | Install `mapqc` in conda env | USER action required |
| B4-real | Real rLN vs FL/MCL split or demote to supplement | USER decision |
| Figshare | Archive `nunez_annotated.h5ad` | USER credentials required |
| P5-E | conda lock / REPRODUCE.md / Singularity.def | USER decision |

## 5. Open items for next session

**Monitor B3 (job 25132400) — check when done (~13:00 CEST Jul 1):**
```bash
sacct -j 25132400 --format=JobID,State,Elapsed,ExitCode
cat .scratch/cytoanvi-benchmark/slurm/out/cytoanvi_p3a_roider_b3_25132400.log | tr '\r' '\n' | grep -E "Epoch|loss|NaN|concordance|complete"
ls -la .scratch/cytoanvi-benchmark/results/roider_full_b3_s{0,1,2}.json
```
- Confirm: no NaN, per-seed concordance ≥ 0.70
- If done: add F-012 to FINDINGS_REGISTRY for roider-full-e1000 B3

**Monitor B5 (job 25132401) — check when done (~06:00 CEST Jul 2):**
```bash
sacct -j 25132401 --format=JobID,State,Elapsed,ExitCode
cat .scratch/cytoanvi-benchmark/slurm/out/cytoanvi_p3b_roider_b5sweep_25132401.log | tr '\r' '\n' | grep -E "cluster|AUROC|holdout|complete"
ls -la .scratch/cytoanvi-benchmark/results/roider_full_b5_sweep_s0.json
```
- Confirm: ∃ type with AUROC > 0.70
- If done: add F-013 to FINDINGS_REGISTRY for roider-full-e1000 B5

**After both jobs complete:**
```bash
python benchmarks/common/aggregate_results.py --manifest .scratch/cytoanvi-benchmark/publication_manifest.json
```
- Then update FINDINGS_REGISTRY + ANALYSIS_MANIFEST with full-cohort results
- Push `feat/cytoanvi` branch and prepare upstream PR
