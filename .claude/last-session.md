# Last Session Summary — 2026-06-30 (autonomous continuation, session 8)

## 1. What was accomplished

**Discovered and fixed harmonypy 0.2.0 Z_corr transposition bug (commit e8a6d5f9):**
- PID 1520357 (scvi env) crashed at harmony baseline after CytoANVI training: `IndexError: boolean index did not match indexed array along axis 0; size of axis is 30 but size of corresponding boolean axis is 100000`
- Root cause: harmonypy 0.2.0 returns `Z_corr` as `(n_cells, n_components)` but code applied `.T` unconditionally, making it `(n_components, n_cells)` = shape `(30, 100000)` → boolean mask of 100000 elements fails
- Fix: `Z_harmony = ho.Z_corr.T if ho.Z_corr.shape[0] == n_comp else ho.Z_corr` in `baselines.py`
- Also added `IndexError` to except clause in `tasks.py` for defensive degradation
- Added L-025 to `.living/learnings.md`
- Killed PID 2539861 (scvi-test env, epoch 183/1000, would have crashed on harmony) and restarted as PID 3186154 writing to `nunez_inductive_b1_s012_v3.log`

**B1 Nuñez v3 progress (PID 3186154, seed-0):**
- At epoch 117/1000, elapsed 27:32, train_loss -81.4, speed ~14.7s/epoch (stable, no NaN)
- ETA: seed-0 done ~01:35 CEST Jul 1; all 3 seeds ~09:45 CEST Jul 1

**Smoke test (job 25129287) still running at session capture (54 min elapsed):**
- Log still at 863 bytes (Python output buffering — no `-u` flag + lightning `\r` progress bars)
- Job is healthy (RUNNING, not PENDING or FAILED)
- Output will appear all at once when 20-epoch B3 training completes

**Living docs updated:**
- `benchmarks/ANALYSIS_MANIFEST.md` — B1 row updated with PID 3186154 v3 restart info
- `todo/TODO_REGISTRY.md` — B1 entry PID updated to 3186154 (v3)
- All changes staged for commit

## 2. Cumulative benchmark status

| Task | Status | Result |
|------|--------|--------|
| B1 Roider | ✓ pub-grade | Δ+0.121±0.040 |
| B1 Nuñez inductive | RUNNING (PID 3186154 v3, seed-0 epoch 117/1000; ETA all-3 ~09:45 CEST Jul 1) | harmony fix (L-025, commit e8a6d5f9) |
| B2 Roider | ✓ | bio +0.108, batch Δ−0.006 |
| B2 Nuñez r0.05 | ✓ | batch Δ−0.005 |
| B3 roider-full | PENDING smoke-test gate (job 25129287 RUNNING, output buffering) | Leiden=47; submit after NaN-free confirmation |
| B4 | Blocked (pseudo split) | plumbing only |
| B5 roider-e1000 | ✓ multiseed | best 0.833±0.122, mean 0.462±0.075 |
| B5 roider-full sweep | PENDING B3 smoke + epoch timing | 47 clusters, ~24h estimated |
| B6 | Blocked | plumbing only |
| B8 | ✅ pub-gate | Δ_hier +0.0862±0.0027 (3-seed, F-011) |
| B9 | Blocked (mapqc) | — |

## 3. SLURM state

| Job ID | Name | Status | Purpose |
|--------|------|--------|---------|
| 25129287 | cytoanvi_smoke_b3_roider | RUNNING (output buffered; ~54 min elapsed at session capture) | 20-epoch B3 timing + NaN check |
| 25089685 | vs_val | PENDING DependencyNeverSatisfied | Stale; USER should cancel |
| 25102610 | cytoanvi_p7_aggregate | PENDING DependencyNeverSatisfied | Stale; USER should cancel |
| 25102547 | cytoanvi_p6_b9_mapqc | PENDING DependencyNeverSatisfied | Stale; blocked on mapqc |

## 4. Open items checklist

| ID | Item | Status |
|----|------|--------|
| P4-A | B1 inductive JSON when complete | RUNNING (PID 3186154 v3, ~09:45 CEST Jul 1 ETA) |
| B3-gate | Submit phase3_b3b5_roider.slurm after smoke test NaN-free | Pending smoke output flush |
| B5-gate | Submit phase3b_b5sweep_roider.slurm after epoch timing | Pending smoke |
| scancel | `scancel 25089685 25102610 25102547` | USER action required |
| mapqc | Install `mapqc` in conda env | USER action required |
| B4-real | Real rLN vs FL/MCL split or demote to supplement | USER decision |
| Figshare | Archive `nunez_annotated.h5ad` | USER credentials required |
| P5-E | conda lock / REPRODUCE.md / Singularity.def | USER decision |

## 5. Open items for next session

**Smoke test (job 25129287) — check log when epoch output appears:**
```bash
cat .scratch/cytoanvi-benchmark/slurm/out/cytoanvi_smoke_b3_roider_25129287.log | tr '\r' '\n' | grep -E "Epoch|loss|NaN|complete|s/it"
```
- Confirm: no NaN, extract epoch timing from epochs 2-10 (in seconds per epoch)
- If OK (no NaN, epoch time ≤~5 min): submit both phase 3 scripts:
  ```bash
  sbatch .scratch/cytoanvi-benchmark/slurm/phase3_b3b5_roider.slurm
  sbatch .scratch/cytoanvi-benchmark/slurm/phase3b_b5sweep_roider.slurm
  ```
- If epoch time significantly > 5 min/epoch: adjust --time upward in phase3_b3b5_roider.slurm

**B1 inductive (PID 3186154 v3) — check when seeds complete (~09:45 CEST Jul 1):**
```bash
ls -la .scratch/cytoanvi-benchmark/results/e1000/nunez_inductive_b1_multiseed.json
cat .scratch/cytoanvi-benchmark/results/e1000/nunez_inductive_b1_multiseed.json | python3 -m json.tool
```
- When all 3 seeds done: update F-003 in FINDINGS_REGISTRY and label-transfer-accuracy.md
- Update ANALYSIS_MANIFEST B1 Nuñez row with final Δ
- Commit findings update

**After B3 + B5 full-cohort results land:**
- Run manifest-mode aggregation: `python benchmarks/common/aggregate_results.py --manifest .scratch/cytoanvi-benchmark/publication_manifest.json`
- Update FINDINGS_REGISTRY with B3/B5 full-cohort results
- Push `feat/cytoanvi` branch and prepare upstream PR
