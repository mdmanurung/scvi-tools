# Session 62 — B2 Roider-full results ingested + crystallization

**Date**: 2026-07-13  
**Branch**: main (CytoANVI + MrTotalVI/MrMultiVI)  
**Trigger**: "Next" — checked SLURM job status and ingested B2 Roider-full results.

---

## 1. Work completed

### SLURM job status check

| Job ID | Name | Status | Outcome |
|--------|------|--------|---------|
| 25211800 | cytoanvi_b2_roider_full | ✅ COMPLETED (ExitCode 0) | Ran 2026-07-12 20:47 → 2026-07-13 07:55 (~11h) |
| 25211796 | cytoanvi_leiden_cal | ⏸ PD (pending) | QOSMaxCpuPerUserLimit — still queued |

### B2 Roider-full results (F-040)

All 3 seeds at max_epochs=1000, n_cells=620,000:

| Model | Total (mean±std) | Bio cons. | Batch corr. |
|-------|------------------|-----------|-------------|
| CytoANVI | **0.5953±0.0027** | **0.6544±0.0053** | 0.5067±0.0018 |
| CytoVI | 0.5360±0.0022 | 0.5461±0.0033 | 0.5208±0.0017 |
| Δ | **+0.0593** | **+0.1083** ✅ | −0.0141 ✅ |

Very low seed variance (std≤0.0053). Fully confirms and extends F-004 (roider-e1000 pilot). The bio gain (+0.108) replicates exactly at full cohort.

### Files updated

- `benchmarks/ANALYSIS_MANIFEST.md` — B2 row updated from "SUBMITTED" to "✅ COMPLETE" with per-seed numbers
- `.living/findings/FINDINGS_REGISTRY.md` — F-040 added
- `CLAUDE.md` — status line updated: B1/B2/B3/B5/B8 complete; B2 Roider-full no longer pending

---

## 2. Key findings

- B2 Roider-full **fully confirms the pilot** (F-004): bio gain +0.108 at 620k cells matches the 5k-cell estimate (+0.109). The result is robust at full cohort.
- **Leiden calibration job still pending** — job 25211796 in queue due to QOSMaxCpuPerUserLimit. Will need to check again next session.
- The B2 full-cohort batch correction is slightly WORSE (−0.014) than CytoVI's — this is expected and desirable: CytoANVI preserves biological signal at the cost of slightly worse mixing.

---

## 3. Outstanding items

| Item | Status |
|------|--------|
| Leiden calibration (job 25211796) | Still pending (QOSMaxCpuPerUserLimit) |
| P1-006: MrTotalVI DA instability root-cause | GPU + real data required |
| P2-005: Macaque CITE-seq validation | Dataset + training compute required |
| 3 deferred code-review findings (DA smoke DRY, elbo_key conftest) | Low priority, no blocker |

---

## 4. Decisions / learnings added

None new this session.

---

## 5. Next session priorities

1. Check Leiden calibration job 25211796 (still pending — check if it ever started).
2. Consider checkpoint commit for all P1/P2/code-review changes (uncommitted changes across 11 files, +566 lines).
3. Fix the 3 deferred code-review findings if time allows (DA smoke DRY, elbo_key conftest).
4. P1-006 (MrTotalVI DA root cause) if GPU access is available.
