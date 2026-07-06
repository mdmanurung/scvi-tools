# 13 — B9: mapQC query-to-reference mapping QC

Status: needs-info
Blocked-by: `mapqc` missing from benchmark environment; real case/control cohort for full scoring

## Task

After CytoANVI query surgery, score mapping quality with [mapQC](https://github.com/theislab/mapqc)
on joint CytoANVI latents (`mapping_qc.build_mapqc_anndata` / `score_query_mapping`).

```bash
# Plumbing smoke (synthetic — builds joint latent, skips mapQC.run)
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task b9 --max-epochs 50 --seed 0

# Full mapQC (real case/control cohort)
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset nunez --task b9 --max-epochs 1000 --seed 0 \
  --mapqc-run   # or non-synthetic dataset auto-enables mapQC
```

Requires `mapqc` / `scvi-tools[cytoanvi-mapping-qc]` to already be installed before queue
submission. Do not install dependencies inside SLURM benchmark jobs.

## Acceptance

- Synthetic B9 returns `status: plumbing_only` with `joint_n_obs` and label-transfer metrics
- Real data: `status: mapqc_complete` with `query_control_mapqc.frac_dist_to_ref` documented

## Known failure points (patched 2026-06-18)

| Failure | Cause | Fix |
|---------|-------|-----|
| AnnData view error in tests | Sliced adata passed to `setup_anndata` | `.copy()` before train |
| Query batch not in registry | Reference-only model on query batch | Use **query model** after surgery |
| mapQC IndexError on synthetic | All neighborhoods filtered (tiny pseudo cohort) | `run_mapqc=False` on synthetic; real data only for scoring |
| Uneven sample sizes | Random sample assignment | Round-robin assignment in `_assign_mapqc_pseudo_samples` |
| ImportError in library | mapQC not installed | `pip install scvi-tools[cytoanvi-mapping-qc]`; fail-fast in B9 before training and in `run_mapqc_on_joint` |

### 2026-06-27 — SLURM job submitted
Phase 6 SLURM script `.scratch/cytoanvi-benchmark/slurm/phase6_b9_mapqc.slurm` submitted as job
**25102525** (dependency: Nuñez P2 25102520). Seed 0, max_epochs=1000, mapQC run enabled.
Pass criteria: UMAP diff < 2σ, pseudo-sample F1 reported. Status: `pending (after P2)`.

### 2026-06-27 — blocked by missing optional dependency
Preflight command failed:

```bash
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python -c "import mapqc"
```

Result: `ModuleNotFoundError: No module named 'mapqc'`.

Policy patch:
- Removed `pip install mapqc` from `.scratch/cytoanvi-benchmark/slurm/phase6_b9_mapqc.slurm`.
- The script now preflights `import mapqc` and exits with a blocker message if the dependency is
  missing.

No Phase 6 job was submitted in the resumed queue. Aggregation proceeds without B9 and records it
as missing/blocked.

### 2026-06-28 — blocked artifact behavior
Phase 6 now writes one stable artifact path,
`.scratch/cytoanvi-benchmark/results/nunez_b9_s0.json`, for both blocked and successful outcomes.
When `mapqc` is unavailable, the JSON carries `status: blocked` and the script exits successfully so
optional B9 does not block required publication phases. Manifest aggregation includes this artifact
as optional/blocked when it exists. B9 remains excluded from publication claims until `mapqc` is
provisioned and a real case/control cohort is scored.

### 2026-06-28 — optional blocked artifact recorded
Submitted Phase 6 as SLURM job **25104251**. The job confirmed `mapqc` is unavailable in the
benchmark environment and wrote
`.scratch/cytoanvi-benchmark/results/nunez_b9_s0.json` with `b9.status == "blocked"`. This satisfies
the optional manifest bookkeeping requirement but remains excluded from publication claims.
