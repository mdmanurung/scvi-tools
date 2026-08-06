# Frozen Plan: MrTotalVI RDX-03 Latent-Integrity v2

Frozen: 2026-07-30

## Objective

Replace the prospective effective-rank terminal gate with a versioned latent-integrity policy,
while preserving exact v1 replay, then validate one fresh immutable 48-fit RDX-03 CPU grid and
hand a valid RDX-03 result to the D1-D5 redesign.

RDX-03 cannot issue `candidate` or terminal `stop`. A scientifically negative but technically
valid RDX-03 result records its decomposition and continues to RDX-04. Scheduler submission
requires separate explicit authority.

## 1. Evidence freeze and amendment

1. Record HEAD, the full dirty status including untracked files, and hashes of every consumed
   source, configuration, fixture, policy, and lineage artifact.
2. Hash but do not modify
   `.scratch/mrtotalvi-v2-redesign/PRESERVED-20260726-partial-run-42of48`.
3. Record the corrected inventory: 42 result/representation/worker-manifest triples plus one
   orphan canonical-human B3 seed-0 result/representation; none is sealed or promotable.
4. Add ADR-0009 to supersede only ADR-0008's effective-rank hard-gate clause for prospective v2
   runs.
5. Add `.scratch/mrtotalvi-v2-redesign/latent-integrity-policy-v2.md` without rewriting the frozen
   v1 plan, challenge, or governance artifacts.
6. Freeze a canonical machine policy before any fit:
   - terminal failure for nonfinite representation/integrity inputs;
   - terminal failure for exactly zero centered variation, with no epsilon threshold;
   - terminal failure if any posterior-scale element is missing, nonfinite, or nonpositive;
   - terminal failure for MrTotalVI residual-gradient coverage other than exactly `1.0`;
   - convergence remains a separate hard failure;
   - effective rank remains recorded, with `< 0.5 * latent_dimension` producing only
     `low_rank_alert`.
7. Retain the preregistered state-recovery, prediction, leakage, stability, and Milo gates. Do not
   add a kNN rescue rule, derive a threshold from prior results, or change direct-metric
   preprocessing.

## 2. Test-first versioned implementation

Implement each behavior as one red-green vertical slice.

1. Preserve `assess_latent_collapse()` and `redesign_run_contract()` as exact v1 behavior. Golden
   test the current contract digest
   `fe650e8c6275568a0c1a2174a9078d0990e6b4c2400443e6828f7544d8e8c26b`.
2. Add explicit v2 entry points, including `assess_latent_integrity_v2()` and
   `redesign_run_contract_v2()`.
3. Dispatch verification from stored schema, policy ID, and digest. Map legacy execution schema
   `mrtotalvi-convergence-execution-v2` to v1 because it has no policy field. Unknown combinations
   fail closed.
4. Register both historical payload variants labeled run-contract v1 by their exact payload
   digests. Do not rewrite or upgrade them.
5. Remove import-time binding to a live default contract in `metric_schema.py`; metric validation
   receives the selected contract adapter explicitly.
6. Add prospective schemas:
   - `mrtotalvi-redesign-run-contract-v2`;
   - execution, fit, aggregate, partial aggregate, and worker v3;
   - `mrtotalvi-redesign-metric-dictionary-v3`;
   - `mrtotalvi-latent-integrity-assessment-v2`.
7. Bind every v3 payload to:
   - `redesign_run_contract_schema_version`;
   - `redesign_run_contract_digest`;
   - `latent_integrity_policy_id`;
   - `latent_integrity_policy_digest`.
8. Seal complete `redesign-run-contract.json` and `latent-integrity-policy.json` payloads in every
   v3 run.
9. Replace v3 fit field `collapse` with representation-specific `latent_integrity`. Keep terminal
   status/reasons separate from effective-rank alerts.
10. Add representation-specific finiteness, exact nonconstant-variation, and
    all-posterior-scales-valid indicators. Existing summaries remain descriptive.
11. Rename aggregate semantics to `terminal_integrity_failed`,
    `effective_rank_screen_flags`, and `all_fits_converged_with_terminal_integrity`.
12. A rank alert never suppresses paired or cross-seed geometry. A representation-specific
    terminal failure suppresses only that representation. Fit non-convergence suppresses all
    geometry from that fit.
13. Invalid latent diagnostics serialize explicit no-calls rather than aborting the 48-fit grid.

Required behavior tests:

- exact v1 contract/digest/verdict replay;
- low-rank valid input fails v1 but passes v2 with an alert;
- invertible anisotropic rescaling changes rank without changing v2 terminal eligibility;
- nonfinite values, exact zero variation, invalid scales, and incomplete gradient coverage fail
  independently;
- `u` and factual `z` failures remain representation-specific;
- rank-alerted representations retain geometry;
- policy/schema/digest tampering and cross-version substitution fail closed;
- exact-grid validation still requires 48 unique cells.

## 3. Engineering and lineage gates

Use Python
`/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python`, with:

- `PYTHONPATH=src:.`;
- `LD_LIBRARY_PATH` prepended with the environment's `lib`;
- `CUDA_VISIBLE_DEVICES=""`;
- writable `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` under `/tmp`.

Before editing, reproduce the focused 31-test baseline. After implementation run:

```bash
env PYTHONPATH=src:. \
  LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib:${LD_LIBRARY_PATH:-} \
  CUDA_VISIBLE_DEVICES="" \
  NUMBA_CACHE_DIR=/tmp/rdx03-numba \
  MPLCONFIGDIR=/tmp/rdx03-mpl \
  /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python \
  -m pytest \
  tests/benchmarks/mrtotalvi/test_convergence.py \
  tests/benchmarks/mrtotalvi/test_convergence_runner_contract.py \
  tests/benchmarks/mrtotalvi/test_redesign_contract.py \
  tests/benchmarks/mrtotalvi/test_redesign_governance.py \
  -q -p no:cacheprovider
```

Then run the complete benchmark, MrTotalVI, and MrMultiVI suites; scoped Ruff; compilation of every
changed Python file; and `git diff --check`.

Recover the exact Python 3.14 interpreter previously used for pure contract validation. Do not
install or guess another interpreter. Failure to recover it blocks scheduler submission.

Independently verify the current authoritative RDX-01 lineage run:

`.scratch/mrtotalvi-v2-redesign/human-lineage-runs/20260731T081355Z-991ec740-b50f4e3a-e6ce6542`

The superseded
`20260726T124847Z-d976773e-607f16dd-c67d109b` run remains immutable historical
v1 evidence. The v2 authority was explicitly approved after the committed S06E
migration-inventory correction and independently proved exact scientific
equivalence before the RDX-03 fixture binding changed.

Rehash the preserved partial directory and prove it remained byte-unchanged. V1 artifacts must
retain their historical classifications; any v2 retrospective sensitivity output must remain
explicitly diagnostic and nonpromotable.

## 4. Independent verification and probe

1. Obtain an independent read-only review of the ADR, policy, compatibility dispatch, verifier,
   tests, and evidence inventory.
2. Close every high- and medium-severity finding.
3. Run one non-authoritative probe:

   ```bash
   python -m benchmarks.mrtotalvi.run_convergence_diagnosis \
     --fixtures mixed --rows B1 --seeds 0
   ```

4. Subset execution must automatically record `run_purpose=probe` and
   `rdx03_completion=prohibited`.
5. Verify the probe with sealed payloads, its code snapshot, policy digest, and current-repository
   compatibility verifier.
6. Confirm effective-rank alerts do not cause terminal failure or geometry no-calls and factual
   human DA remains locked.

## 5. Fresh 48-fit CPU grid

Stop for separate explicit scheduler authorization before `sbatch`.

After authorization:

1. Update `.scratch/mrtotalvi-v2-redesign/relaunch-rdx03-grid.sbatch` to prepend the environment
   `lib` directory and assert the exact Python/import path, CPU-only execution, policy/contract
   digests, RDX-01 lineage, frozen controls, and absence of another active RDX-03 job/process.
2. Reconfirm the old timed-out job is inactive. Preserve stale `.tmp`, failed, and partial runs.
3. Record HEAD, complete dirty status, source snapshot, environment, code/config/data/policy
   digests, and the exact grid.
4. Require exactly four fixtures by B1/B2/B3/D0 by seeds 0/1/2, for 48 unique cells.
5. Submit exactly once:

   ```bash
   sbatch --parsable .scratch/mrtotalvi-v2-redesign/relaunch-rdx03-grid.sbatch
   ```

6. Record the job ID and monitor `squeue`, stdout/stderr, fit-complete count, and `sacct`.
7. Do not edit `benchmarks/mrtotalvi/*.{py,json}` or `src/scvi/**/*.py` while the job is live.
8. Never resubmit automatically. Preserve a failed run and stop for adjudication; never splice
   partial or historical fits.

## 6. Seal and adjudicate RDX-03

1. Require scheduler state `COMPLETED`, exit `0:0`, and a final `sealed` event.
2. Require a new immutable non-`.tmp`, non-`-failed` directory and no `latest` pointer.
3. Verify `status=complete`, exactly 48 unique fits, v3 schemas, and exact contract/policy
   bindings.
4. Verify every result, representation, history, checkpoint, worker manifest, code snapshot,
   configuration, fixture, lineage identity, and checksum.
5. Run internal artifact verification, sealed-code-snapshot verification, and current-repository
   compatibility verification.
6. Independently recompute the aggregate and require exact agreement.
7. Confirm effective rank remains reported but causes neither terminal failure nor geometry
   no-call.
8. Treat non-convergence or terminal integrity failure as valid negative scientific evidence, not
   an execution blockage.
9. Close an independent read-only review before marking RDX-03 complete.

## 7. Downstream decision

1. If the grid is incomplete, unsealed, unverifiable, or lineage-invalid, record `blocked`,
   preserve it, and do not start RDX-04.
2. If the grid is sealed and valid, complete RDX-03 regardless of individual scientific failures.
3. Record D0's convergence/integrity decomposition. Rank alerts alone do not produce
   `d0_convergence_or_latent_integrity_failure`.
4. Hand every valid RDX-03 result to RDX-04; RDX-03 cannot issue `candidate` or terminal `stop`.
5. Execute RDX-04 as vertical red-green slices:
   - D1 exact TotalVI per-modality transform;
   - D2 sample-blind TotalVI `FCLayers` posterior trunk;
   - D3-D5 exact prior and observation-weighting axes;
   - sample-index invariance and registered technical-covariate sensitivity;
   - core, conditioning, and registered-residual gradient coverage;
   - centered hierarchy and target/cell chunk equivalence;
   - frozen legacy MrTotalVI and MrMultiVI oracle regressions.
6. Keep all D-series implementations package-private.
7. Later Stage A/B, Milo, and human-safety launches each require separate scheduler authority.
8. Apply existing thresholds without candidate-specific retuning. `candidate` or `stop` is legal
   only after a complete valid D1-D5 scientific screen and required safety gates.

## Constraints

- Preserve every pre-existing dirty-worktree change and immutable artifact.
- Do not reset, clean, overwrite, add a `latest` pointer, or splice runs.
- Do not change public APIs, defaults, modes, or checkpoint semantics.
- Keep factual W22-versus-W00 DA locked until the later candidate-freeze boundary.
- Do not use the network, install dependencies, use a GPU, commit, push, publish, or write an
  external environment without separate authority.
- Scheduler submission is not authorized by this plan.
