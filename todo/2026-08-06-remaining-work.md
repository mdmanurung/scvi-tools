# Remaining work — orientation brief

**Written 2026-08-06.** Read this before `.claude/last-session.md`, which stops at 2026-07-28 and is
wrong about its own top item. Row-level detail lives in [TODO_REGISTRY.md](TODO_REGISTRY.md).

Two tracks share this repo. **MrTotalVI/MrMultiVI is the active one.** CytoANVI was shelved
2026-07-12; its open items are all data-blocked and are marked `shelved` with revival conditions.

---

## Amendment — later on 2026-08-06, after `BATCH-EMBEDDING` landed

`BATCH-EMBEDDING` edited three files in `src/scvi/external/mrtotalvi/` (`_components.py`, `_model.py`,
`_module.py`) after this brief was written. That **invalidates the "preflight is green, go" framing of
step 2 below**, but less severely than the registry first recorded:

- The sealed snapshot `f1b51341…` is **still valid**. `diff -rq` against it shows exactly those 3
  files differing; `benchmarks/` is byte-identical.
- The grid **never reads the working tree**. `relaunch-rdx03-grid.sbatch:80,93,96` sets
  `PYTHONPATH=$SNAPSHOT_ROOT/src`, `GIT_WORK_TREE=$SNAPSHOT_ROOT`, `cd $SNAPSHOT_ROOT`. The run
  executes from the frozen snapshot, so the batch-embedding feature is simply not in it — correct,
  since it is out of RDX-03's scope.
- What *is* stale is the recorded preflight **result** from job 25374994: the live-tree
  `complete_dirty_status_sha256` (`fe4dcf97…`) moved when those files were edited.
- Reading `_validate_runtime` (`pre_submit.py:2017-2053`) confirms the preflight should still pass:
  the only live-repo binding is `expected_head` (`:2033`), still `cb343ca1` since nothing was
  committed. The dirty-status digest is merely format-checked (`:2035-2039`), and the live
  `source_config_manifest` is compared only to itself and to the contract's schema *string*
  (`:2040-2053`) — never to the snapshot's pinned digest.

**So there is a new step 0: re-run `pre_submit.py --read-only` on a CPU node** to get a current
result. The contract almost certainly does not need re-rendering. Where the dirty-status digest does
bind is later — the authorization payload is built from a passing read-only result (`:2276`) and the
job re-checks it at start, so **freeze the working tree between authorizing and submitting.**

---

## Do next

The entire MrTotalVI programme is chained behind three steps, in this order. Working directory:
`.scratch/mrtotalvi-rdx03-latent-integrity-v2/`.

1. ~~`RDX-03-08a` — re-baseline the status evidence.~~ **DONE 2026-08-06 — and it turned out nothing
   was broken.** The preflight first failed `snapshot runtime identity drifted` (exit 2) from
   `res-hpc-gpu14`. Cause: `native_runtime` went 778 → 779 files, +96,284,520 bytes — a byte-exact
   match for `/usr/lib64/libcuda.so.580.159.04`, which `ldd` resolves on GPU nodes only. Runner
   digests, conda_meta (449/449) and the env-var dict were all identical. Re-run on CPU node
   `res-hpc-exe043` (job 25374994, `libcuda present: NONE`):

   ```
   ready: true      exit 0      head: cb343ca1…      grid_size: 48
   complete_dirty_status_sha256: fe4dcf97…    active.queue_rows: []
   claim_available: true    factual_human_da: locked_not_computed_or_inspected
   ```

   So snapshot `f1b51341…` is **valid**, and the working tree — including this reconciliation's
   `todo/` files — **passes the repository gate as it stands.** There was never a manual
   re-baseline step; the dirty-status digest is computed dynamically at run time.
   **Always run the preflight on a CPU node** — the contract is `cpu_unless_separately_authorized`.

2. **`RDX-03-08b` — run the probe (Issue 08).**
   It was never run. `evidence/` holds directories for issues 01–07 and none for 08, and every
   acceptance box in `issues/08-verify-probe.md` is unchecked. What the last Codex session left is
   *preparation*: source snapshot `launch/source-snapshots/f1b51341…`, `state: "prepared"`,
   288 files, built 2026-07-31 21:54 after fixing three snapshot-builder bugs (PYTHONPATH
   normalisation replacing only the first occurrence; the runner manifest not projecting staging
   paths onto the final digest root; file-mode `0444` stripping the executable bit before hashing).
   The probe is one bounded mixed-fixture/B1/seed-0 run that must record `run_purpose=probe` and
   stay permanently ineligible to complete RDX-03.

3. **`RDX-03-09` — your authorization (Issue 09, HITL).** See below.

Everything after that (`RDX-03-10/11/12` → `RDX-04` D1–D5 screen → `RDX-05…10` → `MRTL-09/10/11`)
is blocked on step 3 and has not started.

**Two things you can stop worrying about.** The `latent_collapse` gate concern that
`.claude/last-session.md` lists as its #1 open item was **adjudicated by ADR-0009 on 2026-07-31**:
effective rank below 0.5×dim is now `low_rank_alert`, not terminal, and terminal integrity is
representation-specific. And the 42/48 grid that died when SLURM allocation 25325917 expired is
preserved read-only and is not promotable — do not try to splice or resume it.

---

## Decisions only you can make

These carry status `ready-for-human` in the registry — `grep ready-for-human todo/TODO_REGISTRY.md`
finds every item that is actionable right now but needs you rather than an agent. Note that
Issue 09 is *not* blocked on anything external; it is waiting only on you.

| Decision | What it unblocks | Context |
|---|---|---|
| **Issue 09: authorize exactly one `sbatch`** of the 48-fit grid (~44 h, 4 fixtures × B1/B2/B3/D0 × seeds 0/1/2) | Everything downstream | Issue 07 was APPROVED 2026-07-31 and records "`sbatch` calls: zero". `launch/authorizations/` holds only the placeholder scheme. The earlier blanket "i approve" is explicitly scoped as conditional on Issues 01–09 all closing, so it does not substitute. No retry, no second submission. Go through `launch/pre_submit.py` — **not** `relaunch-rdx03-grid.sbatch` directly, which is the stale pre-ADR-0009 path |
| **When to commit the uncommitted v2 body** (`V2-LAND`) | Removes the standing risk below | Committing moves `expected_head` off `cb343ca1` and rewrites the whole status baseline. Do it *before* step 1 or *after* `RDX-03-11` — not mid-flight |
| **Whether DE stays in the v1 claim** (`T0.1`) | Scope of the publication claim | See the evidence box below — it is one-sided |

### T0.1 evidence — assembled 2026-08-06 so the decision is a short read

The question is whether eps-space `store_lfc` DE stays in the v1 publication claim. The evidence
already in the repo points one way, and it is stronger than "not ready":

| line of evidence | result |
|---|---|
| Per-cell-type concordance vs stratified pseudobulk, all 12 cell types (F-037) | MrTotalVI Spearman ρ **−0.126 to −0.008 — every value negative**; MRVI −0.098 to +0.027; MrMultiVI −0.037 to +0.012 |
| IFN gene × cell-type direction called correctly (F-037) | MRVI 17.9%, MrTotalVI 19.9%, MrMultiVI 42.9% — **all below the 50% you would get by guessing** |
| Is there a rescue cell type? (F-037) | No. Classical monocytes — the strongest pseudobulk IFN signal — give MRVI ρ=−0.098 and **0/7** IFN genes correct |
| Is this an MrTotalVI defect? (F-036) | **No.** The reference MRVI implementation is itself anti-concordant (ρ=−0.138). The limitation is inherited from the method, not introduced here |
| Ground truth (F-029) | Donor pseudobulk shows IFN **up** at W22; the model narrative said down. The F-020 biological narrative is retracted, and the cross-model IFN "concordance" (F-026) was a convergent artifact |

`benchmarks/ANALYSIS_MANIFEST.md` already draws the conclusion: *"For temporal DE, use PyDESeq2
pseudobulk. The model contribution is in integration quality and cell-type-level resolution, not
temporal DE."*

**Recommendation: drop DE from the v1 claim; keep `store_lfc` as a documented API with its existing
disclaimer.** The framing that matters for a paper is F-036 — because the reference implementation
fails the same way, this is a property of eps-space DE as a method, not a bug in this work. That is
a defensible thing to state, and a much weaker position to defend if DE is left in the claim and a
reviewer runs the pseudobulk comparison themselves. Your call; the numbers are above.

### Standing risk: weeks of work are uncommitted

778 lines of `src/` diff, plus untracked `_counterfactual.py` (75 KB), `_seed.py`,
`benchmarks/mrtotalvi/` (30 modules), `tests/benchmarks/mrtotalvi/` (23 test files),
ADRs 0007/0008/0009, two `docs/plans/` trackers, `docs/review-clear-execute/`, and
`src/scvi/_skills/`. **`src/scvi/external/__init__.py` imports `combine_mrtotalvi_seed_results` from
the untracked `_seed.py`** — a `git clean -fdx` would both break the package import and destroy the
work. This is finished engineering, not a half-done branch: Codex sealed it 2026-07-25 as
package-ready with 964 CPU tests passing, deliberately **not** promoted
(*"not yet demonstrated to outperform legacy MrTotalVI"*). `legacy` remains the default.

---

## Unexplained — and it got much further than it first looked

`.scratch/mrtotalvi-v2-redesign/convergence-runs/20260726T161404Z-…-failed/` reached **45 of 48
fits**. Complete result/representation/worker/checkpoint quadruples exist for all three seeds of
`mixed`, `unequal_cells` and `sealed_500` across B1/B2/B3/D0, plus `canonical_human` B1/B2/B3. The
only missing cell is `canonical_human_if_available--D0` × seeds 0/1/2 — and even its seed-0
checkpoint had already trained to `best-epoch=0399`.

What killed it was the integrity guard, not the science: the worker died in
`_verify_live_sources_against_manifest` with `ValueError: Live worker source differs from snapshot
for benchmarks/mrtotalvi/__init__.py`, because the Jul 30–31 latent-integrity work modified that
file after the run's 2026-07-26 snapshot. The check did its job.

Its config/env/code manifests are dated 2026-07-26 while results and `failure.json` are 2026-08-03,
so a 07-26 run was resumed off cached checkpoints and published as failed on 08-03. `status: failed`,
`evidence_tier: pilot_cache`, unsealed, and under the **pre-ADR-0009 policy** — so it is not
promotable regardless of how close it got, and the governance packet forbids splicing it into a
later run. `sacct` shows no matching SLURM job on 08-03 (only unrelated `s2_exv_*` schisto jobs);
the one allocation spanning that moment is `mrvi_cp` 25349712 (07-31 → 08-04, TIMEOUT, highmem).
**Still open: who launched it, and on what.** Not a gate — but worth knowing, because whatever did
it bypassed the Issue-09 mechanism entirely.

---

## Blocked on data or compute

- **`P2-005` macaque CITE-seq replication** — the macaque panel and homolog map were never frozen.
  This is the single most load-bearing gap: everything measured so far is one confounded cohort with
  NMI(donor, batch) = 0.796, so donor-specific integration cannot be tested on it at all (`T0.2`).
- **`P1-006` DA instability** (MrTotalVI std=9.46 across 3 seeds vs MrMultiVI 0.126). Narrowed in
  session 68 — representation instability is *a contributor, not the mechanism* (18–30%
  neighbourhood-stability gain vs 78% DA-variance reduction). Needs GPU + real schisto data.
- **`P2-008`** — the artifact that set `u_prior="mog"` is unrecoverable, and it measured
  UMAP/centroid geometry, which does not govern Leiden's kNN graph. Until re-run, the u-space
  clustering default is unsupported.
- **CytoANVI, entirely.** No panel-2 ground truth (B3), no external novel-cell-type dataset (B5,
  currently 0.484 — below chance), no real case/control cohort (B4/B6), upstream `mapqc` bug (B9).
  The abandoned Codex review threads are in better shape than they look: all P0/P1 correctness
  findings are closed and `cytoanvi-review-patch-plan.md` is COMPLETE — six cosmetic/perf nits
  remain (`CYTO-NITS`).

---

## Source-of-truth map

Four trackers disagree. Use this table; the contradictions are recorded as `DOC-CONFLICTS` rather
than silently patched, so nothing is lost when those files are next edited.

| Question | Trust | Do **not** trust |
|---|---|---|
| What is open | `todo/TODO_REGISTRY.md` (this reconciliation) | anything else |
| RDX-03 issue state | `.scratch/mrtotalvi-rdx03-latent-integrity-v2/issues/` + `evidence/` | `docs/plans/2026-07-2*.md`, whose headers both still read `Status: not started` although RDX-00/01/02 are sealed |
| Redesign checklists | `docs/review-clear-execute/*/tasks.md` | the `docs/plans/` trackers |
| The rank gate | ADR-0009 (2026-07-31) | `.claude/last-session.md` §3, superseded three days later |
| Model capability readiness | `benchmarks/ANALYSIS_MANIFEST.md` internal-use matrix | `benchmarks/mrtotalvi/README.md`, which describes only a narrower pilot |
| D-041 (VampPrior + frozen prior cuts DA variance) | **Both, they answer different questions**: `.living/decisions.md` says CONFIRMED — true as a measurement (std 9.46 → 0.192); `benchmarks/ANALYSIS_MANIFEST.md` says REFUTED FOR PROMOTION — also true, the artifacts came from a dirty checkout without data/feature/environment/config hashes. Confirmed, not reproducible under the immutability standard | either one alone |
| Stock-TOTALVI effective rank | the 3-seed values (15.49, 17.08) at `.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md:50` | the unmarked n=1 values (15.62 ± 0.66, 17.28) still sitting at `:113` and `:259`, which a review closure log wrongly claims were fixed (`FINDINGS-STALE`) |

`.living/INDEX.md` (2026-07-08), `.living/log/LOG_REGISTRY.md` (session 23),
`.living/findings/FINDINGS_REGISTRY.md` (2026-07-13) and `.claude/last-session.md` (session 68,
2026-07-28) all predate work that has since happened. Treat them as history, not status.
