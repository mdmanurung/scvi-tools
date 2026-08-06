# Next steps — phased plan for MrTotalVI and CytoANVI

**Written 2026-08-06.** Companion to [TODO_REGISTRY.md](TODO_REGISTRY.md) (what is open) and
[2026-08-06-remaining-work.md](2026-08-06-remaining-work.md) (orientation). This file is the
*sequencing*: what to do, in what order, and which gates are yours.

Checklist items marked `[H]` need you, not an agent. Everything else an agent can execute.

---

## Read this first: three facts that shape the whole plan

**1. Task ids.** The operative checklist is
`docs/review-clear-execute/mrtotalvi-v2-redesign/tasks.md`, which uses **RDX-nn** ids. The older
`docs/plans/2026-07-25-mrtotalvi-v2-latent-da.md` uses **MRTL-nn** ids, still reads
`Status: not started`, and is superseded. They overlap; this plan uses RDX ids throughout. The
mapping, so nothing is double-counted:

| Legacy MRTL | Operative RDX | Note |
|---|---|---|
| MRTL-07 Milo bridge + calibrated DA | **RDX-07** | Same work. `TODO_REGISTRY.md` row 29 mislabels this as "MRTL-09"; corrected there |
| MRTL-08 adaptive screen + stop/go | **RDX-05** + **RDX-09** | Split into screen and selection |
| MRTL-11 release validation, docs, promotion | **RDX-06** + **RDX-10** | Split into public mode and final evidence |
| MRTL-09 LEMUR / miloDE / pseudobulk DE comparators | *no RDX equivalent* | Genuinely extra work. MRTL-10's own text calls it optional. **If `T0.1` drops DE from the v1 claim, this becomes moot** |
| MRTL-10 publication-scale simulation + macaque | *no RDX equivalent* | Overlaps `P2-005`; data-blocked |

**2. Execution order is not numeric.** `tasks.md` runs
**RDX-03 → RDX-04 → RDX-05 → RDX-07 → RDX-08 → RDX-09 → RDX-06 → RDX-10.**
RDX-06 (public mode) is late on purpose: it exposes whatever RDX-09 selects, so it cannot run first.

**3. You are signing at least five separate scheduler authorizations, not one.** Issue 12 requires
"separate authority for every later scheduler launch", and the redesign handoff (`:67`, `:93`)
blanket-prohibits submitting scheduler jobs without it. **Neither packet enumerates the launch
points** — the table below is inferred from which phases train models, not quoted. Confirm it
against the packets before budgeting against the number.

| # | Launch | Scale |
|---|---|---|
| 1 | RDX-03 authoritative grid (Issue 09) | 48 fits, ~44 h CPU |
| 2 | RDX-05 Stage A | B1–B3 + D0–D5, seed 0, 3 instances per scenario |
| 3 | RDX-05 Stage B | B1/B2 + ≤2 survivors, seeds 0–2, 10 instances per scenario |
| 4 | RDX-08 canonical human screen | B0–B3 + ≤2 redesigns, seeds 0–2 |
| 5 | RDX-10 factual human Milo | once, only on a `candidate` verdict |

(RDX-07's Milo work is R/miloR and may not need the scheduler; treat as a sixth if it does.)
Issue 09 authorizes **only** launch 1. It explicitly grants no authority for any of the others.

---

# Track A — MrTotalVI / MrMultiVI (active)

## Phase A0 — Unblock the launch

Nothing else in Track A can start until this closes. Working directory
`.scratch/mrtotalvi-rdx03-latent-integrity-v2/`.

- [ ] **A0.1 Re-run `launch/pre_submit.py --read-only` on a CPU node.** Not a GPU node — there
  `ldd` resolves `/usr/lib64/libcuda.so.580.159.04` and the preflight fails
  `snapshot runtime identity drifted` for a reason that is not real. Expect `ready: true`: the only
  live-repo binding is `expected_head` (`pre_submit.py:2033`), still `cb343ca1`.
- [ ] **A0.2 `[H]` Decide `V2-LAND` — commit the v2 body now, or after RDX-03-11.**
  *Recommendation: commit now.* Rationale: committing moves HEAD off the pinned `expected_head`,
  which forces one contract edit + re-review — but A0.1 is being re-run anyway, so the marginal cost
  is one field, not a rebuild. Committing mid-flight (after authorization) is strictly worse. The
  standing risk is real: `src/scvi/external/__init__.py` imports `combine_mrtotalvi_seed_results`
  from the **untracked** `_seed.py`, so a `git clean -fdx` breaks the package import and destroys
  ~30 benchmark modules, 23 test files, ADRs 0007/0008/0009 and `_counterfactual.py` (75 KB).
- [ ] **A0.3** If A0.2 = commit: update `expected_head` in `rdx03-launch-contract.json`, re-render,
  re-run A0.1, and re-review. One edit, one field.
- [ ] **A0.4 Run the Issue 08 probe.** `python -m benchmarks.mrtotalvi.run_convergence_diagnosis
  --fixtures mixed --rows B1 --seeds 0`, CPU-only. Script ready at
  `/exports/para-lipg-hpc/mdmanurung/rdx03-preflight-diag/probe.sbatch`. Acceptance, verbatim from
  `issues/08-verify-probe.md`:
  - [ ] records `run_purpose=probe` and prohibits RDX-03 completion automatically
  - [ ] seals contract + policy payloads, code snapshot, environment, configuration, checksums
  - [ ] internal-artifact, sealed-code-snapshot and live-repository verification all pass
  - [ ] effective-rank alerts cause neither terminal failure nor geometry no-calls
  - [ ] exact-grid validation refuses to classify the one-cell probe as the 48-fit aggregate
  - [ ] factual human DA neither computed nor inspected
  - [ ] preserved immutably, no `latest` pointer, not spliced into later evidence
- [ ] **A0.5 `[H]` Issue 09 — authorize or refuse exactly one submission.** Requires Issues 01–08
  complete with no unresolved high/medium finding. The authorization must be explicit, dated, and
  limited to one fresh 48-fit submission; it grants no authority for retry, resubmission,
  dependency installation, GPU, factual human DA, commit, push, or any later launch. A refusal is a
  valid outcome and must record the reason.
  **Freeze the working tree from here until the job starts** — the authorization payload is built
  from a passing read-only result (`pre_submit.py:2276`) and the job re-checks it at start.

## Phase A1 — RDX-03: complete the convergence diagnosis

- [ ] **A1.1 Issue 10 — submit and monitor exactly one grid.** Through `launch/pre_submit.py`, never
  `relaunch-rdx03-grid.sbatch` directly (stale pre-ADR-0009 path). 4 fixtures × B1/B2/B3/D0 ×
  seeds 0/1/2 = 48 unique cells, ~44 h. Record the job id. Monitor without automatic resubmission.
  Do not edit frozen benchmark or package sources while it is live.
- [ ] **A1.2 Issue 11 — seal and independently recompute.** Requires `COMPLETED`, exit `0:0`, 48
  unique fits with complete result/representation/history/checkpoint/worker quadruples, v3 schema on
  every artifact, all three verifiers passing, durable postflight `matched`, and an independent
  aggregate recomputation agreeing exactly.
- [ ] **A1.3 `[H]` Issue 12 — review and hand off to RDX-04.** RDX-03 may issue neither `candidate`
  nor terminal `stop`. Record D0 convergence and terminal-integrity outcomes **separately** from
  low-rank alerts, and discard no result because of a rank alert.
- [ ] **A1.4** Unblock `P2-006` — the `freeze_prior_after_init` docstrings in
  `mrtotalvi/_model.py:102-105` and `mrmultivi/_model.py` still say "no differential-abundance
  stability improvement is established", contradicting D-041. Fix once A1.2 seals.

**Contingency, decided in advance:** if the grid fails or times out, it is preserved as `blocked`
evidence. **No retry, no splice, no second submission** without a fresh Issue-09 authorization. The
2026-07-26 run that reached 45/48 is the precedent — it is not promotable regardless of how close it
got.

**The question RDX-03 answers** (`tasks.md:73`): *does D0 already pass all downstream gates?* If yes,
`plan.md:108` says retain the current `sample_blind` and **build no redesign at all** — which
collapses RDX-04 and most of RDX-05. That is why the phases below are contingent, not certain.

## Phase A2 — RDX-04: test-first encoder ablations (D1–D5)

Pure CPU development. No scheduler, no grid results needed.

| Candidate | Definition (`plan.md:64-68`) |
|---|---|
| D1 | D0 + exact TotalVI per-modality input normalization only |
| D2 | TotalVI-normalized, TotalVI-`FCLayers`, sample-blind encoder; frozen initialized VampPrior |
| D3 | D2 + trainable MoG |
| D4 | D2 + sample-equal weighting |
| D5 | D2 + trainable MoG + sample-equal weighting |

- [ ] RED: one behavior test for exact TotalVI per-modality input transformation
- [ ] GREEN: implement the reusable transformation without changing old branches
- [ ] RED: one behavior test for the TotalVI-`FCLayers` sample-blind posterior
- [ ] GREEN: implement the new package-private encoder
- [ ] sample-index invariance + technical-covariate sensitivity tests
- [ ] core, conditioning, and all-residual-row gradient tests
- [ ] D1–D5 exact-axis configuration tests
- [ ] centered hierarchy + target/cell chunk equivalence tests
- [ ] legacy MrTotalVI and MrMultiVI frozen-oracle tests still pass
- [ ] refactor only while all new tests are green; scoped Ruff + focused tests pass
- [ ] `EncoderUZ.forward()` and all old encoder branches left unchanged (`plan.md:119`)
- [ ] D1–D5 constructed **only** through benchmark-private configuration until selection
  (`plan.md:120`)

> **Schedule lever, with the catch.** The governance constraint on D1–D5 is *exposure*
> (package-private until selection) and *scheduler authority* — **not** implementation order. So
> RDX-04 could legally start today, in parallel with A0/A1. The reason not to is economic, not
> procedural: if RDX-03 shows D0 passing every gate, `plan.md:108` says build no redesign, and this
> entire phase is wasted. **Starting RDX-04 early is a bet that D0 fails.** Worth taking only if you
> already expect it to.

## Phase A3 — RDX-05: adaptive known-truth screen

`[H]` **Authorizations 2 and 3.** Stage A and Stage B are separate launches.

- [ ] extend each frozen scenario to the paired-donor DA fixture
- [ ] test independent truth / training / evaluation RNG streams
- [ ] atomic staged workflow + exact-grid validation
- [ ] `[H]` **Stage A**: B1–B3 and D0–D5, seed 0, three instances per scenario
- [ ] apply hard disqualification gates **without retuning** — contract, convergence, nonfinite,
  collapse, leakage, factual-`z` failures
- [ ] select at most two redesigns through the frozen rule
- [ ] `[H]` **Stage B**: B1/B2 plus survivors, seeds 0–2, ten instances per scenario
- [ ] verify every immutable run manifest and code/config/data hash; seal the Stage A/B aggregate

## Phase A4 — RDX-07: Milo bridge

The largest piece of genuinely new engineering. `benchmarks/mrtotalvi/README.md` states plainly:
*"The current runner scores latent and decoder recovery. It does not implement Milo inference,
calibrated DA FDR/power, LEMUR, miloDE, or pseudobulk DE."* Requires an R/miloR toolchain.

- [ ] small known-result miloR fixture
- [ ] cell-order-locked `SingleCellExperiment` export with PCA, B0/B1, B2 `u`/`z`, eligible
  redesign `u`/`z`; assert named reduced dims and exact cell IDs
- [ ] frozen settings: `buildGraph(k=30, d=20)`,
  `makeNhoods(prop=0.1, k=30, d=20, refined=TRUE, refinement_scheme="graph")`,
  replicate column `donor_timepoint`
- [ ] reorder design rows exactly to neighborhood-count columns
- [ ] primary paired GLMM `~ timepoint + (1|donor)`, graph-overlap FDR, Fisher solver, REML, TMM,
  `fail.on.error=FALSE`; require **zero** failed or NA primary fits
- [ ] W22-minus-W00 sign and donor-pairing tests; convergence / NA-fit / separation diagnostics
- [ ] run null, DE-only, DA-only, mixed, rare-state, imbalance, continuous, confounded scenarios
- [ ] compute FDP, power, localization, seed-stability endpoints; stop a representation on any
  failed/NA primary fit; seal the aggregate

## Phase A5 — RDX-08: canonical human non-inferiority screen

`[H]` **Authorization 4.**

- [ ] confirm RDX-01 source/cell/feature/split hashes unchanged
- [ ] `[H]` train B0–B3 and ≤2 redesigns at seeds 0–2
- [ ] record held-out prediction, `u`, factual-`z`, rank, runtime, memory endpoints
- [ ] frozen within-donor timepoint-label permutations
- [ ] frozen human-geometry null, DA-only, DE-only perturbations
- [ ] run Milo **without opening factual W22-minus-W00 results**
- [ ] apply every frozen non-inferiority and DA-safety gate; seal or `stop`

## Phase A6 — RDX-09: selection

- [ ] disqualify every hard-gate failure
- [ ] apply the frozen localization / predictive-loss / stability / complexity tie-break
- [ ] recompute selection with result order permuted
- [ ] recompute selection using an independent implementation
- [ ] freeze one candidate configuration/code hash, or issue `stop`
- [ ] record why every nonselected candidate failed or lost the tie-break
- [ ] complete **without inspecting factual human DA**

## Phase A7 — RDX-06: public mode (only on a `candidate` verdict)

- [ ] D1 passed → expose only `sample_blind_scaled`; a D2 family passed → expose only
  `sample_blind_totalvi`; **only D0 passed → add no new public mode**
- [ ] constructor, metadata, topology-specific loading tests; refuse unknown and cross-topology
  semantic overrides; save/load and re-save round trips
- [ ] update ADR, docstrings, user guide, API docs, examples, changelog
- [ ] full legacy / v2 / MrMultiVI regressions

## Phase A8 — RDX-10: final evidence

`[H]` **Authorization 5** (factual Milo, only if `candidate`).

- [ ] `[H]` if `candidate`: run factual paired W22-minus-W00 Milo **once, without retuning**;
  if `stop`: skip factual human DA entirely
- [ ] scoped Ruff + compilation; full MrTotalVI and MrMultiVI tests; benchmark and optional Zarr
  tests; docs and wheel build; no-dependency wheel import smoke; non-GPU repository suite
- [ ] read-only compatibility / API / lineage / statistical review; address findings, rerun gates
- [ ] verify all final manifests and hashes after sealing; update tracker and issue evidence
- [ ] write the final `candidate` / `stop` / `blocked` report

## Track A — work that is NOT blocked by any of the above

> **Source-freeze rule — read before starting any of these.** From A0.5 (authorization) until
> A1.1's job actually starts, **nothing under `src/scvi/**` or `benchmarks/mrtotalvi/**` may be
> edited.** Those trees are exactly what the launch snapshot's `source_config_manifest` globs
> (`pre_submit.py:1804`), and the authorization payload binds the working-tree state. Documentation
> outside those trees is always safe. Items below are tagged `[free]` (safe any time) or
> `[frozen A0.5→A1.1]`.

- [ ] **`[H]` `T0.1` — decide whether `store_lfc` DE stays in the v1 claim.** Independent of the
  entire RDX programme. Evidence is one-sided and assembled: all 12 cell types negative Spearman
  (−0.126 to −0.008); IFN direction correct 17.9–42.9%, all below the 50% of guessing; no rescue
  cell type; and F-036 shows the **reference MRVI implementation fails identically** (ρ=−0.138).
  *Recommendation: drop DE from the v1 claim, keep `store_lfc` as a documented API with its
  disclaimer.* "A property of eps-space DE as a method" is defensible; leaving DE in the claim is
  not, once a reviewer runs the pseudobulk comparison. **Dropping DE would also make MRTL-09
  (LEMUR/miloDE comparators) moot** — MRTL-10 already calls it optional.
- [ ] **`[H]` `FINDINGS-STALE`** — spot-check five recomputed percentages in
  `.scratch/mrtotalvi-z-vs-totalvi/FINDINGS.md`. Arithmetic is the assistant's, not the original
  analyst's. No MrTotalVI value changed; no conclusion affected.
- [ ] **`ANNBATCH`** — MrTotalVI-specific AnnBatch datamodule with protein streaming, sample
  registry, save/load reconstruction, parity tests. Upstream's generic datamodule emits RNA `X`,
  batch/label/sample, size factor and covariates but **no protein tensor or registry**.
  **Split this in two:** *implementation* is `[frozen A0.5→A1.1]` (it lands in `src/scvi/**`);
  *integration into the benchmark path* waits for **A1.2 (RDX-03-11) to seal**. RDX-03 must run on
  the validated in-memory AnnData loader — swapping the loader mid-programme would change the very
  source set the grid is pinned to.
- [ ] **`[free]` `CLAUDE.md` rewrite** — highest leverage per minute in this file. It is auto-loaded
  into every session and still says `Active branch: feat/cytoanvi`, frames the repo as a CytoANVI
  fork, and never mentions MrTotalVI. It mis-orients every new session before any tracker is read.
- [ ] **`[free]` `PREFLIGHT-HOST-DOC`** — one line in `launch/README.md`: the preflight must run on a
  CPU node. Outside both frozen trees. Cost of its absence on 2026-08-06: several hours of
  false-alarm diagnosis.
- [ ] **`[free]` `DOC-CONFLICTS`** — five recorded tracker contradictions, fixed opportunistically.
- [ ] **`[frozen A0.5→A1.1]` `P2-006` docstrings** — `mrtotalvi/_model.py:102-105` and
  `mrmultivi/_model.py`. Inside `src/scvi/**`, so this is the one "documentation" item that is *not*
  free. Also gated on A1.2 for its content (see A1.4).
- [ ] **`[free]` `ORPHAN-0803`** — confirm who launched the 45/48 run and on what. Read-only
  forensics. Not a gate, but whatever did it bypassed the Issue-09 mechanism entirely, which is
  worth knowing before authorizing.

## Track A — blocked on data or compute

Not schedulable. Listed so they are not mistaken for actionable.

| Item | Blocked on | Unblocks |
|---|---|---|
| `P2-005` macaque CITE-seq replication | macaque panel + homolog map, never frozen | The single most load-bearing gap — everything measured so far is **one** cohort at NMI(donor,batch)=0.796 |
| `T0.2` DA reliability on a second cohort | same | Donor-specific integration cannot be tested on the schisto cohort at all |
| `P1-006` DA instability root cause | GPU + real schisto data | MrTotalVI std=9.46 vs MrMultiVI 0.126. Narrowed: representation instability is a *contributor, not the mechanism* |
| `P2-008` MoG-vs-VampPrior default | original T/NK data, currently unlocated | The `u_prior="mog"` default is unsupported until re-run |

---

# Track B — CytoANVI (shelved 2026-07-12)

Nothing here is actionable without a revival trigger. The plan is therefore: a cheap revival
checklist, and the triggers that would fire it.

## B0 — Revival checklist (run this first, whatever the trigger)

- [ ] **Does it still build?** `pytest tests/cytoanvi/ -v` and
  `pytest tests/benchmarks/test_cytoanvi_smoke.py -v`. The track has been untouched since
  2026-07-12 while `src/scvi/**` kept moving — `BATCH-EMBEDDING` landed as recently as today. This
  is step 1 because every other item assumes a working package.
- [ ] Confirm `src/cytoanvi/` imports cleanly and `ruff check` is still clean.
- [ ] Reconcile `benchmarks/ANALYSIS_MANIFEST.md` and `publication_manifest.json` against whichever
  gate the trigger reopened.

## B1 — Revival triggers

| Trigger | Unblocks | Current state |
|---|---|---|
| Independent expert-gated **Panel-2 labels** | B3 accuracy claim | B3 reports 0.671±0.008 *concordance* — model-vs-model agreement, **not** ground-truth accuracy. The ≥0.80 gate is not met and cannot be honestly evaluated without these labels |
| External **novel-cell-type dataset** | B5 novelty AUROC | 0.484±0.019 — below chance. Diagnostic (jobs 25149032/33/34) showed CytoANVI-latent kNN-OOD at 0.906, so the **TTA method is the failure, not the latent** (F-013/D-040) |
| Real **case/control cytometry cohort** | B4, B6, λ default | Both are plumbing-only today (pseudo-batch split); no biological validation is possible |
| Upstream **`mapqc`** IndexError patch | B9 | `mapqc` 0.1.1 raises in `_get_per_cell_filtering_info` (`mode().iloc[0]` on an empty result). Roider is *additionally* blocked by `ValueError: Category 31 not found in source registry` — missing `extend_categories=True` (F-039), which is **ours to fix** and could be done now |
| Public **PyPI upload** planned | `scvi/` namespace collision (L-035) | Deferred per D-011; decide Replace vs Coexist vs upstream-PR first |
| Preprints posted | real DOIs in docs | Currently "in preparation" |
| Unshelving alone | `CYTO-NITS`, coarse-Leiden, job 25211796 | See B2 |

## B2 — Work that needs only unshelving, no new data

- [ ] `CYTO-NITS` — six residual items in `.scratch/cytoanvi-review-fixes.md`: vectorize the O(n³)
  transitivity check in `validate_reachability_matrix`; vectorize one-time init loops; split
  `run_tree_arches_pipeline` (14 params, `hierarchy.py:288-390`); shorten `__init__` /
  `prepare_query_anndata`; `key in d` idiom nits; hoist `batch_size=256` to a constant.
  **All P0/P1 correctness findings are closed** and `cytoanvi-review-patch-plan.md` is COMPLETE —
  this track is in better shape than the abandoned review threads suggest.
- [ ] Clear the stale `[~]` at `.scratch/cytoanvi-scientific-review-fixes.md:88` (P4-A) — its run
  finished; only the marker is wrong.
- [ ] Coarse-Leiden B3/B5 speed lever (L-041): six `phase3{a,b}_{b3,b5}coarse_*.slurm` scripts with a
  `__RES__` placeholder, never submitted. Calibration job **25211796 has been pending since
  session 67** — check whether it is still queued or was cancelled, and cancel it if the track stays
  shelved.

---

## The one strategic fork worth stating

`T0.1` (drop DE from the v1 claim) and the RDX-03→10 v2 redesign are **independent**. v1 can ship on
the current `legacy` default — which remains the default precisely because v2 is *"not yet
demonstrated to outperform legacy MrTotalVI"* — while the v2 programme proceeds on its own clock.

The RDX programme as specified is eight phases, five scheduler authorizations, and a new R/miloR
bridge. That is the right shape for a method paper claiming DA. It is a lot of machinery to finish
before publishing anything.

**Recommendation:** treat them as two deliverables. Take the `T0.1` decision now, scope the v1 claim
to integration quality and cell-type-level resolution (which `benchmarks/ANALYSIS_MANIFEST.md`
already rates Ready / Ready-with-guardrail), and let RDX-03 run without the paper waiting on it.
The alternative — one combined claim gated on RDX-10 — is defensible but months longer, and every
phase can return `stop`.
