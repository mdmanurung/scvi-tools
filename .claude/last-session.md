# Session 68 — MrTotalVI u/z vs TOTALVI vs Multigrate; three review passes; RDX-03 lost

**Date**: 2026-07-27 → 2026-07-28
**Branch**: main
**Trigger**: "pick up from last work on MrTotalVI from codex" — found a live 48-fit RDX-03 run and
a user question about which embedding to cluster on for sample-robust cell typing.

*(Session 67 notes preserved at `.claude/last-session-67.md.bak`.)*

---

## 1. Work completed

### RDX-03 48-fit run — FOUND LIVE, NOW LOST

Discovered running inside a Codex `bwrap` sandbox on `res-hpc-exe029` (orchestrator PID 675127),
42/48 fits sealed. Its SLURM allocation (`25325917`) ends **2026-07-28 13:06:59** and the extension
was never applied. `canonical_human/B3/seed0` has run **15h46m** and is only at epoch 244/400, with
five fits after it. **The run dies today with 42/48 and nothing promotable** — no resume path
(`run_convergence_diagnosis.py` always `mkdtemp`s fresh), and RDX-03's own review forbids splicing.

`relaunch-rdx03-grid.sbatch` is written and ready (7-day `all` partition, 12 CPUs, detached) if a
rerun is wanted. **But see §3 — the grid's headline gate is now in question.**

### New analysis: `.scratch/mrtotalvi-z-vs-totalvi/` (16 scripts, FINDINGS.md)

Compared stock TOTALVI, MrTotalVI (MoG/VampPrior × u/z) and Multigrate on schisto 10k human
CITE-seq (97,954 cells, 10 donors × 4 timepoints, 24 cell types).

- **3-seed matched TOTALVI baseline trained** (GPU, early-stopped 141/87/108) replacing an n=1
  baseline; effective rank **17.08 ± 0.53** vs MrTotalVI z 10.91 ± 1.18 (Welch t=8.24, p=0.005).
- **Multigrate installed in an isolated venv** (NOT `scvi-test`, which the live run imports).
  Diverged to NaN on **2 of 6 seeds** at epoch ~180 (L-096); survivors are runs whose early
  stopping fired first.
- **Cross-seed stability**: kNN Jaccard 0.119–0.375 across all arms — only ~3–7 of 15 neighbours
  survive a seed change, stock TOTALVI included.
- **Cluster-level donor purity** (closes review F8): **no arm produces patient-specific clusters**,
  TOTALVI included (0–1.3% of clusters ≥80% one donor). `u` reduces *residual* donor structure
  ~30% vs TOTALVI (+0.065 vs +0.094 excess over the 0.152 chance level); Vamp `u` closest to chance.
- **Answer to the user's question**: cluster on `u`, prefer **VampPrior** — but the stakes are
  smaller than assumed, since plain TOTALVI would not have broken clustering here.

### Three review passes (2 mycelium + 1 ultracode workflow)

Reports in `.living/outputs/reviews/`: `2026-07-27-…md`, `2026-07-28-…-pass2.md`,
`2026-07-28-…-pass3-ultra.md`. 14 agents in pass 3 with adversarial refutation of every finding.

**Four instances of one error class were found, each inside the fix for the previous one** —
crystallised as **C-004**:
1. L-094 — multivariate η² (denominator is what compression changes)
2. L-097 — cluster F1 at a single Leiden resolution
3. pass-2 F1 — kNN timepoint purity containing same-sample neighbours
4. pass-3 F7 — `separation_ratio`/`within_dispersion` (r=−0.64/+0.85 with effective rank)

Also caught: **sycophancy drift in the assistant's own scorecard** (omitted the two stability
metrics favouring MoG). Corrected symmetrically — applying the same compression test to VampPrior's
metrics removed whitened CKA (r=+0.63) from its side too. **Clean scorecard is 2–2 with one tie**,
not the 5–2 first reported.

---

## 2. Files modified

| File | Change |
|------|--------|
| `.scratch/mrtotalvi-z-vs-totalvi/` | NEW — 16 scripts, FINDINGS.md, TSVs, UMAP figure, venv |
| `docs/user_guide/models/mr_multimodal.md` | VampPrior DA recipe; corrected `freeze_prior_after_init` claim (it is NOT ignored under MoG) |
| `.living/learnings.md` | L-094, L-095, L-096, L-097 |
| `.living/conventions.md` | **C-004** — never compare embeddings with a statistic whose denominator or operating point encodes the effect under test |
| `.living/outputs/reviews/` | 3 review reports |
| `todo/TODO_REGISTRY.md` | P2-006, P2-007, P2-008 |
| `.scratch/mrtotalvi-v2-redesign/` | RUN-STATUS-20260727.md, relaunch-rdx03-grid.sbatch |

**No file under `src/scvi/` or `benchmarks/mrtotalvi/` was touched** — the live run byte-verifies
all 277 of them; verified intact mid-session.

---

## 3. Outstanding items

| Item | Status |
|------|--------|
| **RDX-03 rank gate may be unsound on `canonical_human`** | Its `latent_collapse` hard gate rejected representations that lose nothing recoverable (kNN retest: B2 z 0.798 vs B1 0.811). Since `latent_collapse` is one of four Stage-A prune gates, the D1–D5 screen could discard viable redesigns. **Raise with the RDX-03 reviewer before rerunning anything.** |
| D1–D5 redesign screen | Never started. Stage A = B1/B2/B3/D0 + D1–D5, seed 0, ≤2 survivors; Stage B = survivors + B1/B2, 3 seeds |
| Review F2 | `celltype` ground truth is an automated ensemble call self-documented as superseded — undisclosed beyond a generic "coarse label" caveat |
| Review F4/F5 | Stale n=1 stock-TOTALVI values in two tables |
| Review F9 / P2-008 | FINDINGS.md recommends re-running the session-67 T/NK comparison; P2-008 records that data as unrecoverable. Reconcile |
| P1-006 | DA instability — this session narrowed it: representation instability is a *contributor, not the mechanism* (18–30% neighbourhood-stability gain vs 78% DA-variance reduction) |
| Leiden calibration job 25211796 | Still pending since session 67 |

---

## 4. Next session priorities

1. **Decide RDX-03's fate.** The run is gone either way; the question is whether to resubmit
   `relaunch-rdx03-grid.sbatch` (~44h) *given* the gate concern above. Resolving the gate first is
   probably cheaper than a second 48-fit grid.
2. ~~Reproducibility is the real blocker~~ — **RETRACTED, measured directly and false.** Clusters
   DO reproduce: at Leiden 1.0, 83–95% of clusters recur across a seed change holding 93–99% of
   cells (ARI 0.67–0.90), despite kNN Jaccard of 0.119–0.375. The neighbourhood churn happens
   *inside* clusters. The kNN-Jaccard proxy measured something finer-grained than the conclusions
   drawn from it — fifth instance of C-004 in this session, first running pessimistic.
   **What survives**: Leiden 2.0 is genuinely unstable (ARI 0.57–0.67, ~1 in 5 clusters not
   recurring), so fine-resolution findings need multi-seed confirmation; the main partition does
   not. Stock TOTALVI is the most reproducible arm at res 1.0 (ARI 0.903).
3. Close review F2/F4/F5/F9.
4. Multi-cohort validation (P2-005 macaque) — everything here is one confounded cohort
   (NMI(donor,batch)=0.796, so donor-specific integration cannot be tested on it at all).

---

## 5. Notes for whoever picks this up

- **Apply C-004 before believing any cross-embedding number.** Four instances in one session, each
  found inside the previous fix. The checklist is in `.living/conventions.md`.
- **A schema-conformant agent response is not evidence of work.** The pass-3 completeness critic
  returned literal `"test"` in every field; re-running it with a mandatory tool-use procedure
  produced the single best finding of all three passes.
- **Positive controls earn their keep.** The cluster-donor-purity measurement included `z` arms
  specifically so the measurement could fail; it didn't, which is why its numbers are trustworthy.
- **`z` cannot equal TOTALVI's latent by construction** (user's question, source-verified):
  `z_base = u` is the identity under isomorphic dims, so `z = u + eps` exactly; and `kl_z` is
  `-log N(eps;0,1)` on a *deterministic* eps under `use_map=True` — pure `eps²/2` shrinkage with no
  entropy term, unlike TOTALVI's `KL(q‖N(0,I))`. Measured: eps carries 2–17% of z's variance,
  `rank(z) ≈ rank(u) + 1`. A `z_u_prior=False` ablation was run to test whether the penalty is the
  cause — see `.scratch/mrtotalvi-z-vs-totalvi/z_u_prior_ablation.tsv`.
- The schisto MrTotalVI models were trained with `labels_key="celltype"`, which silently makes the
  MoG prior label-supervised (L-095). Stock TOTALVI was not. Any celltype-based comparison between
  them is supervised-vs-unsupervised.
