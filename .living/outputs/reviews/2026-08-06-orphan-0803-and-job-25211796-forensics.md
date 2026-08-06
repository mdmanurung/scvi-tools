# Forensics memo: ORPHAN-0803 launcher trace, and job 25211796 status

2026-08-06. Read-only follow-up on `todo/TODO_REGISTRY.md`'s ORPHAN-0803 row and on the Leiden
calibration job referenced in `.claude/last-session.md` (session 67). Both items closed below;
neither investigation reached a smoking-gun proof, but one converges strongly and the other is
unambiguous.

## 1. ORPHAN-0803 — who resumed the 07-26 convergence run and published it failed on 08-03?

**Bottom line: cannot be proven at the process level (no shell/notebook history survives), but
every corroborating signal converges on the same account (`mdmanurung`) acting interactively
inside SLURM job `25349712` (`mrvi_cp`) — not on the `rdx03-grid` authoritative launcher, and not
on any other user or host.** No other candidate exists in the data at all.

### 25349712 (`mrvi_cp`) does not name-match, but its window and footprint do

- `sacct -j 25349712 -X`: User `mdmanurung`, Partition `highmem`, NodeList `res-hpc-exe100`,
  Submit/Start `2026-07-31T11:52:44/45`, End `2026-08-04T11:53:09`, State `TIMEOUT`. WorkDir is
  `/exports/para-lipg-hpc/mdmanurung` — the parent of the `scvi-tools` checkout, not the checkout
  itself.
- Its `SubmitLine` is `sbatch scale/jobs/cpu_medium.sh`. That script (read in full) is a generic
  interactive-JupyterLab launcher — starts `jupyter lab` bound to a random port, prints an SSH
  tunnel command, nothing else. It contains no reference to `run_convergence_diagnosis`,
  `mrtotalvi`, or `convergence-runs`, so the script/name itself does **not** identify it as the
  resumer.
- Its own output log (`~/jobs/logs/jupyterCPU_25349712.log`) is just Jupyter server boot output
  ("Serving notebooks from local directory: `/exports/para-lipg-hpc/mdmanurung`" — i.e. scope
  includes `scvi-tools/` as a subdirectory) plus an unrelated `node`-missing extension warning.
  JupyterLab's own log does not record notebook/kernel cell execution, so it cannot directly show
  what ran inside it.

### No properly-submitted batch job matches at all

- Across **every** `mdmanurung` SLURM job from 2026-07-25 to 2026-08-06 (`sacct -u mdmanurung -X`,
  ~200 jobs), only two have a WorkDir under `.../scvi-tools`: job `25211796` (unrelated — see §2,
  cancelled 07-15) and `25349712` itself (WorkDir is the checkout's *parent*, not the checkout).
  No job was submitted from inside the repo during the incident window.
- Searched for a job named `rdx03-grid` (the name `relaunch-rdx03-grid.sbatch` declares via
  `#SBATCH --job-name=rdx03-grid`) both with `sacct --name=rdx03-grid` over 07-20→08-06 and by
  eyeballing the full all-users job list for 08-01→08-04: **zero matches**, at any time, by any
  user. The authoritative RDX-03 launch path — which gates on one-use claims and SHA-256
  authorization hashes precisely to prevent an unaudited resume like this one — was never invoked
  anywhere near 2026-08-03. Whatever resumed the run bypassed that mechanism entirely.

### The failure artifact itself points at a live, unfrozen repo checkout

- `failure.json` / the dying worker's traceback
  (`worker-logs/canonical_human_if_available--D0--seed0.log`) show the crash came from
  `run_convergence_fit.py` → `_verify_live_sources_against_manifest` raising
  `ValueError: Live worker source differs from snapshot for benchmarks/mrtotalvi/__init__.py`.
  This is a guard that compares the **live** file in the working checkout against the digest
  recorded in the run's 07-26 manifest — meaning the 08-03 resume executed the worker against the
  live `/exports/para-lipg-hpc/mdmanurung/scvi-tools` tree, not an isolated frozen copy, and the
  guard correctly killed it when it found the tree had moved on.
- `benchmarks/mrtotalvi/__init__.py` is **untracked** in git (`git status --porcelain` → `??`,
  `git log` on it returns nothing on any branch) — it only ever existed in this working tree. Its
  mtime is `2026-07-31 11:54:44 CEST` — **99 seconds** after job 25349712's Jupyter server logged
  "Jupyter Server 2.17.0 is running at… 11:53:05". This ties the session to repo-editing activity
  in the opening minutes of its lifetime — TODO_REGISTRY independently attributes this same edit
  to "the Jul 30–31 latent-integrity work," a different activity from a run resume, and three days
  separate it from the 08-03 failure. It is evidence the job was an active editing environment for
  this checkout, **not** direct evidence about what triggered the 08-03 resume itself.
- The failing worker's log carries Lightning's `SLURMEnvironment` warning ("the `srun` command is
  available… but is not used"), i.e. it auto-detected `SLURM_JOB_ID` in its environment but was
  not itself launched via `srun`. That is exactly what you'd expect from a python process spawned
  from an interactive shell/kernel *inside* an existing batch allocation (inheriting its SLURM env
  vars), not from a fresh `sbatch`/`srun` submission — and `relaunch-rdx03-grid.sbatch`, the one
  script built to run this workload properly, was never submitted in this window (previous point).

### What the run artifacts do *not* contain

- No hostname, PID-that-resolves-to-a-host, or username field anywhere in `run-manifest.json`,
  `failure.json`, or the per-fit `workers/*.json` records — the latter carry only
  `process_id`/`parent_process_id` (e.g. `parent_process_id: 2`), which are container/session-
  relative, not globally identifying.
- All 45 completed-fit `checkpoints/*/` subdirectories carry mtimes clustered within **1.5
  seconds** of each other (`15:24:51.2xx`–`15:24:52.7xx` on 08-03), not spread across the original
  run's two days (07-26→07-28). That's a bulk re-materialization/copy at teardown, not incremental
  training — consistent with the TODO_REGISTRY's existing read that the 45 fits were pulled from a
  cache rather than re-trained, but it's a copy operation, so it carries no separate identity
  signal of its own.
- The original run's `.tmp-1oncvmjo` scratch directory referenced in the worker log no longer
  exists (already cleaned up) — nothing left to inspect there.

### Conclusion on §1

I did not find a definitive process-level proof (no bash/notebook history, no host-tagged log
line, no audit field). But I found no alternative candidate anywhere either: job `25349712` is the
*only* SLURM allocation, by any user, on any node, whose window contains 2026-08-03T15:23–15:25;
it is the *only* such allocation with any footprint near the `scvi-tools` checkout; the file whose
drift eventually killed the run was edited 99 seconds into that job's start (though that edit is
tied to separate 07-31 development activity, not proof of the 08-03 resume itself); and the
failure mode's own log signature (SLURM-env-detected-without-srun) is inconsistent with the one script
that *would* properly identify a launcher (`relaunch-rdx03-grid.sbatch`, unused all week) and
consistent with ad hoc interactive execution inside a Jupyter session. Nothing here points to a
different user or a different host — only to the account owner (`mdmanurung`), most likely acting
interactively inside that Jupyter session, rather than through the authoritative launcher. If a
stronger identification is needed, JupyterLab kernel logs or shell history for that session
(neither found under `~/jobs/logs/` or the home directory) would be the next place to check, but
none survive.

**Recommend**: TODO_REGISTRY's ORPHAN-0803 "REMAINING: confirm who launched it and on what" can be
downgraded from "who" (unanswerable now, evidence doesn't survive) to a forward-looking fix:
`run_convergence_fit.py`'s live-source guard did its job here and should stay; what's missing is
that nothing prevents *ad hoc* resume of a run directory outside the `rdx03-grid` authorization
path in the first place. `run-manifest.json` should probably record hostname/SLURM_JOB_ID/PID at
write time so a future occurrence is traceable — it currently has neither field.

## 2. Job 25211796 — Leiden coarse-resolution calibration job status

**Not pending. It is `CANCELLED`, and it never ran at all.**

```
$ squeue --job=25211796
slurm_load_jobs error: Invalid job id specified      # not in the queue

$ sacct -j 25211796 --format=JobID,JobName%40,Partition,Submit,Start,End,State%25,ExitCode,WorkDir -X
25211796  cytoanvi_leiden_calibration  medium  2026-07-12T20:46:42  None  2026-07-15T13:36:11  CANCELLED by 149488  0:0  /exports/para-lipg-hpc/mdmanurung/scvi-tools
```

- Submitted 2026-07-12, never started (`Start = None` — consistent with it sitting queued against
  the `QOSMaxCpuPerUserLimit` block that `.claude/last-session.md` records), and cancelled
  **2026-07-15T13:36:11** — three weeks before today (2026-08-06).
- The `State` field reads `CANCELLED by 149488`; UID `149488` resolves to `mdmanurung`
  (`id`/`getent passwd 149488` both confirm) — i.e. it was cancelled by the account owner, not by
  an admin or a scheduler policy action. (`ExitCode` was a plain `0:0`, unrelated to this UID.)
- Per the task instructions, the job was **not** touched by this investigation (no `scancel` run,
  nothing modified) — this is a status report only.

**Recommend**: `.claude/last-session.md` / `CLAUDE.md`'s "Leiden calibration job 25211796 still
pending" line is stale by three weeks and should be corrected or removed — the job is gone, was
self-cancelled while still queued, and produced no output.
