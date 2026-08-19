# logs_archive

The arm-1 console log, archived here because the raw file (115 MB) exceeds
GitHub's hard 100 MB per-file limit and blocked every push from 07:15 on
2026-08-19 — see `logs/failsafe_commit.out` for the failsafe's own
"push failed; commit is local" warnings.

- `*.log.gz` — the raw log, gzip -9, **lossless**. `gunzip -c` reproduces the
  original byte for byte (verified with `cmp` before committing).
- `*.stripped.log` — the same log with tqdm progress lines removed: 11,176
  lines of milestones, evals, epoch losses and validation AUROCs, 676 KB.
  This is the one to read.

The raw `logs/*.log` files are no longer tracked; anything that force-adds them
will re-break pushes once a run passes ~100 MB of tqdm output.
