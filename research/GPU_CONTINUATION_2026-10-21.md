# Restricted V2 GPU continuation — 2026-10-21

**Qualification incomplete. Public OOM-risk API remains disabled.**

The separately frozen continuation attempted 15 previously
unrun recipes. Combined with the preserved original run, 46/200
recipes have been attempted; 154 remain unrun and 2
have unresolved outcomes.

| Metric | Combined observed result |
|---|---:|
| FIT | 37 |
| Execution OOM | 5 |
| Compile OOM | 2 |
| Execution timeout, unknown terminal FIT/OOM | 1 |
| LIKELY_FIT attempts | 34 |
| LIKELY_FIT failures | 0 |
| Unrun | 154 |

Fresh CUDA tracing/replay passed all 224 prediction rows again. The full
regression suite passed all 373 tests on both Windows and WSL Linux with zero
failures, errors, or skips. Original frozen V2 artifact bytes, source runner,
policy, threshold, recipe allowlist, timeout, and historical run1 evidence were
preserved. All original 99 recorded artifact hashes were verified again.

## Protocol and acceptance

The continuation used a new directory and a source-hashed pre-run protocol.
Each previously unattempted recipe runs once in original order in a fresh CUDA
process, retaining the 180-second limit and original idle guards. A timeout is
never FIT or terminal OOM evidence. Only an uncertain execution timeout allows
continuation after the child is killed/reaped and a persisted fresh idle
snapshot proves recovery. Any LIKELY_FIT failure, invalid identity/replay,
unsupported environment, failed recovery, other failure, twelve new execution
timeouts, or the one-hour resource budget stops acquisition. The campaign cap
is checked before idle gating, so idle gating and shutdown may add a few seconds.

The first run's HOLDOUT-0012 timeout remains unknown and is never retried or
overwritten. Complete attempted coverage alone cannot pass qualification:
all individual outcomes must also pass the original acceptance criteria.
Allchecks result: `False`. No new independent accuracy claim
or statistical bound is supported by this repeated historical population.

## Evidence and verification

Repository evidence: `experiments/oom_risk_gpu_qualification_2026-10-21_continuation1/`.
It retains the continuation protocol, fresh GPU replay, individual JSON outcomes,
raw logs, timeout-recovery snapshots, final idle snapshot, aggregate summary,
Windows/Linux JUnit records, and a persisted independent audit. The verifier
recomputed selection, all aggregate counts and qualification gates, checked raw
events and child provenance, prior evidence hashes, frozen model bytes, and both
test reports. The evidence ZIP includes the original verified bundle plus all
continuation evidence and the added source/tests. Every ZIP member is SHA-256
verified after write and ZIP CRC verification passes.

Unresolved and unrun recipes remain blockers. Public exposure must stay disabled
until the complete runtime qualification, including resolved outcomes, passes.

Continuation stop reason: `COMPILE_TIMEOUT`.
