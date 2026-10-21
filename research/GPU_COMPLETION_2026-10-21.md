# Restricted V2 GPU completion — 2026-10-21

**Qualification remains incomplete. Public OOM-risk API stays disabled.**

Across the original run and two append-only continuations, 96/200
recipes have been attempted; 104 remain unrun. There are
5 unresolved outcomes. All 76 tested
LIKELY_FIT recipes have 1 failures.

| Outcome | Combined count |
|---|---:|
| FIT | 83 |
| EXECUTION_OOM | 5 |
| COMPILE_OOM | 3 |
| EXECUTE_TIMEOUT | 1 |
| COMPILE_TIMEOUT | 3 |
| INVALID_CHILD_PROVENANCE_OR_REPLAY | 1 |

The completion campaign added 50 trials. Stop reason:
`INVALID_CHILD_PROVENANCE_OR_REPLAY`. Attempted coverage complete:
`False`; qualification passed: `False`.
Timeouts remain unknown terminal FIT/OOM outcomes and cannot count as passing
qualification. Previous timeouts were never retried or overwritten. All prior
evidence, frozen model/threshold/templates/policy and original runner are intact.

Fresh CUDA tracing/replay passed all 224 rows again. All 379 regression tests
passed on Windows CPU and WSL Linux CPU with zero failures, errors, or skips.
The final GPU telemetry passed the original idle guards. Independent persisted
audit verified both prior evidence hash chains, selection/deduplication, raw
result events, phase/provenance, fresh scores/features, recovery telemetry,
aggregate gates/counts, frozen model bytes, and test XML. This is repeated
historical-population software qualification, not new independent accuracy
evidence. Full qualification remains blocked by unknown or unrun outcomes.

## Protocol and evidence

The original 180-second subprocess limit and exact CUDA environment were
unchanged. Continuation1 stopped on a compile timeout. Its evidence was audited
and frozen before a separate completion protocol broadened acquisition recovery
to uncertain compilation and execution timeouts, only after process kill/reap
and persisted clean idle telemetry. Acceptance criteria remained unchanged:
unknowns block all_checks_pass even with full attempted coverage. Any LIKELY_FIT
failure, unexpected failure, invalid provenance, unsupported/busy environment,
or failed recovery stops execution. Each campaign has one-hour/twelve-timeout
resource caps. No previous recipe is retried and no model is refit.

Evidence: `experiments/oom_risk_gpu_qualification_2026-10-21_completion1/`.
The accompanying ZIP contains the verified preceding evidence bundle, complete
new evidence, added source/tests/protocol and this audit verifier. ZIP CRC and
SHA-256 readback of every member pass. Public exposure remains disabled until
the complete runtime qualification passes.
