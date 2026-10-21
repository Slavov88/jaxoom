# Restricted V2 runtime acquisition — 2026-10-21

**Public OOM-risk API remains disabled. Full qualification has not passed.**

| Gate | Current observed result |
|---|---:|
| Unique recipes with runtime acquisition attempted | 200/200 |
| Recipes not run | 0 |
| Unresolved outcomes | 5 |
| LIKELY_FIT recipes tested | 157 |
| LIKELY_FIT runtime failures | 0 |
| Total launches, including one preflight-only failure | 201 |
| Latest fresh GPU replay | 224 rows passed |
| Windows / Linux regression tests | 383 / 383 passed |

Outcomes: `{"COMPILE_OOM": 5, "COMPILE_TIMEOUT": 4, "EXECUTE_TIMEOUT": 1, "EXECUTION_OOM": 15, "FIT": 175}`.
Attempted coverage complete: `True`.
Qualification passed: `False`.
Latest campaign stop: `None`.

## Preserved failures and telemetry repair

The original run and both stopped continuations remain byte-for-byte preserved.
Their timeout outcomes remain unknown, never terminal FIT/OOM claims. After an
uncertain compile/execute timeout, new acquisition may proceed only after the
child is killed/reaped and the original clean idle gate passes; every unknown
still prevents qualification. Each campaign retains the 180-second child limit,
one-hour/twelve-timeout resource caps, and stops on any likely-fit workload
failure, invalid identity, unsupported environment, or failed recovery.

Completion1 also rejected HOLDOUT-0084 when NVML returned `[Unknown Error]` in
a numeric telemetry field. Its raw child result proves phase preflight, no CUDA
phase event, and gpu_runtime_executed=false. That launch remains infrastructure
failure evidence; it is excluded from runtime-result counts and could not
qualify the recipe. The separate source-hashed resume entry point wraps only
telemetry acquisition: at most three entirely new samples, one second apart,
for that explicit numeric error. No zero substitution, stale sample, relaxed
idle guard, driver/allocator change, or unrelated-error retry is permitted.
The original q.run_one and q.load_plan source checks remain unchanged. The
preflight-only recipe receives one new workload launch along with never-launched
recipes. Prior workload timeouts are not retried. Raw launch totals and unique
recipe coverage are reported separately.

Frozen V2 model bytes, coefficients, normalization, threshold, templates,
allowlist, feature semantics, runtime settings and original runner remain
unchanged. New fresh CUDA replay passed all 224 rows again; the full Windows CPU
and WSL Linux CPU suites each passed all 383 tests with zero errors/failures/skips.
The final GPU snapshot passed the original idle gate.

## Unresolved recipes

| Panel / candidate | Family | Outcome | Frozen score |
|---|---|---|---:|
| oom_risk_heldout_predictions_2026-09-13.json / HOLDOUT-0012 | attention | EXECUTE_TIMEOUT | 0.705258944917 |
| oom_risk_heldout_predictions_2026-09-13.json / HOLDOUT-0028 | convolution | COMPILE_TIMEOUT | 0.26293723561 |
| oom_risk_heldout_predictions_2026-09-13.json / HOLDOUT-0029 | convolution | COMPILE_TIMEOUT | 0.29885873258 |
| oom_risk_heldout_predictions_2026-09-13.json / HOLDOUT-0031 | convolution | COMPILE_TIMEOUT | 0.568444887066 |
| oom_risk_heldout2_predictions_2026-09-13.json / H2-0038 | convolution | COMPILE_TIMEOUT | 0.367816066978 |

Independent persisted audit verified all prior artifact hash chains, exact
pending selection, unique identities, raw child events and provenance, fresh
scores/features, timeout recovery snapshots, aggregate counts/gates, source and
model hashes, and both JUnit reports. This is software qualification on the
historical population, not independent new accuracy evidence. Missing or
unresolved outcomes remain blockers. No public API was enabled.

Wrapper provenance limitation: child_entrypoint_sha256 is declared by the parent
from the pre-run contract, not independently computed by each child. Current
wrapper bytes match that frozen hash; no mismatch was observed. The original
runner is checked by q.load_plan per child. This distinction is retained in
the audit and is not upgraded to independent per-child wrapper measurement.

The evidence ZIP nests all previous verified bundles and adds the complete
resume evidence, telemetry entry point, tests, protocol, report and verifier.
ZIP CRC and SHA-256 readback of every new member pass.
