# Fresh restricted V2 qualification with one uniform runtime bound

Frozen before this campaign's outcomes. The user requested the next qualification
step and a few commits. This is integration qualification on the exact historical
population, not new independent prediction accuracy evidence.

## Fixed controls and population

Preserve every earlier outcome, including historical 180-second timeouts and the
separate 600-second diagnostics. Do not use those results to satisfy new coverage.
Use exactly the original 200 primary candidate identities/specifications/features/
scores and the same 18 smallest structural family/dtype pilots, without selecting
by observed outcomes. Pilots run first and must all FIT; remaining candidates
follow original candidate order. One fresh CUDA process and zero-initialized
inputs per recipe. Frozen V2 bytes, coefficients, templates, legacy feature
semantics, threshold 0.20 and applicability policy are unchanged. No retraining,
seed changes, new cases, dynamic budget, allocator/XLA overrides or device swap.

Require RTX 3050 Laptop GPU 4096 MiB, driver 566.07, JAX/jaxlib 0.11.0, CUDA/BFC,
preallocation false, default precision/x64, one GPU, no donation/sharding. The
original idle guards apply before every child and independently inside it.
Explicit CUDA selection, no CPU fallback. Retain at most three entirely new
telemetry samples only for NVML's explicit numeric Unknown Error condition;
never substitute zero, reuse stale samples or relax telemetry guards.

## Bound, resources and stop conditions

Every child gets the same 600-second outer timeout. This predeclared bound covers
the longest observed diagnostic (403.314 s) with additional startup margin; it
does not guarantee completion. The campaign launch budget is three hours, checked
after idle gating with a full child timeout reserved. Require at least 512 MiB
free on the workspace host filesystem before every launch. No installs, downloads
or new dependencies; stop rather than delete user data if resources fail.

One attempt per identity, no retry/resume within this campaign, no carried-over
runtime result. Stop on ANY timeout, missing/duplicate/corrupt evidence, source/
model/policy/feature/score mismatch, unsupported/busy state, failed recovery, any
LIKELY_FIT non-FIT result, a non-FIT pilot or an unexpected failure. After every
returned result retain a fresh idle recovery snapshot. Timeouts remain unknown
terminal FIT/OOM observations and cannot pass. Compilation OOM and execution OOM
stay distinct; either is acceptable only for a non-LIKELY_FIT non-pilot recipe
with complete returned evidence. Allocation warnings do not establish OOM.

## Evidence and acceptance

Commit protocol, code and negative controls before execution. Require full
Windows CPU and WSL Linux CPU regression results, bound to the tested source
hashes, before launch. Pin their JUnit hashes and validation manifest in the plan.
Then run fresh 224-row CUDA tracing/replay: exact feature equality, score tolerance
1e-12, zero decision mismatches, pure tracing and absent public exports. Only after
replay passes may runtime children launch.

Each child measures its entry point and imported wrappers/adapter/policy/replay
sources, plan bytes and replay bytes BEFORE telemetry/CUDA initialization. Check
source/content hashes, original reference plan and exact catalog/pilot membership;
emit the actual measured attestation before target phases and return it with the
result. The parent validates actual child data without replacing measured fields.

| Gate | Required result |
|---|---|
| Frozen source/reference/policy/model identity | Exact match |
| Full Windows and Linux regression evidence | Zero failures/errors/skips |
| Fresh CUDA tracing/replay | 224/224, zero mismatches |
| Original runtime pilots | 18/18 FIT |
| Unique candidate coverage | Original 200 identities, once each |
| Every child | Valid source/plan/replay/spec/features/score/CUDA provenance |
| LIKELY_FIT outcomes | Every one FIT |
| Other non-pilot outcomes | FIT or terminal phase-correct compile/execution OOM |
| Timeout/missing/corrupt/unsupported evidence | Zero |
| Recovery/final idle and final source identity | Pass |
| Independent persisted evidence audit | Consistent, complete passing campaign |
| Public OOM-risk API | Disabled; release/export remains a separate decision |

The runner reports runtime checks separately from the independent qualification
verdict. A pass applies only to the exact restricted scope/environment and this
declared 600-second protocol, not arbitrary callables, portability or a new error
bound. Main session owns execution, acceptance and commits; a worker may own
isolated tests or read-only review. Preserve raw logs, commands/errors, measured
hashes, phases, telemetry/recovery, test evidence and final manifest. Commit and
report failed/budget-stopped campaigns as incomplete; never retry until passing.
