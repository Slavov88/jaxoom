# GPU timeout diagnostics and repository progress — 2026-10-21

**Diagnostic status: DIAGNOSTICS_COMPLETE. Public OOM-risk API remains disabled.**

The restricted adapter/policy and complete CUDA acquisition are now committed.
The new diagnostic entry point fixes the missing per-child wrapper measurement:
each child independently hashes eight source files, the diagnostic plan and
fresh replay before telemetry/CUDA initialization or workload execution. The
parent checks the child's actual emitted event and result; it does not replace
measured hashes with expected values. Historical runners and evidence are intact.

## Results under the separately frozen 600-second diagnostic bound

Exactly the five uncertain recipes that timed out under the historical
180-second acquisition were selected, in original order, for one attempt each.
No known result or LIKELY_FIT workload was rerun, and no seed, allocator, driver,
recipe, template, model, threshold or policy was changed.

| Candidate | Historical observation | New diagnostic observation | Child wall seconds |
|---|---|---|---:|
| HOLDOUT-0012 | EXECUTE_TIMEOUT | EXECUTION_OOM | 186.508 |
| HOLDOUT-0028 | COMPILE_TIMEOUT | COMPILE_OOM | 393.695 |
| HOLDOUT-0029 | COMPILE_TIMEOUT | EXECUTION_OOM | 306.327 |
| HOLDOUT-0031 | COMPILE_TIMEOUT | COMPILE_OOM | 403.314 |
| H2-0038 | COMPILE_TIMEOUT | FIT | 333.414 |

Diagnostic attempts: 5/5; terminal results accepted by the
original phase/provenance checks: 5/5;
unresolved in this diagnostic stage: 0.
Raw outcome counts: `{"COMPILE_OOM": 2, "EXECUTION_OOM": 2, "FIT": 1}`.
The outer deadline includes child startup; child wall time covers its internal
preflight/target phases and is not a benchmark or a general latency guarantee.

HOLDOUT-0012 returned an execution OOM after the former outer timeout. Notably,
HOLDOUT-0029 previously stopped during compilation, whereas its new run completed
compilation and returned execution OOM. Compilation allocation warnings therefore
could not have established its terminal phase or outcome. These are new runs;
none of the historical timeout records has been relabeled or overwritten.

The historical acquisition remains 200 unique recipes: 175 FIT, 15 execution
OOM, 5 compile OOM and 5 unresolved timeouts, with all 157 LIKELY_FIT recipes FIT.
The diagnostic stage does **not** retroactively pass that 180-second
qualification or enable an API. Any future qualification/release decision must
explicitly specify its runtime bound and evidence policy before execution.
This repeated historical population is not new independent accuracy evidence.

## Validation and provenance

- Fresh CUDA tracing/replay: all 224 rows passed again, with public exports absent.
- Full regression suite: 447 tests passed on Windows CPU and 447 on WSL Linux CPU,
  zero failures/errors/skips. Earlier pre-auditor 417-test reports are also retained.
- Portable persisted auditor checks all prior artifact hash chains, exact selection,
  child-measured source/plan/replay hashes, raw events against saved rows,
  features/scores/backend/phase evidence, recovery telemetry and summary counts.
- Every child exit or kill/reap is followed by saved idle telemetry; final GPU
  telemetry passed the original guards. Original environment remains RTX 3050
  Laptop GPU, driver 566.07, JAX/jaxlib 0.11.0, CUDA/BFC, preallocation false.
- Frozen V2 file SHA-256 remains
  `5b2eeedf9a180af1c79678e94ecf3401b0256b4906e1d8f5180f058d3b354ae3`.
- Scoped Git attributes preserve exact bytes despite core.autocrlf. The first
  two commits' 676 files were read back and matched the workspace byte-for-byte;
  final delivery verifies all changed committed blobs again.

## Commits and reproducible audit

- `8be9224`: restricted WorkloadSpec, policy and replay evidence.
- `774fa22`: complete historical CUDA acquisition tooling and evidence.
- `eaa8c5e`: child-side measured source/plan/replay attestation and diagnostic protocol.
- `9415ce4`: portable evidence auditor with negative controls.
- The commit containing this report records the diagnostic/test evidence and status.

The source and runtime metadata are pinned in the pre-run plan. The independently
measured wrapper fix applies prospectively to these diagnostics; the older
resume records retain their original parent-declared hash limitation.

With the exact JAX 0.11.0 environment and PYTHONPATH=src, the following command
checks persisted evidence without probing or running a GPU:

```text
python -m experiments.oom_risk_gpu_diagnostic_audit --run-dir experiments/oom_risk_gpu_diagnostics_2026-10-21_run1
```

An optional `--output PATH` writes a new audit JSON exclusively; existing files
are never overwritten. Evidence consistency and diagnostic resolution are
reported separately from original qualification, which remains false. The
public API remains disabled pending a fresh qualification/release decision.
