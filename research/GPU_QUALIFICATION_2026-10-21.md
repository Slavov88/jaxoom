# Restricted V2 NVIDIA GPU qualification — 2026-10-21

**Decision: GPU_QUALIFICATION_STOPPED; PUBLIC API DISABLED.**
Result status: COMPUTATIONALLY VERIFIED for GPU tracing, the 18 pilots, and the
audited execution prefix. The full 200-recipe qualification did **not** pass.

## Results

| Gate | Observed result | Status |
|---|---|---|
| Device and driver | RTX 3050 Laptop GPU, 4096 MiB, 566.07 | PASS |
| Initial idle state | 0 MiB used, 3964 MiB free, 0% utilization, no compute processes | PASS |
| Fresh GPU tracing/replay | 224 rows; zero feature, score, or decision mismatches | PASS |
| Family/dtype pilots | 18/18 FIT | PASS |
| Main recipe attempts | 31 of 200 attempted | INCOMPLETE |
| Returned FIT outcomes | 27 | OBSERVED |
| Returned execution OOMs | 2; both above the fit threshold | OBSERVED |
| Returned compile OOMs | 1; above the fit threshold | OBSERVED |
| Execution timeout | 1; above the fit threshold | UNKNOWN FIT/OOM |
| LIKELY_FIT trials | 25/25 FIT; zero false-safe execution OOMs | PREFIX ONLY |
| Remaining recipes | 169 not run after frozen stop | NOT_RUN |
| Public API | Absent/disabled | PASS |

The GPU replay maximum absolute score difference was 1.11e-16, below the fixed
1e-12 tolerance; feature vectors matched exactly. All target calls in the replay
received tracers, with no concrete target execution or attempted public JIT/
compiler paths. Subsequent runtime trials did compile and execute on CUDA;
returned FIT outputs were synchronized and checked for CUDA device residency.

The zero false-safe count applies only to this audited prefix. These recipes
overlap historical validation by design, so this is not a new independent
accuracy panel or an improved statistical error bound. A timeout is not FIT or
terminal execution-OOM evidence. The 3 GiB denominator remains a feature
reference, not a fit guarantee or a newly enforced allocator quota.

## Frozen stop

The stopping attempt was primary candidate HOLDOUT-0012:

```text
family: attention
dtype: float16
batch: 2
heads: 16
sequence: 4608
head_dim: 96
frozen V2 score: 0.7052589449165493
LIKELY_FIT threshold: 0.20
last phase: execute
timeout: 180 seconds
recorded outcome: EXECUTE_TIMEOUT
```

Its raw stdout reached initialization, compilation, and execution, but contains
no final result event. Stderr contains repeated BFC allocation warnings for a
1.27 GiB request. There is no returned terminal exception proving execution OOM.
The child was stopped at the planned timeout; warnings were not converted into
an OOM classification. This is an uncertain workload with an unknown terminal
FIT/OOM outcome, not a false-safe observation. The frozen stop was honored;
thresholds, model, timeout, allowlist, and allocator were not changed to continue.

After stopping, the GPU returned to 0 MiB used, 3964 MiB free, 0% utilization.

## Environment and infrastructure

Qualification used WSL Ubuntu/Python 3.12.3, exact JAX/jaxlib and CUDA12 plugin/
PJRT 0.11.0. The CUDA client reported `cuda12090`. Cached dependencies include
cuDNN 9.25.1.1, cuBLAS 12.4.5.8, CUDA runtime 12.4.127, and nvcc 12.9.86;
complete package versions are retained per completed trial and in the isolated
environment's pip installation report. `pip check` reported no broken requirements.

An initial WSL connection timeout/restart occurred before any GPU trial and
removed a temporary environment. It is recorded separately as infrastructure
failure, not a model result. The environment was restored to the persistent path
`/home/slavov/.venvs/jaxoom-cuda-20261021/env` from read-only cached-wheel links,
with a transient 2 GiB installation memory limit and a 512 MiB host-storage stop
gate. Existing freed blocks in the non-sparse WSL disk were reused; host free
space stayed above that gate. No driver or global Python installation changed.

## Provenance, checks, and next work

Evidence directory:
`experiments/oom_risk_gpu_qualification_2026-10-21_run1/`.
It contains the pre-run hashed plan, idle snapshot, GPU replay, individual
results and raw stdout/stderr, stopped summary, and independent persisted audit.

Plan hash:
`3660a87699b31f3bc92df1207204e6568071543fd1ae7792ae3515a9054c6dc5`.
Policy hash:
`26872f92d9bf745dcfd5645007b0e46020c77a48a4c28c20260dccd4e3aada3d`.
The original frozen V2 file SHA-256 remains
`5b2eeedf9a180af1c79678e94ecf3401b0256b4906e1d8f5180f058d3b354ae3`.

The independent audit reloaded the plan, replay and every returned trial,
checked original child identity and source/policy/model hashes, recomputed
scores, checked raw result events, phase consistency, threshold labels and GPU
eligibility, and verified all recorded artifact hashes. It confirms incomplete
coverage and `all_checks_pass=false`. The Windows CPU regression suite passed
367 tests; the restored Linux environment passed all 367 tests with zero
failures/errors/skips, recorded in `regression_results.xml`. No further GPU
trials were launched during these checks.

The remaining bottleneck is the bounded-execution qualification protocol for
uncertain execution timeouts with repeated allocator warnings. A future
continuation needs a separately frozen timeout/recovery protocol and the 169
remaining recipes; it must preserve this stopped run and must not retroactively
mark the unknown outcome as FIT/OOM or claim full coverage. Public exposure
remains disabled until a complete qualification passes.
