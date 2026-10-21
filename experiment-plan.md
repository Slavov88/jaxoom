# Restricted V2 runtime applicability and GPU qualification

## 2026-10-21 continuation protocol (frozen before continuation trials)

User requested continuation after the first stopped prefix. Preserve run1 and
its unknown outcome byte-for-byte. Audit its original artifact hashes and child
provenance before reuse. Reuse the frozen runner, model, policy, threshold,
180-second process limit, zero inputs, and exact runtime. Run fresh GPU replay
before attempting the 169 unrun recipes once each, in original candidate order.
No repeat of previously attempted recipes and no reinterpretation of timeout
warnings as OOM. This completes evidence acquisition; it does not relax the
qualification acceptance criteria. An uncertain EXECUTE_TIMEOUT can be followed
by another recipe only after subprocess termination/reaping and a new clean idle
snapshot. Stop immediately on any LIKELY_FIT failure, other failure, unsupported
state, or failed recovery. Resource caps: one hour campaign and twelve new
execution timeouts, whichever occurs first. No additional seeds or dependencies.
Record new raw logs, recovery snapshots, source hashes, and aggregate counts in
a new directory. Any unknown outcome still prevents all_checks_pass; coverage
and resolved outcomes are separate gates. Public API stays disabled. Regression
tests must cover unknown outcomes, recovery eligibility, deduplication, and
false-safe stops. This protocol is an engineering continuation, not an independent
accuracy study or a new model. Main session owns execution and acceptance.

Frozen before new runtime outcomes: 2026-10-20. Status: PLANNED / qualification
must remain NOT_RUN when the selected GPU has no supported JAX backend.

## Question, hypothesis, and claim

Can the immutable WorkloadSpec adapter reproduce frozen V2 and execute its
admitted workloads under the original NVIDIA runtime conditions? The hypothesis
is feature/score equivalence plus zero newly observed execution OOMs among
LIKELY_FIT decisions in that scope. This is software integration qualification,
not a new general accuracy claim, model improvement, or mathematical result.

The restricted domain is the exact 200 primary-panel WorkloadSpecs, not all
positive dimensions or feature ranges. Supplemental convolution rows remain
outside that population. No outcome, score, or current hardware convenience is
used to expand the admitted set. Historical templates/model stay frozen.

## Control and policy

The baseline is the original V2 model and frozen primary predictions; threshold
0.20, 3 GiB reference denominator (not a quota), calibration multiplier unchanged.
Require exact JAX/jaxlib 0.11.0, NVIDIA RTX 3050 Laptop GPU 4096 MiB, driver
566.07, CUDA/BFC, preallocation false, no memory-fraction/allocator/XLA overrides,
one device, no donation/sharding, default precision/x64 settings, fresh process.
Before a trial require an observation no older than 5 seconds, zero other compute
processes, GPU utilization <=5%, used <=64 MiB, and free >=3900 MiB. Historical
available snapshots were 3964 MiB free, 0 used. These conservative engineering
guards are not new statistical validation of all occupancy values inside them.

Missing, unknown, contradictory, stale, or unsupported states must fail closed.
AMD/ROCm, CPU, other devices/drivers, dynamic budgets, and novel specifications
are not eligible to qualify frozen CUDA V2. Eligibility is not qualification and
must never return LIKELY_FIT. Fresh replay evidence must match the current full
pipeline contract. The public API stays disabled regardless of a preflight pass.

## Frozen stages and metrics

1. Cheapest pilot: discover the user-selected hardware and its supported JAX
   backend. Do not compile or execute target workloads if this fails.
2. Run CPU regression/negative-control tests and verify replay/artifact hashes.
3. Only on an eligible GPU: fresh feature replay, then one smallest-structural
   workload per admitted family/dtype pair, selected without runtime outcomes.
4. Only after all pilots fit: fresh-process execution of all remaining admitted
   primary recipes. One fixed zero initialization, no seed search or retraining.

The user selected the integrated Radeon instead of the occupied NVIDIA device.
Do not use, stop, reset, or contend with the NVIDIA job. Probe AMD explicitly;
do not silently fall back to CUDA or CPU. No GPU job or package/driver swap is
authorized merely by an unsupported probe. No cloud spending is scheduled.

2026-10-21 continuation: the user reports the NVIDIA GPU is free and authorizes
resuming qualification on it. Live inventory confirms the original RTX 3050,
driver 566.07, 3964 MiB free, 0 used, 0% utilization. This supersedes the prior
Radeon-only resource selection; the recorded Radeon failure remains unchanged.
Restore the exact JAX/jaxlib 0.11.0 CUDA12 runtime in an isolated WSL environment
from cached wheels. Do not change drivers, model, templates, scope, thresholds,
or idle guards. Record actual CUDA library/plugin versions. Monitor host storage
during installation; stop installation if C: drops below 512 MiB. The existing
non-sparse 62 GB WSL disk has about 26 GB guest data, so freed allocated guest
blocks may be reused without corresponding host growth; do not assume it.

Record each result with environment, recipe/model/policy/source hashes, command,
phase, raw stdout/stderr, return code, timing, and serialized outputs. Keep
FIT, INITIALIZATION_OOM, COMPILE_OOM, EXECUTION_OOM, timeouts, and unknown results
distinct. A missing output or timeout is not a FIT or execution-OOM observation.

| Check / metric | Population / meaning | Required result |
|---|---|---|
| Applicability negative controls | wrong/missing/stale device, spec, or evidence | all rejected |
| Feature/score/decision replay | all 224 stored prediction rows | zero mismatches at 1e-12 |
| Backend pilot | actual selected GPU | supported JAX GPU device; no CPU fallback |
| Runtime pilot | smallest primary recipe per family/dtype | all FIT |
| Runtime coverage | all 200 exact primary recipes | complete, provenance checked |
| False-safe execution OOM | admitted LIKELY_FIT recipes | zero observed |
| Compilation failures/timeouts | tracked separately | no LIKELY_FIT with a failure; no missing trial |
| Public export check | jaxoom package | disabled |

Kill/stop on unavailable backend, unsupported environment, busy GPU, feature or
source mismatch, any LIKELY_FIT runtime failure, or missing/corrupt evidence.
An unsupported backend ends runtime qualification as BLOCKED, with zero trials;
it cannot pass the table by vacuous truth. If inputs/code require a repair, keep
the failed evidence and use a new result path. No extra seed/benchmark campaigns.

## Ownership, resources, and unrun work

The main session owns the policy, qualification acceptance, evidence, and report.
A worker owns only the policy test file. Reuse the repo's frozen template factory,
replay guards, and phase-separated fresh-process campaign pattern. New code is
limited to the project-specific fail-closed policy and backend preflight; no new
framework dependency is needed. Literature/novelty work is not in scope; official
JAX/AMD installation/support docs are the feasibility references.

Current CPU environment is isolated on D: with JAX/jaxlib 0.11.0. C: is nearly
full; do not install a large GPU stack there. WSL lacks a ROCm runtime. GPU
execution time is unavailable until the selected backend passes stage 1.
No runtime or accuracy value may replace `--` in a report before its stage runs.
