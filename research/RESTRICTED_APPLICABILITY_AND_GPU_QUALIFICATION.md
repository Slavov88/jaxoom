# Restricted applicability and GPU qualification

Date: 2026-10-20. Policy implementation/test status: COMPUTATIONALLY VERIFIED.
GPU execution qualification: **BLOCKED_UNSUPPORTED_GPU_BACKEND / NOT_RUN**.
Public API: **DISABLED**. No new runtime accuracy claim is made.

## Implemented policy

The internal `experiments/oom_risk_applicability.py` policy is versioned as
`restricted-v2-primary-exact-1`. It admits only the exact 200 immutable recipes
from the two primary prediction panels. It excludes novel dimensions, even when
inside historical feature ranges, and excludes the supplemental convolution
panel. This is a finite recipe allowlist, not a generalization rule.

Eligibility requires the original NVIDIA RTX 3050 Laptop GPU (4096 MiB), driver
566.07, one device, GPU backend, JAX/jaxlib 0.11.0, CUDA/BFC settings,
preallocate=false, no memory-fraction/allocator/XLA overrides, default precision
and x64 settings, fresh process, and no donation/sharding. Before execution,
require zero other compute processes, <=5% utilization, <=64 MiB used, >=3900 MiB
free, and a timezone-aware observation aged 0 through 5 seconds. Reject
contradictory memory values and unknown/noncanonical telemetry.

These thresholds are conservative engineering conditions around the historical
idle snapshot (3964 MiB free, 0 used), not a new validation of an occupancy range.
The reference denominator remains 3 GiB; no allocator quota or fit guarantee is
inferred from it.

The evaluator also checks the current full pipeline contract, frozen prediction
hashes, recorded source hashes, complete fresh replay, error counts, numeric
tolerances, tracing-purity details, and disabled public API. Wrong, missing,
stale, malformed, or unsupported state yields `UNSUPPORTED`, no OOM score, and
`gpu_runtime_qualified=false`. The only positive status is
`ELIGIBLE_FOR_QUALIFICATION`; this does not certify a runtime or issue LIKELY_FIT.
Neither eligibility nor a probe ever exports an API automatically.

The immutable policy contract records the entire admitted recipe set, pipeline
hash, exact environment rules, resource/age thresholds, and policy source hash.
The frozen V2 artifact and the earlier fresh-replay implementation are unchanged.

## Actual selected-device preflight

The initial NVIDIA snapshot showed a busy independent Python compute process.
The user then explicitly selected the integrated Radeon instead. No NVIDIA
target workload was executed, stopped, reset, or replaced. CPU tests explicitly
selected `JAX_PLATFORMS=cpu` and disabled CUDA visibility.

Windows inventory identifies AMD Radeon(TM) Graphics, PCI VEN_1002 / DEV_1636,
driver 30.0.13002.19003, in an AMD Ryzen 7 4800H. The actual backend child selected
`JAX_PLATFORMS=rocm` and `jax.devices("rocm")`, with both CUDA visibility settings
disabled. JAX/jaxlib 0.11.0 reported:

```text
Unable to initialize backend 'rocm': Backend 'rocm' is not in the list of known
backends: ['cpu', 'tpu'].
```

CPU fallback was not allowed. The WSL probe found no /dev/kfd, /dev/dri, or ROCm
installation under /opt. Native Windows AMD GPU support is absent in the
[JAX platform table](https://docs.jax.dev/en/latest/installation.html).
The 4800H iGPU is absent from the supported APU families in AMD's
[ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
AMD's [JAX installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html)
requires a supported Linux ROCm configuration. An unsupported-device override,
DirectML substitute, or changing the model's backend would not qualify frozen
CUDA V2 and was not attempted.

The initial and final preflight records retain the raw inventory, child
stdout/stderr/return code, explicit backend environment, policy and model hashes,
command, source hash, and timestamp. They exit nonzero with `all_checks_pass=false`.
**Zero GPU runtime trials ran.** No timeout, CPU result, or missing device was
counted as FIT. Runtime qualification remains blocked, not failed model accuracy.

## Verification and remaining gate

The complete CPU-only suite passed **287 tests**, including the existing fresh
224-row tracing/replay check, 139 new applicability cases, and 7 backend-preflight
controls. Modeled NVIDIA fixtures exercise policy logic only; they are not actual
GPU qualification evidence. A malformed-evidence exception found during the
negative tests was fixed; regression coverage rejects malformed nested evidence,
contradictory purity counts, changed prediction/source hashes, stale timestamps,
busy metrics, and accidental public-enable flags.

The pre-run contract is retained in `experiment-plan.md`. No runtime pilot or
200-recipe execution campaign was launched because selected-device preconditions
failed. To advance frozen V2, an idle supported NVIDIA environment must pass the
frozen pilot and full recipe qualification stages. Using this Radeon would need
supported hardware/runtime and separate AMD applicability/validation, not a
transfer of the CUDA evidence. Public API remains disabled until that work passes.
