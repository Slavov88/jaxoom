# Execution OOM diagnosis

Status: OBSERVED

Environment: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM, driver
566.07, JAX 0.11.0, jaxlib 0.11.0, Python 3.12.3. The primary allocator
configuration was `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

No production predictor, calibration constant, reserve, or risk threshold was
changed.

## False-fit reproduction

The strongest prior false fit was reproduced in a fresh process:

- Workload: attention, batch 1, heads 8, sequence 4096, head dimension 64,
  float32
- Structural estimate: 1,082,261,504 bytes
- Calibrated interval: 728,866,417 to 1,804,833,959 bytes
- Pre-compilation assessment budget: 3,855,823,667 bytes
- Upper headroom: 2,050,989,708 bytes
- Risk: LOW
- Compilation: SUCCESS
- Execution: EXECUTION_OOM

The concise diagnostic was:

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.00GiB
```

The allocator warning reported a rounded request of 1,073,742,080 bytes. The
compiled memory analysis reported:

```text
argument: 8,388,608
output: 8,388,608
temporary: 1,073,741,840
alias: 0
compiler-accounted: 1,090,519,056
```

## Device-budget audit

Immediately before compilation:

| Quantity | Value |
|---|---:|
| `nvidia-smi` free | 4,068,474,880 bytes |
| CUDA `cudaMemGetInfo` free | 3,444,991,591 bytes |
| JAX pool | 10,485,760 bytes |
| JAX bytes in use | 8,388,608 bytes |
| JAX bytes limit | 3,221,225,472 bytes |
| jaxoom effective available | 4,070,572,032 bytes |

After compilation:

| Quantity | Value |
|---|---:|
| `nvidia-smi` free | 1,843,396,608 bytes |
| CUDA `cudaMemGetInfo` free | 1,219,913,319 bytes |
| JAX pool | 2,225,078,272 bytes |
| JAX bytes in use | 8,388,608 bytes |
| JAX peak bytes in use | 1,170,211,072 bytes |
| JAX bytes limit | 3,221,225,472 bytes |
| JAX largest free block | 1,073,741,824 bytes |
| jaxoom effective available | 4,060,086,272 bytes |

The current formula adds unused JAX pool capacity to driver free memory and
does not cap the result using JAX `bytes_limit`. After compilation, the JAX
allocator had only 996,147,200 bytes between its pool size and `bytes_limit`,
while the execution request was 1,073,742,080 bytes. The largest free block was
also 256 bytes smaller than the rounded request.

The formula therefore is not a valid estimate of immediately allocatable JAX
capacity under this allocator state. `bytes_in_use` does not account for all
pool-resident executable state, so `pool_bytes - bytes_in_use` is not a safe
free-pool estimate.

CUDA and `nvidia-smi` also disagreed by approximately 624 MiB after compilation.
The direct CUDA query used the installed `libcudart.so.12` through a temporary
experimental `ctypes` wrapper. No CUDA dependency was added to the package.

## Raw allocation capacity

A fresh process retained 256 MiB JAX device allocations without compiling the
target workload. Allocations succeeded through 3,072 MiB and failed at 3,328
MiB. This matches the reported JAX `bytes_limit` of 3,072 MiB and confirms that
the allocator has a meaningful process-level cap independent of the larger
physical free-memory reading.

This was a capacity probe, not an exact workload runtime peak measurement.

## Execution phases

For the false fit, memory changed as follows:

```text
before inputs       JAX pool 0 MiB
immediately before  JAX pool 10 MiB
after compile       JAX pool 2,122 MiB
execution failure   JAX pool 2,122 MiB
```

The failed allocation was represented by the compiler temporary accounting, but
that accounting did not represent the already-resident post-compilation JAX
pool state that constrained the allocation.

## First versus steady state

Attention sequence 1024, float32, compiled once and executed three times,
FIT on all executions. JAX pool remained about 262 MiB and bytes in use changed
only from about 2 MiB to 4 MiB after the first execution. No large one-time
execution allocation was observed for this fitting case.

The false-fit 4096 case failed on its first execution. The evidence points to
post-compilation allocator capacity and fragmentation rather than a large
second-execution-only allocation.

## Execution threshold brackets

The threshold harness used fresh processes and pressure levels 0, 768, and
1280 MiB. These are coarse brackets, not exact binary-search thresholds.

| Workload | Result over tested pressures |
|---|---|
| Attention 1024 float32 | FIT at 0, 768, 1280 MiB |
| Attention 2048 float32 | FIT at 0, 768, 1280 MiB |
| Attention 3072 float32 | FIT at 0, 768, 1280 MiB |
| Attention 4096 float32 | EXECUTION_OOM at 0, 768, 1280 MiB |
| Attention 2048 float16 | FIT at 0, 768, 1280 MiB |
| MLP, batch 256, width 8192 | FIT at 0, 768, 1280 MiB |
| Training, batch 256, width 8192 | FIT at 0, 768, 1280 MiB |

For float32 attention, the observed sequence-size transition was between 3072
and 4096 for all tested pressure levels. MLP and training controls remained FIT
at the tested pressures, including one case classified LIKELY EXCEEDS under
high pressure.

## Allocator policy

The same 4096 attention case was run under the platform allocator:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: EXECUTION_OOM
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform`: FIT
- default preallocation: initialization failed with CUDA out-of-memory while
  allocating pinned host memory before a usable target-trial row was produced

The platform allocator run had no usable JAX `memory_stats()` values, but the
execution completed. This is strong evidence that allocator policy, pool
residency, or fragmentation contributes materially. It does not isolate which
allocator mechanism is solely responsible.

## Compiler reconciliation

The lowered HLO fingerprint for the false fit was recorded. No
`custom_call_target` was present in the compact lowered-HLO scan. This does not
exclude backend library work after lowering. Compiler memory analysis did
include a temporary allocation of approximately 1 GiB, matching the execution
failure request in scale.

The missing quantity is therefore not simply an unmodeled compiler temporary.
The immediate failure required the allocator to provide that temporary while a
large post-compilation JAX pool was resident and subject to a 3 GiB allocator
limit.

## Root cause

Classification: **multiple contributors, with a confirmed device-budget
accounting limitation and allocator-policy/fragmentation effects**.

Confirmed observations:

1. The current effective-available formula ignores the JAX `bytes_limit`.
2. It treats `pool_bytes - bytes_in_use` as reusable capacity even though
   `bytes_in_use` does not account for all executable pool residency.
3. Compilation increased the JAX pool from about 10 MiB to 2.2 GiB.
4. The execution request was slightly larger than the largest free allocator
   block and larger than the remaining JAX cap.
5. The platform allocator changed the same case from OOM to FIT.

A single global safety reserve would not explain or reliably correct this
behavior.

## Production change

None. The device-budget formula should be corrected in a separate change only
after defining how to represent JAX allocator limits and unavailable future
post-compilation pool growth. The current public behavior remains unchanged in
this milestone.

## Artifacts

- `execution_oom_diagnosis.py`
- `execution_oom_diagnosis_2026-09-08.json`
- `execution_oom_diagnosis_summary_2026-09-08.json`
- `execution_oom_capacity_2026-09-08.json`
- `execution_oom_false_fit_2026-09-08.json`
- `execution_oom_allocator_policy_2026-09-08.json`
