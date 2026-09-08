# Device budget correction

Status: REDUCED

## Bug

The legacy policy was effectively:

```text
legacy_effective_available =
    driver_free_bytes + max(jax_pool_bytes - jax_bytes_in_use, 0)
```

with an optional environment-fraction cap. This treated pool bytes as
independently reusable and did not enforce the JAX allocator `bytes_limit`.
The policy could therefore report more capacity than the allocator was allowed
to manage.

## Corrected policy

The production policy is now:

```text
allocator_limit =
    JAX memory_stats()["bytes_limit"] when available
    otherwise physical_total * configured memory fraction when available

capacity = min(driver_free_bytes, allocator_limit)
            when both are known
capacity = driver_free_bytes when only driver free is known
capacity = allocator_limit when only allocator limit is known

assessment_budget = max(0, capacity - safety reserve)
```

The existing safety reserve remains unchanged: 5% of physical memory, bounded
to 64 MiB through 256 MiB.

`pool_bytes`, `bytes_in_use`, peak usage, and largest free block are now exposed
as separate observations. Pool bytes are not added to driver free memory. The
largest free block is not used as a pre-compilation budget because its future
reusability is not predictable from a snapshot.

## Allocator limit validation

On the RTX 3050 with JAX 0.11.0 and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`:

- JAX `bytes_limit`: 3,221,225,472 bytes
- Raw JAX allocations through 3,072 MiB: FIT
- Raw JAX allocation at 3,328 MiB: OOM

The capacity probe agrees with the observed allocator limit.

## Strongest false fit

Before correction:

- Budget: 3,855,823,667 bytes
- Upper headroom: 2,050,989,708 bytes
- Risk: LOW

After correction:

- Budget: 3,006,477,107 bytes
- Upper headroom: 1,201,643,148 bytes
- Risk: LOW

Compilation still succeeded and execution still produced EXECUTION_OOM while
requesting approximately 1,073,742,080 bytes.

The accounting bug is fixed, but this false fit remains. It is therefore not
explained solely by driver-versus-allocator capacity accounting.

## Frozen 18-trial replay

The same recorded trial outcomes were replayed offline using the corrected
budget policy and the measured 3,221,225,472-byte allocator limit.

| Metric | Legacy | Corrected |
|---|---:|---:|
| False fits | 4 | 4 |
| False compile fits | 0 | 0 |
| False execution fits | 4 | 4 |
| False OOMs | 0 | 0 |
| LOW -> FIT | 7 | 7 |
| LOW -> OOM | 4 | 4 |

Risk counts changed from:

```text
Legacy:
  LOW: 11
  MODERATE: 5
  LIKELY EXCEEDS: 2

Corrected:
  LOW: 11
  MODERATE: 4
  LIKELY EXCEEDS: 3
```

The correction moved one large-demand case from MODERATE to LIKELY EXCEEDS,
without changing the false-fit count.

## Occupancy regression

With the corrected policy, the fixed attention workload remained monotonic:

| External allocation | Corrected budget | Risk |
|---:|---:|---|
| 256 MiB | 3,006,477,107 bytes | MODERATE |
| 768 MiB | 2,642,621,235 bytes | MODERATE |
| 1280 MiB | 1,568,879,411 bytes | LIKELY EXCEEDS |

External GPU users still reduce the budget. The allocator cap binds when driver
free memory is above the JAX limit; driver free memory binds under stronger
contention.

## Allocator modes

The diagnostic results were:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: 4096 attention EXECUTION_OOM
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform`: 4096 attention FIT
- Default preallocation: initialization CUDA OOM before a usable target row

The corrected production snapshot returns partial evidence for the platform
allocator because JAX allocator statistics were unavailable there. It does not
apply BFC-specific pool arithmetic to that mode.

## Binding and provenance

Snapshots now record allocator limit, largest free block, fraction-variable
source, and budget provenance. Budget provenance distinguishes allocator-plus-
driver evidence from driver-only or partial evidence. The current GPU case is
`ALLOCATOR_AND_DRIVER`, with `ALLOCATOR_LIMITED` binding in a fresh process.

## Remaining execution false fits

Four execution false fits remain. Their compiler-accounted temporary demand was
represented in `memory_analysis`, but compilation grew the JAX pool to about
2.2 GiB and left an allocator state in which the requested 1 GiB allocation did
not fit under the 3 GiB JAX limit. The platform allocator fitting the same case
supports allocator residency or fragmentation as the remaining mechanism.

These are now classified as workload-demand or allocator-residency modeling
limitations, not as the original device-capacity accounting bug.

## Pre-compilation guarantee

`jaxoom.assess(..., memory_limit="auto")` still does not compile or execute the
target workload. The correction only changes observational budget derivation.

## Limitations

The corrected budget is conservative when pool reuse is possible and cannot
predict future compiler pool growth or fragmentation. Driver and allocator
snapshots remain time-sensitive. CPU and unsupported accelerator behavior
remains partial when capacity is unavailable.
