# Current-device budget validation

Status: OBSERVED

Environment: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB VRAM, JAX 0.11.0,
CUDA backend, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Method

A parent process took `jaxoom.device_memory()` snapshots and ran
`jaxoom.assess(..., memory_limit="auto")`. A fresh helper subprocess retained
256 MiB, 768 MiB, or 1280 MiB of JAX GPU allocation. The target assessment
used an abstract attention workload and did not compile or execute that target.
Each helper allocation was released before the next level.

## Occupancy pilot

| State | Driver free | Assessment budget | Upper headroom | Risk |
|---|---:|---:|---:|---|
| Before | 3,894 MiB | 3,689 MiB | -173 MiB | MODERATE |
| External 256 MiB | 3,493 MiB | 3,356 MiB | -506 MiB | MODERATE |
| External 768 MiB | 2,725 MiB | 2,520 MiB | -1,342 MiB | MODERATE |
| External 1280 MiB | 1,701 MiB | 1,496 MiB | -2,366 MiB | LIKELY EXCEEDS BUDGET |

The values are rounded for display. Raw values are in
`device_budget_validation_2026-09-08.json`. The fixed workload's risk moved in
the conservative direction at the highest pressure. All post-release snapshots
returned to approximately the baseline level.

## Latency

Median milliseconds from the same run:

| Operation | Median |
|---|---:|
| `estimate()` | 0.516 |
| `assess(memory_limit="auto")` | 60.716 |
| Device query | 55.948 |
| `compile_analyze()` | 1.695 |
| `analyze_donation()` | 9.704 |

The expected ordering did not hold in this environment because each
`nvidia-smi` query took about 56 ms. This is an observed portability tradeoff,
not a reason to claim that auto assessment is always faster than compilation.
No NVML dependency was added because the available environment did not expose
`pynvml` or `nvidia_ml_py`.

## Boundary and false-fit metrics

No compile or execution boundary sweep was run in this pilot. Therefore:

- FIT: NOT MEASURED
- COMPILE_OOM: NOT MEASURED
- EXECUTION_OOM: NOT MEASURED
- false fits: NOT MEASURED
- false OOMs: NOT MEASURED

The artifact intentionally does not convert this small occupancy observation
into an OOM accuracy claim.

## Interpretation and limitations

The effective available policy is documented in the package API and subtracts
a 5% physical-memory reserve bounded to 64 MiB through 256 MiB. Existing JAX
pool bytes not in use may contribute to effective availability when JAX reports
them. Driver used and free values include all visible users; `external_used`
is derived by subtracting JAX pool usage when possible and is approximate when
pool size is unavailable.

Free VRAM can change after the snapshot. Allocator fragmentation, CUDA context
and library overhead, unmodeled compiler workspaces, and calibration
applicability remain limitations. Donation analysis is not run by assessment.
High-risk callable assessments only provide a suggestion to run
`jaxoom.analyze_donation()` separately.
