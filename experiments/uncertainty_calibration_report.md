# Uncertainty calibration report

Status: COMPUTATIONALLY VERIFIED for the recorded JAX 0.6.2 datasets. The
intervals target compiler-accounted memory, not runtime peak memory or OOM
probability.

## Dataset

Two backend-specific datasets were evaluated:

| Backend | Cases | Families | Source |
|---|---:|---|---|
| CPU | 38 | elementwise, matmul, residual, MLP, attention, Transformer, training | `accelerator_calibration_cpu_2026-09-07_v2.csv` |
| NVIDIA CUDA GPU | 38 | same seven families | `accelerator_calibration_gpu_2026-09-07_v2.csv` |

Both datasets use JAX/jaxlib 0.6.2 and contain float32/float16 cases spanning
small tensors through approximately 4 GiB of structural logical memory on the
largest attention case. The GPU dataset includes a JAX device budget of
3,220,832,256 bytes.

The independent runtime dataset contains four attention trials at sequence
lengths 1024, 2048, 4096, and 8192. The last trial failed during GPU compilation
with `RESOURCE_EXHAUSTED`.

## Target and residual definitions

The target is:

```text
compiler_accounted_bytes
 = argument bytes + output bytes + temporary bytes - alias bytes
```

For each calibration row:

```text
ratio = compiler_accounted_bytes / static_peak_bytes
additive residual = compiler_accounted_bytes - static_peak_bytes
```

The signed comparison used elsewhere in the repository remains unchanged:

```text
static_peak_bytes - compiler_accounted_bytes
```

The shipped interval uses nearest-rank empirical ratio quantiles at 10%, 50%,
and 90%. The upper endpoint is the one-sided compiler-accounting bound.

## Methods compared

- `raw_structural`: no calibration
- `global_ratio`: backend-specific compiler/static ratio quantiles
- `global_additive`: backend-specific additive residual quantiles
- `family_ratio`: family-specific ratio diagnostic
- `size_bucket_ratio`: size-bucket ratio diagnostic

Evaluation used leave-one-out and leave-family-out protocols. Family and size
methods are diagnostic comparisons. The public report does not yet expose a
reliable family or size bucket label, so the shipped method is backend-specific
global ratio calibration.

## Leave-one-out results

The reported coverage is held-out coverage, not in-sample quantile coverage.

| Backend | Method | Interval coverage | Upper coverage | Median width / static |
|---|---|---:|---:|---:|
| CPU | raw structural | 36.8% | 47.4% | 0.00 |
| CPU | global ratio | 78.9% | 89.5% | 2.635 |
| CPU | global additive | 78.9% | 89.5% | 1.402 |
| CPU | family ratio | 84.2% | 92.1% | 0.097 |
| CPU | size bucket ratio | 84.2% | 89.5% | 1.969 |
| GPU | raw structural | 36.8% | 63.2% | 0.00 |
| GPU | global ratio | 78.9% | 89.5% | 0.760 |
| GPU | global additive | 78.9% | 89.5% | 1.146 |
| GPU | family ratio | 76.3% | 86.8% | 0.111 |
| GPU | size bucket ratio | 81.6% | 89.5% | 0.417 |

Leave-family-out upper coverage for the shipped global ratio method was 86.8%
for CPU and 84.2% for GPU. This is lower than leave-one-out coverage and shows
that neighboring workload families matter.

## Chosen method

The shipped method is **backend-specific global ratio calibration**.

Reasons:

1. It improves GPU held-out upper coverage from 63.2% to 89.5%.
2. It improves CPU held-out upper coverage from 47.4% to 89.5%.
3. It scales with the structural estimate, unlike a fixed byte correction.
4. It does not require speculative primitive or family classification.
5. Family and size diagnostics are narrower in some cases, but their labels are
   not yet reliable inputs to the public API.

The GPU summary is scoped to NVIDIA CUDA and JAX 0.6.x. The CPU summary is
scoped to the recorded JAX 0.6.x CPU dataset. A different JAX version is marked
as limited rather than silently treated as equivalent.

## Risk classifier

`jaxoom.assess(report, memory_limit=...)` is separate from `estimate()` and
returns a typed `MemoryAssessment`.

Conceptually:

```text
structural > budget
    LIKELY EXCEEDS BUDGET

structural <= budget < calibrated central
    HIGH

calibrated central <= budget < calibrated upper
    MODERATE

calibrated upper <= budget
    LOW
```

The result is a qualitative compiler-accounting risk level. It is not a
probability of runtime OOM. If no backend calibration exists, the API returns an
explicit raw structural fallback and never labels it as calibrated.

## Actual GPU fit/OOM trials

| Sequence | Static | Calibrated upper | Compiler | Actual |
|---:|---:|---:|---:|---|
| 1024 | 69,238,784 | approximately 75.7M | 71,303,168 | FIT |
| 2048 | 272,695,296 | approximately 298.3M | 276,824,064 | FIT |
| 4096 | 1,082,261,504 | approximately 1.18G | 1,090,519,040 | FIT |
| 8192 | 4,312,006,656 | approximately 4.72G | unavailable | OOM during compilation |

Using the JAX device budget, raw structural and calibrated classifications both
had zero false-fit and zero false-OOM cases in this four-trial sample. The OOM
trial failed during compilation, so it does not validate execution-time peak
prediction.

## Failure cases and limitations

- The GPU upper-bound miss rate remains 10.5% under leave-one-out evaluation.
- Leave-family-out coverage is lower, at 84.2% for GPU and 86.8% for CPU.
- CPU intervals are very wide because residual cases have large
  compiler/static ratios.
- The calibration dataset is dominated by one JAX/jaxlib release and one
  NVIDIA GPU environment.
- Runtime observations are too sparse to support OOM probabilities.
- The interval does not model allocator fragmentation, system contention,
  fusion, or runtime workspaces directly.

No correction factors were applied to the structural estimator.
