# Tesla T4 allocator-residency campaign

Status: OBSERVED

Source: `jaxoom_t4_allocator_residency.zip` supplied from Colab.

## Environment

- Tesla T4
- CUDA backend
- JAX 0.11.0
- jaxlib 0.11.0
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- Infrastructure commit: `fe3be15a3ebcfb4fbc5164c14fbfb682c53302c1`

The T4 identity and versions were verified in the notebook.

## Fraction calibration

Requested fractions mapped to measured JAX allocator limits:

| Fraction | Actual bytes limit |
|---:|---:|
| 0.10 | 1,564,475,392 |
| 0.13 | 2,034,237,440 |
| 0.17 | 2,659,188,736 |
| 0.20 | 3,128,950,784 |
| 0.25 | 3,911,188,480 |
| 0.30 | 4,691,329,024 |
| 0.35 | 5,473,566,720 |
| 0.40 | 6,255,804,416 |
| 0.45 | 7,038,042,112 |
| 0.50 | 7,820,279,808 |

Actual `bytes_limit`, not requested fraction, was used for labels.
Driver-free and physical-total fields were unavailable in the Colab fraction
mapping, but JAX allocator limits were present and consistent.

## Threshold campaign

79 probes were reported in the notebook summary. The packaged raw rows contain
79 threshold rows after excluding the calibration map.

Outcomes:

- FIT: 65
- EXECUTION_OOM: 6
- COMPILE_OOM: 4
- EXECUTION_TIMEOUT: 4

Useful FIT/OOM threshold labels: **7**.

| Family | Useful thresholds |
|---|---:|
| Attention | 4 |
| Transformer | 1 |
| Matmul | 2 |
| MLP | 0 |
| Training | 0 |
| Convolution | 0 |
| Autodiff | 0 |

Dtypes:

- float32: 6
- float16: 1

The lower-capacity strategy was substantially more efficient than the RTX
campaign: four execution timeouts and no compile timeouts were reported. It
still did not reach the target of 12 to 20 useful T4 labels.

Threshold brackets were coarse because the available capacity grid was coarse:

- Median bracket width: 547,356,672 bytes
- Smallest bracket width: 469,762,048 bytes
- Largest bracket width: 782,237,696 bytes

The 64 MiB refinement target was not reached.

## Attention 4096 paired result

T4 float32:

```text
3,128,950,784 < required capacity <= 3,911,188,480 bytes
```

The RTX reference threshold was:

```text
3,265,265,664 < required capacity <= 3,307,208,704 bytes
```

The T4 result is not directly comparable at 64 MiB precision because the T4
fraction grid did not bracket the transition closely enough.

## Paired-device comparison

Five configurations had both T4 and RTX useful upper-threshold labels.
Using conservative smallest-FIT upper bounds:

- Median T4/RTX upper-threshold ratio: 1.052
- P90 T4/RTX upper-threshold ratio: 1.183
- Median absolute upper-threshold drift: 295,698,432 bytes

Paired upper ratios ranged from 0.861 to 1.183. The sample is too small and
contains four attention cases plus one matmul case to establish portability.

The T4 threshold upper bound normalized by T4 static quantities also varied:

- T4 upper / structural peak: approximately 2.45 to 3.61
- T4 upper / calibrated compiler upper: approximately 1.47 to 2.17

This does not yet support a stable cross-device normalization.

## Post-compile diagnostics

For paired rows, post-compile pool values were recorded in the raw probes. At
the smallest successful allocator limit, pool residency was often close to the
allocator limit. These values are diagnostic labels only and were not used as
pre-compilation features.

Backend and allocator behavior remain device-specific. No conclusion is drawn
that equal JAXPRs imply equal allocator requirements.

## Model-fitting gate

Not met:

```text
required T4 thresholds: >= 12
observed T4 thresholds: 7
```

No T4-only, RTX-to-T4, T4-to-RTX, pooled, or family-holdout model was fit.
The historical RTX benchmark was not used for tuning.

## Production decision

**NO CHANGE**.

- No residency predictor shipped.
- Structural estimator unchanged.
- Compiler calibration unchanged.
- Device-budget accounting unchanged.
- `assess()` semantics unchanged.

## Artifacts

- `allocator_residency_t4_raw_2026-09-09.json`
- `allocator_residency_t4_summary_2026-09-09.json`
- `allocator_residency_t4_report_2026-09-09.md`
