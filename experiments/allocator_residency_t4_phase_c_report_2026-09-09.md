# T4 allocator-residency Phase C

Status: OBSERVED

Source: `jaxoom_t4_allocator_residency_refined (1).zip`.

## Dataset

The campaign ran on a Tesla T4 with JAX 0.11.0, jaxlib 0.11.0, CUDA, and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Phase C outcomes:

- FIT: 46
- EXECUTION_OOM: 17
- COMPILE_OOM: 7
- EXECUTION_TIMEOUT: 1

The combined T4 dataset contains 10 useful thresholds:

- Existing refined: 7
- New: 3
- Families: attention, transformer, matmul
- Dtypes: float32 6, float16 4

## New thresholds

| Family | Configuration | Dtype | OOM capacity | FIT capacity | Width |
|---|---|---|---:|---:|---:|
| Transformer | S=3072, W=2048 | float16 | 782,237,696 | 939,524,096 | 157,286,400 |
| Matmul | 10240^3 | float16 | 1,409,286,144 | 1,564,475,392 | 155,189,248 |
| Attention | S=3840 | float16 | 939,524,096 | 1,094,713,344 | 155,189,248 |

The new labels improve dtype coverage but add no new family. All three new
thresholds have brackets wider than 64 MiB.

## Candidate-selection diagnosis

The ranking selected 12 candidates, including higher-demand non-attention rows,
but the resulting probes still yielded only three useful thresholds. Several
selected MLP, training, convolution, and autodiff candidates FIT at both
probed capacities. One candidate timed out.

This is an observed limitation of the current static screening envelope: the
RTX-derived calibrated-upper ranges do not accurately locate T4 thresholds for
these families. The selector is not yet a validated capacity predictor.

## Brackets

Across all 10 T4 thresholds:

- Median bracket width: 58,720,256 bytes
- Maximum bracket width: 312,475,648 bytes
- Brackets <=64 MiB: 6/10

The refined original seven remain the high-precision portion of the dataset.

## Paired transfer

The package reports five paired configurations. Their conservative upper-bound
ratios remain available in `allocator_residency_t4_phase_c_paired_comparison.json`.
The new Phase C labels do not materially expand paired-device coverage because
they are not useful RTX-paired thresholds.

No cross-device model was fit.

## Data sufficiency

**STILL INSUFFICIENT**.

The model gate remains unmet:

```text
required: >=12 useful T4 thresholds and >=3 families
observed: 10 useful T4 thresholds and 3 families
```

A model comparison was correctly recorded as `NOT_RUN`.

## Production

**NO CHANGE**.

No production estimator, calibration, device-budget, or `assess()` behavior was
modified.

## Next campaign adjustment

The next candidate search should add T4-specific moderate-scale configurations
rather than rely only on the RTX screening manifest. In particular, it should
screen larger MLP/training/autodiff/convolution variants while using the dense
0.05 to 0.35 T4 capacity map, then select candidates whose static demand is
near the measured T4 boundary.
