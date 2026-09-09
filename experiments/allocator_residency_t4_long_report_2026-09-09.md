# Long T4 allocator-residency campaign

Status: OBSERVED

Source: `jaxoom_t4_allocator_residency_long_campaign.zip`.

## Environment

- Tesla T4
- JAX 0.11.0
- jaxlib 0.11.0
- CUDA
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

## Campaign

Stage A scale search:

- Probes: 74
- FIT: 54
- COMPILE_OOM: 6
- EXECUTION_OOM: 1
- EXECUTION_TIMEOUT: 13

Seven scale-boundary candidates were selected:

- Autodiff: 3
- Convolution: 2
- MLP: 1
- Transformer: 1
- Training: 0

Stage B threshold probes:

- Probes: 18
- FIT: 8
- EXECUTION_OOM: 3
- COMPILE_OOM: 2
- EXECUTION_TIMEOUT: 1
- OTHER_FAILURE: 4

## New thresholds

The package reported two new threshold rows:

1. Transformer, float32, sequence 3072, width 2048
2. Convolution, float32, batch 4, 512x512, 96 to 192 channels

The transformer row duplicates an existing frozen T4 configuration under a
legacy configuration-ID scheme. After configuration-based deduplication, only
one genuinely new threshold was added: convolution.

The convolution bracket was very wide:

```text
627,048,448 < required capacity <= 5,473,566,720 bytes
```

It is retained as an observed FIT/OOM interval but is low precision and not
suitable for sharp model fitting.

## Distinct T4 evidence

After merging the frozen September evidence, Phase C evidence, and this run:

- Distinct useful thresholds: 11
- Attention: 5
- Matmul: 3
- Transformer: 2
- Convolution: 1
- MLP: 0
- Training: 0
- Autodiff: 0
- Float32: 7
- Float16: 4

The modeling gate was not met. More importantly, the missing MLP, training, and
autodiff regimes remain unresolved.

## Scale-search diagnosis

The family-directed scale search reduced selection to seven candidates, but the
scale boundary was often a compile boundary rather than an allocator-capacity
boundary. Training produced no selected OOM candidate. MLP and autodiff
selected candidates predominantly produced compile OOM or did not yield a
usable FIT/OOM pair. Convolution produced one extremely wide interval.

This confirms that workload-scale search is more informative than the previous
static-only ranking, but the current scale grid still does not isolate useful
moderate-compute memory regimes for the missing families.

## Integrity issue corrected

The long-campaign configuration ID used a different JSON field name from the
existing campaign, causing the transformer duplicate to appear as a new row.
The infrastructure now uses the repository-wide configuration-ID convention.
The notebook merge logic also includes the frozen Phase C thresholds and
deduplicates by configuration identity.

## Model fitting

Not run:

```text
modeling gate not met
```

No T4 internal model, family holdout, or cross-device model was fit.

## Production

**NO CHANGE**.

No production code, estimator, calibration, device-budget logic, or
`assess()` semantics changed.
