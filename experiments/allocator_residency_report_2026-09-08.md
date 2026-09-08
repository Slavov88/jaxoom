# Allocator residency validation

Status: OBSERVED

No production allocator-residency model was shipped.

## Dataset

Primary environment:

- NVIDIA GeForce RTX 3050 Laptop GPU
- JAX 0.11.0 and jaxlib 0.11.0
- CUDA backend
- BFC-style allocator
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

The fresh fraction-probe set contained 21 trials across seven workload
configurations from attention, MLP, training, and transformer workloads. It was
combined analytically with the historical 18-trial OOM benchmark, giving 39
recorded rows, but the historical rows are not treated as pristine held-out
data. An attempted expansion to width-12288 MLP/training cases timed out in a
fresh subprocess and was not included as successful threshold evidence.

Fresh outcomes:

- FIT: 19
- EXECUTION_OOM: 2
- COMPILE_OOM: 0

## Attention 4096 threshold

The exact known false-fit configuration was tested by varying
`XLA_CLIENT_MEM_FRACTION`:

| Allocator limit | Fraction | Outcome |
|---:|---:|---|
| 3,221,225,472 | 0.75 | EXECUTION_OOM |
| 3,265,265,664 | 0.76 | EXECUTION_OOM |
| 3,307,208,704 | 0.77 | FIT |

The observed required allocator-capacity bracket is therefore:

```text
3,265,265,664 < required capacity <= 3,307,208,704 bytes
```

Bracket width: 41,943,040 bytes.

The compiler-calibrated upper bound remains 1,804,833,959 bytes, so compiler
upper memory is not a sufficient proxy for end-to-end allocator capacity.

## Post-compile residency

Across the 21 fresh probes:

- Median pool growth: 849,346,560 bytes
- Minimum pool growth: 268,435,456 bytes
- Maximum pool growth: 2,214,592,512 bytes
- Median post-compile pool fraction: 44.7%

Pool growth varied substantially by workload and allocator limit. It cannot be
replaced with one fixed global constant based on this dataset.

## Current compiler-upper baseline

Using the existing prediction rule:

```text
predicted FIT if calibrated_upper <= corrected assessment budget
```

there were 2 fresh false fits. Both were the attention 4096 cases at allocator
fractions 0.75 and 0.76. There were no fresh false OOMs.

## Static predictability

The fresh harness currently records structural size, calibrated interval,
dtype, family, configuration, allocator fraction, and phase memory state.

The exact attention threshold shows a useful allocator-capacity label, but there
is only one configuration with both a nearby OOM and FIT threshold. That is not
enough to estimate a validated global ratio, additive overhead, or
primitive-aware model.

The simple candidate comparison is therefore:

| Model | Result |
|---|---|
| Existing compiler upper | 2 fresh false fits |
| Global allocator ratio | Not identifiable; would overpredict controls |
| Additive allocator overhead | Not identifiable; would overpredict controls |
| Two-part maximum | Equivalent to current upper on this data |
| Primitive-aware model | Not enough held-out threshold labels |

A global ratio large enough to cover attention 4096 would substantially
overpredict the MLP, training, and transformer controls. No candidate met the
required held-out sharpness standard.

## First execution and controls

The existing diagnostic measurements showed three successful executions for
small attention and control workloads with stable pool residency after the
first call. The remaining failure is concentrated near a BFC allocator
capacity/fragmentation boundary rather than a universally large first-execution
allocation.

The platform allocator remains a diagnostic control only. It FIT the attention
4096 case where BFC-style allocation failed. Its JAX allocator statistics were
not available and its rows are excluded from the primary BFC dataset.

## Holdout evaluation

A genuine 70/30 or family-holdout evaluation was not run. The available fresh
set contains only one useful nearby OOM/FIT threshold, so fitting and reporting
held-out model performance would be misleading. The holdout artifact records
this as `NOT RUN`.

## Historical 18-trial benchmark

The historical benchmark remains unchanged and separate. The corrected current
predictor had:

- 4 false fits
- 4 LOW to EXECUTION_OOM outcomes
- 0 false OOMs

The fresh threshold data are not sufficient to claim an improvement or to fit a
production allocator-residency model.

## Pre-compilation guarantee

The residency harness compiles targets only in fresh experimental subprocesses.
Plain `jaxoom.assess(..., memory_limit="auto")` remains compilation-free.

## Production decision

**KEEP EXPERIMENTAL**.

The evidence supports allocator-capacity dependence, but not a sufficiently
sharp pre-compilation allocator-residency model. No attention-specific margin,
global runtime margin, or calibration change was introduced.

## Artifacts

- `allocator_residency_validation.py`
- `allocator_residency_validation_2026-09-08.json`
- `allocator_residency_thresholds_2026-09-08.json`
- `allocator_residency_summary_2026-09-08.json`
- `allocator_residency_report_2026-09-08.md`
