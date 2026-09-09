# T4 allocator-residency refinement

Status: OBSERVED

Source: `jaxoom_t4_allocator_residency_refined.zip`.

## Phase A: refinement

All seven existing useful T4 thresholds were refined in fresh subprocesses.

| Metric | Before | After |
|---|---:|---:|
| Median bracket width | 547,356,672 bytes | 58,720,256 bytes |
| Maximum bracket width | 782,237,696 bytes | 312,475,648 bytes |
| Brackets <=64 MiB | 0/7 | 6/7 |

The seven refined thresholds were:

| Family | Configuration | Dtype | OOM capacity | FIT capacity | Width |
|---|---|---|---:|---:|---:|
| Attention | S=3584 | float32 | 1,916,796,928 | 1,975,517,184 | 58,720,256 |
| Attention | S=3840 | float32 | 2,071,986,176 | 2,111,832,064 | 39,845,888 |
| Attention | S=4096 | float32 | 3,225,419,776 | 3,275,751,424 | 50,331,648 |
| Attention | S=4096 | float16 | 1,681,915,904 | 1,740,636,160 | 58,720,256 |
| Transformer | S=3072, W=2048 | float32 | 1,975,517,184 | 2,034,237,440 | 58,720,256 |
| Matmul | 8192^3 | float32 | 1,623,195,648 | 1,681,915,904 | 58,720,256 |
| Matmul | 10240^3 | float32 | 2,346,713,088 | 2,659,188,736 | 312,475,648 |

The T4 attention-4096 threshold is now:

```text
3,225,419,776 < required capacity <= 3,275,751,424 bytes
```

## Phase B: targeted candidates

The notebook screened and probed 12 additional candidates, but all completed
rows were FIT at both tested capacities. Therefore:

- New useful thresholds: 0
- New MLP thresholds: 0
- New training thresholds: 0
- New convolution thresholds: 0
- New autodiff thresholds: 0

This was a selection failure, not evidence that those families have no T4
thresholds. The candidate selector chose the low end of the RTX screening band,
while the T4 minimum tested capacity was 1,564,475,392 bytes. The next
campaign version corrects this by selecting the highest non-attention static
upper estimates and calibrating lower T4 allocator fractions.

## Paired transfer

Five configurations remain useful on both devices. Using conservative
smallest-FIT upper bounds:

- Median T4/RTX upper ratio: 0.894
- P90 upper ratio: 0.990
- Interval ratio lower bound minimum: 0.811
- Interval ratio upper bound maximum: 1.566

The interval range remains too wide for a portable predictor. The refined
attention-4096 pair is substantially narrower than before, but the matmul pair
and lower-size attention pairs retain uncertainty.

## Data sufficiency

**STILL INSUFFICIENT**.

Current T4 totals remain:

```text
7 useful thresholds
3 families
6 float32
1 float16
```

The model-fitting gate was not met. Model comparison was correctly recorded as
`NOT_RUN`.

## Production

**NO CHANGE**.

No structural estimator, compiler calibration, device budget, or `assess()`
behavior was modified.

## Corrective infrastructure

The refinement notebook is being updated to:

1. calibrate additional lower fractions near 0.05 to 0.10;
2. prioritize high static-demand non-attention candidates;
3. retain a bounded candidate count and checkpointing.
