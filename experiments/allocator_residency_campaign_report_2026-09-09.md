# Targeted allocator-residency campaign

Status: OBSERVED

Production code was not changed.

## September 8 commits

- `cce9b67` `test: expand allocator residency threshold study`
- `2a5210a` `docs: document allocator residency evidence limits`
- CI was green before this campaign: pending verification is reported separately.

## Screening

The cheap abstract screening phase evaluated 138 candidate rows across:

- attention
- MLP
- training
- transformer
- matmul
- convolution
- autodiff
- float32 and float16

Screening results:

- threshold candidates: 26
- screened out as too small: 112
- screened out as too large: 0
- screening failures: 0

The screening used structural peak, largest-buffer metadata, input size,
equation count, primitive counts, and calibrated compiler upper memory. It did
not execute or compile target workloads.

## Threshold probes

The campaign produced 130 fresh subprocess probes:

- FIT: 62
- EXECUTION_OOM: 9
- COMPILE_OOM: 16
- COMPILE_TIMEOUT: 40
- OTHER_FAILURE: 3

Each allocator-fraction subprocess used a fresh process and recorded the actual
allocator limit from JAX snapshots. Timeout was never converted to OOM.

Useful threshold labels require both an OOM/compile-OOM point and a FIT point.
There were 10 useful thresholded configurations:

| Family | Useful thresholds |
|---|---:|
| Attention | 4 |
| MLP | 2 |
| Training | 3 |
| Matmul | 1 |
| Transformer | 0 |
| Convolution | 0 |
| Autodiff | 0 |

Dtypes:

- float32: 9
- float16: 1

Median bracket width: 429,916,160 bytes.

The target of 12 to 20 useful thresholds was not reached. Large training,
transformer, convolution, and autodiff candidates were dominated by compile
timeouts or did not cross a boundary in the safe fraction range.

## Threshold examples

Attention float32:

- S=3584: 1,933,574,144 < required <= 2,363,490,304 bytes
- S=3840: 1,933,574,144 < required <= 2,363,490,304 bytes
- S=4096: 3,265,265,664 < required <= 3,307,208,704 bytes when combined with the validated September 8 threshold probes

MLP float32, width 8192, depth 4:

- batch 64: 1,289,748,480 < required <= 1,503,657,984 bytes
- batch 128: 1,289,748,480 < required <= 1,503,657,984 bytes

Training float32, width 8192:

- batches 64, 128, and 256 produced compile-OOM/FIT brackets, but the brackets were wide because intermediate fractions timed out.

Matmul float32, 8192 by 8192:

- 1,073,741,824 < required <= 1,933,574,144 bytes
- the lower bracket endpoint was compile OOM; an intermediate probe timed out.

## Family results

Attention and MLP yielded the most reproducible thresholds. Training produced
three labels, but with large uncertainty. Matmul produced one label. Transformer
large-scale candidates repeatedly exceeded the 75-second compile cap. The
larger convolution and autodiff candidates either timed out or FIT throughout
the tested range.

This is evidence about experiment feasibility, not evidence that those families
lack allocator thresholds in general.

## Data sufficiency

**STILL INSUFFICIENT**.

The minimum modeling gate was not met:

```text
required: >= 12 independent useful thresholds and >= 3 families
observed: 10 useful thresholds across 4 families
```

The threshold count is close, but the bracket distribution is not yet sharp and
compile timeouts contaminate several large candidates. No model fitting,
holdout evaluation, or historical-benchmark model selection was performed.

## Production

No production changes:

- structural estimator unchanged
- compiler calibration unchanged
- corrected device-budget logic unchanged
- `assess(auto)` remains compilation-free
- no allocator predictor shipped

## Artifacts

- `allocator_residency_campaign.py`
- `allocator_residency_screening_2026-09-09.json`
- `allocator_residency_thresholds_2026-09-09.json`
- `allocator_residency_campaign_summary_2026-09-09.json`
- `allocator_residency_campaign_report_2026-09-09.md`

## Stop decision

The targeted campaign reached 10 useful thresholds but did not meet the
minimum modeling criterion. Further RTX 3050 searches should not proceed
indefinitely. A larger GPU is a more efficient next platform for residency data.
