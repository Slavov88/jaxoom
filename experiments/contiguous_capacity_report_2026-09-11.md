# Separated allocator request/capacity study

Status: **CONTIGUOUS CAPACITY MODEL PROMISING BUT INSUFFICIENT**.

## Frozen evidence and split

The study reused the frozen discriminator evidence without relabeling it: 25 stable RTX 3050 rows, including 20 FIT rows and 5 OOM rows (three attention and two convolution). The old aggregate gate had 5 false-safe and 0 false-reject cases; the old additive top-two and peak-live gates had 0 false-safe and 14 false-reject cases.

The deterministic split contained 8 development rows and 17 evaluation rows. Development included three OOMs (two attention and CONV-1) and five FITs. Evaluation retained two OOMs (one attention and CONV-2) plus 15 FITs. The split was created for this retrospective modeling study; it is not a prospective preregistration.

## Candidate separated rule

The selected experimental rule is:

```text
aggregate_pass = calibrated_upper <= budget
request_upper = largest_buffer * dtype_bytes / 2
capacity_lower = allocator_limit / 3
allocator_pass = request_upper <= capacity_lower
final_pass = aggregate_pass AND allocator_pass
```

Thus the request multiplier is 2 for float32 and 1 for float16. This is a small, interpretable, dtype-scaled largest-buffer candidate derived from the development diagnostic rows. It is not production logic and is not a claim about exact compiler allocation behavior.

The capacity value is 1 GiB in the tested 3 GiB BFC environment. It is treated as an empirical safe-capacity threshold, not literal contiguous free memory.

## Classification results

| Model | Development false-safe | Development false-reject | Evaluation false-safe | Evaluation false-reject | Overall false-safe | Overall false-reject |
|---|---:|---:|---:|---:|---:|---:|
| Aggregate-only | 3 | 0 | 2 | 0 | 5 | 0 |
| Old additive top-two | 0 | 3 | 0 | 11 | 0 | 14 |
| Old additive peak-live | 0 | 3 | 0 | 11 | 0 | 14 |
| 1.10x | 3 | 0 | 2 | 0 | 5 | 0 |
| 1.25x | 2 | 0 | 2 | 0 | 4 | 0 |
| 1.50x | 2 | 0 | 2 | 0 | 4 | 0 |
| New separated rule | 0 | 0 | 0 | 0 | 0 | 0 |

On this small frozen set, the separated rule recovered all 14 FIT rows rejected by the old additive top-two gate and retained all five known OOM detections.

This result is promising but not validation-grade: the classification improvement is driven by a small, highly selected RTX 3050 dataset and several capacity formulas tied on the development split.

## Request proxy study

Available compiler temporary labels covered six attention rows. For the selected dtype-scaled-largest proxy:

```text
N:                          6 compiler-temp rows
underpredictions:           0
median proxy/temp ratio:    1.00
p90 proxy/temp ratio:       1.00
```

The observed failed-allocation-request labels added five rows. The proxy underpredicted one row, CONV-2, by approximately 16 MiB; it overpredicted the other four. This is not systematic underprediction, but the sample is too small to establish a safety envelope.

The three float32 attention OOM rows had a 1,258,291,200-byte proxy and compiler temporary. The float16 FIT diagnostic rows had a proxy and compiler temporary of 838,860,800 bytes or 761,266,176 bytes, respectively.

## Capacity diagnostics

Eight rows had largest-free-block diagnostics. The selected 1 GiB capacity threshold:

```text
capacity diagnostic rows:       8
capacity > largest-free block:  4
```

The overpredictions occurred primarily on float16 FIT rows with a measured 512 MiB largest free block. Those rows nevertheless FIT, so largest-free-block is not a complete capacity label for this task. The threshold should not be interpreted as a literal allocator block guarantee.

For representative float32 OOM rows, the largest free block was 1 GiB and the failed request was approximately 1.17 GiB. CONV-1 failed during compilation; CONV-2 failed during execution. These are reported as separate failure modes.

## Capacity candidates

The study compared small interpretable capacity families:

```text
allocator_limit / 3
0.28 * allocator_limit
allocator_limit / 3 - 0.05 * structural_peak
allocator_limit / 3 - 0.10 * structural_peak
```

Combined with the selected request proxy, all four produced the same 0/0 classification result on this small set. The capacity parameter is therefore not identified by the available outcomes. This is a central reason not to call the model validated.

## Historical attention sweep

The separated rule preserved the historical FIT points and rejected the historical OOM points:

| S | Historical outcome | Separated prediction |
|---:|---|---|
| 2048 | FIT | pass |
| 3072 | FIT | pass |
| 4096 | FIT / unstable provenance | pass |
| 4608 | EXECUTION_OOM | reject |
| 5120 | EXECUTION_OOM | reject |

## Convolution

Both known aggregate-pass convolution OOMs were rejected:

```text
CONV-1 B=2048: COMPILE_OOM
CONV-2 B=1024: EXECUTION_OOM
```

CONV-2 is the only observed request-label underprediction and was still rejected because its request proxy exceeded the 1 GiB capacity threshold.

## Double counting

The separated rule does not add calibrated upper and top-two-live. It compares a request proxy against an independently parameterized empirical capacity threshold. On this dataset, this removes the old additive gate's 14 FIT false rejects. However, the capacity threshold is based on a sparse same-device diagnostic sample and does not yet establish an independent physical capacity measurement.

## Monotonicity and purity

Static checks passed for increasing sequence length and decreasing budget. Predictions use only static analysis, calibration-independent request features, and allocator-limit metadata.

```text
target compilation: NO
target execution:    NO
compiler diagnostics: NO
allocator diagnostics: NO
```

## Decision

**CONTIGUOUS CAPACITY MODEL PROMISING BUT INSUFFICIENT.** The separated architecture materially improves the frozen classification tradeoff, but the request envelope has only six compiler-temp labels and the capacity candidates are not identified. More same-device targeted capacity diagnostics are required before held-out separated-gate validation.

No production API changed.
