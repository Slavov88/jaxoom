# Allocator gate discriminator study

Status: **ALLOCATOR GATE TOO CONSERVATIVE**.

## Design

A static-only search screened 1,632 attention geometries over B, H, S, D, float16, and float32. Twenty-three candidates were frozen before runtime validation. The primary discriminator region was aggregate PASS / top-two-live REJECT. Predictions used only JAXPR analysis, calibration, and the fixed 3 GiB budget; target compilation and execution were not used for selection.

The selected set covered B={1,2,4}, H={4,8,12,16}, D={32,64,96,128}, and both dtypes. It contained 17 aggregate-PASS/top-two-REJECT rows and six near-pass rows. Static screening found no top-two-vs-peak-live decision disagreements in this search; three top-two-vs-largest-buffer disagreement cases were included.

## Runtime result

There were 46 fresh-process executions: 20 stable FIT configurations and three stable EXECUTION_OOM configurations. No unstable rows or timeouts occurred.

Among the 17 aggregate-PASS/top-two-REJECT rows:

- 3 were OOM; these are allocator-gate true positives relative to aggregate-only.
- 14 FIT; these are top-two false rejects.

Thus the frozen gate removed all three aggregate false-safe outcomes in this selected region, but rejected 14 of 17 configurations that actually FIT. The extra term is informative in this narrow adversarial region, but its sharpness cost is too high.

## Candidate comparison

| Candidate | False-safe | False-reject | Unique OOM true positives |
|---|---:|---:|---:|
| Aggregate-only | 3 | 0 | 0 |
| Largest-buffer | 3 | 0 | 0 |
| Top-two-live | 0 | 14 | 3 |
| Peak-live | 0 | 14 | 3 |
| 1.10x | 3 | 0 | 0 |
| 1.25x | 3 | 0 | 0 |
| 1.50x | 3 | 0 | 0 |

Peak-live produced the same classifications as top-two-live for every selected row. Therefore top-two did not demonstrate superiority over peak-live here. Largest-buffer and all multiplier baselines passed every selected row and retained the three false-safe OOMs.

## Mechanistic diagnostics

Six post-prediction diagnostic rows were collected. The three float32 OOM rows exposed a 1,258,291,200-byte compiler temporary and a runtime request of approximately 1.17 GiB. The two-largest-live proxy was exactly 1,258,291,200 bytes on those rows. Float16 FIT diagnostics exposed an 838,860,800-byte compiler temporary while the static top-two proxy was 1.5 times larger. These diagnostics support a real temporary-allocation mechanism, but not a sufficiently selective production rule at the frozen threshold.

The capacity quantity `budget - calibrated_upper` was compared with allocator diagnostics and is retained only as an allocator safety capacity proxy. In representative OOM rows, the largest free block was 1 GiB while the failed request was about 1.17 GiB; this is consistent with fragmentation pressure but does not establish literal contiguous capacity. FIT diagnostic rows had a 512 MiB largest free block and completed, further showing that this quantity is not a standalone runtime guarantee.

## Double counting and matched analysis

Across the previous multi-regime rows plus this discriminator set, calibrated upper and top-two-live were very strongly positively related (Pearson r approximately 0.9998 for calibrated upper versus top-two-live). Calibrated upper versus largest-buffer had r approximately 0.9458. The discriminator set included aggregate-score-matched strata, but static quantization produced limited score spread within the primary group; it did not provide three clean, independently controlled aggregate-score groups. The result supports mechanistic relevance of top-two-live, while also showing that the frozen sum is overly conservative for many FIT geometries and may double-count correlated demand.

The matched top-two geometry rows did not show an additional failure pattern. No robust top-two-vs-peak-live separation was found.

## Non-attention disagreements

A separate frozen static screen found two feasible non-attention disagreement cases: CONV-1 at B=2048 and CONV-2 at B=1024. Both were stable OOM across two fresh runs. CONV-1 failed during compilation; CONV-2 failed during execution. Aggregate-only passed both, while top-two-live and peak-live rejected both. Largest-buffer rejected CONV-1 but passed CONV-2, producing one additional largest-buffer false-safe. These results show that the mechanism is not exclusively attention-specific, but they do not offset the 14 FIT false rejects in the primary attention discriminator set.

## Planner impact

Offline application to the existing planner boundary traces produced a median first-rejection/recommended-batch ratio of 0.842 and a minimum ratio of 0.0037. Three recommendations were reduced by more than 25%, and one by more than 50%, under this hypothetical rule. This is diagnostic only; planner behavior was not changed.

## Decision

**ALLOCATOR GATE TOO CONSERVATIVE.** The gate has genuine discriminator wins, but most aggregate-PASS/top-two-REJECT cases FIT, and peak-live performs identically on this set. A new development round would be required before any device-transfer study.

No production code changed.
