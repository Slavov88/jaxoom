# Cross-device temporary-memory drift

**Status: COMPUTATIONALLY VERIFIED for the recorded RTX 3050 and Tesla T4 artifacts.**

## Environments and integrity

The source environment was the NVIDIA GeForce RTX 3050 Laptop GPU calibration
run. The validation environment was a Google Colab Tesla T4 with 15,360 MiB
reported VRAM, JAX 0.11.0, jaxlib 0.11.0, Python 3.13.15, driver 580.82.07,
and CUDA driver compatibility 13.0. The validation was pinned to jaxoom
commit `c5925ff7404953c3640756d9b15bea898e60477d`.

The returned raw artifact contains 46 unique matched cases, 23 float32 cases,
23 float16 cases, and 9 workload families. All 46 compiler cases succeeded.
All 46 static estimates matched exactly. For every device, compiler accounting
satisfied:

```text
argument bytes + output bytes + temporary bytes - alias bytes
```

All component byte counts were nonnegative. The copied artifacts are the raw
T4 JSON, summary JSON, environment JSON, and runtime smoke JSON.

## Frozen calibration transfer

The frozen RTX 3050-derived ratios were 0.673466, 1.000000, and 1.667651.
Recomputed T4 results are:

- Interval coverage: 36/46 = 78.3%
- Upper coverage: 40/46 = 87.0%
- Upper misses: 6/46 = 13.0%
- Median interval width/static: 0.9942

The six upper misses are two float32 convolutions and four autodiff cases.

| Case | Dtype | Static | RTX compiler | T4 compiler | Frozen upper | Shortfall | T4 temp | Temp delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| convolution-128x128x16x32 | f32 | 3,164,160 | 5,280,000 | 5,331,456 | 5,276,713 | 54,743 | 2,167,296 | 51,456 |
| convolution-256x256x32x64 | f32 | 25,239,552 | 42,090,752 | 42,295,808 | 42,090,752 | 205,056 | 17,056,256 | 205,056 |
| autodiff-b128-w256 | f32 | 786,436 | 1,310,740 | 4,981,284 | 1,311,500 | 3,669,784 | 4,325,904 | 3,670,544 |
| autodiff-b256-w512 | f32 | 3,145,732 | 3,145,748 | 7,340,580 | 5,245,982 | 2,094,598 | 4,719,120 | 4,194,832 |
| autodiff-b128-w256 | f16 | 458,754 | 917,538 | 4,588,066 | 765,041 | 3,823,025 | 4,260,368 | 3,670,528 |
| autodiff-b256-w512 | f16 | 1,835,010 | 2,621,474 | 5,767,714 | 3,060,155 | 2,707,559 | 4,456,976 | 3,146,240 |

All alias values were zero on both devices. The convolution shortfalls are
small ratio-tail effects. The autodiff shortfalls are dominated by additional
T4 temporary memory.

## Controls and stratification

Upper coverage by family was:

| Family | Upper coverage |
|---|---:|
| Attention | 8/8 |
| Autodiff | 0/4 |
| Convolution | 2/4 |
| Elementwise | 12/12 |
| Matmul | 4/4 |
| MLP | 6/6 |
| Residual | 4/4 |
| Training | 2/2 |
| Transformer | 2/2 |

The convolution float16 configurations with the same two shapes were controls
for the float32 convolution misses and were covered. The matrix contains no
nonfailing autodiff configuration, so an autodiff family control is not
available. The closest evidence is the exact same transformation at two sizes
and two dtypes, all four of which miss on T4.

Dtype coverage was 19/23 for float32 and 21/23 for float16. Size-bucket upper
coverage was 2/7 below 10 MiB, 27/28 for 10 to 100 MiB, 9/9 for 100 to 500 MiB,
and 2/2 above 500 MiB. The small-workload result is therefore strongly
associated with the autodiff cases, not with a broad failure across large
workloads.

## Temporary and compiler drift

Across all cases:

- Median absolute compiler drift: 0.048%
- P90 absolute compiler drift: 49.98%
- Maximum absolute compiler drift: 400.04%
- Temporary ratio median: 1.018
- Temporary ratio P90: 6.601
- Maximum temporary ratio: 9.001
- Temporary zero to nonzero transitions: 4
- Temporary nonzero to zero transitions: 0
- Nonzero alias deltas: 0

Family-level drift was near zero for elementwise and residual workloads. MLP,
matmul, and training showed moderate compiler drift without upper-bound misses.
The exceptional drift was localized to autodiff, where T4 temporary memory was
roughly 3.1 to 4.2 MiB higher than the RTX 3050 controls.

## IR inspection

The returned T4 artifact did not include StableHLO or optimized HLO. A compact
StableHLO summary was therefore collected only on the available RTX 3050
reference environment. It is not a cross-device IR comparison.

The reference autodiff cases had the same operation pattern across sizes:
2 `dot_general`, 1 `tanh`, 1 `reduce`, 1 `transpose`, and the corresponding
arithmetic and broadcast operations. Float16 added 3 `convert` operations.
The convolution cases each exposed one `convolution` operation. No evidence in
the available IR identifies a device-specific algorithm or custom call.

Consequently, the data supports backend lowering or workspace differences, but
cannot distinguish a T4-specific lowering choice from a library workspace
policy. A T4 IR capture would be required for that distinction.

## Ratio, additive, and hybrid diagnostics

These are analysis-only alternatives. No package behavior was changed.
Bounds use nearest-rank 0.10, 0.50, and 0.95 quantiles fitted on one device and
tested on the other.

| Fit to test | Method | Upper coverage | Median width/static | P90 width/static | Maximum width/static |
|---|---|---:|---:|---:|---:|
| RTX 3050 to T4 | Existing ratio | 40/46 = 87.0% | 0.994 | 0.994 | 0.994 |
| RTX 3050 to T4 | Additive residual | 43/46 = 93.5% | 2.314 | 24.687 | 169.285 |
| RTX 3050 to T4 | Hybrid max bound | 45/46 = 97.8% | 2.607 | 24.687 | 169.285 |
| T4 to RTX 3050 | Ratio fitted on T4 | 46/46 = 100.0% | 2.470 | 2.470 | 2.470 |
| T4 to RTX 3050 | Additive residual | 44/46 = 95.7% | 2.313 | 24.667 | 169.143 |
| T4 to RTX 3050 | Hybrid max bound | 46/46 = 100.0% | 4.081 | 24.667 | 169.143 |

The additive residual upper quantile improves cross-device upper coverage, but
its residual lower tail is highly negative because some workloads structurally
overpredict compiler accounting. The resulting interval is excessively wide.
The hybrid bound improves upper coverage, but also has a P90 width above 24
 times the structural estimate under this two-device sample. It is not a
shipping solution.

The reverse-direction results are in-sample for the fitted device and are
reported only as a leave-one-device-out diagnostic. They do not establish a
new calibration.

## Runtime smoke

The three T4 runtime smoke cases were `FIT`. Attention execution was below the
frozen upper bound. The MLP and training compiler-accounted values exceeded the
frozen upper bound for these smoke configurations, while sampled allocator
observations remained below it. This is consistent with the distinction
between compiler accounting and sampled execution observations.

## Conclusion

Structural analysis transferred exactly across the RTX 3050 and T4. Compiler
calibration transferred for most families, but autodiff showed a strong
operation-specific temporary-memory increase on T4. The convolution misses
were small upper-quantile effects rather than a comparable workspace jump.

The evidence favors a combination of device-specific workspace behavior and a
small-workload weakness in a pure ratio bound. The additive diagnostic improves
coverage only by producing impractically wide intervals. It does not establish
that a simple additive or hybrid rule is suitable for shipping.

**Calibration decision: retain ratio calibration unchanged and retain
exact-tested provenance.** No constants or structural analysis were modified.
