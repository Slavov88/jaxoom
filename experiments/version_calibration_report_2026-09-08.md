# Version-aware calibration

**Status: COMPUTATIONALLY VERIFIED for the recorded environments.**

This milestone keeps the structural JAXPR estimator unchanged and makes
compiler-accounted calibration explicit by JAX version family. It does not
predict runtime peaks or OOM probability.

## Environment

The expanded current-JAX run used Python 3.12.3, JAX 0.11.0, jaxlib 0.11.0,
CUDA on one NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB physical VRAM, and
an effective JAX device limit of 3,221,225,472 bytes. It used
`XLA_FLAGS=--xla_gpu_autotune_level=0` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. The legacy comparison dataset used JAX
0.6.2 on the same GPU model.

## Dataset construction

The current-JAX calibration set contains 46 independent abstract-input
configurations: 9 workload families, 23 float32 cases, and 23 float16 cases.
Families are elementwise, matmul, residual, MLP, attention, Transformer-like,
training, convolution, and autodiff. Dimensions vary by case. Repeated runs of
the same configuration were not counted as separate calibration samples.

A grouped holdout contains 10 configurations from the MLP and autodiff
family/dtype groups. These groups were reserved because the committed
cross-version study identified them as underprediction-sensitive. The training
partition contains 36 configurations. This is a targeted diagnostic holdout,
not a random population sample.

The raw current-JAX artifact is
`jax_0_11_gpu_calibration_2026-09-08_v2.json`. The earlier 42-case pilot is
retained as `jax_0_11_gpu_calibration_2026-09-08.json`.

## Original transfer failure

Using the committed 14-case cross-version artifacts and the shipped JAX 0.6.x
GPU upper ratio:

- upper coverage on JAX 0.11.0: **9/14 = 64.3%**
- miss rate: **5/14 = 35.7%**

| Case | Family | Dtype | Static bytes | Legacy upper | JAX 0.11 compiler bytes | Shortfall | Shortfall/compiler | Temp bytes | Alias bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| convolution-f32 | convolution | float32 | 3,164,160 | 3,460,812 | 5,280,000 | 1,819,188 | 34.5% | 2,115,840 | 0 |
| grad-f32 | autodiff | float32 | 655,360 | 716,802 | 1,310,720 | 593,918 | 45.3% | 655,360 | 0 |
| mlp-f16 | MLP | float16 | 4,194,304 | 4,587,536 | 5,505,024 | 917,488 | 16.7% | 1,572,864 | 0 |
| mlp-f32 | MLP | float32 | 8,388,608 | 9,175,072 | 9,961,472 | 786,400 | 7.9% | 2,097,152 | 0 |
| value-and-grad-f32 | autodiff | float32 | 524,292 | 573,446 | 1,310,756 | 737,310 | 56.3% | 655,376 | 0 |

## What drifted

Structural bytes matched in all 14 cases. The compiler-side drift was not a
single uniform multiplier. On the matched cases, argument and alias bytes were
stable. Alias deltas were zero in every case. Temporary-memory deltas were the
main positive change in MLP and autodiff:

| Family | Mean old ratio | Mean new ratio | Temporary deltas in matched cases |
|---|---:|---:|---:|
| attention | 0.889 | 0.888 | -8,208, 0 |
| autodiff | 1.350 | 2.250 | +524,288, +524,288 |
| MLP | 1.063 | 1.250 | +1,048,576, +1,048,576 |
| other matched families | mostly unchanged | mostly unchanged | mostly 0 |

The evidence supports a family-specific temporary-memory explanation for the
largest transfer misses. It does not support changing the structural model.

## Candidate methods

Ratios are `compiler_accounted_bytes / static_peak_bytes`. Bounds use nearest
rank quantiles at 0.10, 0.50, and 0.95.

| Candidate | Lower | Central | Upper | Current expanded upper coverage | Holdout upper coverage |
|---|---:|---:|---:|---:|---:|
| Legacy JAX 0.6 shipped | 0.3333 | 1.0000 | 1.0938 | 41/46 = 89.1% | 7/10 = 70.0% |
| JAX 0.11 version-specific | 0.6735 | 1.0000 | 1.6677 | 44/46 = 95.7% | 9/10 = 90.0% |
| Pooled 0.6.2 + 0.11.0 | 0.6735 | 1.0000 | 1.4286 | 42/46 = 91.3% | 7/10 = 70.0% |
| Conservative version envelope | 0.3333 | 1.0000 | 1.6677 | 44/46 = 95.7% | 9/10 = 90.0% |

The pooled bound is narrower but fails the targeted holdout because pooling
reduces the upper quantile below the current-JAX autodiff and MLP behavior.
The version-specific bound restores the preferred approximate 90% holdout
coverage without using the maximum observed ratio.

## Cross-validation

For the JAX 0.11.0 version-specific candidate:

| Procedure | Upper coverage | Interval coverage |
|---|---:|---:|
| Leave-one-out | 43/46 = 93.5% | 38/46 = 82.6% |
| Leave-family-out | 41/46 = 89.1% | 37/46 = 80.4% |

The full current-JAX interval width is `1.6677 - 0.6735 = 0.9942` times the
structural estimate. On the held-out training-derived interval the width is
1.3343 times structural because the training lower quantile is 0.3333.

## Transfer in both directions

Using expanded datasets and the 95th-quantile candidates:

- Legacy 0.6.2 shipped calibration on current 0.11.0 data: **41/46 = 89.1%**.
- Current 0.11.0 calibration on legacy 0.6.2 data: **38/38 = 100.0%**.
- The original matched 14-case legacy-to-current result remains **9/14 = 64.3%**.

The directions are asymmetric. The expanded legacy set contains additional
high-ratio cases, but those were not part of the shipped legacy calibration.

## Version policy

Built-in calibration now has explicit provenance:

- `EXACT_TESTED`: the active JAX version is listed in the calibration entry.
- `VERSION_FAMILY_MATCH`: the major.minor family matches, but the exact version
  was not tested. The result names the tested versions in its limitations.
- `UNCALIBRATED`: no validated entry exists for the backend and active version.

The shipped entries are JAX 0.6.2 CPU, JAX 0.6.2 GPU, and JAX 0.11.0 GPU.
JAX 0.11.0 CPU remains uncalibrated because no expanded current-CPU dataset was
created. `estimate()` remains version-independent. `assess()` marks an
uncalibrated result as not calibrated and uses a structural fallback risk.

No stale calibration is selected for an unknown JAX family.

## Runtime sanity checks

The existing experimental execution harness was run under JAX 0.11.0 with the
new GPU calibration:

| Workload | Static | New upper | Compiler accounting | Sampled execution observation | Status |
|---|---:|---:|---:|---:|---|
| attention, sequence 512, float32 | 17,842,176 | 29,754,514 | 18,874,368 | 3,145,216 | FIT |
| MLP, batch 256, width 1024, depth 2, float32 | 8,388,608 | 13,989,266 | 9,961,472 | 8,912,384 | FIT |
| training, batch 128, width 512, float32 | 4,200,452 | 7,004,886 | 4,069,692 | 4,725,248 | FIT |

The observation source was sampled allocator `bytes_in_use` during repeated
calls after warmup. It is not an exact execution peak. These three smoke cases
produced no regression. The known sequence-8192 attention compile boundary was
not rerun in this milestone.

## Storage and limitations

Only quantile parameters and provenance are shipped in
`src/jaxoom/calibration_defaults.py`. Raw datasets remain under
`experiments/`. Evidence is limited to one GPU model, one JAX 0.11.0 release,
one allocator configuration, and abstract compiler cases. The holdout is
small and targeted. No claim is made for other JAX releases, GPU models, or
runtime OOM behavior.
