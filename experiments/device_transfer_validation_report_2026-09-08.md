# Cross-device GPU validation

**Status: COMPUTATIONALLY VERIFIED.**

## Environments

- Source: NVIDIA GeForce RTX 3050 Laptop GPU, JAX 0.11.0, jaxlib 0.11.0
- Validation: NVIDIA Tesla T4, 15,360 MiB, JAX 0.11.0, jaxlib 0.11.0
- T4 Python: 3.13.15
- T4 driver: 580.82.07
- Pinned validation commit: `c5925ff7404953c3640756d9b15bea898e60477d`

## Matched matrix

The matrix contained 46 unique cases, 23 float32 cases, 23 float16 cases, and
9 workload families. All 46 compiler cases succeeded. Static estimates matched
exactly in 46/46 cases.

## Frozen calibration transfer

The RTX 3050 calibration was evaluated without modification:

- Interval coverage: 36/46 = 78.3%
- Upper coverage: 40/46 = 87.0%
- Upper misses: 6/46 = 13.0%
- Median interval width/static: 0.9942

Upper misses were concentrated in four autodiff cases and two float32
convolution cases. Attention, elementwise, matmul, MLP, residual, training,
and Transformer cases had complete upper coverage.

## Stratified results

- float32 upper coverage: 19/23 = 82.6%
- float16 upper coverage: 21/23 = 91.3%
- Below 10 MiB: 2/7
- 10 to 100 MiB: 27/28
- 100 to 500 MiB: 9/9
- Above 500 MiB: 2/2

## Device drift

- Median absolute compiler drift: 0.048%
- P90 absolute compiler drift: 49.98%
- Maximum absolute compiler drift: 400.04%
- Temporary ratio median/P90/maximum: 1.018, 6.601, 9.001
- Temporary zero to nonzero transitions: 4
- Alias deltas: 0

The structural result transferred exactly. Compiler differences were localized,
with the strongest effect in small autodiff workloads.

## Runtime smoke

Attention, MLP, and training smoke cases all completed with status `FIT`.
Sampled allocator observations are not exact runtime peaks. Two smoke compiler
values exceeded the frozen upper bound, which is consistent with the compiler
accounting versus runtime observation distinction.

## Policy

The calibration remains exact-tested and is not generalized to all NVIDIA GPUs.
The detailed temporary-memory investigation is in
`device_temporary_drift_report_2026-09-08.md`.
