# Cross-version JAX validation

**Status: COMPUTATIONALLY VERIFIED for the recorded environments.**

This study compares JAX 0.6.2 and JAX 0.11.0 using the same 14 abstract-input
workloads. It checks JAXPR summaries, the structural estimator, compiler memory
accounting, and transfer of the existing calibration intervals. It is not a
runtime peak-memory or OOM benchmark.

## Environments

| Environment | Python | JAX | jaxlib | Backend | Device |
|---|---:|---:|---:|---|---|
| old CPU | 3.11 | 0.6.2 | 0.6.2 | CPU | recorded in JSON artifact |
| new CPU | 3.12.3 | 0.11.0 | 0.11.0 | CPU | recorded in JSON artifact |
| old GPU | 3.11 | 0.6.2 | 0.6.2 | CUDA | RTX 3050 Laptop GPU |
| new GPU | 3.12.3 | 0.11.0 | 0.11.0 | CUDA | RTX 3050 Laptop GPU |

The GPU runs used `XLA_FLAGS=--xla_gpu_autotune_level=0` to avoid unrelated
autotuning allocation failures. Full environment metadata is retained in the
four JSON artifacts.

## Results

All 14 cases passed in each environment. The cases cover elementwise,
matmul, residual, reduction, MLP, attention, Transformer, autodiff,
convolution, and FFT workloads, including selected float16 cases.

| Comparison | Result |
|---|---:|
| Structural estimates unchanged, CPU | 14/14 |
| Structural estimates unchanged, GPU | 14/14 |
| JAXPR summaries unchanged, CPU | 11/14 |
| JAXPR summaries unchanged, GPU | 11/14 |
| Median compiler-memory ratio, new/old CPU | 1.000 |
| Median compiler-memory ratio, new/old GPU | 1.000 |
| Largest absolute compiler difference, CPU | 4,177,920 bytes |
| Largest absolute compiler difference, GPU | 1,048,576 bytes |

The JAXPR summary differences were representation-level changes. The largest
observed one was a changed equation structure in the value-and-grad case.
Structural estimates remained identical for every matched case.

Compiler accounting was mostly stable, but individual cases changed. On CPU,
the largest difference was in float16 attention. On GPU, MLP and selected
autodiff cases increased by 1 MiB in JAX 0.11.0. These are compiler-accounting
changes, not evidence of a runtime peak change.

## Calibration transfer

The shipped calibration constants were not changed.

| Dataset applied to | GPU upper-bound coverage | GPU interval coverage |
|---|---:|---:|
| JAX 0.6.2 | 11/14 | 11/14 |
| JAX 0.11.0 | 9/14 | 9/14 |

The JAX 0.11.0 rows are reported as **limited** because the stored calibration
was fit on JAX 0.6.x evidence. The transfer result is useful as a diagnostic,
but it is not a new calibration dataset and does not justify changing the
constants. The CPU comparison produced 13/14 coverage for both versions in
this matched harness.

## Runtime and XProf

Runtime execution-window profiling remains experimental and is not part of the
public API. The existing 0.6.2 study used sampled allocator `bytes_in_use`
during repeated calls to an already compiled executable. It did not establish
an exact execution-only device trace.

XProf 2.23.1 was available in the JAX 0.11.0 environment. Trace capture was
attempted, but the current-JAX GPU pilot encountered allocator and pinned-host
allocation failures before producing interpretable memory data. This is an
inconclusive pilot, not evidence of an XProf API incompatibility. The earlier
successful trace-capture pilot likewise returned no usable memory-viewer data.

## Reproduction

```bash
PYTHONPATH=src python experiments/jax_version_validation.py \
  --output experiments/jax_version_validation_YYYY-MM-DD.json
```

The checked-in artifacts are:

- `jax_version_validation_2026-09-08_jax062_cpu.json`
- `jax_version_validation_2026-09-08_jax011_cpu.json`
- `jax_version_validation_2026-09-08_jax062_gpu.json`
- `jax_version_validation_2026-09-08_jax011_gpu.json`

Interpretation is limited to these workloads, versions, compiler settings, and
one GPU model. Independent-GPU validation remains the next high-value runtime
experiment.
