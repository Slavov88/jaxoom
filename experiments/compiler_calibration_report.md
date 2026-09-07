# Compiler calibration report

Status: COMPUTATIONALLY VERIFIED for the recorded CPU environment only.

## Environment

- Python: 3.13.14
- JAX / jaxlib: 0.6.2 / 0.6.2
- Backend: CPU
- Device: `TFRT_CPU_0`
- Calibration output: `compiler_calibration_2026-09-07_v2.csv` and `.json`
- Command: `PYTHONPATH=src py -3.13 experiments/compiler_calibration.py --output-prefix experiments/compiler_calibration_2026-09-07_v2`
- Inputs: abstract `ShapeDtypeStruct` values; benchmark functions were not numerically executed

These are CPU compiler-calibration measurements, not GPU-memory accuracy claims.

## Methodology

The static quantity is the existing **sequential JAXPR live-value peak**. The compiler quantity is:

```text
compiler_accounted_bytes = argument_size_in_bytes
                         + output_size_in_bytes
                         + temp_size_in_bytes
                         - alias_size_in_bytes
```

It is compiler accounting, not exact runtime peak memory. Signed error is defined as:

```text
static_peak_bytes - compiler_accounted_bytes
```

Positive values mean static overprediction; negative values mean static underprediction.

The matrix contained 52 successful cases across elementwise, matmul, residual, MLP, reduction, nested, and autodiff families, with multiple shapes/depths. There were no failed or unavailable compiler rows.

## Aggregate results

Relative values are signed `(static - compiler) / compiler`.

| Family | N | Mean | Median | P90 absolute | Bias | Over | Under | Worst over | Worst under |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| elementwise | 18 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| matmul | 3 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| residual | 10 | 310.0% | 200.0% | 800.0% | 310.0% | 100.0% | 0.0% | 800.0% | — |
| MLP | 6 | -11.1% | -10.5% | 21.4% | -11.1% | 0.0% | 100.0% | — | -21.4% |
| reduction | 8 | 12.4% | -0.03% | 99.8% | 12.4% | 12.5% | 50.0% | 99.8% | -0.10% |
| nested | 2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| autodiff | 5 | 5.7% | -0.05% | 20.0% | 5.7% | 40.0% | 60.0% | 20.0% | -11.2% |
| **overall** | **52** | **63.2%** | **0.0%** | **200.0%** | **63.2%** | **25.0%** | **25.0%** | **800.0%** | **-21.4%** |

The overall mean is dominated by the deliberately long residual/reuse cases; the median is zero because several families match compiler accounting on this CPU configuration.

## Fusion findings

The elementwise-chain hypothesis was **not supported by this matrix**. At shape `[64, 64]`, depths 1, 2, 4, 8, 16, and 32 all produced:

| Depth | Static | Compiler-accounted | Relative difference |
|---:|---:|---:|---:|
| 1 | 32,768 | 32,768 | 0.0% |
| 2 | 32,768 | 32,768 | 0.0% |
| 4 | 32,768 | 32,768 | 0.0% |
| 8 | 32,768 | 32,768 | 0.0% |
| 16 | 32,768 | 32,768 | 0.0% |
| 32 | 32,768 | 32,768 | 0.0% |

The current liveness model releases intermediate chain values promptly, so its estimate does not grow with chain depth. This does not prove the model captures fusion; it means this benchmark does not distinguish the two.

## Temporary-memory findings

Compiler temporary memory was nonzero in the MLP and autodiff cases. The largest MLP underprediction was:

```text
mlp-b32-w32-d4
static:              22,528 bytes
compiler accounted:  28,672 bytes
temporary:            8,192 bytes
relative error:       -21.43%
```

The MLP family underpredicted in all six cases. This is the clearest dangerous direction observed: compiler workspace/category accounting is absent from the structural model.

## Autodiff findings

| Mode | Equations | Static | Compiler | Temp | Relative |
|---|---:|---:|---:|---:|---:|
| forward | 3 | 8,192 | 8,196 | 2,048 | -0.05% |
| grad | 9 | 12,288 | 10,240 | 2,048 | 20.0% |
| value_and_grad | 9 | 12,292 | 10,260 | 2,048 | 19.8% |
| jvp | 11 | 16,384 | 18,456 | 6,144 | -11.2% |
| vjp | 11 | 14,340 | 14,368 | 2,048 | -0.19% |

Autodiff behavior is structure-dependent rather than consistently conservative or optimistic. The sample is too small for broad claims.

## Nested JAXPR findings

Two `jax.nn.relu` cases were included. Both were marked `confidence=limited` due to `custom_jvp_call`, yet both matched compiler accounting exactly in this environment. This is encouraging but insufficient evidence for recursive nested-JAXPR semantics.

## Residual/reuse findings

Long-lived retained values cause strong static overprediction:

```text
residual-d16-16: static 18,432 bytes vs compiler 2,048 bytes (+800%)
residual-d16-64: static 294,912 bytes vs compiler 32,768 bytes (+800%)
```

The discrepancy scales with retained logical values and is the dominant overprediction mechanism in this matrix.

## Donation experiment

For a `(1024, 1024)` float32 update:

| Configuration | Argument | Output | Temp | Alias | Accounted |
|---|---:|---:|---:|---:|---:|
| without donation | 8,388,608 | 4,194,304 | 0 | 0 | 12,582,912 |
| with donation | 8,388,608 | 4,194,304 | 0 | 4,194,304 | 8,388,608 |

Donation appears through `alias_size_in_bytes` and reduces compiler-accounted bytes by one output-sized buffer. The static estimator was intentionally unchanged.

## Worst cases

Worst underprediction:

```text
MLP batch 32, depth 4: -21.43%
```

Worst overprediction:

```text
Residual depth 16: +800%
```

No compiler-unavailable or benchmark-failure rows occurred in this run.

## Architectural conclusion

**Choice C: primitive/workspace modeling should be next.**

The evidence does not justify a fusion heuristic or recursive nested-JAXPR redesign yet. Nested cases were small and matched, while the most important dangerous errors came from compiler temporary memory in MLP/autodiff workloads. The next milestone should investigate which primitive/configuration patterns produce `temp_size_in_bytes` and add explicit, evidence-backed workspace accounting or uncertainty—not hand-tuned correction factors.
