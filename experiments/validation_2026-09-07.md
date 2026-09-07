# MVP validation — 2026-09-07

Status: COMPUTATIONALLY VERIFIED for the recorded environment only.

## Configuration

- Commit: `86f6ec69a5347cf3552fc1dbefb4dc0bcf338a73`
- Python: 3.13.14
- JAX / jaxlib: 0.6.2 / 0.6.2
- Backend/device: CPU, one `CpuDevice(id=0)`
- Command: `PYTHONPATH=src py -3.13 experiments/compile_validation.py`
- Inputs: abstract float32 arrays; no numerical workload execution
- Seed: not applicable; tracing and compilation are shape-only for these cases

## Recorded compiler comparison

`structural` is the sequential JAXPR live-value peak. The other columns are
`Compiled.memory_analysis()` categories and are not an equivalent peak metric.

| case | structural | argument | output | temp | alias |
|---|---:|---:|---:|---:|---:|
| elementwise | 8,192 | 4,096 | 4,096 | 0 | 0 |
| matmul | 14,336 | 12,288 | 2,048 | 0 | 0 |
| mlp | 65,536 | 49,152 | 2,048 | 16,384 | 0 |
| residual | 12,288 | 4,096 | 4,096 | 0 | 0 |
| grad | 16,384 | 4,096 | 4,096 | 0 | 0 |

Raw output is retained in `validation_2026-09-07.txt`.

## Findings

- Elementwise and matmul cases have structural values equal to argument plus
  output in this CPU compilation, but this is not evidence of general exactness.
- The residual case has a 12,288-byte structural peak versus 8,192 bytes of
  argument plus output compiler categories. The extra logical live value is a
  concrete fusion/liveness discrepancy for this model.
- The MLP structural peak is 65,536 bytes, while compiler categories sum to
  67,584 bytes because compiler temporary storage is reported separately.
- The gradient JAXPR exposes the forward product, a dropped reduction result,
  broadcast, and backward products/addition. The structural model sees these
  logical values, but does not claim to model compiler reuse or autodiff runtime
  storage exactly.
- `custom_jvp_call`/nested JAXPR constructs occur for `jax.nn.relu`; the MVP
  reports them and lowers confidence rather than recursively pretending to have
  complete semantics.
- `jax.ShapeDtypeStruct` supports shape-only analysis, including a tested
  100,000 x 1,024 float16 input, without allocating that array.

No fusion heuristic was added: the residual discrepancy is evidence to retain
as a validation target, not justification for speculative compiler modeling.
