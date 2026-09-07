# jaxoom

**Know before you OOM.**

`jaxoom` provides static and compiler-backed memory analysis for JAX programs.
The static analyzer traces a function to JAXPR and estimates sequential logical
live-value memory without numerically executing the workload.

**Status: experimental.** This project does not provide exact GPU peak-memory
prediction or OOM guarantees.

## Quick start

```bash
python -m pip install -e ".[test]"
```

Analyze shapes without allocating the corresponding array:

```python
import jax
import jax.numpy as jnp
import jaxoom

x = jax.ShapeDtypeStruct((4096, 4096), jnp.float32)


def f(x):
    y = jnp.sin(x)
    return y + x

report = jaxoom.estimate(f, x, memory_limit="4 GiB")
report.print()
```

Output from the current test environment:

```text
JAXOOM STATIC ANALYSIS
================================

Model                         sequential JAXPR live-value estimate
Estimated structural peak    192.00 MiB
Equations analyzed           2
Confidence                   structural
Memory limit                 4.00 GiB
Assessment                   LIKELY FIT
```

## Static and compiler analysis

`estimate()` uses the sequential JAXPR model and does not compile the function.

```python
static = jaxoom.estimate(f, x)
compiler = jaxoom.compile_analyze(f, x)
comparison = jaxoom.compare_memory(static, compiler)
```

`compile_analyze()` lowers and compiles the function, then reads the backend's
memory categories. The reported compiler quantity is:

```text
argument bytes + output bytes + temporary bytes - alias bytes
```

It is compiler accounting, not observed runtime peak memory. Compiler analysis
may be unavailable or differ across JAX versions and backends.

## Preliminary GPU validation

The repository includes a backend-portable calibration harness and an isolated
runtime/OOM experiment. One WSL2 run used an NVIDIA GeForce RTX 3050 Laptop GPU
with 4 GiB VRAM, JAX 0.6.2, and the default allocator:

| Attention sequence | Structural | Compiler accounting | Allocator peak | Result |
|---:|---:|---:|---:|---|
| 1024 | 69,238,784 B | 71,303,168 B | 104,857,600 B | FIT |
| 2048 | 272,695,296 B | 276,824,064 B | 310,378,496 B | FIT |
| 4096 | 1,082,261,504 B | 1,090,519,040 B | 1,124,073,472 B | FIT |
| 8192 | 4,312,006,656 B | unavailable | unavailable | OOM during compilation |

This is a small environment-specific validation run, not a GPU accuracy
benchmark. The full measurements and configuration are in
`experiments/gpu_calibration_report.md`.

## Limitations

- Sequential JAXPR logical liveness is not exact XLA or runtime memory.
- Compiler fusion, aliasing, scheduling, allocator behavior, and workspaces are not modeled by the structural estimator.
- Compiler temporary memory can cause underprediction.
- Residual graphs can be substantially overpredicted.
- Nested JAXPR and control-flow handling is limited.
- Runtime OOM behavior depends on allocator settings and system state.
- GPU validation remains limited to the recorded environments.

The repository does not yet implement runtime profiling as a public API,
checkpoint planning, batch-size optimization, or automatic memory optimization.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m compileall -q src
python examples/basic_estimate.py
```

Calibration scripts use abstract inputs for compiler comparisons where possible.
They record environment metadata and should not be treated as production
benchmarks without reviewing their reports.

See `CONTRIBUTING.md` for the small development workflow. The project is
licensed under the MIT License.
