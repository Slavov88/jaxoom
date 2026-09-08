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

## Calibrated assessment

Calibration is opt-in and does not change `estimated_peak_bytes`:

```python
interval = jaxoom.calibrate(static)
assessment = jaxoom.assess(static, memory_limit="16 GiB")
assessment.print()
```

The built-in summaries use empirical compiler/static ratios from recorded
JAX 0.6.2 CPU, JAX 0.6.2 CUDA, and JAX 0.11.0 CUDA runs. Selection is
version-aware. An exact tested version is marked `EXACT_TESTED`, an untested
minor release in a tested family is marked `VERSION_FAMILY_MATCH`, and an
unknown family is `UNCALIBRATED`. The range is compiler-accounted memory, not a
runtime peak interval, and the risk level is not an OOM probability. See
`experiments/version_calibration_report_2026-09-08.md` for the validation.

## Preliminary GPU validation

The repository includes a backend-portable calibration harness and an isolated
runtime/OOM experiment. An expanded WSL2 run used an NVIDIA GeForce RTX 3050
Laptop GPU with 4 GiB VRAM, JAX 0.6.2, and the default allocator:

| Workload family | FIT | Compile OOM | Execution OOM |
|---|---:|---:|---:|
| Attention | 8 | 1 | 0 |
| MLP | 6 | 0 | 0 |
| Transformer | 4 | 0 | 0 |
| Training-like | 4 | 0 | 0 |

The sequence-8192 float32 attention case failed during compilation. This is a
small environment-specific validation run, not a GPU accuracy benchmark.
Allocator high-water counters include compilation and backend overhead, so they
are not execution-only runtime peaks. A follow-up study sampled allocator
`bytes_in_use` during compiled execution, but did not establish an exact device
memory trace. Full measurements and interpretation are in
`experiments/runtime_validation_report_2026-09-07_v3.md` and
`experiments/execution_memory_validation_report_2026-09-08.md`.

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
