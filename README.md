# jaxoom

**Know before you OOM.**

`jaxoom` is an early static memory analyzer for JAX. The first milestone traces a function to a JAXPR and computes a conservative **sequential JAXPR live-value peak** for ordinary dense-array computations.

## Status

This is v0.1 exploratory software. It does not predict exact XLA or GPU memory.

## Local development

```bash
python -m pip install -e ".[test]"
pytest
```

JAX is the only runtime dependency. The supported JAX version should be checked against the environment where the analysis is run.

## Example

```python
import jax
import jax.numpy as jnp
import jaxoom


def f(x):
    y = x @ x.T
    return y + jnp.sin(y)

report = jaxoom.estimate(
    f,
    jax.ShapeDtypeStruct((1024, 1024), "float32"),
    memory_limit="16 GiB",
)
report.print()
```

The result is a typed `MemoryReport` with `estimated_peak_bytes`, peak contributors, equation information, limitations, and a conservative analysis-confidence label.

## Current limitations

The model counts logical dense JAXPR values and treats each equation as requiring its inputs and newly materialized outputs simultaneously. It does not model compiler fusion, buffer aliasing, scheduling, allocator behavior, device workspaces, control-flow execution semantics, sharding, or runtime observation. Nested/control-flow JAXPR constructs are detected and lower confidence rather than being treated as fully understood.

This is a structural estimate, not an exact GPU-memory oracle and not a guarantee that a workload will or will not OOM.

## Roadmap

Next milestones are compiler-memory comparison, broader nested-JAXPR handling, and batch-size/donation advice—only after validation of this structural model.
