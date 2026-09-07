# Contributing

## Development setup

```bash
python -m pip install -e ".[test]"
```

The current test environment uses Python 3.13 and JAX 0.6.2. GPU support is
not required for the package or test suite.

## Checks

```bash
python -m pytest
python -m compileall -q src
python examples/basic_estimate.py
```

Keep the structural estimator's assumptions explicit. New compiler or runtime
measurements should record the JAX version, backend, device, allocator settings,
command, and raw machine-readable output. Do not report benchmark results as
general accuracy claims without a reproducible comparison.

GPU experiments are optional and should run outside ordinary CPU CI.
