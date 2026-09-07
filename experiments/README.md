# Experiments

These scripts record compiler accounting and limited runtime observations. They
are separate from the public `jaxoom` API and do not provide benchmark claims by
themselves.

CPU calibration:

```bash
PYTHONPATH=src python experiments/compiler_calibration.py \
  --output-prefix experiments/compiler_calibration_YYYY-MM-DD
```

Backend-portable larger calibration:

```bash
PYTHONPATH=src python experiments/accelerator_calibration.py \
  --output-prefix experiments/accelerator_calibration_YYYY-MM-DD
```

The runtime validation harness launches each trial in a fresh subprocess. It is
intended for an environment with a CUDA-capable JAX installation:

```bash
PYTHONPATH=src python experiments/runtime_validation.py \
  --output experiments/runtime_validation_YYYY-MM-DD.json
```

Stored CSV and JSON files include the environment and raw measured categories.
Reports distinguish structural JAXPR memory, compiler accounting, and allocator
counters.
