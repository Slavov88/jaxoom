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

The runtime validation harness launches each trial in a fresh subprocess. It
records compiler accounting and allocator counters at explicit phases. The
allocator `peak_bytes_in_use` field is a process high-water counter that includes
compilation, not an execution-only runtime peak. It is intended for an
environment with a CUDA-capable JAX installation:

```bash
PYTHONPATH=src python experiments/runtime_validation.py \
  --output experiments/runtime_validation_YYYY-MM-DD.json
```

The 2026-09-07 expanded run is recorded in
`runtime_validation_2026-09-07_v3.json`, with interpretation in
`runtime_validation_report_2026-09-07_v3.md`. It covers attention, MLP,
transformer, and training-like workloads. The report deliberately does not
turn allocator counters into runtime-peak or OOM-probability claims.

The execution-window study in `execution_memory_validation.py` samples
`bytes_in_use` only while compiled calls run, after a separate warmup. It does
not provide an exact device-memory trace. XProf was tested, but its memory
viewer did not return usable data for the pilot trace. Results are recorded in
`execution_memory_validation_2026-09-08.json` and its summary and report.

Calibration evaluation:

```bash
PYTHONPATH=src python experiments/uncertainty_calibration.py \
  --cpu-csv experiments/accelerator_calibration_cpu_2026-09-07_v2.csv \
  --gpu-csv experiments/accelerator_calibration_gpu_2026-09-07_v2.csv \
  --output-prefix experiments/uncertainty_calibration_YYYY-MM-DD
```

Stored CSV and JSON files include the environment and raw measured categories.
Reports distinguish structural JAXPR memory, compiler accounting, allocator
counters, and calibrated compiler intervals.
