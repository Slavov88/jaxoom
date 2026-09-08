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

Cross-version validation uses matched abstract-input cases to compare JAXPR
summaries, structural estimates, compiler accounting, and calibration transfer
between JAX 0.6.2 and JAX 0.11.0:

```bash
PYTHONPATH=src python experiments/jax_version_validation.py \
  --output experiments/jax_version_validation_YYYY-MM-DD.json
```

The recorded comparison is in `jax_version_validation_report_2026-09-08.md`.
It found identical structural estimates in all 14 matched cases, with
compiler-accounting drift in selected workloads. The expanded version-aware
calibration study is recorded in `version_calibration_report_2026-09-08.md`.
It keeps raw datasets under `experiments/` and stores only quantile parameters
and provenance in the package.

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

Portable cross-device validation uses the frozen JAX 0.11.0 GPU dataset as a
reference and reruns the same logical cases on the active CUDA device:

```bash
PYTHONPATH=src:experiments python experiments/device_transfer_validation.py \
  --reference experiments/jax_0_11_gpu_calibration_2026-09-08_v2.json \
  --output-prefix experiments/device_transfer_validation_<gpu>_YYYY-MM-DD
```

The harness records paired compiler categories, frozen-calibration coverage,
device metadata, family/dtype/size summaries, and a Markdown report. It
refuses to label the RTX 3050 as an independent device. Use
`--allow-same-device-smoke` only for a local harness smoke test.
