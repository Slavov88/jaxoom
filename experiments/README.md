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

The device-budget correction replay compares the legacy driver-plus-pool
policy with the allocator-aware policy using the frozen 18-trial dataset. The
replay, occupancy regression, and remaining false fits are recorded in
`device_budget_correction_summary_2026-09-08.json` and
`device_budget_correction_report_2026-09-08.md`.

Allocator-residency validation varies `XLA_CLIENT_MEM_FRACTION` in fresh
subprocesses and records the allocator capacity bracket for end-to-end FIT:

```bash
PYTHONPATH=src:experiments python experiments/allocator_residency_validation.py \
  --output experiments/allocator_residency_validation_YYYY-MM-DD.json
```

The 2026-09-08 RTX 3050 study found an attention-4096 FIT/OOM bracket between
3,265,265,664 and 3,307,208,704 bytes. The fresh probes were not sufficient to
validate a general pre-compilation residency model, so production behavior was
left unchanged. See `allocator_residency_report_2026-09-08.md`.

The execution OOM diagnostic harness instruments device, CUDA runtime, and
JAX allocator state around inputs, compilation, and repeated execution:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src:experiments \
  python experiments/execution_oom_diagnosis.py \
  --threshold-output experiments/execution_oom_diagnosis_YYYY-MM-DD.json
```

The RTX 3050 false-fit reproduction found that compilation grew the JAX pool to
about 2.2 GiB while the JAX allocator limit was 3 GiB. The failed first
execution requested about 1 GiB, while the remaining allocator capacity was
about 996 MiB. A platform-allocator diagnostic FIT where the default BFC-style
configuration failed. This identifies allocator state and JAX capacity limits
as contributors. It does not justify a production predictor change yet. See
`execution_oom_diagnosis_report_2026-09-08.md`.

The OOM boundary harness runs each target trial in a fresh subprocess and
keeps intrinsic and controlled-contention tracks separate:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src:experiments \
  python experiments/oom_boundary_validation.py \
  --output experiments/oom_boundary_validation_YYYY-MM-DD.json
```

The RTX 3050/JAX 0.11.0 run on 2026-09-08 recorded 18 trials: 7 FIT,
1 COMPILE_OOM, and 10 EXECUTION_OOM. Four were false fits under the frozen
upper-bound rule, all execution OOMs. The report deliberately does not turn
this small, single-device dataset into a general accuracy claim. Raw trials,
summary, reserve sensitivity, and snapshot-race measurements are stored under
`oom_boundary_*_2026-09-08.*`.

The returned Tesla T4 validation is recorded in
`device_transfer_validation_t4_2026-09-08.json` and summarized in
`device_transfer_validation_report_2026-09-08.md`. It matched static estimates
in all 46 cases and achieved 87.0% frozen upper coverage. Temporary-memory
drift was concentrated in small autodiff cases, so calibration remains
exact-tested rather than generalized across NVIDIA GPUs. The diagnostic
analysis is in `device_temporary_drift_report_2026-09-08.md`.

Current-device budget validation uses a fresh subprocess to hold separate JAX
allocations while the parent takes non-compiling snapshots and assessments:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src:experiments \
  python experiments/device_budget_validation.py \
  --output experiments/device_budget_validation_YYYY-MM-DD.json
```

The 2026-09-08 RTX 3050 pilot is recorded in
`device_budget_validation_2026-09-08.json`. It observed monotonic reductions in
assessment budget and upper-bound headroom under 256 MiB, 768 MiB, and 1280 MiB
of retained external allocation. The 1280 MiB condition moved the fixed
attention workload from `MODERATE` to `LIKELY EXCEEDS BUDGET`; the pilot did not
claim a production OOM rate or run a broad compile boundary sweep.

Donation validation uses the public advisor over a focused pure-JAX workload
matrix:

```bash
PYTHONPATH=src python experiments/donation_validation.py \
  --output-prefix experiments/donation_validation_YYYY-MM-DD
```

The recorded JAX 0.6.2 CPU, JAX 0.11.0 CPU, and RTX 3050 JAX 0.11.0 runs are
summarized in `donation_validation_summary_2026-09-08.json` and
`donation_validation_report_2026-09-08.md`. The result measures compiler-
accounted before and after memory only. It does not establish that the caller
can safely reuse a donated input.
