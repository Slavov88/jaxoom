# Execution memory validation report

**Status: OBSERVED.** This report records an isolated WSL2 CUDA run using JAX
0.6.2 on an NVIDIA GeForce RTX 3050 Laptop GPU.

## Environment

- Backend: CUDA, `cuda:0`
- Physical VRAM: 4,294,967,296 bytes
- JAX allocator limit: 3,220,832,256 bytes
- JAX and jaxlib: 0.6.2
- XProf: 2.23.1, installed only in the isolated WSL environment
- Allocator settings: default environment
- Trial isolation: one fresh subprocess per trial

The raw intrinsic dataset is
`execution_memory_validation_2026-09-08.json`. A separate contention result is
in `execution_memory_contention_2026-09-08.json`.

## Measurement method

Each intrinsic trial followed this sequence:

1. Create and synchronize input arrays.
2. Lower and compile the workload.
3. Run one warmup execution and synchronize it.
4. Start a background poller for the allocator `bytes_in_use` counter.
5. Execute the already compiled function three times, synchronizing each result.
6. Keep the final result live while taking a post-execution snapshot.
7. Stop polling, delete allocations, and record the final snapshot.

`execution_peak_bytes` is the maximum sampled `bytes_in_use` during the
execution window, including the final post-execution snapshot. It is a sampled
allocator observation. It can miss allocations shorter than the polling period
and is not an exact device-memory trace.

`peak_bytes_in_use` remains a process high-water counter. It includes
compilation and warmup allocations and is reported separately as
`allocator_peak_bytes_in_use`.

## XProf investigation

`jax.profiler.start_trace` and `stop_trace` produced XPlane and Perfetto trace
files for a compiled execution pilot. The XProf CLI processed the trace, but
`get_memory_profile` returned unavailable values for memory capacity and peak
usage. `get_peak_allocations` reported that no memory viewer data was available
for the session. No stable execution-memory number could therefore be extracted
from this XProf trace.

`jax.profiler.device_memory_profile` was treated only as a snapshot API. It was
not used as an execution trace or peak estimator. No profiler traces are stored
in the repository.

## Intrinsic dataset

There were 24 FIT trials and no intrinsic OOM failures:

| Family | Trials | Dtypes |
|---|---:|---|
| Attention | 6 | float32, float16 |
| MLP | 6 | float32, float16 |
| Transformer-like | 4 | float32, float16 |
| Training-like | 4 | float32, float16 |
| Convolution | 2 | float32, float16 |
| FFT | 2 | float32 |

The structural estimates ranged from 4,464,640 to 1,082,261,504 bytes. The
sampled execution observations ranged from 2,621,440 to 260,046,848 bytes.

Outcome counts for the intrinsic dataset:

| Outcome | Count |
|---|---:|
| FIT | 24 |
| COMPILE_OOM | 0 |
| EXECUTION_OOM | 0 |
| OTHER_FAILURE | 0 |

## Comparisons

The sampled execution observation was compared with the existing compiler
accounting and calibrated compiler interval. These quantities have different
semantics. Execution observations include live input and output buffers, while
compiler accounting includes compiler-reported temporary storage and may include
buffers that are not simultaneously live at the sampled observation.

Across the 24 FIT trials:

- Median execution-over-compiler ratio: 0.8574
- Range of execution-over-compiler ratio: 0.0154 to 1.0001
- Median signed execution-minus-compiler difference: -3,146,162 bytes
- Execution observation was at or below the structural estimate in 20 of 24
  trials.
- The calibrated compiler upper interval covered the sampled execution
  observation in 23 of 24 trials, or 95.8%.
- The one upper-bound miss was `training-b512-w1024-float32`: execution
  observation 20,987,904 bytes versus calibrated upper bound 20,652,876 bytes.

The calibrated upper bound was not designed as an execution bound. The observed
coverage is therefore a comparison result, not a validation claim for a general
runtime guarantee.

## Cross-family observations

The calibrated upper interval covered all sampled execution observations in
attention, MLP, transformer-like, convolution, and FFT trials. It missed one of
four training-like trials by 335,028 bytes. This small dataset does not support
family-specific calibration changes.

Convolution and FFT were included as workspace-sensitive families. No case
showed a meaningful sampled execution observation above compiler accounting.
The largest observed execution-over-compiler ratio was 1.0001 in a small
transformer-like trial. This does not establish that backend workspaces never
exceed compiler accounting.

## Dtype observations

Both float32 and float16 were tested for attention, MLP, transformer-like,
training-like, and convolution workloads. The sampled observations generally
scaled with the corresponding input and output sizes. No stable deviation was
identified that would justify changing the structural estimator or shipped
calibration constants.

## Contention experiment

No intrinsic execution OOM was observed. A separate controlled contention trial
compiled attention with sequence 4096 first, then retained a 2,200,000,000-byte
float32 background allocation before execution. The compiled execution failed
with `RESOURCE_EXHAUSTED` while trying to allocate 1,073,741,840 bytes.

This result is classified as `CONTENTION_EXECUTION_OOM`. It is not part of the
intrinsic dataset and does not show that the workload alone would OOM. It shows
that available execution budget can differ materially from physical VRAM and
from the post-compilation allocator state.

## Conclusions

**Status: PARTIALLY VALIDATED.** A repeatable execution window can be isolated
and sampled using allocator `bytes_in_use` while a compiled executable runs.
Warmup and compilation are recorded separately. The result is useful as an
execution-window allocator observation, but it is not an exact peak because the
poller can miss short-lived allocations.

**Status: OBSERVED.** XProf was installed and exercised, but the tested XProf
memory tools did not provide usable memory-viewer data for the JAX trace.
Programmatic XProf execution-peak extraction is therefore not included.

**Status: OBSERVED.** The existing calibrated compiler upper interval covered 23
of 24 sampled execution observations in this environment. One small miss was
observed. This is not evidence of a runtime guarantee, and the sample is too
small to change calibration constants.

**Status: OBSERVED.** External contention produced an execution-time OOM after
successful compilation. Contention must remain separate from intrinsic workload
memory results.

## Limitations

- The execution peak is sampled allocator `bytes_in_use`, not a hardware trace.
- Short-lived temporary allocations can be missed.
- Repeated calls define the execution window, so the observation is not a
  single-invocation trace in every detail.
- XProf memory extraction was unavailable for the tested trace.
- The GPU evidence is one device, one JAX and jaxlib version, and one allocator
  configuration.
- The intrinsic dataset contains no natural execution OOM.
- No public runtime profiling API is introduced.
