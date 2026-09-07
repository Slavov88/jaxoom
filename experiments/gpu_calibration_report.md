# GPU calibration and OOM report

Status: COMPUTATIONALLY VERIFIED for the recorded WSL2/CUDA environment; runtime memory observations are backend- and allocator-specific.

## GPU environment

- Host OS: Windows 11 26200, WSL2
- Guest OS: Ubuntu 24.04.4 LTS
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Physical VRAM: 4,096 MiB
- Driver: 566.07
- Reported CUDA compatibility: 12.7
- Python: 3.12.3
- JAX / jaxlib: 0.6.2 / 0.6.2
- JAX backend: `gpu`
- Device: `cuda:0`
- JAX device memory limit: 3,220,832,256 bytes
- Allocator variables: default/unset for the canonical run

The isolated environment and package versions are recorded in
`gpu_environment_2026-09-07.txt`. The Windows CPU environment was not modified.

## CPU regression

Before GPU work:

```text
22 tests passed
compileall passed
CPU calibration smoke: 52 rows, successful
```

The same realistic calibration harness was run on Windows CPU and WSL2 CUDA
GPU. No structural-estimator code was changed.

## GPU benchmark matrix

The realistic matrix contained 38 successful compiler cases:

| Family | Cases | Dtypes | Scale |
|---|---:|---|---|
| elementwise | 12 | float32/float16 | 16–256 MiB logical tensors |
| matmul | 4 | float32/float16 | tens of MiB |
| residual | 4 | float32/float16 | tens–hundreds of MiB structural |
| MLP | 6 | float32/float16 | tens–hundreds of MiB |
| attention | 8 | float32/float16 | tens MiB to ~773 MiB structural |
| transformer | 2 | float32/float16 | tens MiB |
| training-like | 2 | float32/float16 | tens MiB |

Artifacts:

```text
accelerator_calibration_gpu_2026-09-07_v2.csv/.json
accelerator_calibration_cpu_2026-09-07_v2.csv/.json
```

The largest static case was approximately 4.01 GiB: float32 attention with
sequence length 4096, 8 heads, and head dimension 64.

## Compiler calibration

GPU aggregate, 38 cases:

```text
median absolute error:          1,559,088 bytes
p90 absolute error:             67,108,864 bytes
median absolute relative error: 4.37%
p90 absolute relative error:    200%
overprediction frequency:       26.3%
underprediction frequency:      36.8%
worst overprediction:            268,435,456 bytes (+800%)
worst underprediction:           -25,166,592 bytes (-24.33%)
```

The large relative p90 is driven by small compiler-accounted denominators in
residual cases. The absolute error is more relevant to the 4 GiB device.

Selected GPU/CPU comparisons for identical logical workloads:

| Case | GPU compiler | CPU compiler | GPU temp | CPU temp |
|---|---:|---:|---:|---:|
| attention S=512 f32 | 18,874,368 | 19,136,512 | 16,777,216 | 17,039,360 |
| attention S=4096 f32 | 1,090,519,056 | 1,107,296,256 | 1,073,741,840 | 1,090,519,040 |
| MLP B=1024 W=4096 D=4 f32 | 293,602,048 | 293,601,280 | 33,555,200 | 33,554,432 |
| residual D=16 f32 | 33,554,432 | 33,554,432 | 0 | 0 |
| training B=512 W=1024 f32 | 19,936,340 | 19,968,044 | 4,195,368 | 4,227,072 |

Compiler values are not runtime peak values.

## Scale effects

Observed size buckets on GPU:

| Bucket | N | Median absolute relative error |
|---|---:|---:|
| <10 MiB | 1 | 5.29% |
| 10–100 MiB | 26 | 1.48% |
| 100–500 MiB | 9 | 8.57% |
| 500 MiB–1 GiB | 1 | 48.47% |
| >1 GiB | 1 | 0.76% |

The sample is sparse in the largest buckets. At the largest successful
attention case (static 1,082,261,504 bytes), the compiler difference was
-8,257,552 bytes (-0.76%). Scale alone did not produce monotonic relative error.

Using the JAX device limit as the target budget, the median absolute
budget-normalized error was 0.048%; the maximum was 8.33% (the depth-16
residual case). The absolute-error/budget metric is less distorted by the
small residual compiler denominators.

## Workspace hypothesis

Hypothesis: GPU underprediction is primarily explained by compiler temporary
memory.

**OBSERVED:** all 14 GPU underpredictions had nonzero compiler temporary bytes,
but temporary size was not sufficient to explain direction or magnitude.

- Correlation between compiler temporary bytes and positive static shortfall
  among underpredictions: `r = 0.10`.
- Correlation between compiler temporary bytes and signed error over all cases:
  `r = 0.20`.
- Attention had very large temporary allocations but often static
  overprediction.
- MLP and training-like cases consistently underpredicted.

Conclusion: temporary memory is a necessary visible feature of the observed
underpredictions, but not the primary standalone predictor in this matrix.

## Primitive composition

The machine-readable rows include primitive counts. Underpredicted MLP and
training cases contain repeated `dot_general`, broadcast, and elementwise
operations. Attention cases contain `dot_general`/einsum, transpose/reshape,
softmax-related reductions, and large temporary allocations. No correction rule
was added.

## Runtime profiling

JAX 0.6.2 exposed:

```text
jax.profiler.save_device_memory_profile
jax.profiler.start_trace / stop_trace
jax.profiler.device_memory_profile
```

The experimental runtime harness used `save_device_memory_profile` on successful
trials. The resulting files were gzip-compressed protobuf profiles of about
1.3 KiB. `file` identified them as gzip data; without an XProf/pprof frontend in
this environment, no full visual timeline was claimed. The profile is not
committed; only the harness and observations are retained.

`Device.memory_stats()['peak_bytes_in_use']` was also recorded, but is labeled
an allocator peak counter for the process—not an exact execution-interval peak.

## OOM classification

The bounded increasing-sequence attention search ran each trial in a fresh
subprocess under default JAX allocator behavior:

| Sequence | Static | Compiler | Allocator peak | Actual |
|---:|---:|---:|---:|---|
| 1024 | 69,238,784 | 71,303,168 | 104,857,600 | FIT |
| 2048 | 272,695,296 | 276,824,064 | 310,378,496 | FIT |
| 4096 | 1,082,261,504 | 1,090,519,040 | 1,124,073,472 | FIT |
| 8192 | 4,312,006,656 | unavailable | unavailable | OOM during compilation |

The 8192 case failed with `RESOURCE_EXHAUSTED` while allocating 2 GiB during
GPU compilation. Static analysis correctly classified it as over the JAX device
memory limit (3,220,832,256 bytes). Compiler analysis was unavailable because
compilation itself failed.

Observed classification for these four trials:

```text
static false-fit: 0/4
static false-OOM: 0/4
compiler false-fit: 0/3 available compiler cases
```

This is too small and too easy a boundary to establish production OOM
classification accuracy. An exploratory 70%, 80%, and 90% static safety-margin
analysis gives the same classifications for all four trials: the three
successful cases are below 70% of budget, and the failed case is above 100%.
It therefore provides no evidence for selecting a production margin.

## Allocator findings

Canonical run: all allocator variables unset (default JAX behavior). For the
successful S=4096 f32 trial:

```text
Default preallocation:
  allocator peak bytes in use: 1,124,073,472
  pool bytes:                   3,220,832,256

XLA_PYTHON_CLIENT_PREALLOCATE=false:
  allocator peak bytes in use: 1,132,462,080
  pool bytes:                   2,181,038,080
```

The diagnostic setting reduced the reserved pool but slightly increased the
observed allocator peak. It did not change compiler accounting. This is one
workload/configuration only.

## Donation

GPU compiler-side donation for a `(1024, 1024)` float32 update:

| Configuration | Alias | Compiler-accounted |
|---|---:|---:|
| Without donation | 0 | 12,582,912 |
| With donation | 4,194,304 | 8,388,608 |

No GPU donation fit-boundary experiment was attempted; the result confirms the
CPU observation that donation is visible through alias bytes.

## Architectural decision

**Choice D: compiler-calibrated uncertainty bounds should be next.**

Reasoning:

1. GPU structural error is directionally mixed: 26.3% overprediction versus
   36.8% underprediction.
2. Workspace size alone has weak correlation with shortfall (`r=0.10` among
   underpredictions).
3. Residual graphs still produce extreme overprediction (+800%), while MLP and
   training cases underpredict by up to 24.3%.
4. Nested/liveness changes are not isolated as the dominant GPU failure mode.
5. The one actual OOM boundary was correctly screened by the raw structural
   estimate, but the sample is insufficient for a correction model.

A calibrated, backend-aware uncertainty representation is therefore better
supported by the evidence than immediate primitive correction rules. No safety
margin or correction factor has been shipped.
