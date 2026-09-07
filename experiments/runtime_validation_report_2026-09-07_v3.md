# Runtime validation report

**Status: OBSERVED.** This report records one isolated WSL2 GPU run. It is not a
runtime accuracy benchmark and does not establish a runtime peak-memory bound.

## Configuration

- Device: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB physical VRAM
- JAX and jaxlib: 0.6.2
- Backend: CUDA, `cuda:0`
- Allocator limit reported by JAX: 3,220,832,256 bytes
- Platform: WSL2 Linux 6.6.87.2
- Allocator settings: default environment, no explicit preallocation override
- Trial isolation: one fresh subprocess per trial
- Workloads: attention, MLP, transformer residual attention, and training-like
  value-and-gradient updates
- Dtypes: `float32` and `float16`
- Raw artifact: `runtime_validation_2026-09-07_v3.json`

## Results

| Family | FIT | Compile OOM | Execution OOM | Other failure |
|---|---:|---:|---:|---:|
| Attention | 8 | 1 | 0 | 0 |
| MLP | 6 | 0 | 0 | 0 |
| Transformer | 4 | 0 | 0 | 0 |
| Training-like | 4 | 0 | 0 | 0 |
| **Total** | **22** | **1** | **0** | **0** |

The `float32`, sequence-8192 attention case failed during compilation with
`RESOURCE_EXHAUSTED`. Its structural estimate was 4,312,006,656 bytes and its
calibrated upper compiler interval was 4,716,273,729 bytes. This is consistent
with a budget-exceeding compilation in this environment, but it is one trial,
not an OOM-probability estimate.

## Measurement interpretation

Every successful trial recorded JAX compiler accounting from the compiled
executable and allocator counters before input creation, after input creation,
after lowering, after compilation, after execution, and after cleanup. The raw
allocator counter `peak_bytes_in_use` is a process high-water mark. It includes
compilation allocations and backend or framework allocations, so it is not an
execution-interval runtime peak.

In the 22 FIT trials, compiler accounting was available in all 22 cases. The
allocator high-water mark was at or below the structural estimate in 2 cases and
at or below the calibrated compiler upper interval in 3 cases. This comparison is
not evidence that the calibrated interval failed as a compiler interval. It shows
that allocator high-water marks from this harness are not a valid runtime target
for that interval. The gap is especially large for small workloads because fixed
compilation and backend overhead dominates the workload memory.

The compiler-accounted quantity remains the existing quantity:

```text
argument + output + temporary - alias
```

No estimator semantics or calibration constants were changed by this
experiment.

## Conclusions

**Status: OBSERVED.** This run broadens the prior attention-only trial set to 23
trials across four workload families and two dtypes. It reproduces a compile-time
failure at sequence 8192 and observed FIT behavior below that boundary.

**Status: REDUCED.** The current allocator-counter harness cannot identify an
execution-only runtime peak because the allocator high-water counter cannot be
reset after compilation. It therefore cannot validate the public calibrated
compiler interval as a runtime interval.

**Status: CONJECTURED.** A useful runtime integration will require either a
programmatic execution-interval memory trace or a separately validated allocator
measurement protocol that excludes compilation and reports its semantics
explicitly. Broader independent GPU validation remains necessary before making
runtime claims.
