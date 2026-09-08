# RTX 3050 and Tesla T4 compiler introspection

**Status: COMPUTATIONALLY VERIFIED for the 14-case paired diagnostic run.**

## Reproducibility

Both devices used the same case manifest, JAX 0.11.0, jaxlib 0.11.0, and:

```text
--xla_gpu_autotune_level=0
```

The T4 run used repository commit `2ed40dd81536f132b50fba311155dab0cb27b8d2`.
All 14 cases completed. Each case has case-addressed XLA dump files and one
parsed memory-usage report.

The T4 archive is retained externally as the source ZIP. The committed compact
artifacts are:

- `compiler_introspection_t4_2026-09-08.json`
- `compiler_introspection_t4_environment_2026-09-08.json`
- `compiler_drift_cases_t4_returned_2026-09-08.json`

## First observed divergence

| Stage | RTX and T4 comparison |
|---|---:|
| JAXPR fingerprint | 14/14 equal |
| StableHLO fingerprint | 14/14 equal |
| Lowered HLO fingerprint | 14/14 equal |
| Compiled HLO fingerprint | 0/14 equal |

The first observed divergence is therefore after the common lowered HLO, in the
device-target compiled HLO/backend lowering stage. This is an observation about
the captured artifacts, not a claim about an unobserved internal pass boundary.

## Compiled target evidence

The case-addressed optimized HLO dumps identify:

- T4 matmul, MLP, attention, autodiff, and training cases: `__cublas$lt$matmul`
- RTX 3050 counterparts: direct `dot` instructions in the optimized HLO dump
- Both devices' convolution cases: `__cudnn$convForward`

The autodiff cases therefore enter the T4 compiled representation as two
cuBLASLt custom calls, while the RTX counterparts retain dot instructions.

## Buffer assignment and temporary memory

The preallocated temporary allocation in the buffer-assignment dump matches
the compiler temporary field for every inspected autodiff case.

| Case | RTX temp | T4 temp | T4 minus RTX | T4 preallocated temp |
|---|---:|---:|---:|---:|
| autodiff-b128-w256-float32 | 655,360 | 4,325,904 | 3,670,544 | 4,325,904 |
| autodiff-b256-w512-float32 | 524,288 | 4,719,120 | 4,194,832 | 4,719,120 |
| autodiff-b128-w256-float16 | 589,840 | 4,260,368 | 3,670,528 | 4,260,368 |
| autodiff-b256-w512-float16 | 1,310,736 | 4,456,976 | 3,146,240 | 4,456,976 |

Alias bytes were zero for the selected cases on both devices. The additional
T4 compiler-accounted bytes in these autodiff cases are therefore directly
represented by the additional preallocated temporary allocation.

## Convolution controls

Under this controlled introspection configuration, both float32 convolution
cases had identical compiler accounting on RTX and T4:

- 5,280,000 bytes for the smaller case
- 42,090,752 bytes for the larger case

The float16 convolution controls also matched exactly. Both devices used the
same cuDNN convolution custom-call target.

This does not erase the previous transfer-study convolution misses. The earlier
T4 transfer environment did not record `XLA_FLAGS`, while this paired diagnostic
explicitly uses `--xla_gpu_autotune_level=0`. The convolution discrepancy is
therefore not reproduced under the controlled diagnostic configuration and
should not be attributed to a device-only mechanism from this run.

## Conclusion

The paired evidence is sufficient to locate the first observed divergence at
device-target compiled HLO/backend lowering, after JAXPR and lowered StableHLO
or HLO remain identical. For the failing autodiff cases, the T4 compiled HLO
uses cuBLASLt matmul custom calls and the buffer assignment contains a larger
preallocated temporary allocation. That allocation accounts directly for the
additional compiler temporary bytes.

This is compiler-accounted memory, not runtime peak memory or OOM evidence.
No calibration constants, structural estimator code, or public API were
changed.
