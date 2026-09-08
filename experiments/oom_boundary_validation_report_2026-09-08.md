# OOM boundary validation

Status: OBSERVED

Environment: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB VRAM, JAX 0.11.0,
jaxlib 0.11.0, CUDA backend, Python 3.12.3. Trials used
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Each target trial ran in a fresh
subprocess. The contention holder ran in a separate process.

The frozen predictor was not changed: structural JAXPR estimate, the existing
JAX 0.11 GPU calibration, the existing risk rules, current driver free memory
plus unused JAX pool bytes, and the 5% reserve bounded to 64 MiB through 256
MiB.

## Dataset

| Track | Trials | Families | Pressure levels |
|---|---:|---|---|
| Intrinsic | 12 | attention, MLP, training, transformer | 0 MiB |
| Contention | 6 | attention | 0, 768, 1280 MiB |

The intrinsic set contained five attention configurations, three MLP
configurations, two training configurations, and two transformer
configurations. Attention was the only family that reached the device boundary
in this small run.

## Outcomes

| Track | FIT | COMPILE_OOM | EXECUTION_OOM | OTHER_FAILURE |
|---|---:|---:|---:|---:|
| Intrinsic | 7 | 1 | 4 | 0 |
| Contention | 0 | 0 | 6 | 0 |
| Total | 7 | 1 | 10 | 0 |

## Risk confusion table

| Risk | FIT | COMPILE_OOM | EXECUTION_OOM | OTHER_FAILURE |
|---|---:|---:|---:|---:|
| LOW | 7 | 0 | 4 |
| MODERATE | 0 | 0 | 5 |
| HIGH | 0 | 0 | 0 |
| LIKELY EXCEEDS BUDGET | 0 | 1 | 1 |

MODERATE and HIGH were not treated as predicted OOM for the false-OOM metric.

## False fits and false OOMs

The frozen operational definitions were:

- Predicted FIT: calibrated upper bound is at or below the assessment budget.
- Predicted exceeds: structural estimate is above the assessment budget.
- FALSE FIT: predicted FIT but COMPILE_OOM or EXECUTION_OOM.
- FALSE OOM: predicted exceeds but FIT.

Results:

- Predicted FIT: 11/18
- False fits: 4/18 overall, or 4/11 among predicted FIT rows
- False-fit compile OOM: 0
- False-fit execution OOM: 4
- False OOMs: 0

False-fit trials were intrinsic attention sequence 4096, intrinsic attention
sequence 5120, and contention attention sequence 4096 at 0 MiB and 768 MiB
requested pressure. In all four cases compilation succeeded and the execution
failed while allocating a large attention temporary.

## Reliability by risk

- LOW: 11 trials, 7 FIT, 4 EXECUTION_OOM
- LIKELY EXCEEDS: 2 trials, 1 COMPILE_OOM, 1 EXECUTION_OOM

This is not production accuracy evidence. It is a small, device- and
version-specific diagnostic dataset.

## Occupancy response

For attention sequence 4096, upper-bound headroom changed as follows:

| Requested pressure | Assessment budget | Upper headroom | Risk | Outcome |
|---:|---:|---:|---|---|
| 0 MiB | 3.58 GiB | 1.91 GiB | LOW | EXECUTION_OOM |
| 768 MiB | 2.52 GiB | 855 MiB | LOW | EXECUTION_OOM |
| 1280 MiB | 1.45 GiB | -237 MiB | MODERATE | EXECUTION_OOM |

For attention sequence 6144, risk changed from MODERATE at 0 and 768 MiB to
LIKELY EXCEEDS at 1280 MiB. In both cases budget and headroom worsened
monotonically. The actual outcome was already EXECUTION_OOM without pressure.

## Device-budget and demand diagnostics

The calibrated upper bound covered the compiler-accounted memory for 17/17
trials that compiled successfully. This means the observed false fits were not
explained by the compiler-accounted quantity exceeding the frozen upper bound.
They were execution failures despite successful compilation, consistent with
runtime-only allocation, backend workspace, allocator behavior, or a volatile
budget. The experiment does not identify one cause conclusively.

The 8192 attention case was predicted LIKELY EXCEEDS and failed during
compilation. The 6144 attention case under 1280 MiB pressure was predicted
LIKELY EXCEEDS and failed during execution.

## Safety-reserve sensitivity

This was an offline sensitivity analysis. No trials were rerun and no shipped
policy was changed.

| Reserve | False fits | False OOMs | LOW FIT | LOW OOM |
|---|---:|---:|---:|---:|
| 0 MiB | 7 | 0 | 7 | 7 |
| 64 MiB | 4 | 0 | 7 | 4 |
| 128 MiB | 4 | 0 | 7 | 4 |
| Current 5%-bounded, 214.7 MiB | 4 | 0 | 7 | 4 |
| 256 MiB | 4 | 0 | 7 | 4 |

The current reserve is not contradicted by this dataset. The data do not
justify tuning it, especially because the false fits are execution failures
that the reserve alone did not remove.

## Snapshot race

Ten idle double-snapshot pairs showed zero driver-free-memory delta. Warmed pair
latency was approximately 112 to 172 ms because each snapshot invokes
`nvidia-smi`; the first pair included process startup and took 4.6 seconds.
This does not measure an active contention race. Trial snapshots were taken
within approximately 0.02 to 0.05 ms of the compilation start in the child
process, excluding the driver query itself that produced the snapshot.

## Timing

Median values across boundary trials:

- snapshot to compile start: 0.022 ms
- successful compilation: 11,412 ms
- successful execution: 27.5 ms
- device query in the earlier pilot: 55.9 ms

The small assessment-to-compile interval is not a guarantee that VRAM remains
reserved. Device query latency remains a product limitation.

## Donation rescue

Not run. Donation was not needed to answer the primary frozen-policy boundary
question, and assessment does not invoke donation analysis automatically.

## Limitations

This study uses one GPU, one JAX/jaxlib version, one allocator configuration,
and a small hand-selected workload set. Several execution OOMs occurred well
below the calibrated upper bound. Memory-analysis values are compiler
accounting, not exact runtime peaks. External allocations and driver state can
change between observations. The study has no independent held-out workload
split large enough to support tuning or generalization claims.
