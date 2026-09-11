# Allocator capacity boundary study

Decision: **CAPACITY THRESHOLDS NOT STABLE ENOUGH**.

## Environment and protocol

Primary environment:

```text
RTX 3050 Laptop GPU, 4096 MiB
Driver 566.07
JAX/jaxlib 0.11.0
CUDA, BFC-style allocator
XLA_PYTHON_CLIENT_PREALLOCATE=false
capacity control: XLA_CLIENT_MEM_FRACTION
```

Each probe used a fresh subprocess, one workload, compile, first execution, and explicit synchronization. Capacity was controlled through `XLA_CLIENT_MEM_FRACTION`; observed allocator limits were approximately fraction times physical 4 GiB.

## Workload panel

Nine workloads were frozen before threshold probing:

```text
attention: S=1024, 2048, 4096, 4608, 5120 float32
attention: B=2,H=12,S=2560,D=64 float32
attention: B=2,H=16,S=2560,D=96 float16
convolution: CONV-1 B=2048
convolution: CONV-2 B=1024
```

The panel includes small FIT controls, attention FIT/OOM transitions, float16,
float32, and both known convolution failures.

## Capacity search

```text
Total probes:       81
FIT:                50
EXECUTION_OOM:      25
COMPILE_TIMEOUT:     6
COMPILE_OOM:         0
Infrastructure:      0
```

Five workloads produced usable monotone brackets. Four did not:

- S=1024 and S=2048 were FIT at the lowest tested capacity.
- B=2,H=12,S=2560,D=64 float32 was FIT at every tested capacity, conflicting with its frozen historical EXECUTION_OOM label at nominal capacity.
- CONV-1 timed out during compilation at all tested capacities; no bracket was obtained.

## Threshold brackets

`required capacity` is retained as:

```text
last_oom_capacity < required_capacity <= first_fit_capacity
```

| Workload | Request upper | Last OOM | First FIT | Width | Failure stage |
|---|---:|---:|---:|---:|---|
| ATT B2 H16 S2560 D96 f16 | 0.39 GiB | 1.00 GiB | 1.10 GiB | 104 MiB | execution |
| ATT S4096 f32 | 1.00 GiB | 3.40 GiB | 3.50 GiB | 102 MiB | execution |
| ATT S4608 f32 | 1.27 GiB | 3.40 GiB | 3.50 GiB | 102 MiB | execution |
| ATT S5120 f32 | 1.56 GiB | 3.60 GiB | 3.70 GiB | 104 MiB | execution |
| CONV-2 B1024 | 1.12 GiB | 3.70 GiB | 3.80 GiB | 102 MiB | execution |

No compile-OOM threshold was identified. CONV-1 remains `THRESHOLD_INCOMPLETE_TIMEOUT`.

## Capacity ratios

The interval is `request_upper / first_fit` through
`request_upper / last_oom`:

| Workload | Ratio interval |
|---|---:|
| ATT f16 | 0.355–0.391 |
| ATT S4096 | 0.286–0.294 |
| ATT S4608 | 0.362–0.372 |
| ATT S5120 | 0.422–0.434 |
| CONV-2 | 0.296–0.304 |

Ratios vary materially by geometry and dtype. They do not cluster tightly
around one universal constant.

## Limit/3 hypothesis

The current hypothesis predicts:

```text
required capacity = 3 * request_upper
```

It was unsafe on two of five bracketed workloads:

```text
ATT S4096 float32
CONV-2 B1024
```

It was conservative on the remaining three. Therefore `allocator_limit / 3`
is not supported as a generally safe threshold by this boundary sample.

## Capacity candidates

| Model | Development unsafe misses | Held-out unsafe misses | Interpretation |
|---|---:|---:|---|
| `3 * request` | 1/3 | 1/2 | unsafe |
| `3.25 * request` | 1/3 | 1/2 | unsafe |
| `3.50 * request` | 0/3 | 0/2 | conservative envelope |
| `request + structural` | 3/3 | 2/2 | unsafe |
| `request + 3*structural` | 0/3 | 0/2 | very conservative |

The `3.50 * request` envelope was selected from development rows and held on
the two-row evaluation split, but this is not sufficient identification.
The held-out set contains only two bracketed workloads, and no compile-OOM
threshold.

## Compile versus execution

All usable brackets ended in execution OOM below the threshold. The known
compile-OOM convolution workload could not be bracketed because compilation
exceeded the probe timeout. One unified compile/execution threshold model is
therefore unsupported.

## Request proxy audit

The existing request proxy remains:

```text
largest_buffer * dtype_bytes / 2
```

Existing diagnostic evidence showed zero compiler-temp underpredictions across
six rows. Observed allocation-request evidence showed one approximately 16 MiB
underprediction out of five rows, on CONV-2. That error is small relative to
most brackets but should not be silently absorbed into the capacity model.

## Stability finding

The B=2,H=12,S=2560,D=64 float32 workload was previously a stable OOM
classifier row, including repeated fresh-process diagnostics. In this capacity
campaign it FIT repeatedly at capacities from approximately 1.8 GiB through
3.0 GiB. This nominally identical configuration conflict is direct evidence
that the threshold is not stable under the current experimental conditions.

The result may reflect allocator/autotuner state or another environmental
factor, but it prevents treating the observed threshold brackets as a stable
physical law.

## Pre-compilation guarantee

Capacity-model predictions require only static features, the configured
allocator limit, and the frozen request proxy:

```text
target compilation: NO
target execution:    NO
runtime snapshots:   NO
```

Runtime capacity probes are validation labels only.

## Decision

**CAPACITY THRESHOLDS NOT STABLE ENOUGH.** Five execution thresholds were
bracketed to approximately 102–104 MiB, but one frozen workload contradicted
its prior outcome, CONV-1 could not be bracketed, and compile-OOM behavior was
not measured. The `allocator_limit / 3` hypothesis is unsafe on two measured
brackets.

No production code changed.
