# OOM-Risk Production Integration Design

**Date:** 2026-09-13  
**Status:** DESIGN ONLY — NO PUBLIC API CHANGE  
**Result status:** COMPUTATIONALLY VERIFIED for the replay audit; production readiness is blocked.

## Decision

**V2_FEATURE_PIPELINE_NOT_PRODUCTION_REPRODUCIBLE**

The frozen V2 model is confirmatorily validated, but its feature pipeline is not yet a defensible generic production pipeline. In particular:

- `config_numeric_count`, `config_numeric_max`, and `config_numeric_log_product` came from experiment workload dictionaries. The current public callable API has no canonical workload-configuration object from which to derive them.
- `dtype_bytes` is likewise supplied as experiment metadata and is not uniquely defined for arbitrary multi-dtype JAX functions.
- The frozen schema calls a feature `largest_over_budget`, while the experiment rows expose `largest_buffer_over_budget`; the frozen V2 model imputes the missing schema field to zero, and its coefficient is currently zero. A production implementation that computes the apparent semantic feature would not reproduce the frozen feature vector exactly, even though the current score remains numerically identical.

No public API is exported and no production behavior changes.

## Confirmatory evidence carried into this design

The exact frozen V2 artifact is:

```text
model hash: 9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96
dataset hash: b0eb4498401077cf58f13734e6eb7c172240aa8cba9ccb8273417115cac6d61c
primary threshold: p_fit = 0.20
held-out LIKELY_FIT: 157
held-out false-safe: 0
exact one-sided 95% upper bound: 1.89%
strict near-duplicate sensitivity bound: 2.35%
```

Validated scope:

```text
NVIDIA RTX 3050 Laptop GPU
JAX 0.11.0 / jaxlib 0.11.0
GPU/CUDA, CUDA/BFC
XLA_PYTHON_CLIENT_PREALLOCATE=false
fixed 3 GiB validation budget
```

This is an empirical execution-OOM result, not a universal fit guarantee.

## Proposed API comparison

### Design A: dedicated API

```python
assess_oom_risk(fn, *args, workload_config=..., memory_limit=...)
```

This is the preferred eventual shape because empirical risk has a different epistemic status from `estimate()` and `assess()`:

- `estimate()` is a structural JAXPR memory analysis.
- `assess()` compares a static/calibrated interval with a budget.
- OOM risk is a versioned empirical classifier with a bounded applicability domain.

### Design B: extend `assess()`

This would be backward compatible only syntactically, but risks making a deterministic structural assessment appear to be a validated probability. It also makes unsupported empirical scope difficult to communicate. This design is rejected for the first integration.

### Proposed result, not implemented

```python
OomRiskAssessment(
    status=OomRiskStatus.LIKELY_FIT | UNCERTAIN | UNSUPPORTED,
    oom_score=float | None,
    fit_threshold=0.20 | None,
    applicability=...,
    model_version="v2" | None,
    model_hash=... | None,
    validation_scope=...,
    reason=...,
)
```

`LIKELY_FIT` is intentionally not named `SAFE`, `WILL_FIT`, or `NO_OOM`.

The eventual decision rule would be:

```text
if applicability is valid and p_oom <= 0.20:
    LIKELY_FIT
elif applicability is valid:
    UNCERTAIN
else:
    UNSUPPORTED
```

An unsupported environment must override the numerical score.

## Frozen V2 feature audit

| Feature | Proposed source | Classification | Result |
|---|---|---|---|
| `structural_peak_over_budget` | `MemoryReport.estimated_peak_bytes / 3 GiB` | generic static | reproducible |
| `calibrated_upper_over_budget` | structural peak multiplied by frozen V1 static upper ratio `1.66765051935476`, divided by 3 GiB | calibration-derived | requires risk-specific frozen calibration |
| `largest_over_budget` | largest `MemoryReport.largest_buffers` byte count / 3 GiB | generic static, but schema/source mismatch | blocked for exact replay |
| `top_two_peak_live_over_budget` | two largest `MemoryReport.peak.live_buffers` / 3 GiB | generic static | reproducible |
| `peak_live_over_budget` | structural peak / 3 GiB | generic static | reproducible |
| `dtype_bytes` | experiment `dtype` metadata | workload metadata | no canonical arbitrary-callable definition |
| `config_numeric_count` | numeric values in experiment `configuration` | experiment-generator-derived | no canonical public source |
| `config_numeric_max` | maximum numeric configuration value | experiment-generator-derived | no canonical public source |
| `config_numeric_log_product` | `log1p(product(max(1, value)))` over configuration values | experiment-generator-derived | no canonical public source |

The existing `calibrate()` registry must not be substituted silently: it describes compiler/static ratios and is not the frozen OOM-risk calibration artifact.

## Feature replay audit

The internal structured adapter replayed all 224 stored prediction rows from the first and second held-out panels.

```text
rows replayed:                         224
feature mismatched values:             224
feature mismatch: largest_over_budget  224
model probability mismatches:            0
p_fit=0.20 status mismatches:             0
max absolute score difference:           0
hidden target compilation:               NO
hidden target execution:                 NO
```

The score match is explained by the frozen V2 coefficient for `largest_over_budget` being exactly zero. This is not sufficient to claim feature-pipeline reproducibility: the model schema and the source feature are still inconsistent.

For arbitrary public callables, replay is additionally blocked by the three configuration-derived features and the single-dtype assumption.

## Applicability design

A first production version would need strict guards for:

```text
backend == GPU/CUDA
device identity contains RTX 3050 Laptop GPU
JAX == 0.11.0
jaxlib == 0.11.0
XLA_PYTHON_CLIENT_PREALLOCATE == "false"
XLA memory-fraction overrides unset
CUDA/BFC default allocator
fixed validated budget == 3 GiB
```

Unknown environment state must return `UNSUPPORTED`, not silently predict.

A future feature-domain guard could use inclusive ranges from the frozen training feature distribution, but those ranges have not been validated as an OOD policy and should not be added implicitly.

## Compile-risk interaction

The experiment-only compile filter remains separate. It is not a production gate and must not be combined with execution-OOM probability. The first production design should not claim that `LIKELY_FIT` means successful compilation plus successful execution.

## Artifact design

The proposed immutable JSON model payload is recorded at:

```text
experiments/oom_risk_production_model_v2_design_2026-09-13.json
```

It contains the schema, normalization, coefficients, intercept, model hash, threshold, validation metadata, and scope. It is marked `DESIGN_ONLY_NOT_PUBLIC`.

Direct logistic inference is sufficient:

```python
z = intercept + dot(coefficients[1:], standardized_features)
p = stable_sigmoid(z)
```

No sklearn runtime dependency or pickle is needed.

## Readiness gates

| Gate | Result | Reason |
|---|---|---|
| A. Frozen artifact reproducibility | PASS | V2 hash recomputes exactly from canonical model fields |
| B. Feature reproducibility for structured adapter | FAIL | `largest_over_budget` schema/source mismatch |
| B. Feature reproducibility for arbitrary callable | BLOCKED | configuration features lack canonical public inputs |
| C. Logistic inference equivalence | PASS | zero probability mismatches on 224 replay rows |
| D. Strict environment applicability | PASS as design | exact guard is defined; current shell without explicit PREALLOCATE is unsupported |
| E. No hidden compilation/execution | PASS | adapter uses static `jaxoom.estimate()` only |
| F. Non-guarantee semantics | PASS | proposed statuses are asymmetric and scoped |
| G. Backward compatibility | PASS BY NO CHANGE | no public API was added |

## Statistical documentation wording

A future supported API may use wording equivalent to:

> On two model-independent held-out validation batches within the validated RTX 3050/JAX 0.11.0 scope, frozen V2 at `p_fit=0.20` classified 157 workloads as `LIKELY_FIT` with zero observed execution-OOM false-safes. The exact one-sided 95% upper bound was 1.89%; a stricter near-duplicate sensitivity analysis gave 2.35%. This is an empirical bound for the validated workload population, not a universal guarantee for arbitrary JAX programs or devices.

## Final design decision

```text
V2_FEATURE_PIPELINE_NOT_PRODUCTION_REPRODUCIBLE
NO PUBLIC PRODUCTION CHANGE
```

The next implementation milestone should first resolve the feature contract: either provide a validated structured workload adapter whose configuration and dtype semantics are explicit, or redevelop a production-compatible model using only generic features exposed by arbitrary static JAX analysis.
