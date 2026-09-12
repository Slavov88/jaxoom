# Pre-compilation execution-OOM risk model V1

**Status: COMPUTATIONALLY VERIFIED (experiment-only). Decision: LOGISTIC OOM MODEL PROMISING BUT DATA-LIMITED. Production: NO PRODUCTION CHANGE.**

## Reproduction and scope

- Commit baseline: `6c14e2853554edb80ed6aeb9a455e99a7f3558be`
- RTX 3050 Laptop GPU / CUDA-BFC / JAX 0.11.0 / `PREALLOCATE=false`
- 191 normalized rows, 63 workload groups
- 142 stable eligible rows; 7 execution-OOM groups
- Outcomes: 136 FIT, 50 EXECUTION_OOM, 3 COMPILE_OOM, 2 COMPILE_TIMEOUT

Repeated runs retain repeat identity and are grouped by canonical workload geometry. Compiler/runtime diagnostics and actual outcomes are labels/provenance only, never model inputs.

## Grouped V1 result

Selected model: `D_memory_shape__L2`, 11 features, training-fold-only imputation/normalization and five grouped folds.

| metric | result |
|---|---:|
| ROC-AUC | 0.817 |
| PR-AUC | 0.558 |
| Brier | 0.155 |
| ECE | 0.125 |

The grouped threshold-tree baseline achieved ROC-AUC 0.547. At frozen abstention thresholds `p<=0.1` / `p>=0.9`, coverage was 0% with zero false-safe and false-positive confident decisions. Budget monotonicity had 3 violations.

## Frozen model

`oom_risk_model_v1_frozen_2026-09-12.json` records the exact feature schema, means, scales, coefficients, intercept, regularization, dataset hash, and model hash:

```text
model hash:   1338e8570cd36a70bfa273e7c39e8063d1c1c8f2e2c4ba3cb78dbb264f535d05
dataset hash: 2c3f823b3627be0631419da4a94428bd926abdbecb40edb8612d83f63da6ff50
```

## Limitations

The seven execution-OOM groups are too few for useful calibration or family-generalized confident predictions. This artifact is the frozen prospective selector for the V2 expansion and must not be retrained using expansion outcomes.
