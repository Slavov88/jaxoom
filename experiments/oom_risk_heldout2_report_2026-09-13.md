# Second Fully Held-Out OOM Validation — 2026-09-13

**Status: COMPUTATIONALLY VERIFIED**

## Decision

**OOM RISK MODEL CONFIRMATORILY VALIDATED**

Production decision: **NO PRODUCTION CHANGE**.

## Frozen protocol
- V2 hash: `9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96`
- V3 hash: `7663490267c552569c5b9f1b93cf05dcc113282317c899f55c8e30e06f4b2da8`
- Primary V2 threshold: `p_fit=0.20`
- Primary plan hash: `71d900039f182a6f8db257cab203be98483d4eb10fd33ed0f901d808c641279b`
- Primary prediction hash: `07ba2839a7f875bde5aef4b5de51a3588c088e93d43c391f264a46d634896822`
- Primary selection used model probabilities: `NO`
- No retraining, feature changes, normalization changes, or threshold tuning.

## Batch 2 panel
- Selected: 110; eligible: 107
- Families: `{'attention': 22, 'convolution': 18, 'mlp': 11, 'training': 11, 'transformer': 14, 'autodiff': 12, 'scan': 8, 'matmul': 7, 'reduction': 7}`
- Dtypes: `{'float16': 60, 'float32': 50}`
- Scales: `{'small': 30, 'medium': 32, 'large': 30, 'near_device': 18}`
- Overlap: 0 group IDs and 0 exact fingerprints with prior data.

## Batch 2 outcomes

```text
FIT: 97
EXECUTION_OOM: 10
COMPILE_OOM: 2
COMPILE_TIMEOUT: 1
```

## Batch 2 metrics

| Model | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| V2 | 0.997 | 0.970 | 0.051 | 0.143 |
| V3 | 0.996 | 0.957 | 0.045 | 0.140 |

## Batch 2 V2 risk coverage

| p_fit | N | coverage | false-safe | upper 95% |
|---:|---:|---:|---:|---:|
| 0.10 | 0 | 0.0% | 0 | 100.0% |
| 0.15 | 53 | 49.5% | 0 | 5.5% |
| 0.20 | 87 | 81.3% | 0 | 3.4% |
| 0.25 | 93 | 86.9% | 0 | 3.2% |
| 0.30 | 98 | 91.6% | 2 | 6.3% |
| 0.35 | 101 | 94.4% | 4 | 8.8% |
| 0.40 | 103 | 96.3% | 6 | 11.2% |

## Batch 2 V3 risk coverage

| p_fit | N | coverage | false-safe | upper 95% |
|---:|---:|---:|---:|---:|
| 0.10 | 4 | 3.7% | 0 | 52.7% |
| 0.15 | 78 | 72.9% | 0 | 3.8% |
| 0.20 | 91 | 85.0% | 0 | 3.2% |
| 0.25 | 94 | 87.9% | 0 | 3.1% |
| 0.30 | 98 | 91.6% | 2 | 6.3% |
| 0.35 | 101 | 94.4% | 4 | 8.8% |
| 0.40 | 103 | 96.3% | 6 | 11.2% |

## Batch-2 family safety at p_fit=.20

| Family | Eligible | V2 likely-fit | V2 false-safe | V3 likely-fit | V3 false-safe |
|---|---:|---:|---:|---:|---:|
| attention | 21 | 10 | 0 | 11 | 0 |
| autodiff | 12 | 11 | 0 | 12 | 0 |
| convolution | 16 | 13 | 0 | 13 | 0 |
| matmul | 7 | 7 | 0 | 7 | 0 |
| mlp | 11 | 11 | 0 | 11 | 0 |
| reduction | 7 | 7 | 0 | 7 | 0 |
| scan | 8 | 7 | 0 | 7 | 0 |
| training | 11 | 10 | 0 | 11 | 0 |
| transformer | 14 | 11 | 0 | 12 | 0 |

## Pooled primary result (batch 1 + batch 2)

- Eligible rows: **190** (FIT 175, EXECUTION_OOM 15)
- V2 @ p_fit=.20: **{'p_fit': 0.2, 'likely_fit_n': 157, 'coverage': 0.8263157894736842, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.01890020551196819}**
- V3 @ p_fit=.20: **{'p_fit': 0.2, 'likely_fit_n': 158, 'coverage': 0.8315789473684211, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.018781714429240905}**

The exact one-sided Clopper–Pearson bound is the primary error statement.

## Batch replication
- Environment comparable: `True`
- Batch 2 V2 @ .20: `{'p_fit': 0.2, 'likely_fit_n': 87, 'coverage': 0.8130841121495327, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.033847610681619573}`
- Pooled V2 @ .20: `{'p_fit': 0.2, 'likely_fit_n': 157, 'coverage': 0.8263157894736842, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.01890020551196819}`

## Supplemental convolution stress
- Selected: 24; outcomes: `{'FIT': 20, 'COMPILE_TIMEOUT': 4}`
- This slice is not included in the pooled population bound.

## Compile filter
- Used for primary selection: `False`
- Batch-2 compile-failure recall: `1.0`
- Compile-success retention: `0.9439252336448598`

## Power interpretation

With zero failures, approximate LIKELY_FIT counts required for one-sided 95% upper bounds are 59 (<5%), 149 (<2%), and 299 (<1%).

## Monotonicity
- Batch-2 V2/V3 violations: `0 / 0`

## Near-duplicate sensitivity
- Strict clustering retained `152` of `190` rows; V2 p_fit=.20: `{'p_fit': 0.2, 'likely_fit_n': 126, 'coverage': 0.8289473684210527, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.023495238866670053}`

The confirmatory claim is scoped to the model-independent primary workload population and V2 at p_fit=.20. It is not a universal population guarantee and does not justify production integration by itself.
