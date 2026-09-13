# Fully Held-Out OOM Risk Validation — 2026-09-13

**Status: COMPUTATIONALLY VERIFIED**

## Decision

**OOM RISK MODEL VALIDATION PROMISING BUT UNDERPOWERED**

Production decision: **NO PRODUCTION CHANGE**.

## Frozen models and protocol
- V2 model hash: `9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96`
- V3 model hash: `7663490267c552569c5b9f1b93cf05dcc113282317c899f55c8e30e06f4b2da8`
- Plan SHA-256: `4e8319b7b9864043509ca89857a26f1920cfbe6bf1c64197c9f9987fe15a174f`
- Prediction SHA-256: `1a929787d91053b70e7a70170ea5ae30c85c035a130bd321da412bba752abad5`
- Model-independent selection: `True`
- Seed: `20260913`; selected groups: 90
- No retraining, feature changes, coefficient changes, normalization changes, or post-outcome threshold tuning.

## Panel design
- Families: `{'attention': 18, 'convolution': 15, 'mlp': 9, 'training': 9, 'transformer': 12, 'autodiff': 10, 'scan': 7, 'matmul': 5, 'reduction': 5}`
- Dtypes: `{'float32': 45, 'float16': 45}`
- Scale strata: `{'small': 23, 'medium': 27, 'large': 26, 'near_device': 14}`
- Training group overlaps: `0`
- Exact fingerprint overlaps: `0`
- Near-duplicate pairs within panel / against training: `6 / 28`

## Outcomes

```text
FIT: 78
EXECUTION_OOM: 5
COMPILE_OOM: 3
COMPILE_TIMEOUT: 4
```

Eligible FIT/EXECUTION_OOM rows: **{h['eligible_rows']}** (FIT {h['eligible_fit']}, EXECUTION_OOM {h['eligible_execution_oom']})

## Metrics

| Model | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| V2 | 1.000 | 1.000 | 0.039 | 0.146 |
| V3 | 1.000 | 1.000 | 0.039 | 0.179 |

## Risk coverage

| p_fit | V2 N | V2 coverage | V2 false-safe | V2 upper 95% | V3 N | V3 coverage | V3 false-safe | V3 upper 95% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0 | 0.0% | 0 | 100.0% | 0 | 0.0% | 0 | 100.0% |
| 0.15 | 39 | 47.0% | 0 | 7.4% | 35 | 42.2% | 0 | 8.2% |
| 0.20 | 70 | 84.3% | 0 | 4.2% | 67 | 80.7% | 0 | 4.4% |
| 0.25 | 72 | 86.7% | 0 | 4.1% | 70 | 84.3% | 0 | 4.2% |
| 0.30 | 78 | 94.0% | 0 | 3.8% | 74 | 89.2% | 0 | 4.0% |
| 0.35 | 79 | 95.2% | 1 | 5.9% | 78 | 94.0% | 0 | 3.8% |
| 0.40 | 80 | 96.4% | 2 | 7.7% | 78 | 94.0% | 0 | 3.8% |

The bound is a one-sided exact Clopper–Pearson 95% upper bound. Zero observed false-safes are not treated as zero underlying risk.

## Primary operating points
- V2 p_fit=.20: {'p_fit': 0.2, 'likely_fit_n': 70, 'coverage': 0.8433734939759037, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.04189334406688858}
- V2 p_fit=.25: {'p_fit': 0.25, 'likely_fit_n': 72, 'coverage': 0.8674698795180723, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.040753686230639914}
- V3 p_fit=.20: {'p_fit': 0.2, 'likely_fit_n': 67, 'coverage': 0.8072289156626506, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.043727554781889444}
- V3 p_fit=.25: {'p_fit': 0.25, 'likely_fit_n': 70, 'coverage': 0.8433734939759037, 'false_safe_n': 0, 'false_safe_rate': 0.0, 'false_safe_upper_95': 0.04189334406688858}

V2 has higher coverage than V3 at both candidate thresholds with zero observed false-safes. Neither model dominates over the entire grid: V3 avoids V2's false-safes at p_fit=.35 and .40.

## Family safety at p_fit=.20

| Family | Eligible N | V2 likely-fit N | V2 false-safe | V3 likely-fit N | V3 false-safe |
|---|---:|---:|---:|---:|---:|
| attention | 15 | 7 | 0 | 5 | 0 |
| autodiff | 10 | 10 | 0 | 10 | 0 |
| convolution | 11 | 10 | 0 | 9 | 0 |
| matmul | 5 | 5 | 0 | 5 | 0 |
| mlp | 9 | 9 | 0 | 9 | 0 |
| reduction | 5 | 5 | 0 | 5 | 0 |
| scan | 7 | 6 | 0 | 6 | 0 |
| training | 9 | 9 | 0 | 9 | 0 |
| transformer | 12 | 9 | 0 | 9 | 0 |

Convolution is represented by 15 selected groups and 11 eligible groups; it has zero observed false-safes for both models, but its family-specific upper bound is wide because no convolution execution OOM occurred.

## Low-risk false-safe audit

No new V2 `p(OOM) <= 0.25 -> EXECUTION_OOM` cases.

## End-to-end failures

Compile failures are excluded from the primary execution-OOM metric and are retained separately. None were classified LIKELY_FIT at p_fit=.20 by either model:

```json
[
  {
    "candidate_id": "HOLDOUT-0008",
    "family": "attention",
    "likely_fit_v2_at_20": false,
    "likely_fit_v3_at_20": false,
    "outcome": "COMPILE_OOM",
    "v2_probability": 0.43687094799608367,
    "v3_probability": 0.5199839520551712
  },
  {
    "candidate_id": "HOLDOUT-0012",
    "family": "attention",
    "likely_fit_v2_at_20": false,
    "likely_fit_v3_at_20": false,
    "outcome": "COMPILE_TIMEOUT",
    "v2_probability": 0.7052589449165493,
    "v3_probability": 0.7899310128642206
  },
  {
    "candidate_id": "HOLDOUT-0015",
    "family": "attention",
    "likely_fit_v2_at_20": false,
    "likely_fit_v3_at_20": false,
    "outcome": "COMPILE_OOM",
    "v2_probability": 0.9279227334673099,
    "v3_probability": 0.9622418268063968
  },
  {
    "candidate_id": "HOLDOUT-0028",
    "family": "convolution",
    "likely_fit_v2_at_20": false,
    "likely_fit_v3_at_20": false,
    "outcome": "COMPILE_TIMEOUT",
    "v2_probability": 0.2629372356095438,
    "v3_probability": 0.310238880856618
  },
  {
    "candidate_id": "HOLDOUT-0029",
    "family": "convolution",
    "likely_fit_v2_at_20": false,
    "likely_fit_v3_at_20": false,
    "outcome": "COMPILE_TIMEOUT",
    "v2_probability": 0.29885873257974666,
    "v3_probability": 0.35499051397448744
  },
  {
    "candidate_id": "HOLDOUT-0031",
    "family": "convolution",
    "likely_fit_v2_at_20": false,
    "likely_fit_v3_at_20": false,
    "outcome": "COMPILE_TIMEOUT",
    "v2_probability": 0.5684448870661212,
    "v3_probability": 0.6651588622678493
  },
  {
    "candidate_id": "HOLDOUT-0032",
    "family": "convolution",
    "likely_fit_v2_at_20": false,
    "likely_fit_v3_at_20": false,
    "outcome": "COMPILE_OOM",
    "v2_probability": 0.9885327196303109,
    "v3_probability": 0.9957182766324691
  }
]
```

## Additional checks
- V2/V3 budget monotonicity violations: `0 / 0`
- Permutation mean AUC: `0.432 / 0.436` (V2 / V3)
- Model-only inference ms per 1,000 rows: `0.461 / 0.502`

Predictions used no target compilation, target execution, runtime allocator diagnostics, or compiler memory analysis.

## Limitation

The held-out panel contains only five stable EXECUTION_OOM groups. The overall p_fit=.20 bound is below 5%, but family-specific bounds remain wide; this is insufficient for a broad production safety claim.
