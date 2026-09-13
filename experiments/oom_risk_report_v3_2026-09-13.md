# OOM risk model V3: low-risk enrichment and compile-feasibility filtering

**Status: COMPUTATIONALLY VERIFIED. Decision: OOM RISK MODEL V3 PROMISING BUT DATA-LIMITED. Production: NO PRODUCTION CHANGE.**

## Repository and baseline

The live repository was verified before changes. HEAD and `origin/master` were `4520d69efdece191cbb844e35e0d6db8122a3e54`; the working tree was clean. The latest successful CI before this milestone was run 34706269266 on the V2 commit:

<https://github.com/Slavov88/jaxoom/actions/runs/34706269266>

The project test environment passed the full pytest suite, `python -m compileall -q src experiments`, and `git diff --check` before and after the experiment implementation.

V1 was reproduced at 191 rows / 63 groups / 7 execution-OOM groups with ROC-AUC 0.817, PR-AUC 0.558, Brier 0.155, ECE 0.125, 0% `.1/.9` coverage, and 3 budget-monotonicity violations. V2 remains frozen with 123 groups / 19 execution-OOM groups and zero budget-monotonicity violations.

## Frozen models

No V1 or V2 model was retrained before candidate selection.

```text
V1 model hash: 1338e8570cd36a70bfa273e7c39e8063d1c1c8f2e2c4ba3cb78dbb264f535d05
V2 model hash: 9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96
```

Schemas, normalization, coefficients, intercepts, regularization, and dataset hashes are preserved in the V1 freeze artifact and existing V2 model artifact.

## Static candidate pool

- Unique static candidates after excluding all V1/V2 groups: 3,709
- Static errors: 0
- Compile-filter rejected: 336
- Compile-filter retained: 3,373
- Selected new groups: 60
- No capacity sweeps were used

Selected categories:

```text
LOW_RISK_CHALLENGER:       20
MODERATE_BOUNDARY:         34
MODEL_DISAGREEMENT:         5
HIGH_RISK_COMPILE_AUDIT:    6
```

Family distribution: attention 20, autodiff 2, convolution 4, matmul 2, MLP 13, reduction 2, scan 2, training 2, transformer 13. No family exceeded 40% of the selected set.

The plan was frozen before runtime outcomes and contains no outcome fields:

```text
experiments/oom_risk_expansion_plan_v3_2026-09-13.json
```

## Compile-feasibility filter

The filter is experiment-only. It uses static shape/workspace rules to reject obviously enormous compile-risk geometry, while retaining a six-group rejected high-risk audit slice.

Historical grouped evidence contained 5 compile failures across 63 workload groups. The rule achieved:

```text
compile-failure recall:       1.000
compile-success retention:   0.892
```

On the prospective selected set:

```text
selected rejected audit groups: 6
compile failures among them:   6
prospective failure recall:    1.000
accepted-group success rate:   1.000
compile-failure waste rate:   10.0% (6/60)
```

The previous V2 campaign had 26/60 compile OOM or timeout outcomes (43.3%). The V3 campaign reduced this to 6/60, while retaining challenging boundary candidates and an explicit audit slice.

## Prospective frozen-model results

The 60 new groups produced 54 stable FIT/EXECUTION_OOM groups after excluding compile failures.

| metric | frozen V1 | frozen V2 |
|---|---:|---:|
| ROC-AUC | 0.893 | 0.919 |
| PR-AUC | 0.763 | 0.824 |
| Brier | 0.196 | 0.210 |

Prospective one-sided FIT coverage on the new stable groups:

| p_fit | V1 coverage / false-safe | V2 coverage / false-safe |
|---|---:|---:|
| 0.10 | 1.9% / 0 | 0.0% / 0 |
| 0.15 | 13.0% / 0 | 5.6% / 0 |
| 0.20 | 46.3% / 0 | 46.3% / 0 |
| 0.25 | 46.3% / 0 | 46.3% / 0 |
| 0.30 | 46.3% / 0 | 46.3% / 0 |
| 0.40 | 96.3% / 21 | 100% / 23 |
| 0.50 | 100% / 23 | 100% / 23 |

The frozen V2 model therefore achieved the target-like prospective observation of approximately 46% FIT coverage with zero false-safe predictions at `p_fit=0.20`. This is campaign-conditional evidence from active sampling, not a population-calibrated guarantee.

## New outcomes

Across 60 fresh-process runs:

```text
FIT:              31
EXECUTION_OOM:    23
COMPILE_OOM:       5
COMPILE_TIMEOUT:   1
UNSTABLE:          0
OTHER_FAILURE:     0
```

The expansion increased the combined corpus to 183 independent groups and 42 execution-OOM groups.

## Low-risk false-safe audit

There were **no** new cases satisfying:

```text
frozen V2 p(OOM) <= 0.25
actual outcome == EXECUTION_OOM
```

Consequently, the dedicated low-risk false-safe table is empty. This is encouraging but does not prove the low-risk region is safe outside the actively selected campaign.

## Dataset V3

- Total rows: 311
- Independent groups: 183
- Stable binary rows: 230
- Stable binary groups: 144
- FIT groups: 106
- EXECUTION_OOM groups: 42
- Compile-OOM rows: 20 including historical data
- Compile-timeout rows: 17 including historical data

## V3 feature set

```text
structural_peak_over_budget
calibrated_upper_over_budget
largest_over_budget
top_two_peak_live_over_budget
peak_live_over_budget
dtype_bytes
config_numeric_count
config_numeric_max
config_numeric_log_product
largest_over_structural
top_two_over_structural
```

The final schema adds large-buffer and top-two structural ratios to the monotone V2 core. It excludes runtime, allocator-diagnostic, compiler-diagnostic, and outcome-derived fields. The selected model has 11 features, nonnegative coefficients for all demand/budget and structural-ratio terms, and model hash:

```text
7663490267c552569c5b9f1b93cf05dcc113282317c899f55c8e30e06f4b2da8
```

## Grouped V3 model comparison

Metrics are grouped out-of-fold metrics over 230 stable binary rows.

| model | ROC-AUC | PR-AUC | Brier | ECE | budget violations |
|---|---:|---:|---:|---:|---:|
| V2 compact constrained | 0.913 | 0.781 | 0.157 | 0.116 | 0 |
| V1-rich constrained | 0.915 | 0.790 | 0.152 | 0.128 | 0 |
| V3 compact L2 | 0.916 | 0.787 | 0.156 | 0.115 | 0 |
| V3 compact constrained | 0.916 | 0.787 | 0.156 | 0.115 | 0 |
| V3 group-balanced constrained | 0.926 | 0.791 | 0.169 | 0.208 | 0 |
| V3 threshold tree | 0.664 | 0.391 | 0.387 | 0.387 | not applicable |

The group-balanced model ranks best but is substantially worse calibrated. The ordinary and constrained V3 compact models are effectively identical; the constrained formulation is retained because it preserves the safety monotonicity requirement.

## Risk-coverage frontier

Grouped out-of-fold V3 compact constrained results:

| p_fit | FIT coverage | false-safe count |
|---|---:|---:|
| 0.05 | 0.0% | 0 |
| 0.10 | 0.0% | 0 |
| 0.15 | 2.2% | 0 |
| 0.20 | 20.9% | 1 |
| 0.25 | 40.4% | 1 |
| 0.30 | 53.0% | 3 |
| 0.40 | 84.3% | 38 |
| 0.50 | 95.7% | 61 |

V3 does not yet provide 40% grouped-OOS FIT coverage with zero false-safe predictions. The frozen V2 prospective result is stronger on this particular campaign, but it must be confirmed prospectively on another frozen set.

## Calibration and active sampling

V3 compact constrained ECE is 0.115 on the enriched grouped OOF corpus. The new campaign deliberately enriched difficult candidates, so all probability calibration results are campaign-conditional. No population OOM probability claim is made.

## Family and leave-family-out results

Leave-family-out results for the V3 constrained model:

- attention: ROC-AUC 0.918, PR-AUC 0.857, Brier 0.178 (`N=131`)
- convolution: ROC-AUC 0.667, PR-AUC 0.696, Brier 0.316 (`N=16`)
- transformer: ROC-AUC 0.967, PR-AUC 0.874, Brier 0.224 (`N=22`)

Autodiff, MLP, training, scan, reduction, and matmul holdouts lack enough class variation for meaningful AUCs. Convolution remains materially weaker and cross-family generalization is not fully resolved.

## Permutation sanity

Group-label permutation over 20 repetitions produced mean AUC **0.497**, consistent with chance-level performance.

## Speed

```text
Static pool extraction: approximately 41.6 s for 4,000 requested candidates
V3 model inference:     approximately 0.303 ms per 1,000 rows
Feature count:          11
Serialized model:       1,160 bytes
```

## Pre-compilation guarantee

The V3 scoring path uses static feature artifacts only:

```text
target compilation:             NO
target execution:               NO
runtime allocator diagnostics:  NO
```

The runtime campaign itself is separate and exists only to collect labels.

## Artifacts

```text
experiments/oom_risk_expansion_v3.py
experiments/oom_risk_expansion_campaign_v3.py
experiments/oom_risk_candidate_pool_v3_2026-09-13.json
experiments/compile_feasibility_filter_v1_2026-09-13.json
experiments/oom_risk_expansion_plan_v3_2026-09-13.json
experiments/oom_risk_expansion_results_v3_2026-09-13.json
experiments/oom_risk_model_v3.py
experiments/oom_risk_model_v3_2026-09-13.json
experiments/oom_risk_dataset_v3_2026-09-13.json
experiments/oom_risk_cv_v3_2026-09-13.json
experiments/oom_risk_models_v3_2026-09-13.json
experiments/oom_risk_risk_coverage_v3_2026-09-13.json
experiments/oom_risk_summary_v3_2026-09-13.json
tests/test_oom_risk_v3.py
```

## Decision

**OOM RISK MODEL V3 PROMISING BUT DATA-LIMITED**

The compile filter substantially reduced wasted runs, and frozen V2 achieved 46% prospective FIT coverage with zero false-safe decisions at `p_fit=0.20`. The expanded corpus now contains 42 execution-OOM groups, and V3 preserves zero budget-monotonicity violations. However, the V3 grouped frontier still has one false-safe at approximately 40% coverage, calibration remains campaign-conditional, and convolution/cross-family evidence is limited.

## Production

**NO PRODUCTION CHANGE.**

## Next technical decision

**fully held-out OOM risk validation**. Freeze the current V3 artifact and evaluate it on a new, independently selected group set before considering any production use. No further expansion or production integration was started in this milestone.
