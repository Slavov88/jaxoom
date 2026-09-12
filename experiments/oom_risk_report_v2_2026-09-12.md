# OOM risk model V2 expansion and evaluation

**Status: COMPUTATIONALLY VERIFIED. Decision: OOM RISK MODEL V2 PROMISING BUT DATA-LIMITED. Production: NO PRODUCTION CHANGE.**

## Scope

This milestone froze V1, generated and selected static candidates before runtime outcomes, ran a bounded fresh-process GPU campaign, evaluated frozen-V1 prospective performance, and fit experiment-only V2 models. No production APIs or runtime behavior changed.

Primary environment: RTX 3050 Laptop GPU, driver 566.07, JAX/jaxlib 0.11.0, CUDA/BFC, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Capacity sweeps were not used.

## Frozen V1

- V1 dataset hash: `2c3f823b3627be0631419da4a94428bd926abdbecb40edb8612d83f63da6ff50`
- V1 model hash: `1338e8570cd36a70bfa273e7c39e8063d1c1c8f2e2c4ba3cb78dbb264f535d05`
- Feature schema: the 11 features in `oom_risk_model_v1_frozen_2026-09-12.json`
- Normalization and coefficients were copied exactly; no retraining occurred before selection.

## Static candidate pool and selection

- Candidate pool generated: 2,365 unique static candidates
- Static errors: 0
- Selected frozen candidates: 60
- Selection used only frozen V1 predictions, static aggregate ratios, family, and metadata
- Risk distribution: 18 low-risk, 27 uncertain, 15 high-risk
- Family distribution: attention 27, convolution 7, training 4, MLP 3, autodiff 3, transformer 4, scan 3, reduction 6, matmul 3
- Largest family share: attention 45%; no family exceeded 50%
- The frozen plan contains no runtime outcome fields.

The selection plan is `oom_risk_expansion_plan_v2_2026-09-12.json`. Static tracing used `jaxoom.estimate` and did not compile or execute candidates.

## Prospective V1 performance

On the 34 newly executed stable FIT/EXECUTION_OOM groups (compile failures excluded):

| metric | frozen V1 |
|---|---:|
| ROC-AUC | 0.973 |
| PR-AUC | 0.867 |
| Brier | 0.110 |
| log loss | 0.368 |

At frozen thresholds `p<=0.1` / `p>=0.9`:

- coverage: 8.8% (2 likely-FIT, 1 likely-OOM)
- false-safe likely-FIT: 0
- false-positive likely-OOM: 0
- uncertain: 31

This is prospective **campaign-conditional** evidence because active selection changed the class prevalence. It is not population-level calibration evidence. There were no frozen-V1 confident errors among the stable new rows.

## New runtime outcomes

Across 60 genuinely new workload groups, using one fresh subprocess per candidate:

- FIT: 22
- EXECUTION_OOM: 12
- COMPILE_OOM: 12
- COMPILE_TIMEOUT: 14
- UNSTABLE: 0
- OTHER_FAILURE: 0

The campaign stopped at the preselected 60-group cap. The high compile-timeout/compile-OOM rate is an infrastructure/model-scope limitation, not a reason to manufacture more capacity variations.

## V2 dataset

- Total rows: 251
- Total independent groups: 123
- Stable binary rows: 176
- Stable binary groups: 90
- FIT groups: 75
- EXECUTION_OOM groups: 19
- Compile OOM groups: 15 including historical data
- Compile timeout groups: 16 including historical data

The desired 25–40 independent execution-OOM groups was not reached; only 19 are available. This is the main reason the result remains data-limited.

## V2 feature set

The compact V2 schema contains nine features:

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
```

External pressure and redundant calibrated-central terms were removed. The first five are demand-to-budget ratios. Both ordinary compact L2 and sign-constrained L2 had zero budget-monotonicity violations; the constrained version was retained as the explicit safety formulation.

## Grouped model comparison

All metrics are grouped out-of-fold metrics over 176 stable binary rows. The tree threshold is selected inside each grouped training fold.

| model | ROC-AUC | PR-AUC | Brier | ECE | coverage at .1/.9 | false-safe FIT | false-positive OOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1-schema L2 retrained | 0.856 | 0.680 | 0.150 | 0.104 | 1.1% | 0 | 0 |
| V2 compact L2 | 0.845 | 0.650 | 0.155 | 0.088 | 0% | 0 | 0 |
| V2 compact L1 | 0.810 | 0.610 | 0.153 | 0.068 | 2.8% | 0 | 0 |
| V2 group-balanced L2 | 0.888 | 0.721 | 0.178 | 0.212 | 0.6% | 0 | 0 |
| V2 sign-constrained L2 | 0.845 | 0.650 | 0.155 | 0.088 | 0% | 0 | 0 |
| grouped threshold tree | 0.638 | 0.451 | 0.256 | 0.256 | 100% | 19 | 26 |

Group-balanced weighting improves ranking but worsens calibration substantially. No model provides useful coverage at the original conservative thresholds.

## Monotonicity

- V1: 3 violations
- V2 compact ordinary L2: 0 violations
- V2 sign-constrained L2: 0 violations

The selected V2 artifact has nonnegative fitted coefficients for all five demand-to-budget features. This is a feature/model formulation improvement, not a production safety guarantee.

## V2 abstention and calibration

For the selected sign-constrained model:

| FIT / OOM thresholds | coverage | false-safe FIT | false-positive OOM |
|---|---:|---:|---:|
| .10 / .90 | 0.0% | 0 | 0 |
| .15 / .85 | 4.0% | 0 | 0 |
| .20 / .80 | 30.1% | 1 | 0 |
| .25 / .75 | 52.8% | 6 | 0 |
| .30 / .70 | 78.4% | 22 | 0 |
| .40 / .60 | 92.0% | 33 | 0 |
| .50 / .50 | 100.0% | 40 | 1 |

The first operating point with approximately 40% coverage has six false-safe confident-FIT OOMs. Thus the target of useful coverage with near-zero false-safe FIT decisions was not met.

ECE is 0.088 for V2 compact and 0.088 for the constrained model, an improvement over V1's 0.125 retrospective ECE but still campaign-conditioned and based on only 19 OOM groups. No population-calibration claim is made.

## Family and leave-family-out results

Selected V2 leave-family-out results include:

- attention: ROC-AUC 0.883, PR-AUC 0.780, Brier 0.170 (`N=111`)
- convolution: ROC-AUC 0.667, PR-AUC 0.696, Brier 0.376 (`N=16`)
- transformer: ROC-AUC 1.000, PR-AUC 0.500, Brier 0.132 (`N=9`)

Autodiff, MLP, training, scan, reduction, and matmul holdouts lack enough class variation for meaningful AUC interpretation. Cross-family generalization remains unresolved.

## Leakage and instability audit

- No runtime outcomes were present in the frozen candidate pool or selection plan.
- Each selected candidate used one fresh subprocess.
- Workload groups remained unique across the selection plan.
- Grouped folds kept each `workload_group_id` together.
- Compile failures were not merged into execution OOM.
- No mixed FIT/OOM group occurred in the new campaign.

## Speed and purity

- V2 features: 9
- Serialized V2 model: 1,068 bytes
- Measured inference: approximately 0.016 ms per 1,000 rows
- Candidate feature extraction: static JAX tracing only
- Target compilation during inference: NO
- Target execution during inference: NO
- Runtime allocator diagnostics during inference: NO

## Artifacts

```text
experiments/oom_risk_model_v1_frozen_2026-09-12.json
experiments/oom_risk_candidate_pool_v2_2026-09-12.json
experiments/oom_risk_expansion_plan_v2_2026-09-12.json
experiments/oom_risk_expansion_results_v2_2026-09-12.json
experiments/oom_risk_dataset_v2_2026-09-12.json
experiments/oom_risk_cv_v2_2026-09-12.json
experiments/oom_risk_models_v2_2026-09-12.json
experiments/oom_risk_model_v2_2026-09-12.json
experiments/oom_risk_summary_v2_2026-09-12.json
```

## Decision

**OOM RISK MODEL V2 PROMISING BUT DATA-LIMITED**

Prospective V1 ranking is encouraging, V2 removes the budget-monotonicity violations, and the compact model improves ECE. However, only 19 independent execution-OOM groups are available, conservative V2 coverage is zero, and the first approximately 40% coverage point has six false-safe FIT decisions. This is not validated for production or for population-calibrated probability use.

## Production

**NO PRODUCTION CHANGE.**

## Next technical decision

**second targeted dataset expansion** — only if further data collection is justified, with emphasis on independent execution-OOM groups and compile-versus-execution separation. Do not begin it automatically in this milestone.
