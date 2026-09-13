# Research status

## Current frontier

The second fully held-out prospective validation is complete. A production-integration design audit was performed, but no public risk API or production gate is enabled.

## Confirmed results

- New outcomes: 31 FIT, 23 EXECUTION_OOM, 5 COMPILE_OOM, 1 COMPILE_TIMEOUT.
- Combined corpus: 183 independent groups and 42 execution-OOM groups.
- Frozen V2 prospective FIT policy at `p_fit=0.20`: 46.3% coverage, zero false-safe OOMs on 54 new stable groups.
- V3 constrained grouped OOF: ROC-AUC 0.916, PR-AUC 0.787, ECE 0.115.
- V3 budget monotonicity violations: 0.
- Compile-failure waste declined from 26/60 to 6/60.
- Production decision: **NO PRODUCTION CHANGE**.
- Held-out panel: 90 groups; 83 stable FIT/EXECUTION_OOM rows (78 FIT, 5 EXECUTION_OOM).
- Frozen V2 at `p_fit=0.20`: 70/83 LIKELY_FIT, 84.3% coverage, zero false-safe; exact one-sided 95% upper bound 4.2%.
- Frozen V3 at `p_fit=0.20`: 67/83 LIKELY_FIT, 80.7% coverage, zero false-safe; exact one-sided 95% upper bound 4.4%.
- Held-out compile failures: 3 COMPILE_OOM and 4 COMPILE_TIMEOUT; none were LIKELY_FIT at `p_fit=0.20`.
- V2 and V3 held-out budget monotonicity violations: 0.
- Batch 2 primary: 110 groups; 107 eligible (97 FIT, 10 EXECUTION_OOM).
- Batch 2 V2 at `p_fit=0.20`: 87/107 LIKELY_FIT, 81.3% coverage, zero false-safe; exact upper bound 3.4%.
- Pooled primary batches: 190 eligible, 157 V2 LIKELY_FIT, zero false-safe; exact upper bound 1.9%.
- Supplemental convolution stress: 24 groups, 20 FIT, 4 COMPILE_TIMEOUT; kept outside the pooled bound.
- Primary environments matched on JAX 0.11.0, jaxlib 0.11.0, GPU backend, and allocator settings.

## Failed or unresolved directions

- Family-specific bounds remain wide, especially attention and convolution; no family-specific false-safe was observed at `p_fit=0.20`.
- Strict near-duplicate clustering reduces the pooled V2 LIKELY_FIT count to 126 and raises the upper bound to approximately 2.35%.
- Calibration remains campaign-conditional; these panels are not representative workload-prevalence samples.
- V2 is the primary confirmatory model and has higher coverage than V3 at `p_fit=0.20` in batch 2; both have zero primary false-safes.
- Production replay covered 224 held-out rows: zero logistic score mismatches and zero threshold-status mismatches.
- Feature replay found 224 `largest_over_budget` schema/source mismatches; the current coefficient is zero, so scores still match, but exact feature reproducibility fails.
- `config_numeric_count`, `config_numeric_max`, `config_numeric_log_product`, and `dtype_bytes` lack canonical definitions for arbitrary public callables.

## Next highest-value work

Resolve the production-compatible feature contract: either define a validated structured workload adapter or redevelop a model using only generic static features. Do not export a public API until this blocker is resolved.
