# Research status

## Current frontier

The first fully held-out prospective validation of frozen V2/V3 is complete. The panel was selected by a deterministic, model-independent family/scale design and executed in fresh subprocesses. No production risk gate is enabled.

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

## Failed or unresolved directions

- The held-out panel contains only five stable EXECUTION_OOM groups; family-specific confidence bounds remain wide.
- The random stratified panel produced no convolution execution OOM, so convolution safety is not strongly bounded despite 15 selected convolution groups.
- Calibration remains campaign-conditional; this panel is not a representative workload prevalence sample.
- V2 dominates V3 at the candidate `.20` and `.25` operating points on coverage with equal observed safety, but V3 avoids V2 false-safes at `.35` and `.40`.

## Next highest-value work

Second fully held-out validation batch, with additional independent execution-OOM cases and stronger family-specific safety bounds. Do not begin it automatically; do not retrain or integrate production behavior.
