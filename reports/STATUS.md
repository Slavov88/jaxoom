# Research status

## Current frontier

V3 low-risk/boundary enrichment is complete. A static compile-feasibility filter selected 60 fresh-process GPU workloads from 3,709 candidates.

## Confirmed results

- New outcomes: 31 FIT, 23 EXECUTION_OOM, 5 COMPILE_OOM, 1 COMPILE_TIMEOUT.
- Combined corpus: 183 independent groups and 42 execution-OOM groups.
- Frozen V2 prospective FIT policy at `p_fit=0.20`: 46.3% coverage, zero false-safe OOMs on 54 new stable groups.
- V3 constrained grouped OOF: ROC-AUC 0.916, PR-AUC 0.787, ECE 0.115.
- V3 budget monotonicity violations: 0.
- Compile-failure waste declined from 26/60 to 6/60.
- Production decision: **NO PRODUCTION CHANGE**.

## Failed or unresolved directions

- V3 grouped OOF still has one false-safe at approximately 40% FIT coverage.
- Population calibration is not identified under active sampling.
- Convolution and several family holdouts remain underpowered.
- Compile filter is experiment-only and has not been independently validated beyond this campaign.

## Next highest-value work

Fully held-out OOM-risk validation using the frozen V3 artifact. Do not retrain or expand the model before that evaluation, and do not integrate production behavior.
