# Research status

## Current frontier

The V2 experiment-only OOM-risk expansion is complete. A frozen V1 model selected 60 genuinely new static workload groups; 60 fresh-process GPU runs produced 22 FIT, 12 EXECUTION_OOM, 12 COMPILE_OOM, and 14 COMPILE_TIMEOUT outcomes.

## Confirmed results

- Frozen-V1 prospective performance on 34 new stable binary groups: ROC-AUC 0.973, PR-AUC 0.867, Brier 0.110.
- Expanded dataset: 123 total groups, 90 stable binary groups, 19 execution-OOM groups.
- V2 compact/sign-constrained L2 grouped ROC-AUC 0.845, PR-AUC 0.650, ECE 0.088.
- V2 compact and sign-constrained formulations have zero budget-monotonicity violations.
- Production decision: **NO PRODUCTION CHANGE**.

## Failed or unresolved directions

- V2 conservative `.1/.9` abstention has zero coverage.
- At approximately 40% coverage, V2 has six false-safe confident-FIT OOMs.
- The target of 25–40 independent execution-OOM groups was not reached; only 19 exist.
- Family holdouts outside attention/convolution remain underpowered.
- The campaign generated many compile failures/timeouts among high-risk candidates; compile and execution mechanisms remain separate.

## Next highest-value work

Second targeted dataset expansion is the next technical option, but should not begin automatically. If authorized, prioritize independent execution-OOM groups in plausible boundary regimes, not allocator-capacity sweeps or repeated copies of existing graphs. Keep compile-OOM and execution-OOM labels separate.
