# Research status

## Current frontier

NVIDIA runtime acquisition reached all 200 unique primary recipes on 2026-10-21 across append-only, separately frozen continuations. Outcomes: 175 FIT, 15 execution OOM, 5 compile OOM, 1 execution timeout and 4 compile timeouts. All 157 LIKELY_FIT recipes fit; zero recipes remain unrun. Five uncertain outcomes remain unresolved, so full qualification has NOT passed and the public OOM-risk API stays disabled. One additional launch failed on telemetry before workload initialization and remains separate infrastructure evidence (201 total launches). Fresh GPU replay passed all 224 rows; Windows and Linux each passed 383 regression tests. See `research/GPU_RUNTIME_ACQUISITION_2026-10-21.md`.

## Confirmed results

- COMPUTATIONALLY VERIFIED (2026-10-21 completion): unique runtime acquisition 200/200; 157/157 LIKELY_FIT recipes FIT, zero false-safe execution OOMs on this repeated historical population. Five timeouts remain unknown, not terminal FIT/OOM evidence. All earlier stopped-run artifacts and original frozen V2 bytes are preserved.
- Telemetry recovery is bounded and fail-closed: only explicit NVML numeric `[Unknown Error]` samples may be replaced by at most three entirely new samples; idle guards are unchanged. Proven preflight-only HOLDOUT-0084 infrastructure failure was retained and its one workload retry FIT. No prior workload timeout was retried. The source-hashed wrapper is parent-declared per trial, with final source bytes checked; independent per-child wrapper measurement remains a provenance limitation.
- Full current regression suite: 383 Windows CPU and 383 WSL Linux CPU tests passed, zero failures/errors/skips. Fresh GPU replay passed all 224 rows in each new campaign. Final GPU returned to the required idle state.
- COMPUTATIONALLY VERIFIED (2026-10-21): exact feature equality and zero score/decision mismatches on 224 fresh GPU traces; 18/18 pilot FIT. Audited prefix: 25/25 LIKELY_FIT trials FIT, no false-safe OOMs; not a new independent accuracy panel.
- Qualification runner regression controls cover malformed logs/timeouts, child identity and model/policy/feature provenance, CUDA identity, and conservative stop handling. Windows CPU suite: 367 passed; restored Linux CPU suite: 367 passed, zero failures/errors/skips, with explicit JUnit evidence retained.
- COMPUTATIONALLY VERIFIED (2026-10-20): 224 fresh abstract traces; exact feature equality; zero frozen-V2 score or threshold-decision mismatches at 1e-12 tolerance; no concrete target execution or attempted public compilation calls.
- JAX/jaxlib 0.11.0 CPU regression suite: 141 tests passed (39 new restricted-adapter checks). Final replay evidence: `experiments/oom_risk_workload_replay_2026-10-20_final.json`.
- COMPUTATIONALLY VERIFIED (2026-10-20 applicability work): full CPU-only suite now passes 287 tests, including 139 applicability and 7 preflight controls. Exact primary-recipe allowlist, strict environment/freshness guards, and replay-evidence integrity fail closed.
- Frozen V2 artifact bytes, coefficients, normalization, threshold, and historical experiment evidence remain unchanged. `largest_over_budget` is explicitly the legacy zero feature.
- Immutable, versioned WorkloadSpec inputs bind template generation and metadata together; unknown fields, unsupported dtypes/callables, and changed templates/model bytes fail closed.
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
- The September design audit found 224 `largest_over_budget` schema/source mismatches. That historical result is preserved; the new restricted adapter instead reproduces the actual legacy zero feature and passes fresh replay.
- `config_numeric_count`, `config_numeric_max`, `config_numeric_log_product`, and `dtype_bytes` now have explicit restricted-spec semantics, but still lack canonical definitions for arbitrary public callables.
- The historical 3 GiB value was a feature-normalization reference, not an enforced runtime allocator cap. Dynamic budgets and a 3 GiB fit claim remain unsupported.
- CPU tracing/replay passes do not qualify GPU runtime applicability or every specification permitted by the schema. Legacy scan depth remains metadata-only to preserve V2.
- Actual integrated Radeon probe found no JAX ROCm backend; WSL exposes no ROCm device nodes/runtime. Runtime qualification is NOT_RUN with zero trials. This is an unsupported-environment blocker, not evidence about model accuracy. NVIDIA was left untouched after the user's Radeon selection.
- The preceding Radeon result remains historical. On 2026-10-21 the user authorized the idle NVIDIA GPU; its qualification STOPPED at HOLDOUT-0012 (attention float16, batch 2, heads 16, sequence 4608, head_dim 96; score 0.7052589449165493). Execution emitted allocation warnings but no terminal result by 180 s. Outcome remains UNKNOWN FIT/OOM; 169 recipes unrun.

## Next highest-value work

Keep the public API disabled and preserve all stopped NVIDIA evidence. Runtime acquisition is complete, but five uncertain timeout outcomes still block qualification. Any diagnostic resolution needs its own frozen bounded protocol; allocation warnings alone cannot classify terminal OOM. Harden the telemetry wrapper with independent per-child source measurement before future qualification claims. Generic callables, dynamic budgets, or AMD support still require separate validation. See the continuation protocols and `research/GPU_RUNTIME_ACQUISITION_2026-10-21.md`.
