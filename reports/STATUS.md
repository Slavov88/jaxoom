# Research status

## Current frontier

2026-11-08 BATCH 3 (COMPUTATIONALLY VERIFIED): 180 unseen recipes, predictions frozen before runtime; 125 V2 LIKELY_FIT, all FIT (zero false-safe); exact bound 2.37% alone, 1.06% pooled with the 157 earlier. Attention (16) and convolution (20) per-family bounds stay wide. Looser runner than the uniform qualification. See `research/BATCH3_CONFIRMATORY_RESULTS_2026-11-08.md`. Next: attention/convolution-focused panel, then the separate release review.

2026-11-08 (COMPUTATIONALLY VERIFIED): run 2 reproduced run 1 exactly (200/200 per-recipe outcomes identical, both audits pass). Batch-3 fresh panel (180 unseen recipes, 125 LIKELY_FIT) is frozen, with predictions committed before any runtime result; runtime pending. Draft protocol: `research/FRESH_PANEL_PROTOCOL_DRAFT_2026-11-08.md`.

2026-11-07 RESULT (COMPUTATIONALLY VERIFIED): the fresh uniform 600 s campaign completed 200/200 recipes, 18/18 pilots FIT, 157/157 LIKELY_FIT recipes FIT, zero false-safe OOMs, zero timeouts (176 FIT, 17 execution OOM, 7 compile OOM). Independent audits passed on Windows and Linux. Single run, same population as before; public API stays disabled. See `research/GPU_UNIFORM_QUALIFICATION_RESULTS_2026-11-07.md`. Next: reproduce once, then the separate release review.

2026-11-07 preflight (OBSERVED snapshot): the storage blocker has cleared (C: 13.98 GB free vs 512 MiB required), the RTX 3050 is idle, the frozen validation manifest still validates against current sources, and WSL starts. The 200-recipe campaign is still NOT_RUN, with zero GPU trials. Residual risk: the WSL swap file is on D:, which has about 3 MB free; the first WSL start failed on swap creation and a retry succeeded, so free D: space or move the swap before a 3-hour run. Release criteria were predeclared in `research/RELEASE_DECISION_CRITERIA_2026-11-07.md`. The new preflight lives outside the source-hashed set (`experiments/oom_risk_gpu_uniform_launch_preflight.py`).

The fresh uniform 600-second qualification runner and portable auditor are implemented and committed, with 102 new regression tests. Full Windows and Linux CPU suites each passed 549 tests, zero failures/errors/skips, with source-bound JUnit evidence. The new 200-recipe GPU campaign is NOT_RUN: launch preflight measured 475.6 MiB free on C:, below the frozen 512 MiB requirement. Public OOM-risk exports stay disabled. See `research/GPU_UNIFORM_QUALIFICATION_READINESS_2026-10-21.md`; storage/telemetry measurements are timestamped snapshots and need a fresh check before launch.

The five formerly unresolved recipes now have terminal results in a separately frozen 600-second diagnostic stage: 2 execution OOM, 2 compile OOM and 1 FIT. The original 180-second qualification remains incomplete; no historical timeout was relabeled and the public OOM-risk API stays disabled. New children independently measure source/plan/replay hashes before GPU work. Portable persisted audits passed on Windows and Linux, as did all 447 regression tests on each platform. See `research/GPU_TIMEOUT_DIAGNOSTICS_2026-10-21.md`.

Historical NVIDIA acquisition covered all 200 unique primary recipes: 175 FIT, 15 execution OOM, 5 compile OOM and 5 timeouts under its original bound. All 157 LIKELY_FIT recipes fit. One preflight-only telemetry failure remains separate infrastructure evidence (201 launches). These original records are preserved in full; the longer diagnostic results supplement them without upgrading the old qualification.

## Confirmed results

- COMPUTATIONALLY VERIFIED (2026-10-21 diagnostics): all five selected uncertain recipes returned terminal results within the fixed 600-second bound, with independently measured child source/plan/replay hashes. HOLDOUT-0012 execution OOM in 186.508 s; HOLDOUT-0028 compile OOM in 393.695 s; HOLDOUT-0029 execution OOM in 306.327 s; HOLDOUT-0031 compile OOM in 403.314 s; H2-0038 FIT in 333.414 s. No LIKELY_FIT recipe was rerun. These wall times are observations, not general latency guarantees.
- Both portable persisted audits passed: exact selection, historical artifact hash chains, source/plan/replay measurements, raw/result identity, phase-correct terminal outcomes and recovery telemetry. Earlier resume records retain their parent-declared wrapper limitation; the new diagnostic entry point fixes that measurement prospectively, without rewriting history.
- Full current regression suite: 447 Windows CPU and 447 WSL Linux CPU tests passed with zero failures/errors/skips. Fresh diagnostic GPU replay passed all 224 rows, public exports absent. Frozen V2 and all historical source/evidence bytes are unchanged; final GPU returned to the original idle state.
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

Free space on D: (WSL swap) or relocate the swap, rerun the launch preflight immediately before launch, then execute the fresh 200-recipe campaign under `research/GPU_UNIFORM_QUALIFICATION_PROTOCOL_2026-10-21.md`. The runner and auditor are ready; CPU tests do not substitute for fresh GPU replay/runtime coverage. Retain public API disablement even if restricted qualification passes, pending the separate release decision.

Keep the public API disabled and preserve the original qualification and separate diagnostics. All five diagnostic outcomes are now terminal, but the old 180-second gate remains incomplete. A fresh qualification/release decision must predeclare a suitable bound and use independently attested children across the intended exact scope; the selective longer diagnostics cannot retroactively qualify the historical campaign. H2-0038 FIT after the old timeout also confirms that allocation warnings are insufficient to classify terminal OOM. Generic callables, dynamic budgets, or AMD support still require separate validation. See `research/GPU_TIMEOUT_DIAGNOSTIC_PROTOCOL_2026-10-21.md` and the diagnostic results report.
