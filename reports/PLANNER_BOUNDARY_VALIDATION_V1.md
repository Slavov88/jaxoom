# Planner Boundary Validation v1

## STATUS

**PARTIAL.** Eight real planner recommendations were validated on CUDA across four workload families. Seven recommendations fit; one was **FALSE_SAFE**. The false-safe case prevents a positive safety claim and must be investigated before planner tuning.

## ENVIRONMENT

- Planner commit: `135a646`
- Validation harness commit: `3035876b1e914355789b4ee9d67674497861ed2b`
- OS: WSL2 Linux 6.6.87.2 on Windows
- Python: 3.12.3
- JAX / jaxlib: 0.11.0 / 0.11.0
- Backend: CUDA
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB
- NVIDIA driver: 566.07; reported CUDA capability: 12.7
- Allocator setting: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- Inputs: deterministic `jnp.ones`, float32, abstract `jax.ShapeDtypeStruct` during planning
- Validation: one fresh subprocess per batch probe

Planning used no target compilation or target execution. Compilation and execution occurred only in the explicit validation subprocesses.

## WORKLOADS

| ID | Family | Key dimensions | Mode | dtype |
|---|---|---|---|---|
| MLP-1 | MLP | input 256, hidden 768, depth 3 | inference | f32 |
| MLP-2 | MLP | input 384, hidden 1024, depth 2 | inference | f32 |
| TRAIN-1 | training | input 256, hidden 768, output 256 | value-and-grad | f32 |
| TRAIN-2 | training | input/output 512, one dense layer | value-and-grad | f32 |
| ATT-1 | attention | sequence 4096, heads 8, head dimension 64 | QK-softmax-V forward | f32 |
| ATT-2 | attention | sequence 5120, heads 8, head dimension 64 | QK-softmax-V forward | f32 |
| CONV-1 | convolution | 64x64, 3 channels, 16/32 filters, two stages | forward | f32 |
| CONV-2 | convolution | 96x96, 3 channels, 16/32/16 filters, three stages | forward | f32 |

The training functions use `jax.value_and_grad`; they are not forward-only proxies.
The attention functions materialize score matrices and softmax weights. The convolution functions use `jax.lax.conv_general_dilated`.

## BOUNDARY RESULTS

A boundary record always probes the recommendation `B` and the adjacent larger batch `B+1`. An empirical maximum is exact only when an OOM bracket was found. For conservative rows, `>=` denotes the highest tested fitting batch, not the physical maximum.

| ID | Family | Budget Mode | Planned Batch | Fits? | Next/Riskier Batch | Outcome | Empirical Max | Classification |
| -- | ------ | ----------- | ------------: | ----- | -----------------: | ------- | ------------: | -------------- |
| MLP-1 | MLP | explicit 128 MiB | 12,587 | YES | 12,588 | FIT | >=12,588 | CONSERVATIVE_SAFE |
| MLP-2 | MLP | explicit 128 MiB | 9,632 | YES | 9,633 | FIT | >=9,633 | CONSERVATIVE_SAFE |
| TRAIN-1 | training | explicit 128 MiB | 5,986 | YES | 5,987 | FIT | >=5,987 | CONSERVATIVE_SAFE |
| TRAIN-2 | training | explicit 128 MiB | 6,549 | YES | 6,550 | FIT | >=6,550 | CONSERVATIVE_SAFE |
| ATT-1 | attention | explicit 3 GiB | 1 | YES | 2 | OOM | 1 | EXACT_BOUNDARY |
| ATT-2 | attention | explicit 3 GiB | 1 | **NO** | 2 | OOM | -- | **INVALID/INCONCLUSIVE; FALSE_SAFE** |
| CONV-1 | convolution | explicit 512 MiB | 409 | YES | 410 | FIT | >=410 | CONSERVATIVE_SAFE |
| CONV-2 | convolution | explicit 512 MiB | 272 | YES | 273 | FIT | >=273 | CONSERVATIVE_SAFE |

## SAFETY

- Total recommendations: **8**
- Recommendations that fit: **7/8**
- Exact boundaries: **1**
- Conservative-safe boundaries: **6**
- Inconclusive/invalid cases: **1**
- False-safe count: **1**
- False-safe rate: **12.5%**

The false-safe is ATT-2: the planner recommended batch 1, but isolated execution failed with `RESOURCE_EXHAUSTED` while allocating approximately 1.56 GiB. Batch 2 also failed. The planner's structural upper estimate for batch 1 was below the 3 GiB explicit budget, so this is a genuine planner-underestimation failure, not merely a rejected-risk candidate.

ATT-1 produced a useful exact boundary: batch 1 FIT and batch 2 failed during execution with an approximately 2 GiB allocation request.

## FALSE-SAFE CASES

| ID | Planned batch | Planner upper estimate | Budget | Actual result | Failure |
|---|---:|---:|---:|---|---|
| ATT-2 | 1 | 2,833,099,692 bytes | 3,221,225,472 bytes | OOM | JAX `RESOURCE_EXHAUSTED` during execution |

This case is preserved in `experiments/validation_ATT-2_2026-09-09_v2.json`. No safety margin or planner constant was changed after observing it.

## CONSERVATISM

Among the seven recommendations with a tested fitting batch and a ratio, the planned/highest-tested-fit ratios were:

- median: **0.99985**
- minimum: **0.99634**

These are **lower-bound utilization ratios**, because six rows stopped after `B+1` for runtime control. They must not be interpreted as utilization of the true empirical maximum.

## MLP

Both MLP recommendations fit. Their adjacent batches also fit, so both are conservative-safe at the tested resolution. Larger upward searches were intentionally not repeated after the attention false-safe was found.

## TRAINING

Both workloads execute `jax.value_and_grad` over actual dense losses. Planned batches 5,986 and 6,549 fit, as did their adjacent batches. These are conservative-safe at the tested resolution.

## ATTENTION

ATT-1 is an exact measured boundary at the chosen explicit budget: `1 FIT -> 2 OOM`.

ATT-2 is a false-safe: `1 OOM -> 2 OOM`. This demonstrates that structural/calibrated planning can underestimate runtime workspace or allocator requirements for large attention score matrices. Attention safety is therefore not validated generally.

## CONVOLUTION

Both multi-stage convolution workloads fit at their recommendations and adjacent batches. They are conservative-safe at the tested resolution. No exact physical maximum was pursued because individual convolution probes were expensive and the false-safe gate had already failed.

## EXPLICIT-BUDGET MONOTONICITY

**PASS (observed and computationally verified).** Each of the eight workloads was planned at three increasing explicit budgets. Recommendations were non-decreasing for every workload, including the cases that returned no fit at the smallest budget.

| ID | Recommendations at increasing budgets | Monotonic |
|---|---|---|
| MLP-1 | 1,125 -> 6,037 -> 12,587 | PASS |
| MLP-2 | 1,036 -> 4,720 -> 9,632 | PASS |
| TRAIN-1 | 696 -> 2,963 -> 5,986 | PASS |
| TRAIN-2 | 818 -> 3,274 -> 6,549 | PASS |
| ATT-1 | 1 -> 1 -> 2 | PASS |
| ATT-2 | none -> 1 -> 1 | PASS |
| CONV-1 | 102 -> 204 -> 409 | PASS |
| CONV-2 | 68 -> 136 -> 272 | PASS |

Raw record: `experiments/planner_explicit_monotonicity_v1_2026-09-09.json`.

## AUTO-DEVICE PRESSURE

**PASS (planning monotonicity).** Pressure was held in a fresh process for each state. The same MLP-1 workload used `memory_limit="auto"`; greater real GPU occupancy produced a smaller recommendation.

| Pressure | Observed Free VRAM | Frozen Auto Budget | Recommended Batch |
| -------- | -----------------: | -----------------: | ----------------: |
| baseline | 4,008,706,048 | 3,006,477,107 | 292,916 |
| +1 GiB | 2,928,672,768 | 2,713,924,403 | 264,363 |
| +2 GiB | 1,854,930,944 | 1,640,182,579 | 159,567 |

The pressure allocation was a real device allocation, not a smaller explicit budget. Each planning call used one frozen device-budget snapshot. The experiment validated the recommendation response but did not execute each auto recommendation.

Raw record: `experiments/auto_pressure_MLP-1_2026-09-09_v5.json`.

## PLANNER CHANGES

**None.** The planner and calibration model were not redesigned or tuned. Changes are limited to an empirical harness, fresh-process probing, result serialization, and CPU/CI-safe unit tests. The existing one-snapshot auto-budget behavior was preserved.

## TESTS

- Boundary harness unit tests: **13 passed** (`tests/test_batch_planner.py` and `tests/test_planner_boundary_validation.py`)
- Full repository suite: **77 passed** with `pytest -q` in the CUDA JAX 0.11.0 environment.
- Destructive GPU OOM probes are opt-in experiment commands and are not ordinary CI tests.

## COMMITS

- `9f61cd6` — add isolated planner boundary harness and pre-run plan
- `c2ca450` — always probe the adjacent batch and add harness tests
- `3035876` — isolate auto-pressure snapshots in fresh processes

## FILES / RECORDS

- `experiments/PLANNER_BOUNDARY_PLAN_V1.md`
- `experiments/planner_boundary_validation_v1.py`
- `experiments/pilot_MLP-1_2026-09-09.json`
- `experiments/validation_MLP-2_2026-09-09.json`
- `experiments/validation_TRAIN-1_2026-09-09.json`
- `experiments/validation_TRAIN-2_2026-09-09.json`
- `experiments/validation_ATT-1_2026-09-09_v2.json`
- `experiments/validation_ATT-2_2026-09-09_v2.json`
- `experiments/validation_CONV-1_2026-09-09.json`
- `experiments/validation_CONV-2_2026-09-09.json`
- `experiments/planner_explicit_monotonicity_v1_2026-09-09.json`
- `experiments/auto_pressure_MLP-1_2026-09-09_v5.json`
- `reports/planner_boundaries_v1.json`

## LIMITATIONS

1. The acceptance gate failed because ATT-2 was false-safe.
2. Six conservative cases only tested `B+1`; their reported maxima are lower bounds.
3. Only one GPU and one JAX/jaxlib version were tested.
4. Runtime device snapshots include allocator and driver observations; precise execution-only peak memory is not claimed.
5. Auto-pressure recommendations were not separately compiled/executed.
6. A pre-existing `batch_planner_runtime_gap_validation.py` experiment was not incorporated into this report because its workload matrix and checkpoint were separate from this controlled eight-row matrix.

## FINAL VERDICT

**PARTIAL — NOT VALIDATED FOR SAFETY.** The current planner demonstrates useful behavior on seven of eight tested recommendations, preserves explicit-budget monotonicity, and responds monotonically to real device pressure. However, one large attention workload is a verified false-safe. The planner must not receive a broad “recommended batches fit” claim until ATT-2's runtime-memory underestimation is understood and the same matrix is rerun.

## NEXT STEP

Investigate ATT-2's runtime workspace/allocator discrepancy using the preserved raw probe, without changing planner constants first. Then make one targeted correction only if justified, add ATT-2 plus held-out attention workloads to regression coverage, rerun the exact baseline matrix, and keep the status PARTIAL unless false-safe recommendations are eliminated without breaking monotonicity.
