# ATT-2 False-Safe Analysis v1

## STATUS

**ROOT_CAUSE_IDENTIFIED_NO_FIX.** ATT-2 is reproducibly safe according to the planner's explicit 3 GiB compiler-calibrated interval, but fails during first GPU execution. The immediate mechanism is a large attention-specific XLA temporary allocation after compilation, combined with JAX allocator-pool residency/capacity. No planner constants were changed.

This report preserves the original baseline report. The investigation records are under `experiments/att2_false_safe/`.

## ATT-2 REPRODUCTION

Five fresh-process repetitions were run with the recorded ATT-2 shape:

- batch: 1
- Q/K/V: `(1, 8, 5120, 64)` float32
- explicit budget: 3,221,225,472 bytes (3 GiB)
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

Result: **5/5 OOM**, all during `FIRST_EXECUTION` after successful compilation. The exception was consistently `JaxRuntimeError: RESOURCE_EXHAUSTED`, requesting `1.56 GiB`.

## FAILURE PHASE

The probe synchronously blocked inputs, compiled the lowered executable, captured compiler memory analysis, executed once, and synchronously blocked the result. ATT-2 therefore fails at:

**FIRST EXECUTION**, not tracing or compilation.

## ATT-1 VS ATT-2

| Quantity | ATT-1 | ATT-2 |
|---|---:|---:|
| Sequence length | 4,096 | 5,120 |
| Heads | 8 | 8 |
| Head dimension | 64 | 64 |
| Input shape | `(1,8,4096,64)` | `(1,8,5120,64)` |
| Static peak | 1,082,261,504 | 1,698,856,960 |
| Calibrated upper | 1,804,833,959 | 2,833,099,692 |
| Compiler temporary | 1,073,741,840 | 1,677,721,600 |
| Compiler accounted | 1,107,296,272 | 1,719,664,640 |
| Five fresh outcomes | 1 FIT, 4 OOM | 5 OOM |

ATT-1 is allocator-sensitive under this repeated control: it occasionally fits when its approximately 1 GiB temporary can use the largest available allocator block. ATT-2 needs approximately 1.56 GiB, which is larger than the observed 1 GiB largest free block after compilation.

The JAXPR is identical across CPU and CUDA backends: 15 equations and SHA-256 `88dd35cc...970e93`. The difference is backend/runtime memory behavior, not tracing structure.

## PLANNER MEMORY BREAKDOWN

For ATT-2 batch 1:

- Planner explicit budget: **3,221,225,472 bytes**
- Recommendation: **1**
- Structural peak: **1,698,856,960 bytes**
- Calibrated central: **1,698,856,960 bytes**
- Calibrated upper: **2,833,099,692 bytes**
- Planner risk: **LOW**
- Inputs: 3 × 10,485,760 bytes = 31,457,280 bytes
- Compiler outputs: 10,485,760 bytes
- Compiler temporary: 1,677,721,600 bytes
- Compiler accounted total: 1,719,664,640 bytes
- Alias bytes: 0

The static peak consists primarily of two simultaneously live score-shaped buffers plus Q/K inputs and reduction state. Each score buffer has shape `(1, 8, 5120, 5120)` and size **838,860,800 bytes**. Two such buffers total **1,677,721,600 bytes = 1.5625 GiB**.

## COMPILER MEMORY ANALYSIS

`Compiled.memory_analysis()` succeeds for all five ATT-2 compilations:

```
arguments:  31,457,280 bytes
outputs:    10,485,760 bytes
temporaries: 1,677,721,600 bytes
aliases:    0
accounted:  1,719,664,640 bytes
```

The compiler estimate is close to the static estimate, not substantially larger. This rejects a simple “static accounting forgot the attention score tensor” explanation.

The CPU control compiled and executed successfully. Its compiler analysis reported:

```
arguments:  31,457,280 bytes
outputs:    10,485,760 bytes
temporaries: 0 bytes
accounted:  41,943,040 bytes
```

This confirms a CUDA/XLA backend-specific temporary allocation rather than a JAXPR-only memory requirement.

## OOM ALLOCATION ANALYSIS

The observed requested allocation is approximately **1,675,037,245 bytes** from the textual `1.56 GiB` error. The compiler temporary is **1,677,721,600 bytes**, exactly two score-buffer sizes. This is strong evidence that the failed request is attention score/softmax staging or an equivalent compiler temporary with the same byte size.

The runtime state immediately after compilation was:

| Quantity | Value |
|---|---:|
| Driver free VRAM before compile | 3,935,305,728 |
| Driver free VRAM after compile | 1,710,227,456 |
| JAX pool bytes | 2,283,798,528 |
| JAX allocator limit | 3,221,225,472 |
| Largest allocator free block | 1,073,741,824 |
| Requested allocation | ~1,675,037,245 |
| Request minus largest block | ~601,295,421 |

The allocator has approximately 937 MiB of headroom relative to its configured limit, and its largest free block is 1 GiB. A 1.56 GiB request cannot be satisfied even though driver-level free VRAM is about 1.71 GiB. The planner's explicit budget does not model this post-compilation allocator state.

## DIMENSION SWEEP

A one-axis sequence sweep at batch 1, heads 8, head dimension 64 produced:

| Sequence | Structural | Calibrated upper | Compiler temp | Actual |
|---:|---:|---:|---:|---|
| 2,048 | 276,889,600 | 461,755,085 | 268,435,456 | FIT |
| 3,072 | 616,660,992 | 1,028,375,024 | 603,979,776 | FIT |
| 4,096 | 1,090,650,112 | 1,818,823,226 | 1,073,741,824 | FIT |
| 4,608 | 1,377,976,320 | 2,297,982,926 | 1,358,954,512 | OOM |
| 5,120 | 1,698,856,960 | 2,833,099,692 | 1,677,721,600 | OOM |

The reduced counterexample is therefore approximately sequence 4,608: its calibrated upper prediction remains below 3 GiB, but execution fails. The transition is attention-score scaling, not batch scaling.

## ALLOCATOR CONTROLS

- Baseline `XLA_PYTHON_CLIENT_PREALLOCATE=false`: 5/5 ATT-2 OOM at first execution.
- `XLA_CLIENT_MEM_FRACTION=0.8`, preallocation false: 3/3 OOM at first execution with the same 1.56 GiB request.
- `XLA_PYTHON_CLIENT_PREALLOCATE=true`: the WSL2 service terminated before usable probe records were produced. No FIT/OOM conclusion is assigned.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`: the WSL2 service terminated before usable probe records were produced. No FIT/OOM conclusion is assigned.

The completed fraction control supports allocator sensitivity but does not rescue ATT-2. The two WSL failures are infrastructure results, not workload outcomes.

## FREE-VRAM CONTROLS

An exact ATT-2 external-holder control was attempted, but the WSL2 service repeatedly terminated before producing a usable record. It is therefore **NOT CHECKED** rather than inferred. Baseline ATT-2 probes began with approximately 3.94 GiB driver-free VRAM.

## EXPLICIT-BUDGET THRESHOLD

For the fixed ATT-2 shape, the planner returned no feasible batch through 2.5 GiB. It began recommending batch 1 at 2.75 GiB, because the calibrated upper estimate is 2,833,099,692 bytes. It continued recommending batch 1 at 3–4 GiB.

This threshold is an estimator/calibration threshold, not an empirical execution threshold: ATT-2 still failed at 3 GiB.

## ROOT CAUSE

The evidence supports a combined mechanism:

1. **Attention-specific scaling:** the failed allocation is exactly two `(1,8,5120,5120)` float32 score buffers. The sequence sweep crosses from FIT to OOM between 4,096 and 4,608.
2. **CUDA/XLA temporary allocation:** GPU compiler analysis reports a 1.6 GiB temporary; CPU reports zero temporary bytes for the same JAXPR.
3. **Allocator/runtime capacity:** after compilation, the JAX pool occupies 2.284 GiB and the largest free block is only 1 GiB. The first execution requests 1.56 GiB.

This is not primarily an incorrect static tensor-size calculation. The planner accounts for the large score-shaped live buffers and its GPU compiler interval is close to compiler accounting. The missing safety context is post-compilation allocator residency and usable contiguous capacity.

## REQUIRED CORRECTION MAGNITUDE

No global correction factor is installed.

Useful diagnostics are:

- `requested allocation - largest free block`: approximately **601 MiB**;
- `JAX allocator limit - post-compile pool`: approximately **894 MiB**;
- naive `(post-compile pool + requested allocation) / static peak`: approximately **2.33×**.

The last ratio is not a proposed calibration factor: it combines runtime residency and a temporary allocation and would be unsafe to generalize.

## PLANNER CHANGE

**None.** No global safety margin, attention special case, or calibration change was added.

## ORIGINAL 8 REVALIDATION

Not rerun. No planner code changed, so the original V1 results remain the immutable baseline. ATT-2 remains false-safe in that baseline.

## HELD-OUT ATTENTION VALIDATION

Not applicable. No correction was implemented and no new workload campaign was started.

## FALSE-SAFE RATE

Baseline V1 remains **1/8 = 12.5%** false-safe. This investigation does not claim improvement.

## CONSERVATISM

Not changed. No planner recommendations were altered.

## EXPLICIT-BUDGET MONOTONICITY

The V1 result remains **PASS**. No planner code changed. The ATT-2 budget curve itself is non-decreasing in the expected sense: no batch through 2.5 GiB, batch 1 from 2.75 GiB upward.

## AUTO-DEVICE PRESSURE

The V1 real-pressure result remains **PASS**. No planner code changed. A new pressure run was not required for this root-cause investigation.

## TESTS

- Full repository suite: **77 passed** under CUDA JAX 0.11.0.
- Investigation scripts compile successfully.
- GPU OOM probes remain opt-in and isolated from ordinary CI.

## COMMITS

- `4bee85f` — previous V1 boundary validation baseline
- Investigation changes: pending commit

## FILES / RECORDS

- `reports/ATT2_FALSE_SAFE_ANALYSIS_V1.md`
- `reports/att2_false_safe_analysis_v1.json`
- `experiments/att2_false_safe/investigate.py`
- `experiments/att2_false_safe/probe_repetitions.py`
- `experiments/att2_false_safe/dimension_sweep.py`
- `experiments/att2_false_safe/budget_curve.py`
- `experiments/att2_false_safe/att2_reproduction_v2.json`
- `experiments/att2_false_safe/att1_control_v3.json`
- `experiments/att2_false_safe/att2_cpu_control_v1.json`
- `experiments/att2_false_safe/attention_sequence_sweep_v1.json`
- `experiments/att2_false_safe/att2_explicit_budget_curve_v1.json`
- `experiments/att2_false_safe/att2_xla_client_fraction_08_v1.json`
- `experiments/att2_false_safe/att2_jaxpr_gpu_v1.json`
- `experiments/att2_false_safe/att2_jaxpr_cpu_v1.json`

## LIMITATIONS

1. Exact external free-VRAM sensitivity for ATT-2 was not completed because the WSL2 service terminated during holder controls.
2. Preallocation and Python memory-fraction controls likewise produced infrastructure failures rather than usable workload outcomes.
3. ATT-1 was allocator-sensitive across five repetitions, so its original exact-boundary label should be interpreted as a single-run baseline result, not deterministic runtime behavior.
4. No planner correction was attempted and no V2 safety validation was run.

## FINAL VERDICT

**A. ROOT_CAUSE_IDENTIFIED_NO_FIX.** ATT-2's false-safe is explained sufficiently to block global retuning: GPU XLA allocates an attention-specific temporary matching two score buffers, while post-compilation JAX allocator residency leaves only a 1 GiB largest free block. The explicit 3 GiB planning budget is not equivalent to usable post-compilation capacity.

## NEXT STEP

If product work is authorized, design a separate, mechanistic runtime-capacity correction milestone. It should model compiler temporary/residency or classify this calibration domain as unsupported before changing constants. Any fix must first be tested on ATT-2 and ATT-1, then on held-out attention workloads, while preserving the original monotonicity and auto-pressure checks.
