# Empirical planner-boundary validation plan v1

Status before execution: **CONJECTURED**.

## Hypothesis

For fixed JAX workloads and explicit memory budgets, the batch-size planner's recommended batch is executable, and its first rejected/riskier candidate is a useful conservative boundary. Under `memory_limit="auto"`, occupying more GPU memory should not increase the recommendation.

## Strongest baseline

The current planner at commit `135a646`, without planner changes. Existing calibration remains unchanged. The prior 2026-09-09 validation only tested recommended batches and hit configured upper bounds, so it is not treated as boundary evidence.

## Smallest discriminating pilot

Run one workload from each family through the current planner with a larger search range, then separately compile/execute the recommended batch and a larger/riskier batch in fresh subprocesses. A recommendation that fails is a false-safe and stops any positive validation claim.

## Main matrix

At least two distinct workloads each for MLP, training/autodiff, attention, and convolution. Each row records explicit budget, planned batch, next/riskier probe, empirical upward bracket, execution outcome, device snapshots, and provenance. Real CUDA execution is separate from pre-compilation planning.

## Primary metrics

- false-safe: planner recommends a batch as fit, but its isolated compile/execute probe fails;
- exact boundary: recommended batch fits and next practical tested batch fails;
- conservative-safe: recommended and larger tested batch fit;
- utilization: planned batch / empirical maximum safe batch;
- explicit-budget monotonicity: recommendations are non-decreasing with budget;
- auto-pressure monotonicity: recommendations are non-increasing as occupied VRAM increases.

## Kill criteria

Stop positive claims if any recommended batch is false-safe. Preserve and inspect every such case. Do not tune constants before archiving the baseline. If the GPU/JAX environment is unavailable, report the milestone as PARTIAL rather than substituting estimator-only evidence.

## Resource controls

- CUDA JAX 0.11.0 in the existing WSL2 environment;
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` held fixed;
- one fresh subprocess per actual probe;
- fixed deterministic inputs and f32 unless a workload explicitly records otherwise;
- binary/upward probing rather than exhaustive batch scans;
- raw JSON output retained with environment and commit metadata.
