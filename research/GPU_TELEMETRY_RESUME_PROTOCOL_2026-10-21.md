# Telemetry-only recovery protocol, frozen before resume

Completion1 stopped after 50 launches when HOLDOUT-0084 returned an int parsing
error for NVML `[Unknown Error]` during preflight. Raw stdout has one result
event, phase preflight, gpu_runtime_executed=false, and no phase events; the
workload was never initialized. Preserve the rejected parent result and audit
it against raw child identity, exact error, and absence of target execution.
This is an infrastructure failure, not a false-safe workload OOM. The launch
counts in history; it cannot qualify a recipe. For acquisition only, permit this
single proven preflight-only recipe to run once alongside the 104 never launched
recipes. Never retry any prior workload runtime, compile, or initialization.

Use an explicitly source-hashed child entry point that invokes the unchanged
original q.run_one with only the GPU telemetry function wrapped. A numeric
`[Unknown Error]` ValueError allows up to three entirely new nvidia-smi samples,
one second apart. Never substitute zero, ignore missing fields, reuse an older
sample, relax idle thresholds, or retry unrelated errors. Valid samples receive
their own fresh timestamp and must pass the original policy immediately before
CUDA initialization. All source/model/policy/replay guards remain in force;
original q.load_plan verifies original runner bytes. Record wrapper source hash
per trial and telemetry retry events in stdout. Retry caps are within the
unchanged 180-second child deadline. Unknown compile/execute outcomes remain
unknown; all unknowns still block qualification. Each recipe gets at most one
new launch. Fresh CUDA replay, negative-control tests, full regressions and
persisted chain audit required. One-hour/twelve-new-timeout cap, likely-fit
failures and invalid/recovery states stop. Keep public API disabled. Record
unique recipe coverage separately from launch count (one preflight-only launch
is excluded from runtime-result counts and remains in infrastructure evidence).
