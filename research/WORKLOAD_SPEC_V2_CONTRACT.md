# Restricted WorkloadSpec adapter for frozen V2

Date: 2026-10-20  
Result status: COMPUTATIONALLY VERIFIED for fresh CPU tracing and frozen-model replay.  
Production decision: **PUBLIC API DISABLED**. No new GPU runtime validation.

## Implemented contract

`experiments/oom_risk_workload_spec.py` is an internal experiment module, outside
the installed `jaxoom` package. It is not exported by `jaxoom`, `jaxoom.api`, or
`jaxoom.__all__`. There is no public `assess_oom_risk` entry point.

`WorkloadSpec(family, configuration, dtype)` supports only the nine historical
template families: attention, convolution, mlp, training, autodiff, transformer,
matmul, reduction, and scan. The exact fields depend on the family; unknown or
missing fields fail. Only positive Python integer dimensions and float16/float32
are accepted. Booleans, floats, mixed dtypes, and arbitrary callables fail.
Transformer width must be divisible by heads. Dimensions are bounded by
2**31 - 1 and depth/layers by 64. These are implementation/resource guards,
**not** validated statistical applicability or extrapolation bounds.

The constructor defensively copies configuration into an immutable, sorted tuple.
Serialization requires exactly version, family, configuration, and dtype.
The function, abstract argument shapes, and metadata all come from that one
specification. There is no independent workload_config or memory_limit input.

The existing experiment template factory is reused only after checking its
normalized source SHA-256. A template edit fails closed rather than silently
changing the model's input program. This preserves historical quirks: training
uses the historical MLP template; scan depth is retained as metadata but does
not alter its computation. Fixing either semantic requires a new template/model
and fresh evidence, not an unannounced change to frozen V2.

## Frozen preprocessing and inference

- Original V2 model file SHA-256:
  `5b2eeedf9a180af1c79678e94ecf3401b0256b4906e1d8f5180f058d3b354ae3`.
- Original V2 model hash:
  `9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96`.
- The artifact bytes, original coefficient hash, feature order, and duplicated
  intercept consistency are verified before replay.
- `largest_over_budget` is explicitly the legacy constant zero. All 176 eligible
  training rows lack this field; the frozen mean and coefficient are zero. The
  adapter does not substitute the semantically different largest-buffer ratio.
- Other memory features come from fresh `jaxoom.estimate` tracing. Metadata
  features come from the same immutable specification; no dtype fallback exists.
- The calibration multiplier remains `1.66765051935476`, and the reference
  denominator remains 3 GiB. Neither is taken from the current calibration registry.
- The threshold remains 0.20. Missing, extra, or nonfinite inference features
  fail rather than silently imputing an active feature.
- Exact JAX/jaxlib 0.11.0 is required for frozen feature tracing. Replaying on a
  different version must fail rather than claiming historical equivalence.

The complete pipeline contract includes the source hashes, schema/version,
original model, feature definitions, multiplier, reference denominator, threshold,
and input guards. A separate contract hash records this implementation without
renaming or modifying the frozen model.

## Replay gate

Run from the repository with JAX/jaxlib 0.11.0 and `PYTHONPATH=src`:

```text
python -m experiments.oom_risk_workload_replay --output NEW_RESULT_PATH.json
```

The CLI exclusively creates its result path; it refuses to replace existing
evidence. It checks the exact bytes and embedded hashes of the three frozen
prediction panels, then freshly traces all 90 + 110 + 24 rows. Supplemental H2
candidate IDs overlap the primary panel, so identities are (panel, candidate_id).
Saved features and probabilities are comparison targets only, never inputs to
the new feature extractor. No runtime outcomes are used in this replay.

The gate requires 224 completed rows, no errors, no feature/score/decision
mismatches at absolute tolerance 1e-12, and a passing tracing-only guard. Each
target invocation must receive JAX tracers. Public JIT/compiler/donation paths
and concrete target calls are forbidden during replay. This is a guard on the
target path, not a claim that no JAX internals can ever compile primitives.

Results retain every new feature vector and score, source/model/prediction hashes,
the invocation, environment, timestamp, base Git commit, and working-tree state.
Partial/error runs fail the gate and the CLI exits nonzero. Passing never
automatically exports an API.

## Evidence and limits

The first loader pilot failed before tracing because it assumed candidate IDs
were globally unique. Its failed record is retained. The corrected first replay
completed 224 traces: exact feature equality, zero score/decision mismatches,
maximum probability difference 1.11e-16, 224 tracer-only calls, no concrete target
calls, and no attempted public compilation calls. A final replay after failure
handling and artifact-integrity tests passed with identical results in
`experiments/oom_risk_workload_replay_2026-10-20_final.json`. Its contract hash is
`84dc466cddc79b12590a7034f830fd189b57f36c28e3e4347c3df26d7cfb5471`.
The complete JAX/jaxlib 0.11.0 CPU suite passed **141 tests**, including 39 new
restricted-adapter checks. Syntax also compiles on the package's Python 3.10
minimum. No existing tracked experiment, model, research-design, or public-package
file was modified; only the status report is updated alongside the new files.

The September production-design JSON/report remain historical evidence and are
not rewritten. Their failed structured-adapter gate describes the old semantic
largest-buffer adapter. The new restricted adapter preserves the actual legacy
zero feature; it does not repair that old artifact in place. Use the new replay
command instead of rerunning the historical production-design writer.

**3 GiB is a feature reference, not a measured allocator quota.** The held-out
campaigns cleared allocator fraction overrides and ran on the 4 GiB RTX 3050
without enforcing a 3 GiB cap. This work does not establish fit within a 3 GiB
runtime allocation limit, dynamic-budget validity, or validity under changed
external memory occupancy.

Fresh CPU tracing establishes feature-pipeline equivalence. It does not establish
new GPU runtime accuracy, validate all specifications admitted by the schema,
or transfer the historical campaign-conditional false-safe bound to arbitrary
programs, configurations, devices, or memory budgets. Runtime applicability is
explicitly NOT_QUALIFIED. A public restricted assessment still needs an explicit
applicability policy and GPU integration qualification; generic callables need a
new feature contract/model and prospective validation. Compile risk stays separate.
