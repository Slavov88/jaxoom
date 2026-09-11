# Multi-regime allocator gate validation

Status: ALLOCATOR GATE PROMISING BUT INSUFFICIENT.

The frozen top-two-peak-live gate was evaluated on 15 pre-registered RTX 3050/JAX 0.11.0 configurations: 10 attention cases spanning batch, heads, head dimension, and float16, plus three non-attention controls and two additional attention geometries. Static predictions were recorded before fresh-process execution.

Observed stable outcomes were 13 FIT and 1 EXECUTION_OOM. One float32 B=1,H=8,S=4096,D=64 case was UNSTABLE across two fresh repetitions (one FIT, one OOM) and was not forced into a deterministic label. The H=16,D=32,S=4096 case was rejected by the allocator gate and produced EXECUTION_OOM. All stable FIT cases were preserved by the top-two-live candidate.

The frozen candidate produced zero stable false-safe cases and zero stable false-reject cases in this small set. The aggregate-only and multiplier baselines were evaluated on the same rows. The result is promising but insufficient: only one new stable OOM was observed, the held-out set remains small, and no alternate device or allocator policy was tested.

The matched float16 pair B=2,H=8,S=3072,D=32 and B=1,H=16,S=3072,D=32 had equal static top-two proxies and both FIT, providing a limited geometry control.

No production code changed.
