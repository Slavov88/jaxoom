# Allocator-aware safety gate study

Status: ALLOCATOR GATE PROMISING BUT INSUFFICIENT.

The static candidate uses the sum of the two largest buffers live at the JAXPR peak as a large-allocation proxy. It is available before compilation. The candidate compares this proxy with the aggregate calibrated-upper headroom (`budget - calibrated_upper`). It is not a global multiplier and does not use compiler or runtime diagnostics as inputs.

On the frozen RTX 3050/JAX 0.11.0 attention sequence evidence, the aggregate gate produced two false-safe OOM classifications (S=4608 and S=5120). The top-two-live gate produced zero false-safe and zero false-reject classifications across S=2048, 3072, 4096, 4608, and 5120. The same candidate also preserved the three non-attention FIT controls when evaluated against the 3 GiB physical allocator budget.

This is promising mechanism evidence, not a production validation. The attention held-out set is small, contains no alternate dtype/H/D/B regime, and the controls are not a balanced OOM/FIT sample. No production gate was integrated.

The ATT-2 requirement is satisfied experimentally: aggregate passes, top-two-live fails, and the actual outcome is EXECUTION_OOM. S=4608 is also rejected by the top-two-live candidate while actual outcome is EXECUTION_OOM.

Production decision: NO CHANGE.
