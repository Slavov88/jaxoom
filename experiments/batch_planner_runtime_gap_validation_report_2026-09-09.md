# Batch planner runtime-gap validation

Status: COMPUTATIONALLY VERIFIED PARTIAL on an RTX 3050 Laptop GPU with JAX 0.11.0 and CUDA.

Three explicit scenarios completed: MLP, training-like, and attention, each with a 32 MiB planner budget. The planner recommendations were 920, 271, and 152 respectively. Each recommendation FIT in a fresh subprocess. Geometric runtime searches reached 7,360, 2,168, and 1,216 respectively, all FIT, with no OOM boundary. Therefore the runtime gaps are lower bounds of 8.0x for all three cases.

The convolution scenario timed out during runtime probing. Auto-budget scenarios were not completed after the bounded run exceeded its wall-clock budget. No exact runtime boundary was found.

Planning itself performed no target compilation or execution. Runtime compilation and execution occurred only in fresh validation subprocesses. No calibration, planner margin, or residency logic was changed.

Decision: VALIDATION INSUFFICIENT. The observed lower bounds already indicate strong conservatism, but they do not establish the actual runtime maximum.
