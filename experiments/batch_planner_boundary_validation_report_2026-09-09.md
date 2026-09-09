# Batch planner boundary validation

Status: COMPUTATIONALLY VERIFIED on RTX 3050 Laptop GPU, JAX 0.11.0, CUDA, and .

Twelve explicit-budget scenarios across MLP, training-like, attention, and convolution used 16, 32, and 64 MiB budgets with batch range 1--2048. There were 9 interior boundaries, 1 upper-bound control, and 2 nothing-fits controls. All nine recommended batches executed FIT in post-planning validation.

The next adjacent batch FIT in all nine interior cases despite exceeding the calibrated-upper criterion. False-safe count was 0/9; conservative-miss count was 9/9. No calibration was tuned.

Auto-budget planning used 0, 512 MiB, and 1 GiB bounded JAX pressure. Budgets were 3,006,477,107; 2,779,984,691; and 1,706,242,867 bytes, with recommendations 8192, 8192, and 7745. Occupancy monotonicity held.

Median planner evaluations were 17, median planner latency 167 ms, median post-planning compile latency  [recorded in the JSON], and median post-planning execution latency [recorded in the JSON]. Planning performed no target compilation or execution. Compilation and execution were explicit post-planning harness steps.

The mechanics are validated, but this sample indicates the calibrated upper criterion is conservative.
