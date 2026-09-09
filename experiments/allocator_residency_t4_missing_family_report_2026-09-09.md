# T4 scan-based missing-family residency study

Status: OBSERVED. This artifact records the completed negative result from the
Tesla T4 campaign. It is not a production residency model.

The scan representation bounded the outer graph while structural memory grew:
MLP had one outer equation and a two-equation body; training had 20--23 outer
equations and a ten-equation body; autodiff had 8--11 outer equations and a
ten-equation body. The scale artifact does not contain complete compile latency
for every scale point, so latency decoupling is not quantified here.

The scale stage used 100 probes: 76 FIT, 21 EXECUTION_OOM, and 3 other
failures. The threshold stage used 22 probes: 13 FIT, 3 COMPILE_OOM, 4
EXECUTION_OOM, 1 EXECUTION_TIMEOUT, and 1 other failure. Timeouts were not
converted to OOM labels.

Four new threshold rows were produced: two autodiff scan rows, one training
scan row, and one low-precision convolution refinement. No MLP threshold was
obtained. The final merged dataset has 14 distinct thresholds: attention 5,
matmul 3, transformer 2, convolution 1, training scan 1, autodiff scan 2,
and MLP scan 0. Dtypes are float32 9 and float16 5. Median bracket width is
155,189,248 bytes and P90 width is 312,475,648 bytes.

The strict evidence gate was not met. Model fitting and family holdout were
not run. Residency modeling is stopped. Production decision: NO CHANGE.
