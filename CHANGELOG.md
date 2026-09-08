# Changelog

## Unreleased

- Added static sequential JAXPR live-value analysis.
- Added compiler-backed memory accounting and static/compiler comparison.
- Added CPU and NVIDIA GPU calibration experiments.
- Added isolated experimental fit-boundary and OOM validation.
- Added opt-in empirical compiler-accounted memory intervals and qualitative budget risk assessment.
- Added cross-version validation artifacts for JAX 0.6.2 and JAX 0.11.0.
- Added version-aware compiler calibration selection and explicit uncalibrated fallback behavior.
- Added compiler-confirmed positional buffer donation advice with explicit caller-safety warnings.
- Added observational current-device memory snapshots and conservative auto budgets for pre-compilation assessment.
- Corrected auto budgets to respect known JAX allocator limits without adding pool bytes to driver free memory.
