"""Built-in calibration summaries generated from recorded accelerator runs.

Sources are the dated CSV artifacts in ``experiments/``. These summaries target
compiler-accounted bytes for JAX 0.6.x and are not runtime peak models.
"""
from __future__ import annotations

from .types import CalibrationSummary


CALIBRATIONS: tuple[CalibrationSummary, ...] = (
    CalibrationSummary(
        backend="cpu",
        hardware_scope="JAX CPU backend",
        jax_version_family="0.6",
        dataset_version="accelerator_calibration_cpu_2026-09-07_v2",
        sample_count=38,
        families=("attention", "elementwise", "matmul", "mlp", "residual", "training", "transformer"),
        method="nearest-rank empirical compiler/static ratio quantiles",
        lower_ratio=0.3333333333333333,
        central_ratio=1.0231318880949498,
        upper_ratio=2.96875,
        coverage_target=0.90,
        applicability="calibrated for the recorded JAX 0.6.x CPU dataset",
        limitations=(
            "CPU calibration is based on 38 recorded configurations.",
            "The upper ratio is dominated by a small number of residual configurations.",
            "This bounds compiler accounting, not runtime peak memory.",
        ),
    ),
    CalibrationSummary(
        backend="gpu",
        hardware_scope="NVIDIA CUDA GPU",
        jax_version_family="0.6",
        dataset_version="accelerator_calibration_gpu_2026-09-07_v2",
        sample_count=38,
        families=("attention", "elementwise", "matmul", "mlp", "residual", "training", "transformer"),
        method="nearest-rank empirical compiler/static ratio quantiles",
        lower_ratio=0.3333333333333333,
        central_ratio=1.0,
        upper_ratio=1.0937538146972656,
        coverage_target=0.90,
        applicability="calibrated for the recorded NVIDIA CUDA and JAX 0.6.x dataset",
        limitations=(
            "GPU calibration is based on one NVIDIA RTX 3050 CUDA environment and 38 configurations.",
            "The upper ratio has 89.5% leave-one-out coverage in the recorded dataset.",
            "This bounds compiler accounting, not runtime peak memory or OOM probability.",
        ),
    ),
)
