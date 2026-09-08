"""Built-in calibration summaries generated from recorded accelerator runs.

Sources are the dated CSV artifacts in ``experiments/``. These summaries target compiler-accounted bytes for tested JAX version
families and are not runtime peak models. They are intentionally explicit
rather than a generic compatibility registry.
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
        tested_jax_versions=("0.6.2",),
        tested_jaxlib_versions=("0.6.2",),
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
        tested_jax_versions=("0.6.2",),
        tested_jaxlib_versions=("0.6.2",),
    ),
    CalibrationSummary(
        backend="gpu",
        hardware_scope="NVIDIA CUDA GPU",
        jax_version_family="0.11",
        dataset_version="jax_0_11_gpu_calibration_2026-09-08_v2",
        sample_count=46,
        families=("attention", "autodiff", "convolution", "elementwise", "matmul", "mlp", "residual", "training", "transformer"),
        method="nearest-rank empirical compiler/static ratio quantiles",
        lower_ratio=0.6734660838594787,
        central_ratio=1.0,
        upper_ratio=1.6676505193119118,
        coverage_target=0.95,
        applicability="calibrated for the recorded NVIDIA CUDA and JAX 0.11.0 dataset",
        limitations=(
            "GPU calibration is based on one NVIDIA RTX 3050 CUDA environment and 46 configurations.",
            "The upper ratio is the 95th nearest-rank ratio quantile; it is not an OOM probability.",
            "The held-out grouped validation used 10 configurations and covered 9/10 upper bounds.",
            "This bounds compiler accounting, not runtime peak memory or OOM probability.",
        ),
        tested_jax_versions=("0.11.0",),
        tested_jaxlib_versions=("0.11.0",),
    ),
)
