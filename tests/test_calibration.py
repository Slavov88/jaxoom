import jax

import jaxoom
from jaxoom.types import CalibrationSummary, MemoryRiskLevel


def test_backend_calibration_has_provenance():
    report = jaxoom.estimate(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"))
    interval = jaxoom.calibrate(report, backend="gpu")
    assert interval.dataset_version == "accelerator_calibration_gpu_2026-09-07_v2"
    assert interval.lower_bytes <= interval.central_bytes <= interval.upper_bytes
    assert interval.coverage_target == 0.90
    assert interval.sample_count == 38


def test_risk_levels_use_interval_and_budget():
    summary = CalibrationSummary(
        backend="test",
        hardware_scope="test backend",
        jax_version_family="0.6",
        dataset_version="test-data",
        sample_count=10,
        families=("test",),
        method="test ratio",
        lower_ratio=1.0,
        central_ratio=1.0,
        upper_ratio=2.0,
        coverage_target=0.9,
        applicability="test",
        limitations=(),
    )
    report = jaxoom.estimate(lambda x: x, jax.ShapeDtypeStruct((25,), "float32"))
    moderate = jaxoom.assess(report, report.estimated_peak_bytes, summary=summary)
    low = jaxoom.assess(report, report.estimated_peak_bytes * 2, summary=summary)
    likely = jaxoom.assess(report, report.estimated_peak_bytes // 2, summary=summary)
    assert moderate.risk is MemoryRiskLevel.MODERATE
    assert low.risk is MemoryRiskLevel.LOW
    assert likely.risk is MemoryRiskLevel.LIKELY_EXCEEDS_BUDGET
    assert moderate.headroom_to_upper_bytes < 0


def test_uncalibrated_backend_is_explicit():
    report = jaxoom.estimate(lambda x: x, jax.ShapeDtypeStruct((4,), "float32"))
    interval = jaxoom.calibrate(report, backend="tpu")
    assessment = jaxoom.assess(report, "1 KiB", backend="tpu")
    assert interval.dataset_version is None
    assert interval.applicability == "unavailable"
    assert not assessment.calibrated
    assert assessment.risk is MemoryRiskLevel.HIGH


def test_zero_structural_memory_is_stable():
    report = jaxoom.estimate(lambda x: x, jax.ShapeDtypeStruct((0,), "float32"))
    interval = jaxoom.calibrate(report, backend="gpu")
    assert interval.lower_bytes == interval.central_bytes == interval.upper_bytes == 0
