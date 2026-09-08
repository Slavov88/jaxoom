import jax

import jaxoom
from jaxoom.types import CalibrationSummary, MemoryRiskLevel


def test_exact_version_calibration_has_provenance():
    report = jaxoom.estimate(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"))
    interval = jaxoom.calibrate(report, backend="gpu", jax_version="0.6.2")
    assert interval.dataset_version == "accelerator_calibration_gpu_2026-09-07_v2"
    assert interval.applicability == "EXACT_TESTED"
    assert interval.lower_bytes <= interval.central_bytes <= interval.upper_bytes
    assert interval.coverage_target == 0.90
    assert interval.sample_count == 38


def test_version_family_calibration_is_explicit():
    report = jaxoom.estimate(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"))
    interval = jaxoom.calibrate(report, backend="gpu", jax_version="0.11.3")
    assert interval.dataset_version == "jax_0_11_gpu_calibration_2026-09-08_v2"
    assert interval.applicability == "VERSION_FAMILY_MATCH"


def test_unvalidated_version_does_not_use_legacy_calibration():
    report = jaxoom.estimate(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"))
    interval = jaxoom.calibrate(report, backend="gpu", jax_version="0.12.0")
    assessment = jaxoom.assess(report, "16 GiB", backend="gpu", jax_version="0.12.0")
    assert interval.dataset_version is None
    assert interval.applicability == "UNCALIBRATED"
    assert not assessment.calibrated


def test_version_family_parser_rejects_malformed_versions():
    from jaxoom.calibration import _version_family

    assert _version_family("0.11.0") == "0.11"
    assert _version_family("0.11.3.dev1") == "0.11"
    assert _version_family("unknown") is None


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
    moderate = jaxoom.assess(report, report.estimated_peak_bytes, summary=summary, jax_version="0.6.2")
    low = jaxoom.assess(report, report.estimated_peak_bytes * 2, summary=summary, jax_version="0.6.2")
    likely = jaxoom.assess(report, report.estimated_peak_bytes // 2, summary=summary, jax_version="0.6.2")
    assert moderate.risk is MemoryRiskLevel.MODERATE
    assert low.risk is MemoryRiskLevel.LOW
    assert likely.risk is MemoryRiskLevel.LIKELY_EXCEEDS_BUDGET
    assert moderate.headroom_to_upper_bytes < 0


def test_uncalibrated_backend_is_explicit():
    report = jaxoom.estimate(lambda x: x, jax.ShapeDtypeStruct((4,), "float32"))
    interval = jaxoom.calibrate(report, backend="tpu")
    assessment = jaxoom.assess(report, "1 KiB", backend="tpu")
    assert interval.dataset_version is None
    assert interval.applicability == "UNCALIBRATED"
    assert not assessment.calibrated
    assert assessment.risk is MemoryRiskLevel.HIGH


def test_zero_structural_memory_is_stable():
    report = jaxoom.estimate(lambda x: x, jax.ShapeDtypeStruct((0,), "float32"))
    interval = jaxoom.calibrate(report, backend="gpu")
    assert interval.lower_bytes == interval.central_bytes == interval.upper_bytes == 0
