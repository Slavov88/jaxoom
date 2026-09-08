"""Inspectably calibrated compiler-memory intervals and budget risk."""
from __future__ import annotations

import re

import jax

from .analysis.tensor_size import parse_memory_limit
from .calibration_defaults import CALIBRATIONS
from .types import CalibrationSummary, MemoryAssessment, MemoryInterval, MemoryReport, MemoryRiskLevel


def calibrate(
    report: MemoryReport,
    *,
    backend: str | None = None,
    jax_version: str | None = None,
    summary: CalibrationSummary | None = None,
) -> MemoryInterval:
    """Return an empirical interval for compiler-accounted memory.

    The structural report remains unchanged. Built-in summaries are selected by
    backend and are limited to the recorded JAX 0.6.x CPU/CUDA datasets.
    """
    version = jax_version or jax.__version__
    selected = summary or _select_summary(backend or jax.default_backend(), version)
    if selected is None:
        return _fallback_interval(
            report,
            f"no validated calibration is available for backend {backend or jax.default_backend()} and JAX {version}",
        )

    limitations = list(selected.limitations)
    applicability = _applicability(selected, version)
    if applicability == "VERSION_FAMILY_MATCH":
        limitations.append(
            f"calibration family {selected.jax_version_family}.x was tested on {', '.join(selected.tested_jax_versions) or 'unspecified versions'}"
        )

    static = report.estimated_peak_bytes
    if static == 0:
        lower = central = upper = 0
        limitations.append("structural estimate is zero, so multiplicative calibration is zero")
    else:
        lower = max(0, round(static * selected.lower_ratio))
        central = max(0, round(static * selected.central_ratio))
        upper = max(central, round(static * selected.upper_ratio))
    return MemoryInterval(
        lower_bytes=lower,
        central_bytes=central,
        upper_bytes=upper,
        coverage_target=selected.coverage_target,
        calibration_method=selected.method,
        calibration_scope=selected.hardware_scope,
        dataset_version=selected.dataset_version,
        sample_count=selected.sample_count,
        applicability=applicability,
        limitations=tuple(limitations),
    )


def assess(
    report: MemoryReport,
    memory_limit: str | int,
    *,
    backend: str | None = None,
    jax_version: str | None = None,
    summary: CalibrationSummary | None = None,
) -> MemoryAssessment:
    """Compare a calibrated interval with a memory budget.

    Risk is qualitative. It is not an estimated probability of OOM.
    """
    limit = parse_memory_limit(memory_limit)
    if limit is None:
        raise ValueError("memory_limit is required for risk assessment")
    interval = calibrate(report, backend=backend, jax_version=jax_version, summary=summary)
    calibrated = interval.dataset_version is not None and interval.applicability != "UNCALIBRATED"
    if not calibrated:
        risk = (
            MemoryRiskLevel.LIKELY_EXCEEDS_BUDGET
            if report.estimated_peak_bytes > limit
            else MemoryRiskLevel.HIGH
        )
        limitations = interval.limitations + ("risk is an uncalibrated structural fallback",)
    elif report.estimated_peak_bytes > limit:
        risk = MemoryRiskLevel.LIKELY_EXCEEDS_BUDGET
        limitations = interval.limitations
    elif interval.central_bytes > limit:
        risk = MemoryRiskLevel.HIGH
        limitations = interval.limitations
    elif interval.upper_bytes > limit:
        risk = MemoryRiskLevel.MODERATE
        limitations = interval.limitations
    else:
        risk = MemoryRiskLevel.LOW
        limitations = interval.limitations
    return MemoryAssessment(
        structural_peak_bytes=report.estimated_peak_bytes,
        interval=interval,
        memory_limit_bytes=limit,
        risk=risk,
        calibrated=calibrated,
        headroom_to_upper_bytes=limit - interval.upper_bytes,
        limitations=limitations,
    )


def _version_family(version: str) -> str | None:
    match = re.match(r"^(\d+\.\d+)(?:\.|$)", version)
    return match.group(1) if match else None


def _select_summary(backend: str, version: str) -> CalibrationSummary | None:
    candidates = [item for item in CALIBRATIONS if item.backend == backend]
    for item in candidates:
        if version in item.tested_jax_versions:
            return item
    family = _version_family(version)
    return next((item for item in candidates if item.jax_version_family == family), None)


def _applicability(summary: CalibrationSummary, version: str) -> str:
    if version in summary.tested_jax_versions:
        return "EXACT_TESTED"
    if _version_family(version) == summary.jax_version_family:
        return "VERSION_FAMILY_MATCH"
    return "UNCALIBRATED"


def _fallback_interval(report: MemoryReport, reason: str) -> MemoryInterval:
    return MemoryInterval(
        lower_bytes=None,
        central_bytes=report.estimated_peak_bytes,
        upper_bytes=report.estimated_peak_bytes,
        coverage_target=None,
        calibration_method="raw structural fallback",
        calibration_scope="none",
        dataset_version=None,
        sample_count=None,
        applicability="UNCALIBRATED",
        limitations=(reason, "compiler-accounted calibration is unavailable"),
    )
