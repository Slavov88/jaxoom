"""Public and internal typed result structures."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class BufferInfo:
    """One abstract array value tracked by the sequential model."""

    variable_id: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    producer: str
    birth: int
    last_use: int
    is_input: bool = False
    is_output: bool = False


@dataclass(frozen=True)
class PeakPoint:
    """Memory state at the conservative peak candidate."""

    equation: int | None
    primitive: str | None
    live_bytes: int
    live_buffers: tuple[BufferInfo, ...]


@dataclass(frozen=True)
class MemoryReport:
    """Structured result of static sequential JAXPR analysis."""

    estimated_peak_bytes: int
    largest_buffers: tuple[BufferInfo, ...]
    peak: PeakPoint
    equations_analyzed: int
    confidence: str
    model: str
    unsupported_constructs: tuple[str, ...]
    limitations: tuple[str, ...]
    memory_limit_bytes: int | None = None
    assessment: str | None = None

    def render(self) -> str:
        from .reports.console import render_report

        return render_report(self)

    def print(self) -> None:
        print(self.render())

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True)
class CompilerMemoryReport:
    """Memory categories returned by the compiled JAX program, when available."""

    argument_bytes: int | None
    output_bytes: int | None
    temporary_bytes: int | None
    alias_bytes: int | None
    compiler_accounted_bytes: int | None
    backend: str
    platform: str | None
    jax_version: str
    jaxlib_version: str
    available: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class MemoryRiskLevel(str, Enum):
    """Qualitative risk under a supplied memory budget."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    LIKELY_EXCEEDS_BUDGET = "LIKELY EXCEEDS BUDGET"


@dataclass(frozen=True)
class CalibrationSummary:
    """Provenance and empirical quantiles for one calibration scope."""

    backend: str
    hardware_scope: str
    jax_version_family: str
    dataset_version: str
    sample_count: int
    families: tuple[str, ...]
    method: str
    lower_ratio: float
    central_ratio: float
    upper_ratio: float
    coverage_target: float
    applicability: str
    limitations: tuple[str, ...]
    tested_jax_versions: tuple[str, ...] = ()
    tested_jaxlib_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryInterval:
    """A compiler-accounted memory interval around a structural estimate."""

    lower_bytes: int | None
    central_bytes: int
    upper_bytes: int
    coverage_target: float | None
    calibration_method: str
    calibration_scope: str
    dataset_version: str | None
    sample_count: int | None
    applicability: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class MemoryAssessment:
    """Budget comparison kept separate from the structural memory report."""

    structural_peak_bytes: int
    interval: MemoryInterval
    memory_limit_bytes: int
    risk: MemoryRiskLevel | None
    calibrated: bool
    headroom_to_upper_bytes: int
    limitations: tuple[str, ...]

    def render(self) -> str:
        from .reports.console import render_assessment

        return render_assessment(self)

    def print(self) -> None:
        print(self.render())

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True)
class MemoryComparison:
    """Comparison with signed error defined as static minus compiler bytes."""

    static_peak_bytes: int
    compiler_accounted_bytes: int | None
    signed_difference_bytes: int | None
    absolute_difference_bytes: int | None
    relative_difference: float | None
    static_overpredicts: bool | None
    static_underpredicts: bool | None
    limitations: tuple[str, ...]
