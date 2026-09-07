"""Public and internal typed result structures."""
from __future__ import annotations

from dataclasses import dataclass


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
