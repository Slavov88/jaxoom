"""Public and internal typed result structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
