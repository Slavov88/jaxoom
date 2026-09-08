"""Public analysis API."""
from __future__ import annotations

from typing import Any, Callable

from .analysis.liveness import build_lifetimes
from .calibration import assess as _assess
from .calibration import calibrate as _calibrate
from .analysis.peak import calculate_peak
from .analysis.tensor_size import parse_memory_limit
from .compiler.memory_analysis import compile_analyze as _compile_analyze
from .donation import analyze_donation as _analyze_donation
from .tracing import trace
from .types import (
    CalibrationSummary,
    CompilerMemoryReport,
    MemoryAssessment,
    MemoryComparison,
    MemoryInterval,
    MemoryReport,
    DonationReport,
)


def compile_analyze(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> CompilerMemoryReport:
    """Compile a function and return backend-reported memory categories."""
    return _compile_analyze(fn, args, kwargs)


def analyze_donation(
    fn: Callable[..., Any],
    *args: Any,
    max_compilations: int = 32,
    **kwargs: Any,
) -> DonationReport:
    """Recommend compiler-confirmed positional buffer donation opportunities."""
    return _analyze_donation(fn, *args, max_compilations=max_compilations, **kwargs)


def compare_memory(static_report: MemoryReport, compiler_report: CompilerMemoryReport) -> MemoryComparison:
    """Compare static bytes with compiler accounting using static minus compiler."""
    from .compiler.comparison import compare_memory as _compare_memory

    return _compare_memory(static_report, compiler_report)


def calibrate(
    report: MemoryReport,
    *,
    backend: str | None = None,
    jax_version: str | None = None,
    summary: CalibrationSummary | None = None,
) -> MemoryInterval:
    """Add an empirical compiler-accounted memory interval to a report."""
    return _calibrate(report, backend=backend, jax_version=jax_version, summary=summary)


def assess(
    report: MemoryReport,
    memory_limit: str | int,
    *,
    backend: str | None = None,
    jax_version: str | None = None,
    summary: CalibrationSummary | None = None,
) -> MemoryAssessment:
    """Assess qualitative memory risk under an explicit budget."""
    return _assess(report, memory_limit, backend=backend, jax_version=jax_version, summary=summary)


def estimate(
    fn: Callable[..., Any],
    *args: Any,
    memory_limit: str | int | None = None,
    **kwargs: Any,
) -> MemoryReport:
    """Estimate a function's sequential JAXPR live-value peak.

    The callable is traced with JAX abstract values. It is not intentionally
    numerically executed. Concrete inputs are accepted, and
    ``jax.ShapeDtypeStruct`` can be used to avoid allocating large arrays.
    """
    closed = trace(fn, args, kwargs)
    model = build_lifetimes(closed)
    primitives = tuple(str(eqn.primitive) for eqn in closed.jaxpr.eqns)
    peak = calculate_peak(model, len(primitives), primitives)
    limit = parse_memory_limit(memory_limit)
    assessment = None
    if limit is not None:
        assessment = "LIKELY FIT" if peak.peak.live_bytes <= limit else "LIKELY OOM"
    confidence = "limited" if model.unsupported_constructs else "structural"
    return MemoryReport(
        estimated_peak_bytes=peak.peak.live_bytes,
        largest_buffers=peak.largest_buffers,
        peak=peak.peak,
        equations_analyzed=len(primitives),
        confidence=confidence,
        model="sequential JAXPR live-value estimate",
        unsupported_constructs=model.unsupported_constructs,
        limitations=(
            "JAXPR logical liveness is not compiled buffer liveness.",
            "Compiler fusion, buffer aliasing, scheduling, allocator behavior, and workspaces are not modeled.",
        ),
        memory_limit_bytes=limit,
        assessment=assessment,
    )
