"""Static/compiler comparison utilities."""
from __future__ import annotations

from ..types import CompilerMemoryReport, MemoryComparison, MemoryReport


def compare_memory(static_report: MemoryReport, compiler_report: CompilerMemoryReport) -> MemoryComparison:
    """Compare bytes with signed difference ``static - compiler``."""
    compiler = compiler_report.compiler_accounted_bytes
    if compiler is None:
        return MemoryComparison(
            static_peak_bytes=static_report.estimated_peak_bytes,
            compiler_accounted_bytes=None,
            signed_difference_bytes=None,
            absolute_difference_bytes=None,
            relative_difference=None,
            static_overpredicts=None,
            static_underpredicts=None,
            limitations=("compiler-accounted bytes are unavailable.",),
        )
    signed = static_report.estimated_peak_bytes - compiler
    relative = signed / compiler if compiler else None
    return MemoryComparison(
        static_peak_bytes=static_report.estimated_peak_bytes,
        compiler_accounted_bytes=compiler,
        signed_difference_bytes=signed,
        absolute_difference_bytes=abs(signed),
        relative_difference=relative,
        static_overpredicts=signed > 0,
        static_underpredicts=signed < 0,
        limitations=("signed difference is static minus compiler-accounted bytes.",),
    )
