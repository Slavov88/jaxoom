"""Plain-text report rendering."""
from __future__ import annotations

from ..analysis.tensor_size import format_bytes
from ..types import MemoryReport


def render_report(report: MemoryReport) -> str:
    lines = [
        "JAXOOM STATIC ANALYSIS",
        "=" * 32,
        "",
        f"Model                         {report.model}",
        f"Estimated structural peak    {format_bytes(report.estimated_peak_bytes)}",
        f"Equations analyzed           {report.equations_analyzed}",
        f"Confidence                   {report.confidence}",
    ]
    if report.memory_limit_bytes is not None:
        lines += [f"Memory limit                 {format_bytes(report.memory_limit_bytes)}", f"Assessment                   {report.assessment}"]
    lines += ["", "Peak"]
    lines += [f"equation                     {report.peak.equation if report.peak.equation is not None else 'entry'}"]
    lines += [f"primitive                    {report.peak.primitive or '<entry>'}", f"live values                  {len(report.peak.live_buffers)}", "", "Largest buffers"]
    for index, buffer in enumerate(report.largest_buffers[:10], 1):
        shape = "[" + ",".join(str(dim) for dim in buffer.shape) + "]"
        lines.append(f"{index}. {buffer.dtype}{shape:<24} {format_bytes(buffer.nbytes):>12}  ({buffer.producer})")
    lines += ["", "Limitations", "- This is a sequential JAXPR live-value estimate.", "- It is not an exact prediction of XLA or runtime GPU peak memory.", "- Compiler fusion, aliasing, scheduling, and workspaces are not modeled."]
    if report.unsupported_constructs:
        lines.append("- Nested or control-flow constructs detected: " + ", ".join(report.unsupported_constructs))
    return "\n".join(lines)
