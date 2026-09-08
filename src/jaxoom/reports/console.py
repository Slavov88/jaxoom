"""Plain-text report rendering."""
from __future__ import annotations

from ..analysis.tensor_size import format_bytes
from ..types import DonationReport, MemoryAssessment, MemoryReport


def render_assessment(assessment: MemoryAssessment) -> str:
    interval = assessment.interval
    lower = format_bytes(interval.lower_bytes) if interval.lower_bytes is not None else "unavailable"
    headroom = (
        format_bytes(assessment.headroom_to_upper_bytes)
        if assessment.headroom_to_upper_bytes >= 0
        else f"-{format_bytes(-assessment.headroom_to_upper_bytes)}"
    )
    lines = [
        "JAXOOM MEMORY ASSESSMENT",
        "=" * 32,
        "",
        f"Structural estimate         {format_bytes(assessment.structural_peak_bytes)}",
        f"Calibrated central          {format_bytes(interval.central_bytes)}",
        f"Calibrated range             {lower} to {format_bytes(interval.upper_bytes)}",
        f"Conservative upper           {format_bytes(interval.upper_bytes)}",
        f"Memory budget                {format_bytes(assessment.memory_limit_bytes)}",
        f"Headroom to upper            {headroom}",
        f"Risk                         {assessment.risk.value if assessment.risk else 'UNAVAILABLE'}",
        "",
        "Calibration",
        f"Scope                       {interval.calibration_scope}",
        f"Method                      {interval.calibration_method}",
        f"Target coverage             {interval.coverage_target:.0%}" if interval.coverage_target is not None else "Target coverage             unavailable",
        f"Dataset                     {interval.dataset_version or 'unavailable'}",
        f"Samples                     {interval.sample_count or 'unavailable'}",
        "",
        "Limitations",
    ]
    lines.extend(f"- {limitation}" for limitation in assessment.limitations[:4])
    return "\n".join(lines)


def render_donation(report: DonationReport) -> str:
    baseline = report.baseline.compiler_accounted_bytes
    lines = [
        "JAXOOM DONATION ANALYSIS",
        "=" * 32,
        "",
        f"Backend                      {report.backend}",
        f"JAX version                  {report.jax_version}",
        f"Baseline compiler memory    {format_bytes(baseline) if baseline is not None else 'unavailable'}",
        f"Compiler evaluations        {report.compiler_evaluations}",
        "",
        "Best candidate",
    ]
    if report.best is None:
        lines.append("No compiler-confirmed beneficial donation.")
    else:
        best = report.best
        lines.extend(
            [
                f"donate_argnums              {best.argnums}",
                f"Compiler memory             {format_bytes(best.donated_compiler_bytes)}",
                f"Compiler saving             {format_bytes(best.compiler_saving_bytes)}",
                f"Saving                      {best.saving_fraction:.1%}" if best.saving_fraction is not None else "Saving                      unavailable",
                f"Alias gain                  {format_bytes(best.alias_gain_bytes)}" if best.alias_gain_bytes is not None else "Alias gain                  unavailable",
            ]
        )
    lines.extend(["", "Candidates"])
    for candidate in report.candidates[:10]:
        saving = format_bytes(candidate.compiler_saving_bytes) if candidate.compiler_saving_bytes is not None else "unavailable"
        lines.append(f"{candidate.argnums!s:<28} {candidate.status:<20} {saving}")
    lines.extend(["", "Warnings"])
    lines.extend(f"- {warning}" for warning in report.limitations[:3])
    return "\n".join(lines)


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
