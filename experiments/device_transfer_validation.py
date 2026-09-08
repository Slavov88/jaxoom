"""Run a paired compiler-calibration transfer check on another CUDA GPU.

The reference JSON must come from the frozen JAX 0.11.0 RTX 3050 dataset. The
script reruns the same logical configurations on the active device and writes
paired results, summary metrics, and a concise report. It refuses to call a
same-model run an independent-device result unless explicitly requested for a
smoke test.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from accelerator_calibration import cases, environment, run_case
from jaxoom.calibration_defaults import CALIBRATIONS


FROZEN_REFERENCE_DEVICE = "NVIDIA GeForce RTX 3050 Laptop GPU"


def frozen_calibration() -> dict[str, Any]:
    matches = [
        summary
        for summary in CALIBRATIONS
        if summary.backend == "gpu"
        and summary.jax_version_family == "0.11"
        and "0.11.0" in summary.tested_jax_versions
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact JAX 0.11 GPU calibration, found {len(matches)}")
    summary = matches[0]
    return {
        "lower_ratio": summary.lower_ratio,
        "central_ratio": summary.central_ratio,
        "upper_ratio": summary.upper_ratio,
        "dataset_version": summary.dataset_version,
    }


def nearest_percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def load_reference(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    return payload.get("summary", {}).get("environment", {}), {row["name"]: row for row in rows if row.get("status") == "ok"}


def device_label(env: dict[str, Any]) -> str:
    return str(env.get("device_kind") or env.get("device") or "unknown device")


def size_bucket(static_bytes: int) -> str:
    mib = static_bytes / 1024**2
    if mib < 10:
        return "<10 MiB"
    if mib < 100:
        return "10-100 MiB"
    if mib < 500:
        return "100-500 MiB"
    return ">500 MiB"


def pair_case(reference: dict[str, Any], current: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    static = current.get("static_peak_bytes")
    reference_compiler = reference.get("compiler_accounted_bytes")
    current_compiler = current.get("compiler_accounted_bytes")
    lower = round(static * calibration["lower_ratio"]) if static is not None else None
    central = round(static * calibration["central_ratio"]) if static is not None else None
    upper = round(static * calibration["upper_ratio"]) if static is not None else None
    return {
        "name": current["name"],
        "family": current["family"],
        "dtype": current["dtype"],
        "configuration": json.loads(current["configuration"]),
        "size_bucket": size_bucket(static),
        "static_bytes_reference": reference.get("static_peak_bytes"),
        "static_bytes_current": static,
        "static_bytes_delta": static - reference.get("static_peak_bytes", static),
        "reference_compiler_bytes": reference_compiler,
        "current_compiler_bytes": current_compiler,
        "compiler_drift_bytes": current_compiler - reference_compiler if current_compiler is not None and reference_compiler is not None else None,
        "compiler_relative_drift": current_compiler / reference_compiler - 1 if current_compiler is not None and reference_compiler else None,
        "reference_argument_bytes": reference.get("compiler_argument_bytes"),
        "current_argument_bytes": current.get("compiler_argument_bytes"),
        "argument_delta": current.get("compiler_argument_bytes") - reference.get("compiler_argument_bytes") if current.get("compiler_argument_bytes") is not None and reference.get("compiler_argument_bytes") is not None else None,
        "reference_output_bytes": reference.get("compiler_output_bytes"),
        "current_output_bytes": current.get("compiler_output_bytes"),
        "output_delta": current.get("compiler_output_bytes") - reference.get("compiler_output_bytes") if current.get("compiler_output_bytes") is not None and reference.get("compiler_output_bytes") is not None else None,
        "reference_temp_bytes": reference.get("compiler_temp_bytes"),
        "current_temp_bytes": current.get("compiler_temp_bytes"),
        "temp_delta": current.get("compiler_temp_bytes") - reference.get("compiler_temp_bytes") if current.get("compiler_temp_bytes") is not None and reference.get("compiler_temp_bytes") is not None else None,
        "reference_alias_bytes": reference.get("compiler_alias_bytes"),
        "current_alias_bytes": current.get("compiler_alias_bytes"),
        "alias_delta": current.get("compiler_alias_bytes") - reference.get("compiler_alias_bytes") if current.get("compiler_alias_bytes") is not None and reference.get("compiler_alias_bytes") is not None else None,
        "frozen_lower_bytes": lower,
        "frozen_central_bytes": central,
        "frozen_upper_bytes": upper,
        "current_inside_interval": lower <= current_compiler <= upper if current_compiler is not None else None,
        "current_under_upper": current_compiler <= upper if current_compiler is not None else None,
    }


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in pairs if row["current_compiler_bytes"] is not None]
    relative = [abs(row["compiler_relative_drift"]) for row in valid if row["compiler_relative_drift"] is not None]
    signed = [row["compiler_relative_drift"] for row in valid if row["compiler_relative_drift"] is not None]
    temp_ratios = [row["current_temp_bytes"] / row["reference_temp_bytes"] for row in valid if row["reference_temp_bytes"] not in (None, 0) and row["current_temp_bytes"] is not None]
    calibration = frozen_calibration()
    interval_width = calibration["upper_ratio"] - calibration["lower_ratio"]

    def grouped(key: str) -> dict[str, Any]:
        result = {}
        for value in sorted({row[key] for row in valid}):
            group = [row for row in valid if row[key] == value]
            result[str(value)] = {
                "count": len(group),
                "upper_coverage": sum(row["current_under_upper"] for row in group) / len(group),
                "interval_coverage": sum(row["current_inside_interval"] for row in group) / len(group),
                "median_relative_drift": statistics.median(abs(row["compiler_relative_drift"]) for row in group),
            }
        return result

    return {
        "sample_count": len(valid),
        "interval_coverage_count": sum(row["current_inside_interval"] for row in valid),
        "interval_coverage": sum(row["current_inside_interval"] for row in valid) / len(valid),
        "upper_coverage_count": sum(row["current_under_upper"] for row in valid),
        "upper_coverage": sum(row["current_under_upper"] for row in valid) / len(valid),
        "upper_miss_rate": sum(not row["current_under_upper"] for row in valid) / len(valid),
        "median_width_over_static": interval_width,
        "p90_width_over_static": interval_width,
        "median_signed_relative_drift": statistics.median(signed) if signed else None,
        "median_absolute_relative_drift": statistics.median(relative) if relative else None,
        "p90_absolute_relative_drift": nearest_percentile(relative, 0.90),
        "maximum_absolute_relative_drift": max(relative) if relative else None,
        "temporary_ratio_median": statistics.median(temp_ratios) if temp_ratios else None,
        "temporary_ratio_p90": nearest_percentile(temp_ratios, 0.90),
        "temporary_ratio_maximum": max(temp_ratios) if temp_ratios else None,
        "temporary_zero_to_nonzero": sum(row["reference_temp_bytes"] == 0 and row["current_temp_bytes"] > 0 for row in valid),
        "temporary_nonzero_to_zero": sum(row["reference_temp_bytes"] > 0 and row["current_temp_bytes"] == 0 for row in valid),
        "alias_delta_nonzero": sum(row["alias_delta"] != 0 for row in valid),
        "by_family": grouped("family"),
        "by_dtype": grouped("dtype"),
        "by_size_bucket": grouped("size_bucket"),
    }


def write_report(path: Path, reference_env: dict[str, Any], current_env: dict[str, Any], summary: dict[str, Any], independent: bool) -> None:
    independence = "independent device" if independent else "same-device smoke only"
    lines = [
        "# Device transfer validation",
        "",
        f"**Status: {'COMPUTATIONALLY VERIFIED' if independent else 'SMOKE ONLY'}.**",
        "",
        f"This run is classified as **{independence}**. It evaluates the frozen JAX 0.11.0 GPU calibration derived from the RTX 3050 dataset.",
        "",
        "## Environments",
        "",
        f"- Reference: {device_label(reference_env)}, JAX {reference_env.get('jax_version')}, jaxlib {reference_env.get('jaxlib_version')}",
        f"- Current: {device_label(current_env)}, JAX {current_env.get('jax_version')}, jaxlib {current_env.get('jaxlib_version')}",
        "",
        "## Frozen transfer metrics",
        "",
        f"- Cases: {summary['sample_count']}",
        f"- Interval coverage: {summary['interval_coverage_count']}/{summary['sample_count']} = {summary['interval_coverage']:.1%}",
        f"- Upper coverage: {summary['upper_coverage_count']}/{summary['sample_count']} = {summary['upper_coverage']:.1%}",
        f"- Upper miss rate: {summary['upper_miss_rate']:.1%}",
        f"- Median width/static: {summary['median_width_over_static']:.4f}",
        "",
        "## Compiler drift",
        "",
        f"- Median absolute relative drift: {summary['median_absolute_relative_drift']:.4f}",
        f"- P90 absolute relative drift: {summary['p90_absolute_relative_drift']:.4f}",
        f"- Maximum absolute relative drift: {summary['maximum_absolute_relative_drift']:.4f}",
        f"- Temporary ratio median/P90/max: {summary['temporary_ratio_median']}, {summary['temporary_ratio_p90']}, {summary['temporary_ratio_maximum']}",
        f"- Alias deltas nonzero: {summary['alias_delta_nonzero']}",
        "",
        "Family, dtype, and size breakdowns are stored in the summary JSON.",
        "",
        "This harness does not infer cross-device validity from a same-device smoke run.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-same-device-smoke", action="store_true")
    args = parser.parse_args()

    reference_env, reference_rows = load_reference(args.reference)
    calibration = frozen_calibration()
    current_env = environment()
    current_label = device_label(current_env)
    independent = current_label != FROZEN_REFERENCE_DEVICE
    if not independent and not args.allow_same_device_smoke:
        raise SystemExit("active device matches the RTX 3050 reference; use --allow-same-device-smoke only for a smoke test")
    if current_env.get("jax_version") != "0.11.0":
        raise SystemExit(f"expected JAX 0.11.0, found {current_env.get('jax_version')}")

    selected = cases()[: args.limit] if args.limit else cases()
    rows = []
    for case in selected:
        current = run_case(case, current_env)
        if current.get("status") != "ok":
            rows.append({"name": case.name, "family": case.family, "dtype": case.dtype, "status": current.get("status"), "error_message": current.get("error_message")})
            continue
        reference = reference_rows.get(case.name)
        if reference is None:
            raise SystemExit(f"reference is missing matched case {case.name}")
        rows.append(pair_case(reference, current, calibration))

    summary = summarize(rows)
    payload = {
        "status": "COMPUTATIONALLY VERIFIED" if independent else "SMOKE ONLY",
        "independent_device": independent,
        "reference_environment": reference_env,
        "current_environment": current_env,
        "frozen_calibration": calibration,
        "summary": summary,
        "cases": rows,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_prefix.with_name(args.output_prefix.name + "_summary").with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_prefix.with_name(args.output_prefix.name + "_report").with_suffix(".md"), reference_env, current_env, summary, independent)
    print(json.dumps({"status": payload["status"], "cases": summary["sample_count"], "upper_coverage": summary["upper_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
