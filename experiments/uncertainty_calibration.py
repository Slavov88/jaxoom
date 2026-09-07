"""Evaluate simple compiler-memory calibration strategies.

The script uses recorded CSV measurements and held-out evaluation. It does not
fit a machine-learning model and does not alter the structural estimator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METHODS = ("raw_structural", "global_ratio", "global_additive", "family_ratio", "size_bucket_ratio")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row for row in rows
        if row.get("status", "ok") == "ok"
        and int(row["static_peak_bytes"]) > 0
        and row.get("compiler_accounted_bytes") not in (None, "")
    ]


def nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def interval(method: str, row: dict[str, Any], training: list[dict[str, Any]], fallback: list[dict[str, Any]]) -> tuple[int, int, int, str]:
    static = int(row["static_peak_bytes"])
    if method == "raw_structural":
        return static, static, static, "raw structural estimate"
    source = training if training else fallback
    if method == "family_ratio":
        source = training or fallback
    if method == "size_bucket_ratio":
        source = training or fallback
    if method in {"family_ratio", "size_bucket_ratio", "global_ratio"}:
        ratios = [int(item["compiler_accounted_bytes"]) / int(item["static_peak_bytes"]) for item in source]
        lower, central, upper = (nearest_rank(ratios, p) for p in (0.10, 0.50, 0.90))
        return round(static * lower), round(static * central), round(static * upper), "nearest-rank compiler/static ratio quantiles"
    residuals = [int(item["compiler_accounted_bytes"]) - int(item["static_peak_bytes"]) for item in source]
    lower, central, upper = (nearest_rank(residuals, p) for p in (0.10, 0.50, 0.90))
    return max(0, round(static + lower)), max(0, round(static + central)), max(0, round(static + upper)), "nearest-rank additive residual quantiles"


def evaluate(rows: list[dict[str, Any]], backend: str, protocol: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        global_training = [item for j, item in enumerate(rows) if j != index]
        for method in METHODS:
            if protocol == "leave_family_out":
                global_training = [item for item in rows if item["family"] != row["family"]]
            if method == "family_ratio" and protocol == "leave_one_out":
                training = [item for j, item in enumerate(rows) if j != index and item["family"] == row["family"]]
            elif method == "size_bucket_ratio":
                training = [item for item in global_training if item.get("size_bucket") == row.get("size_bucket")]
            else:
                training = global_training
            lower, central, upper, description = interval(method, row, training, global_training)
            compiler = int(row["compiler_accounted_bytes"])
            static = int(row["static_peak_bytes"])
            budget_text = row.get("target_budget_bytes")
            budget = int(float(budget_text)) if budget_text not in (None, "", "None") else None
            predicted_safe = upper <= budget if budget is not None else None
            actual_safe = compiler <= budget if budget is not None else None
            results.append({
                "backend": backend,
                "protocol": protocol,
                "method": method,
                "name": row["name"],
                "family": row["family"],
                "size_bucket": row.get("size_bucket", ""),
                "static_peak_bytes": static,
                "compiler_accounted_bytes": compiler,
                "lower_bytes": lower,
                "central_bytes": central,
                "upper_bytes": upper,
                "interval_covered": lower <= compiler <= upper,
                "upper_covered": compiler <= upper,
                "interval_width_bytes": upper - lower,
                "interval_width_ratio": (upper - lower) / static,
                "target_budget_bytes": budget,
                "predicted_safe": predicted_safe,
                "actual_safe": actual_safe,
                "false_fit": predicted_safe is True and actual_safe is False,
                "false_oom": predicted_safe is False and actual_safe is True,
                "method_description": description,
            })
    return results


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    interval_coverage = [bool(row["interval_covered"]) for row in rows]
    upper_coverage = [bool(row["upper_covered"]) for row in rows]
    widths = [int(row["interval_width_bytes"]) for row in rows]
    width_ratios = [float(row["interval_width_ratio"]) for row in rows]
    budget_rows = [row for row in rows if row["target_budget_bytes"] is not None]
    return {
        "n": len(rows),
        "interval_coverage": sum(interval_coverage) / len(interval_coverage),
        "upper_bound_coverage": sum(upper_coverage) / len(upper_coverage),
        "upper_bound_miss_rate": 1 - sum(upper_coverage) / len(upper_coverage),
        "median_interval_width_bytes": statistics.median(widths),
        "median_interval_width_ratio": statistics.median(width_ratios),
        "budget_cases": len(budget_rows),
        "false_fit_rate": (sum(bool(row["false_fit"]) for row in budget_rows) / len(budget_rows)) if budget_rows else None,
        "false_oom_rate": (sum(bool(row["false_oom"]) for row in budget_rows) / len(budget_rows)) if budget_rows else None,
    }


def write_outputs(prefix: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path, json_path = prefix.with_suffix(".csv"), prefix.with_suffix(".json")
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration evaluation: {prefix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in results for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    json_path.write_text(json.dumps({"summary": summary, "rows": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-csv", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    all_results: list[dict[str, Any]] = []
    for backend, path in (("cpu", args.cpu_csv), ("gpu", args.gpu_csv)):
        rows = read_rows(path)
        for protocol in ("leave_one_out", "leave_family_out"):
            all_results.extend(evaluate(rows, backend, protocol))
    grouped: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        for backend in ("cpu", "gpu"):
            for protocol in ("leave_one_out", "leave_family_out"):
                key = f"{backend}:{protocol}:{method}"
                grouped[key] = summarize([row for row in all_results if row["backend"] == backend and row["protocol"] == protocol and row["method"] == method])
    write_outputs(args.output_prefix, all_results, grouped)
    print(json.dumps(grouped, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
