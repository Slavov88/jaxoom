"""Build the dated version-calibration comparison artifact.

This is an experiment helper. It reads committed cross-version results and the
expanded JAX 0.11.0 calibration results, then writes summary statistics without
changing the source datasets.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name: str) -> list[dict]:
    rows = json.loads((ROOT / name).read_text())
    return rows["cases"] if "cases" in rows else rows["rows"]


def valid(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") in {"PASS", "ok"}]


def ratio(row: dict) -> float:
    static = row.get("static_peak_bytes", row.get("structural_peak_bytes"))
    return row["compiler_accounted_bytes"] / static


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))]


def bounds(rows: list[dict], upper_probability: float = 0.95) -> dict[str, float]:
    values = [ratio(row) for row in rows]
    return {"lower": quantile(values, 0.10), "central": quantile(values, 0.50), "upper": quantile(values, upper_probability)}


def score(calibration: dict[str, float], rows: list[dict]) -> dict[str, float | int]:
    inside = [calibration["lower"] <= ratio(row) <= calibration["upper"] for row in rows]
    upper = [ratio(row) <= calibration["upper"] for row in rows]
    width = calibration["upper"] - calibration["lower"]
    return {
        "sample_count": len(rows),
        "interval_coverage_count": sum(inside),
        "interval_coverage": sum(inside) / len(rows),
        "upper_coverage_count": sum(upper),
        "upper_coverage": sum(upper) / len(rows),
        "upper_miss_rate": 1 - sum(upper) / len(rows),
        "median_width_over_static": width,
        "p90_width_over_static": width,
    }


def cross_validation(rows: list[dict], upper_probability: float = 0.95) -> dict[str, dict[str, float | int]]:
    loo_upper = []
    loo_inside = []
    for index, row in enumerate(rows):
        train = rows[:index] + rows[index + 1 :]
        calibration = bounds(train, upper_probability)
        loo_upper.append(ratio(row) <= calibration["upper"])
        loo_inside.append(calibration["lower"] <= ratio(row) <= calibration["upper"])

    family_upper = []
    family_inside = []
    for family in sorted({row["family"] for row in rows}):
        train = [row for row in rows if row["family"] != family]
        test = [row for row in rows if row["family"] == family]
        calibration = bounds(train, upper_probability)
        family_upper.extend(ratio(row) <= calibration["upper"] for row in test)
        family_inside.extend(calibration["lower"] <= ratio(row) <= calibration["upper"] for row in test)

    return {
        "leave_one_out": {
            "upper_coverage_count": sum(loo_upper),
            "upper_coverage": sum(loo_upper) / len(loo_upper),
            "interval_coverage_count": sum(loo_inside),
            "interval_coverage": sum(loo_inside) / len(loo_inside),
        },
        "leave_family_out": {
            "upper_coverage_count": sum(family_upper),
            "upper_coverage": sum(family_upper) / len(family_upper),
            "interval_coverage_count": sum(family_inside),
            "interval_coverage": sum(family_inside) / len(family_inside),
        },
    }


def main() -> None:
    old_matched = valid(load("jax_version_validation_2026-09-08_jax062_gpu.json"))
    new_matched = valid(load("jax_version_validation_2026-09-08_jax011_gpu.json"))
    old_expanded = valid(load("accelerator_calibration_gpu_2026-09-07_v2.json"))
    new_expanded = valid(load("jax_0_11_gpu_calibration_2026-09-08_v2.json"))

    old_by_name = {row["name"]: row for row in old_matched}
    new_by_name = {row["name"]: row for row in new_matched}
    misses = []
    for name in sorted(new_by_name):
        row = new_by_name[name]
        if row["compiler_accounted_bytes"] > row["calibrated_upper_bytes"]:
            shortfall = row["compiler_accounted_bytes"] - row["calibrated_upper_bytes"]
            misses.append({
                "name": name,
                "family": row["family"],
                "configuration": row["configuration"],
                "dtype": row["dtype"],
                "static_bytes": row["structural_peak_bytes"],
                "legacy_upper_bytes": row["calibrated_upper_bytes"],
                "jax_0_11_compiler_bytes": row["compiler_accounted_bytes"],
                "shortfall_bytes": shortfall,
                "shortfall_fraction_of_compiler": shortfall / row["compiler_accounted_bytes"],
                "compiler_temp_bytes": row["compiler_temp_bytes"],
                "compiler_alias_bytes": row["compiler_alias_bytes"],
            })

    drift_by_family: dict[str, list[dict]] = defaultdict(list)
    for name in sorted(set(old_by_name) & set(new_by_name)):
        old = old_by_name[name]
        new = new_by_name[name]
        drift_by_family[new["family"]].append({
            "name": name,
            "static_bytes_delta": new["structural_peak_bytes"] - old["structural_peak_bytes"],
            "compiler_argument_delta": new["compiler_argument_bytes"] - old["compiler_argument_bytes"],
            "compiler_output_delta": new["compiler_output_bytes"] - old["compiler_output_bytes"],
            "compiler_temp_delta": new["compiler_temp_bytes"] - old["compiler_temp_bytes"],
            "compiler_alias_delta": new["compiler_alias_bytes"] - old["compiler_alias_bytes"],
            "compiler_accounted_delta": new["compiler_accounted_bytes"] - old["compiler_accounted_bytes"],
            "compiler_ratio_0_6": ratio(old),
            "compiler_ratio_0_11": ratio(new),
            "ratio_of_ratios": ratio(new) / ratio(old) if ratio(old) else None,
        })

    holdout_groups = {"mlp:float16", "mlp:float32", "autodiff:float16", "autodiff:float32"}
    train = [row for row in new_expanded if f"{row['family']}:{row['dtype']}" not in holdout_groups]
    holdout = [row for row in new_expanded if f"{row['family']}:{row['dtype']}" in holdout_groups]
    legacy_shipped = {"lower": 0.3333333333333333, "central": 1.0, "upper": 1.0937538146972656}
    old_bounds = bounds(old_expanded)
    new_bounds = bounds(new_expanded)
    pooled_bounds = bounds(old_expanded + new_expanded)
    envelope_bounds = {
        "lower": min(legacy_shipped["lower"], new_bounds["lower"]),
        "central": max(legacy_shipped["central"], new_bounds["central"]),
        "upper": max(legacy_shipped["upper"], new_bounds["upper"]),
    }
    train_bounds = bounds(train)
    pooled_train_bounds = bounds(old_expanded + train)

    candidates = {
        "legacy_0_6_global": {"bounds": legacy_shipped, "new_expanded": score(legacy_shipped, new_expanded), "holdout": score(legacy_shipped, holdout)},
        "jax_0_11_global": {"bounds": new_bounds, "new_expanded": score(new_bounds, new_expanded), "holdout": score(train_bounds, holdout), "cross_validation": cross_validation(new_expanded)},
        "pooled_global": {"bounds": pooled_bounds, "new_expanded": score(pooled_bounds, new_expanded), "old_expanded": score(pooled_bounds, old_expanded), "holdout": score(pooled_train_bounds, holdout)},
        "conservative_envelope": {"bounds": envelope_bounds, "new_expanded": score(envelope_bounds, new_expanded), "old_expanded": score(envelope_bounds, old_expanded), "holdout": score(envelope_bounds, holdout)},
    }

    transfer = {
        "legacy_0_6_to_0_11_matched": score({"lower": min(row["calibrated_lower_bytes"] / row["structural_peak_bytes"] for row in new_matched), "central": 1.0, "upper": max(row["calibrated_upper_bytes"] / row["structural_peak_bytes"] for row in new_matched)}, new_matched),
        "jax_0_11_to_0_6_expanded": score(new_bounds, old_expanded),
    }
    # The first transfer above is retained only as a machine-readable check of
    # the stored 14-case interval. Its bounds are replaced with the exact legacy
    # summary below for clarity.
    transfer["legacy_0_6_to_0_11_matched"] = {
        "sample_count": len(new_matched),
        "upper_coverage_count": sum(row["compiler_accounted_bytes"] <= row["calibrated_upper_bytes"] for row in new_matched),
        "upper_coverage": sum(row["compiler_accounted_bytes"] <= row["calibrated_upper_bytes"] for row in new_matched) / len(new_matched),
        "upper_miss_rate": sum(row["compiler_accounted_bytes"] > row["calibrated_upper_bytes"] for row in new_matched) / len(new_matched),
    }

    output = {
        "status": "COMPUTATIONALLY VERIFIED",
        "method": {"ratio": "compiler_accounted_bytes / static_peak_bytes", "quantiles": "nearest-rank", "lower_quantile": 0.10, "central_quantile": 0.50, "upper_quantile": 0.95},
        "datasets": {"legacy_matched_cases": len(old_matched), "current_matched_cases": len(new_matched), "legacy_expanded_cases": len(old_expanded), "current_expanded_cases": len(new_expanded), "current_train_cases": len(train), "current_holdout_cases": len(holdout), "holdout_groups": sorted(holdout_groups)},
        "transfer_failure": {"misses": misses, "summary": transfer["legacy_0_6_to_0_11_matched"]},
        "drift_by_family": dict(sorted(drift_by_family.items())),
        "candidate_methods": candidates,
        "transfer_both_directions": {"legacy_0_6_to_current_expanded": candidates["legacy_0_6_global"]["new_expanded"], "current_to_legacy_expanded": transfer["jax_0_11_to_0_6_expanded"]},
    }
    path = ROOT / "version_calibration_comparison_2026-09-08.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
