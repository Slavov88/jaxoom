"""Inspectable experiment-only ranking for T4 threshold candidates."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FAMILY_PRIORITY = {"training": 7, "mlp": 6, "autodiff": 5, "transformer": 4, "convolution": 3, "matmul": 2, "attention": 1}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def empirical_envelopes(screening: list[dict[str, Any]], thresholds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_id = {row["configuration_id"]: row for row in screening}
    ratios: dict[str, list[float]] = defaultdict(list)
    for row in thresholds:
        static = by_id.get(row["configuration_id"])
        upper = row.get("required_allocator_upper_bytes", row.get("known_fit_capacity"))
        if static and upper and static.get("calibrated_upper_bytes"):
            ratios[row["family"]].append(upper / static["calibrated_upper_bytes"])
    all_ratios = [value for values in ratios.values() for value in values]
    if not all_ratios:
        raise ValueError("no threshold/static pairs available")
    global_low, global_high = min(all_ratios), max(all_ratios)
    envelopes = {}
    for family in set(by_id.get(row["configuration_id"], {}).get("family") for row in thresholds):
        values = ratios.get(family, all_ratios)
        envelopes[family] = {"ratio_low": min(values) * 0.85, "ratio_high": max(values) * 1.15, "observations": len(values)}
    envelopes["__global__"] = {"ratio_low": global_low * 0.85, "ratio_high": global_high * 1.15, "observations": len(all_ratios)}
    return envelopes


def classify_candidate(row: dict[str, Any], envelope: dict[str, float], minimum: int, maximum: int) -> tuple[str, float, float]:
    estimate = row.get("calibrated_upper_bytes")
    if not estimate:
        return "UNCERTAIN", 0.0, 0.0
    low, high = estimate * envelope["ratio_low"], estimate * envelope["ratio_high"]
    if high < minimum * 0.90:
        return "TOO_SMALL", low, high
    if low > maximum * 1.10:
        return "TOO_LARGE", low, high
    return "LIKELY_THRESHOLD", low, high


def rank(screening: list[dict[str, Any]], thresholds: list[dict[str, Any]], minimum: int, maximum: int, exclude: set[str], limit: int) -> dict[str, Any]:
    envelopes = empirical_envelopes(screening, thresholds)
    rows = []
    center = (minimum + maximum) / 2
    for row in screening:
        if row["configuration_id"] in exclude:
            continue
        family = row["family"]
        envelope = envelopes.get(family, envelopes["__global__"])
        status, low, high = classify_candidate(row, envelope, minimum, maximum)
        if status != "LIKELY_THRESHOLD":
            final_status = status
        else:
            midpoint = (low + high) / 2
            distance = abs(math.log(max(midpoint, 1) / center))
            dtype_bonus = 1.0 if row["dtype"] == "float16" else 0.0
            score = FAMILY_PRIORITY.get(family, 0) * 0.25 + dtype_bonus + 1.0 / (1.0 + distance) - row.get("equation_count", 0) / 10000
            final_status = status
        rows.append({**row, "target_status": final_status, "predicted_required_low_bytes": low, "predicted_required_high_bytes": high, "target_score": score if final_status == "LIKELY_THRESHOLD" else None, "envelope": envelope})
    eligible = [row for row in rows if row["target_status"] == "LIKELY_THRESHOLD"]
    eligible.sort(key=lambda row: row["target_score"] or -1, reverse=True)
    selected = []
    family_counts: Counter[str] = Counter()
    # First reserve one high-scoring candidate per underrepresented family, then fill by score.
    for row in eligible:
        if family_counts[row["family"]] == 0 and len(selected) < limit:
            selected.append(row); family_counts[row["family"]] += 1
    for row in eligible:
        if row in selected or len(selected) >= limit:
            continue
        selected.append(row); family_counts[row["family"]] += 1
    return {"status": "OBSERVED", "thresholdable_capacity": {"minimum_bytes": minimum, "maximum_bytes": maximum}, "envelopes": envelopes, "rows": rows, "selected": selected, "selected_family_counts": dict(family_counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--fraction-map", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    screening = load_json(args.screening)["rows"]
    thresholds_data = load_json(args.thresholds)
    thresholds = thresholds_data.get("useful_thresholds", thresholds_data.get("rows", []))
    fraction_rows = [row for row in load_json(args.fraction_map)["rows"] if row.get("bytes_limit") is not None]
    exclude = set(load_json(args.exclude_ids))
    result = rank(screening, thresholds, min(row["bytes_limit"] for row in fraction_rows), max(row["bytes_limit"] for row in fraction_rows), exclude, args.limit)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(result["selected"]), "families": result["selected_family_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
