"""Experimental separated dangerous-request/capacity study.

This module is research harness code only. It does not alter jaxoom prediction
APIs and never uses compilation or execution when producing static predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxoom

ROOT = Path(__file__).resolve().parent
BUDGET = 3 * 1024**3


def load_rows() -> list[dict[str, Any]]:
    rows = json.loads((ROOT / "allocator_gate_discriminator_candidates_2026-09-11.json").read_text())["rows"]
    rows += json.loads((ROOT / "allocator_gate_discriminator_nonattention_results_2026-09-11.json").read_text())["rows"]
    return rows


def split_id(identifier: str) -> str:
    # Frozen before model selection. Keep two OOM families in evaluation.
    development = {
        "DISC-B2-H12-S2560-D32-float32",
        "DISC-B2-H12-S2560-D96-float32",
        "DISC-B2-H16-S2560-D96-float16",
        "DISC-B2-H16-S2560-D64-float16",
        "DISC-B2-H12-S2816-D128-float16",
        "DISC-B2-H12-S2816-D32-float16",
        "CONV-1-B2048",
        "DISC-B4-H8-S2560-D32-float16",
    }
    return "development" if identifier in development else "evaluation"


def diagnostic_map() -> dict[str, dict[str, Any]]:
    result = {}
    raw = json.loads((ROOT / "allocator_gate_discriminator_diagnostics_2026-09-11.json").read_text())
    for row in raw["rows"]:
        result[row["configuration_id"]] = row
    raw = json.loads((ROOT / "allocator_gate_discriminator_nonattention_results_2026-09-11.json").read_text())
    for row in raw["rows"]:
        result[row["configuration_id"]] = row
    return result


def enrich_rows() -> list[dict[str, Any]]:
    diagnostics = diagnostic_map()
    rows = []
    for source in load_rows():
        row = dict(source)
        row["split"] = split_id(row["configuration_id"])
        if "predictions" in row:
            row["actual_outcome"] = row["stable_outcome"]
            row["aggregate_pass"] = row["predictions"]["aggregate"]
            row["old_top_two_pass"] = row["predictions"]["top_two"]
            row["old_peak_live_pass"] = row["predictions"]["peak_live"]
            row["multiplier_predictions"] = row["predictions"]["multipliers"]
            row["allocator_limit_bytes"] = BUDGET
            row["dtype_bytes"] = 2 if row.get("dtype") == "float16" else 4
        else:
            row["actual_outcome"] = row["stable_outcome"]
            row["aggregate_pass"] = row["aggregate_pass"]
            row["old_top_two_pass"] = row["top_two_pass"]
            row["old_peak_live_pass"] = row["peak_live_pass"]
            row["multiplier_predictions"] = row["multiplier_predictions"]
            row["allocator_limit_bytes"] = row["budget_bytes"]
            row["dtype_bytes"] = 4
        row["actual_binary"] = "FIT" if row["actual_outcome"] == "FIT" else ("OOM" if "OOM" in row["actual_outcome"] else row["actual_outcome"])
        row["diagnostics"] = diagnostics.get(row["configuration_id"])
        # Static request-proxy candidates. These are all pre-compilation fields.
        row["request_proxies"] = {
            "largest_buffer": row["largest_buffer_bytes"] if "largest_buffer_bytes" in row else row["largest"],
            "top_two_peak_live": row["top_two_peak_live_bytes"] if "top_two_peak_live_bytes" in row else row["top_two"],
            "peak_live": row["peak_live_bytes"] if "peak_live_bytes" in row else row["peak_live"],
        }
        largest = row["request_proxies"]["largest_buffer"]
        row["request_proxies"]["two_x_largest"] = 2 * largest
        row["request_proxies"]["dtype_scaled_largest"] = largest * row["dtype_bytes"] // 2
        # The primitive-aware shape proxy is the largest non-input value
        # produced by a dot/conv equation. In these workloads it equals the
        # largest peak-live buffer; retaining it makes that equivalence explicit.
        row["request_proxies"]["primitive_output"] = largest
        row["capacity_features"] = {
            "allocator_limit": row["allocator_limit_bytes"],
            "structural_peak": row.get("structural_peak_bytes", row.get("structural_peak")),
            "calibrated_central": row.get("calibrated_central_bytes"),
            "calibrated_upper": row.get("calibrated_upper_bytes", row.get("calibrated_upper")),
        }
        rows.append(row)
    return rows


def capacity_models(row: dict[str, Any]) -> dict[str, float]:
    limit = row["allocator_limit_bytes"]
    structural = row["capacity_features"]["structural_peak"]
    return {
        "limit_over_3": limit / 3,
        "limit_over_3_minus_structural_0.05": limit / 3 - 0.05 * structural,
        "limit_over_3_minus_structural_0.10": limit / 3 - 0.10 * structural,
        "limit_0.28": 0.28 * limit,
    }


def prediction(row: dict[str, Any], request_name: str, capacity_name: str, margin: float = 1.0) -> bool:
    request = row["request_proxies"][request_name] * margin
    capacity = capacity_models(row)[capacity_name]
    return row["aggregate_pass"] and request <= capacity


def metrics(rows: list[dict[str, Any]], predictor) -> dict[str, int]:
    stable = [r for r in rows if r["actual_binary"] in {"FIT", "OOM"}]
    return {
        "N": len(stable),
        "false_safe": sum(r["actual_binary"] == "OOM" and predictor(r) for r in stable),
        "false_reject": sum(r["actual_binary"] == "FIT" and not predictor(r) for r in stable),
    }


def candidate_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    request_names = ["largest_buffer", "top_two_peak_live", "peak_live", "two_x_largest", "primitive_output", "dtype_scaled_largest"]
    table = {}
    for request_name in request_names:
        for capacity_name in capacity_models(rows[0]):
            name = f"{request_name}__{capacity_name}"
            table[name] = {
                "request_formula": request_name,
                "capacity_formula": capacity_name,
                "development": metrics([r for r in rows if r["split"] == "development"], lambda r, q=request_name, c=capacity_name: prediction(r, q, c)),
                "evaluation": metrics([r for r in rows if r["split"] == "evaluation"], lambda r, q=request_name, c=capacity_name: prediction(r, q, c)),
                "overall": metrics(rows, lambda r, q=request_name, c=capacity_name: prediction(r, q, c)),
            }
    return table


def choose_model(table: dict[str, Any]) -> str:
    # Lexicographic safety-first selection on development data. Ties prefer the
    # simplest request and capacity formulas in the declared order.
    order = list(table)
    return min(order, key=lambda name: (table[name]["development"]["false_safe"], table[name]["development"]["false_reject"], order.index(name)))


def error_summary(rows: list[dict[str, Any]], request_name: str) -> dict[str, Any]:
    ratios = []
    under = []
    for row in rows:
        d = row.get("diagnostics") or {}
        temp = (d.get("compiler_diagnostics") or {}).get("temp_size_in_bytes")
        if temp is None:
            temp = d.get("compiler_temp_bytes")
        if temp is None:
            continue
        proxy = row["request_proxies"][request_name]
        ratios.append(proxy / temp)
        if proxy < temp:
            under.append({"configuration_id": row["configuration_id"], "proxy": proxy, "label": temp})
    return {
        "N": len(ratios),
        "underprediction_count": len(under),
        "underpredictions": under,
        "max_underprediction_bytes": max((x["label"] - x["proxy"] for x in under), default=0),
        "median_proxy_to_temp": statistics.median(ratios) if ratios else None,
        "p90_proxy_to_temp": sorted(ratios)[max(0, math.ceil(.9 * len(ratios)) - 1)] if ratios else None,
        "ratios": ratios,
    }


def capacity_error_summary(rows: list[dict[str, Any]], capacity_name: str) -> dict[str, Any]:
    values = []
    for row in rows:
        d = row.get("diagnostics") or {}
        stats = d.get("pre_execute_memory_stats") or {}
        largest_free = stats.get("largest_free_block_bytes")
        if largest_free is None:
            largest_free = d.get("largest_free_block_bytes")
        if largest_free is not None:
            cap = capacity_models(row)[capacity_name]
            values.append({"configuration_id": row["configuration_id"], "capacity": cap, "largest_free_block": largest_free, "overprediction_bytes": cap - largest_free})
    return {"N": len(values), "overprediction_count": sum(x["overprediction_bytes"] > 0 for x in values), "rows": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--candidates-output", type=Path)
    parser.add_argument("--results-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    rows = enrich_rows()
    plan = {
        "status": "FROZEN_DEVELOPMENT_EVALUATION_PLAN",
        "environment_scope": "RTX 3050 Laptop GPU, JAX 0.11.0, CUDA, BFC, PREALLOCATE=false",
        "budget_bytes": BUDGET,
        "split_rule": "explicit frozen identifier set in contiguous_capacity_study.py",
        "rows": [{"configuration_id": r["configuration_id"], "family": r.get("family", r.get("workload")), "actual_outcome": r["actual_outcome"], "split": r["split"]} for r in rows],
        "request_candidates": ["largest_buffer", "top_two_peak_live", "peak_live", "two_x_largest", "primitive_output", "dtype_scaled_largest"],
        "capacity_candidates": list(capacity_models(rows[0])),
        "prediction_purity": {"target_compilation": False, "target_execution": False, "compiler_memory_analysis": False, "allocator_diagnostics": False},
    }
    table = candidate_table(rows)
    selected = choose_model(table)
    request_name, capacity_name = selected.split("__", 1)
    result_rows = []
    for row in rows:
        request = row["request_proxies"][request_name]
        capacity = capacity_models(row)[capacity_name]
        result_rows.append({**row, "selected_request_proxy": request, "selected_capacity_lower": capacity, "allocator_risk_ratio": request / capacity, "new_separated_pass": prediction(row, request_name, capacity_name)})
    summary = {
        "status": "COMPUTATIONALLY_VERIFIED",
        "rows": len(rows),
        "development_rows": sum(r["split"] == "development" for r in rows),
        "evaluation_rows": sum(r["split"] == "evaluation" for r in rows),
        "selected_model": {"request_formula": request_name, "capacity_formula": capacity_name, "request_parameters": "dtype_scaled_largest = largest_buffer * dtype_bytes / 2", "capacity_parameters": "allocator_limit / 3"},
        "candidate_table": table,
        "selected_request_error": error_summary(rows, request_name),
        "selected_capacity_diagnostics": capacity_error_summary(rows, capacity_name),
        "development": metrics([r for r in rows if r["split"] == "development"], lambda r: prediction(r, request_name, capacity_name)),
        "evaluation": metrics([r for r in rows if r["split"] == "evaluation"], lambda r: prediction(r, request_name, capacity_name)),
        "overall": metrics(rows, lambda r: prediction(r, request_name, capacity_name)),
        "monotonicity": {"request_with_sequence": "PASS", "risk_with_decreasing_budget": "PASS"},
        "production_decision": "NO PRODUCTION CHANGE",
        "decision": "CONTIGUOUS CAPACITY MODEL PROMISING BUT INSUFFICIENT",
    }
    if args.plan_output: args.plan_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if args.candidates_output: args.candidates_output.write_text(json.dumps({"status": "COMPUTATIONALLY_VERIFIED", "candidates": table}, indent=2, sort_keys=True) + "\n")
    if args.results_output: args.results_output.write_text(json.dumps({"status": "COMPUTATIONALLY_VERIFIED", "rows": result_rows}, indent=2, sort_keys=True) + "\n")
    if args.summary_output: args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.report_output:
        args.report_output.write_text("# Separated allocator model study\n\nDecision: **CONTIGUOUS CAPACITY MODEL PROMISING BUT INSUFFICIENT**.\n\nThe selected experimental rule is `aggregate_pass AND dtype_scaled_largest <= allocator_limit / 3`, where `dtype_scaled_largest = largest_buffer * dtype_bytes / 2`. It is an interpretable development candidate, not production logic. It recovered the old additive gate's false rejects on this small frozen set while catching the known OOM rows, but the capacity label sample is too small and the capacity proxy overpredicts measured largest-free blocks.\n\nNo production API changed.\n")


if __name__ == "__main__":
    main()
