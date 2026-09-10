"""Experimental allocator-aware gate study; not production prediction logic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import jaxoom

BUDGET = 3 * 1024**3
SEQUENCES = (2048, 3072, 4096, 4608, 5120)


def attention(q, k, v):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def arguments(sequence: int):
    return tuple(jax.ShapeDtypeStruct((1, 8, sequence, 64), jnp.float32) for _ in range(3))


def large_allocation_proxy(report: Any, count: int = 2) -> int:
    """Sum the largest `count` buffers live at the static peak."""
    return sum(sorted((buffer.nbytes for buffer in report.peak.live_buffers), reverse=True)[:count])


def evaluate_gate(report: Any, budget: int, candidate: str) -> dict[str, Any]:
    interval = jaxoom.calibrate(report, backend="gpu", jax_version=jax.__version__)
    headroom = budget - interval.upper_bytes
    proxy = {
        "aggregate": 0,
        "largest_buffer": large_allocation_proxy(report, 1),
        "top_two_live": large_allocation_proxy(report, 2),
        "peak_live": report.peak.live_bytes,
    }[candidate]
    aggregate_pass = interval.upper_bytes <= budget
    allocator_pass = candidate == "aggregate" or proxy <= headroom
    return {"candidate": candidate, "aggregate_pass": aggregate_pass, "allocator_pass": allocator_pass, "passes": aggregate_pass and allocator_pass, "large_allocation_proxy_bytes": proxy, "safe_capacity_bytes": max(0, headroom), "upper_bytes": interval.upper_bytes, "structural_bytes": report.estimated_peak_bytes}


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for sequence in SEQUENCES:
        report = jaxoom.estimate(attention, *arguments(sequence))
        actual = "FIT" if sequence in (2048, 3072, 4096) else "EXECUTION_OOM"
        rows.append({"sequence": sequence, "actual_outcome": actual, "candidates": {name: evaluate_gate(report, BUDGET, name) for name in ("aggregate", "largest_buffer", "top_two_live", "peak_live")}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows()
    split = {"development": [2048, 3072, 5120], "held_out": [4096, 4608]}
    metrics = {}
    for candidate in ("aggregate", "largest_buffer", "top_two_live", "peak_live"):
        values = []
        for split_name, sequences in split.items():
            selected = [row for row in rows if row["sequence"] in sequences]
            false_safe = sum(row["actual_outcome"] == "EXECUTION_OOM" and row["candidates"][candidate]["passes"] for row in selected)
            false_reject = sum(row["actual_outcome"] == "FIT" and not row["candidates"][candidate]["passes"] for row in selected)
            values.append((split_name, {"false_safe": false_safe, "false_reject": false_reject, "rows": len(selected)}))
        metrics[candidate] = dict(values)
    control_rows = []
    for workload_id, filename in (("MLP-2", "validation_MLP-2_2026-09-09.json"), ("TRAIN-1", "validation_TRAIN-1_2026-09-09.json"), ("CONV-1", "validation_CONV-1_2026-09-09.json")):
        source = json.loads((Path(__file__).parent / filename).read_text())
        source_row = source["rows"][0]
        workload = __import__("planner_boundary_validation_v1", fromlist=["workloads"]).workloads()[workload_id]
        report = jaxoom.estimate(workload.fn, *workload.args(source_row["planned_batch"], False))
        control_rows.append({"workload_id": workload_id, "actual_outcome": "FIT", "planned_batch": source_row["planned_batch"], "candidates": {name: evaluate_gate(report, BUDGET, name) for name in ("aggregate", "largest_buffer", "top_two_live", "peak_live")}})
    control_metrics = {}
    for candidate in ("aggregate", "largest_buffer", "top_two_live", "peak_live"):
        control_metrics[candidate] = {"false_safe": 0, "false_reject": sum(not row["candidates"][candidate]["passes"] for row in control_rows), "rows": len(control_rows)}
    candidate_formulas = {"aggregate": "calibrated_upper <= budget", "largest_buffer": "upper + largest_peak_live_buffer <= budget", "top_two_live": "upper + sum(two_largest_peak_live_buffers) <= budget", "peak_live": "upper + peak_live_bytes <= budget"}
    payload = {"status": "OBSERVED", "candidate_formulas": candidate_formulas, "environment": {"backend": jax.default_backend(), "jax_version": jax.__version__, "device": str(jax.devices()[0])}, "budget_bytes": BUDGET, "split": split, "rows": rows, "control_rows": control_rows, "candidate_metrics": metrics, "control_metrics": control_metrics, "production_decision": "NO_CHANGE"}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
