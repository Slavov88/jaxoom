"""Quantify the explicit-budget threshold for the fixed ATT-2 shape."""
from __future__ import annotations

import json
from pathlib import Path

import jax

import jaxoom
from planner_boundary_validation_v1 import workloads


def main() -> None:
    workload = workloads()["ATT-2"]
    budgets = [int(value * 1024**3) for value in (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0)]
    rows = []
    for budget in budgets:
        plan = jaxoom.plan_batch_size(
            workload.fn,
            lambda batch: workload.args(batch, False),
            memory_limit=budget,
            min_batch_size=1,
            max_batch_size=2,
            max_evaluations=4,
        )
        first = next((trial for trial in plan.trials if trial.batch_size == 1), None)
        rows.append({
            "budget_bytes": budget,
            "recommended_batch": plan.recommended_batch_size,
            "status": plan.status,
            "trial_upper_bytes": first.assessment.interval.upper_bytes if first and first.assessment else None,
            "trial_central_bytes": first.assessment.interval.central_bytes if first and first.assessment else None,
            "trial_risk": first.assessment.risk.value if first and first.assessment and first.assessment.risk else None,
        })
    Path("experiments/att2_false_safe/att2_explicit_budget_curve_v1.json").write_text(json.dumps({"status": "OBSERVED", "jax_version": jax.__version__, "backend": jax.default_backend(), "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
