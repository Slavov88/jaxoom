from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).parents[1]
EXP = ROOT / "experiments"


def test_panel_is_model_independent_and_has_no_training_overlap():
    plan = json.loads((EXP / "oom_risk_heldout_plan_2026-09-13.json").read_text())
    assert plan["status"] == "FROZEN_BEFORE_SCORING"
    assert plan["selection_independent_of_model_predictions"] is True
    assert plan["outcomes_excluded"] is True
    assert not any("probability" in key for row in plan["selected_candidates"] for key in row)
    assert plan["overlap_audit"]["training_group_overlaps"] == 0
    assert plan["overlap_audit"]["training_fingerprint_overlaps"] == 0
    assert len(plan["selected_candidates"]) == 90


def test_predictions_are_frozen_after_panel_and_before_outcomes():
    plan = json.loads((EXP / "oom_risk_heldout_plan_2026-09-13.json").read_text())
    pred = json.loads((EXP / "oom_risk_heldout_predictions_2026-09-13.json").read_text())
    assert pred["status"] == "FROZEN_BEFORE_RUNTIME"
    assert pred["plan_hash"] == hashlib.sha256((EXP / "oom_risk_heldout_plan_2026-09-13.json").read_bytes()).hexdigest()
    assert pred["outcomes_excluded"] is True
    assert len(pred["predictions"]) == len(plan["selected_candidates"])
    assert all("outcome" not in row for row in pred["predictions"])


def test_risk_coverage_and_exact_upper_bound():
    summary = json.loads((EXP / "oom_risk_heldout_summary_2026-09-13.json").read_text())
    assert summary["heldout"]["eligible_rows"] == 83
    assert summary["operating_points"]["v2"]["0.2"]["false_safe_n"] == 0
    assert summary["operating_points"]["v2"]["0.2"]["likely_fit_n"] == 70
    assert summary["monotonicity"]["v2"]["violations"] == 0
    assert summary["monotonicity"]["v3"]["violations"] == 0


def test_runtime_outcomes_remain_separate_from_execution_oom_eligibility():
    results = json.loads((EXP / "oom_risk_heldout_results_2026-09-13.json").read_text())["results"]
    assert len(results) == 90
    assert {r["outcome"] for r in results} <= {"FIT", "EXECUTION_OOM", "COMPILE_OOM", "COMPILE_TIMEOUT", "EXECUTION_TIMEOUT", "INITIALIZATION_OOM", "OTHER_FAILURE", "UNSTABLE"}
    eligible = [r for r in results if r["outcome"] in ("FIT", "EXECUTION_OOM")]
    assert len(eligible) == 83
    assert not any(r["outcome"] == "COMPILE_OOM" for r in eligible)
