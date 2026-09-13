from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).parents[1]
EXP = ROOT / "experiments"


def test_batch2_models_and_thresholds_are_frozen():
    manifest = json.loads((EXP / "oom_risk_heldout2_validation_manifest_2026-09-13.json").read_text())
    assert manifest["primary_model"] == "V2"
    assert manifest["thresholds"]["primary"] == 0.20
    assert manifest["selection_independent_of_model_predictions"] is True
    assert manifest["models"]["v2"]["model_hash"] == "9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96"
    assert manifest["models"]["v3"]["model_hash"] == "7663490267c552569c5b9f1b93cf05dcc113282317c899f55c8e30e06f4b2da8"


def test_batch2_panels_are_outcome_free_and_disjoint():
    prior_plan = json.loads((EXP / "oom_risk_heldout_plan_2026-09-13.json").read_text())
    primary = json.loads((EXP / "oom_risk_heldout2_plan_2026-09-13.json").read_text())
    conv = json.loads((EXP / "oom_risk_heldout2_conv_stress_plan_2026-09-13.json").read_text())
    prior = {x["workload_group_id"] for x in prior_plan["selected_candidates"]}
    pids = {x["workload_group_id"] for x in primary["selected_candidates"]}
    cids = {x["workload_group_id"] for x in conv["selected_candidates"]}
    assert primary["selection_independent_of_model_predictions"] is True
    assert conv["selection_independent_of_model_predictions"] is True
    assert primary["overlap_audit"]["prior_group_overlaps"] == 0
    assert conv["overlap_audit"]["prior_group_overlaps"] == 0
    assert not pids & prior
    assert not cids & prior
    assert not pids & cids
    assert all("probability" not in key for row in primary["selected_candidates"] for key in row)
    assert all("outcome" not in row for row in primary["selected_candidates"])


def test_batch2_prediction_hashes_reference_frozen_plans():
    for plan_name, pred_name in (
        ("oom_risk_heldout2_plan_2026-09-13.json", "oom_risk_heldout2_predictions_2026-09-13.json"),
        ("oom_risk_heldout2_conv_stress_plan_2026-09-13.json", "oom_risk_heldout2_conv_stress_predictions_2026-09-13.json"),
    ):
        pred = json.loads((EXP / pred_name).read_text())
        assert pred["plan_hash"] == hashlib.sha256((EXP / plan_name).read_bytes()).hexdigest()
        assert pred["outcomes_excluded"] is True


def test_batch2_outcome_classes_and_pooled_provenance():
    allowed = {"FIT", "EXECUTION_OOM", "COMPILE_OOM", "COMPILE_TIMEOUT", "EXECUTION_TIMEOUT", "INITIALIZATION_OOM", "OTHER_FAILURE", "UNSTABLE"}
    primary = json.loads((EXP / "oom_risk_heldout2_results_2026-09-13.json").read_text())["results"]
    stress = json.loads((EXP / "oom_risk_heldout2_conv_stress_results_2026-09-13.json").read_text())["results"]
    assert len(primary) == 110
    assert len(stress) == 24
    assert {x["outcome"] for x in primary} <= allowed
    assert {x["outcome"] for x in stress} <= allowed
    summary = json.loads((EXP / "oom_risk_heldout2_summary_2026-09-13.json").read_text())
    assert summary["pooled"]["batch1_eligible"] == 83
    assert summary["pooled"]["batch2_eligible"] == 107
    assert summary["pooled"]["eligible"] == 190
    assert "near_duplicate_sensitivity" in summary["pooled"]
