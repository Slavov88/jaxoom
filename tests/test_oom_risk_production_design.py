from pathlib import Path
import json

ROOT = Path(__file__).parents[1]
EXP = ROOT / "experiments"


def test_design_keeps_public_api_unmodified_and_freezes_v2():
    import jaxoom

    design = json.loads((EXP / "oom_risk_production_design_2026-09-13.json").read_text())
    assert design["status"] == "DESIGN_ONLY_NO_PUBLIC_API_CHANGE"
    assert design["selected_model"] == "V2"
    assert design["model_hash"] == "9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96"
    assert design["threshold"] == 0.20
    assert not hasattr(jaxoom, "assess_oom_risk")


def test_design_replay_records_feature_and_score_provenance():
    design = json.loads((EXP / "oom_risk_production_design_2026-09-13.json").read_text())
    replay = design["replay_audit"]
    assert replay["rows"] == 224
    assert replay["model_probability_mismatches"] == 0
    assert replay["status_mismatches_at_p_fit_0.20"] == 0
    assert replay["target_compilation"] is False
    assert replay["target_execution"] is False
    assert replay["feature_mismatches_by_name"]["largest_over_budget"] == 224
    assert design["readiness_gates"]["B_feature_reproducibility_structured_adapter"] == "FAIL"
    assert design["readiness_gates"]["B_arbitrary_callable_feature_reproducibility"] == "BLOCKED"


def test_design_artifact_contains_explicit_scope_and_non_guarantee_metadata():
    model = json.loads((EXP / "oom_risk_production_model_v2_design_2026-09-13.json").read_text())
    assert model["status"] == "DESIGN_ONLY_NOT_PUBLIC"
    assert model["primary_threshold"] == 0.20
    assert model["validated_environment"]["jax"] == "0.11.0"
    assert model["validated_environment"]["jaxlib"] == "0.11.0"
    assert model["validation"]["one_sided_95_upper_bound"] == 0.01890020551196819
    assert model["validation"]["strict_sensitivity_upper_bound"] == 0.023495238866670053


def test_design_does_not_modify_existing_public_api_contract():
    import jaxoom

    public = set(jaxoom.__all__)
    assert {"estimate", "assess", "plan_batch_size", "compile_analyze", "analyze_donation"} <= public
    assert "assess_oom_risk" not in public
