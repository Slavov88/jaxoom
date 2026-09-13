from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = load("oom_risk_model_v3", ROOT / "experiments" / "oom_risk_model_v3.py")


def test_compile_filter_and_prospective_plan_are_frozen():
    filt = json.loads((ROOT / "experiments/compile_feasibility_filter_v1_2026-09-13.json").read_text())
    plan = json.loads((ROOT / "experiments/oom_risk_expansion_plan_v3_2026-09-13.json").read_text())
    assert filt["selected_method"] == "static_shape_rule"
    assert filt["historical_grouped_audit"]["compile_failure_recall_at_rule"] == 1.0
    assert plan["status"] == "FROZEN_BEFORE_RUNTIME"
    assert all("outcome" not in row for row in plan["selected_candidates"])


def test_v3_risk_coverage_has_no_low_risk_prospective_false_safe():
    summary = json.loads((ROOT / "experiments/oom_risk_summary_v3_2026-09-13.json").read_text())
    v2 = summary["prospective"]["V2"]["risk_coverage"]["0.2"]
    assert v2["coverage"] >= 0.4
    assert v2["false_safe"] == 0


def test_v3_model_schema_is_compact_and_monotone():
    model = json.loads((ROOT / "experiments/oom_risk_model_v3_2026-09-13.json").read_text())
    summary = json.loads((ROOT / "experiments/oom_risk_summary_v3_2026-09-13.json").read_text())
    assert len(model["feature_names"]) <= 20
    assert summary["monotonicity"]["v3"] == 0
    assert not set(model["feature_names"]).intersection(V.V1.FORBIDDEN)


def test_v3_dataset_keeps_grouped_integrity():
    rows = json.loads((ROOT / "experiments/oom_risk_dataset_v3_2026-09-13.json").read_text())["rows"]
    eligible = [r for r in rows if r["target_execution_oom"] is not None and not r.get("unstable_group", False)]
    for train, test in V.V1.grouped_folds(eligible, 5):
        assert not {eligible[i]["workload_group_id"] for i in train}.intersection(
            eligible[i]["workload_group_id"] for i in test
        )
