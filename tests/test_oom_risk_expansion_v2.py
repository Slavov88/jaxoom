from pathlib import Path
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = load("oom_risk_expansion_v2", ROOT / "experiments" / "oom_risk_expansion_v2.py")
V = load("oom_risk_model_v2", ROOT / "experiments" / "oom_risk_model_v2.py")


def test_frozen_plan_has_unique_preoutcome_groups():
    plan = json.loads((ROOT / "experiments/oom_risk_expansion_plan_v2_2026-09-12.json").read_text())
    rows = plan["selected_candidates"]
    assert plan["status"] == "FROZEN_BEFORE_RUNTIME"
    assert len({r["candidate_id"] for r in rows}) == len(rows)
    assert len({r["workload_group_id"] for r in rows}) == len(rows)
    assert all("outcome" not in r and "actual_outcome" not in r for r in rows)


def test_candidate_pool_has_static_only_features_and_frozen_scores():
    pool = json.loads((ROOT / "experiments/oom_risk_candidate_pool_v2_2026-09-12.json").read_text())
    valid = [r for r in pool["candidates"] if r["status"] == "STATIC_OK"]
    assert len(valid) >= 2000
    assert all("v1_predicted_probability" in r for r in valid)
    assert all("outcome" not in r and "runtime" not in r for r in valid)
    assert len({r["workload_group_id"] for r in valid}) == len(valid)


def test_v2_core_features_are_precompilation_and_monotone_ratios():
    assert len(V.V2_FEATURES) <= 15
    assert V.RATIO_FEATURES == set(V.V2_FEATURES[:5])
    assert not set(V.V2_FEATURES).intersection(V.M.FORBIDDEN)

    row = {"features": {name: 0.5 for name in V.V2_FEATURES}}
    beta = np.zeros(len(V.V2_FEATURES) + 1)
    beta[1 : 1 + len(V.RATIO_FEATURES)] = 1.0
    x, _, mean, scale = V.prepare([row], [], V.V2_FEATURES)
    assert V.monotonicity([row], V.V2_FEATURES, beta, mean, scale) == 0
