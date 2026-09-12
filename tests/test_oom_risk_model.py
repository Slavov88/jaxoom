from pathlib import Path
import importlib.util

SPEC = importlib.util.spec_from_file_location(
    "oom_risk_model_v1", Path(__file__).parents[1] / "experiments" / "oom_risk_model_v1.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_group_fingerprint_is_order_independent():
    a = {"batch": 2, "sequence": 2560, "heads": 12}
    b = {"heads": 12, "batch": 2, "sequence": 2560}
    assert M.config_group("attention", a, "float32") == M.config_group("attention", b, "float32")
    assert M.fingerprint("attention", a, "float32") == M.fingerprint("attention", b, "float32")


def test_forbidden_diagnostics_are_not_model_features():
    features = {name for names in M.feature_sets().values() for name in names}
    assert not features.intersection(M.FORBIDDEN)


def test_grouped_folds_do_not_split_workload_groups():
    rows = [
        {"workload_group_id": "a", "target_execution_oom": 0},
        {"workload_group_id": "a", "target_execution_oom": 0},
        {"workload_group_id": "b", "target_execution_oom": 1},
        {"workload_group_id": "c", "target_execution_oom": 0},
    ]
    folds = M.grouped_folds(rows, n_splits=3)
    memberships = {}
    for fold, (_, test) in enumerate(folds):
        for i in test:
            memberships.setdefault(rows[i]["workload_group_id"], set()).add(fold)
    assert all(len(folds_seen) == 1 for folds_seen in memberships.values())


def test_abstention_never_calls_mid_probability_oom():
    assert M.abstention([0.01, 0.5, 0.99], 0.1, 0.9) == ["LIKELY_FIT", "UNCERTAIN", "LIKELY_OOM"]
