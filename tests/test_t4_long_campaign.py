from experiments.t4_long_campaign import configuration_id, default_candidates, select_boundary_candidates


def test_long_campaign_candidate_ids_are_unique():
    candidates = default_candidates()
    ids = [row["configuration_id"] for row in candidates]
    assert len(ids) == len(set(ids))
    assert configuration_id("transformer", {"sequence": 3072, "width": 2048, "heads": 8}, "float32") == "21ed3ee099a0a446"


def test_scale_selection_only_uses_real_oom_rows(tmp_path):
    output = tmp_path / "selection.json"
    select_boundary_candidates([
        {"configuration_id": "fit", "family": "mlp", "dtype": "float32", "configuration": {}, "outcome": "FIT"},
        {"configuration_id": "oom", "family": "mlp", "dtype": "float16", "configuration": {}, "outcome": "EXECUTION_OOM"},
        {"configuration_id": "timeout", "family": "mlp", "dtype": "float32", "configuration": {}, "outcome": "EXECUTION_TIMEOUT"},
    ], output, max_per_family=3)
    selected = __import__("json").loads(output.read_text())["selected"]
    assert [row["configuration_id"] for row in selected] == ["oom"]
