"""Regression checks for the restricted, unexported frozen V2 adapter."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import math

import jax
import jaxlib
import pytest

from experiments import oom_risk_workload_spec as adapter
from experiments import oom_risk_workload_replay as replay


def attention(**changes):
    config = dict(batch=2, heads=4, sequence=768, head_dim=64)
    config.update(changes)
    return config


FRESH = pytest.mark.skipif(
    (jax.__version__, jaxlib.__version__) != ("0.11.0", "0.11.0"),
    reason="frozen V2 fresh replay requires exact JAX/jaxlib 0.11.0",
)


def test_spec_copies_canonicalizes_and_roundtrips_without_mutable_aliases():
    config = attention()
    spec = adapter.WorkloadSpec("attention", config, "float32")
    same = adapter.WorkloadSpec("attention", dict(reversed(list(config.items()))), "float32")
    config["sequence"] = 999
    assert spec == same
    assert hash(spec) == hash(same)
    assert spec.spec_hash == same.spec_hash
    assert adapter.WorkloadSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    exported = spec.to_dict()
    exported["configuration"]["sequence"] = 123
    assert dict(spec.configuration)["sequence"] == 768
    with pytest.raises(FrozenInstanceError):
        spec.dtype = "float16"


@pytest.mark.parametrize("value", [0, -1, True, False, 2.0, "2", None, math.inf, math.nan, 2**31])
def test_noncanonical_dimensions_are_rejected(value):
    with pytest.raises(ValueError, match="positive Python integers"):
        adapter.WorkloadSpec("attention", attention(batch=value), "float32")


@pytest.mark.parametrize("dtype", ["bfloat16", "float64", "int32", None, ("float16", "float32")])
def test_single_dtype_contract_rejects_unsupported_and_mixed_dtypes(dtype):
    with pytest.raises(ValueError, match="dtype"):
        adapter.WorkloadSpec("attention", attention(), dtype)


def test_unknown_missing_and_injected_configuration_fields_are_rejected():
    with pytest.raises(ValueError, match="exactly"):
        adapter.WorkloadSpec("attention", attention(ignored_metadata=1000000), "float32")
    config = attention()
    del config["heads"]
    with pytest.raises(ValueError, match="exactly"):
        adapter.WorkloadSpec("attention", config, "float32")
    with pytest.raises(ValueError, match="family"):
        adapter.WorkloadSpec("custom", attention(), "float32")
    with pytest.raises(TypeError, match="mapping"):
        adapter.WorkloadSpec("attention", tuple(attention().items()), "float32")


def test_spec_version_cross_field_constraints_and_resource_guard():
    with pytest.raises(ValueError, match="version"):
        adapter.WorkloadSpec("attention", attention(), "float32", version="next")
    with pytest.raises(ValueError, match="divisible"):
        adapter.WorkloadSpec("transformer", dict(sequence=32, width=65, heads=4, layers=1), "float32")
    with pytest.raises(ValueError, match="resource limit"):
        adapter.WorkloadSpec("mlp", dict(batch=1, width=32, depth=65), "float32")
    spec = adapter.WorkloadSpec("attention", attention(), "float32")
    for extra in ({"fn": "arbitrary"}, {"memory_limit": "4 GiB"}, {"workload_config": attention()}):
        with pytest.raises(ValueError, match="exactly"):
            adapter.WorkloadSpec.from_dict(dict(spec.to_dict(), **extra))


def test_callables_and_independent_metadata_are_not_adapter_inputs():
    spec = adapter.WorkloadSpec("attention", attention(), "float32")
    with pytest.raises(TypeError, match="WorkloadSpec"):
        adapter.build_workload(lambda x: x)
    with pytest.raises(TypeError):
        adapter.extract_features(spec, workload_config=attention(batch=1))
    with pytest.raises(TypeError):
        adapter.extract_features(spec, memory_limit="4 GiB")
    with pytest.raises(ValueError):
        adapter.build_workload(adapter.WorkloadSpec("attention", attention(), "float32", version="bad"))


def test_frozen_model_bytes_and_historical_zero_feature_are_preserved():
    model = adapter.load_frozen_model()
    assert hashlib.sha256(adapter.MODEL_PATH.read_bytes()).hexdigest() == adapter.MODEL_FILE_SHA256
    assert model["intercept"] == model["coefficients"][0]
    assert model["coefficients"][3] == model["means"][2] == 0.0
    dataset = json.loads(adapter.MODEL_PATH.with_name("oom_risk_dataset_v2_2026-09-12.json").read_text())
    eligible = [r for r in dataset["rows"] if r["target_execution_oom"] is not None and not r.get("unstable_group", False)]
    assert len(eligible) == 176
    assert all(r["features"].get("largest_over_budget") is None for r in eligible)
    assert adapter.digest(dataset["rows"]) == model["dataset_hash"]
    calibration = json.loads(adapter.MODEL_PATH.with_name("oom_risk_model_v1_frozen_2026-09-12.json").read_text())
    assert adapter.UPPER_RATIO == calibration["calibration_proxy_for_static_candidates"]["upper_ratio_median"]


@pytest.mark.parametrize("mutation", ["whitespace", "intercept", "coefficient", "schema"])
def test_model_tampering_fails_closed(tmp_path, monkeypatch, mutation):
    raw = adapter.MODEL_PATH.read_bytes()
    model = json.loads(raw)
    if mutation == "whitespace":
        changed = raw + b"\n"
    else:
        if mutation == "intercept":
            model["intercept"] += 1
        elif mutation == "coefficient":
            model["coefficients"][1] += 1
        else:
            model["feature_names"][0] = "other"
        changed = json.dumps(model).encode()
    path = tmp_path / "tampered.json"
    path.write_bytes(changed)
    monkeypatch.setattr(adapter, "MODEL_PATH", path)
    with pytest.raises(RuntimeError, match="artifact bytes"):
        adapter.load_frozen_model()


def test_template_changes_fail_before_workload_construction(monkeypatch):
    from experiments import oom_risk_expansion_v2

    def changed_template(family, config, dtype):
        raise AssertionError("must never run")

    monkeypatch.setattr(oom_risk_expansion_v2, "candidate_functions", changed_template)
    with pytest.raises(RuntimeError, match="templates have changed"):
        adapter.build_workload(adapter.WorkloadSpec("attention", attention(), "float32"))


@pytest.mark.parametrize("bad", ["missing", "nan", "inf", "semantic_largest", "extra"])
def test_inference_never_silently_imputes_active_features(bad):
    model = adapter.load_frozen_model()
    features = {n: 0.0 for n in adapter.FEATURE_NAMES}
    if bad == "missing":
        del features["dtype_bytes"]
    elif bad == "nan":
        features["dtype_bytes"] = math.nan
    elif bad == "inf":
        features["dtype_bytes"] = math.inf
    elif bad == "semantic_largest":
        features["largest_over_budget"] = 0.01
    else:
        features["unknown"] = 0.0
    with pytest.raises(ValueError):
        adapter._score_features(features, model)


def test_incompatible_trace_version_fails_before_tracing(monkeypatch):
    monkeypatch.setattr(jaxlib, "__version__", "unsupported")
    with pytest.raises(RuntimeError, match="0.11.0"):
        adapter.extract_features(adapter.WorkloadSpec("attention", attention(), "float32"))


@FRESH
def test_all_224_rows_match_fresh_traces_scores_decisions_and_purity():
    result = replay.run_replay()
    assert result["fresh_tracing_and_replay_pass"], result["errors"]
    assert result["completed"] == result["rows"] == 224
    assert result["feature_mismatches"] == 0
    assert result["score_mismatches"] == result["decision_mismatches"] == 0
    assert result["max_feature_delta"] == 0.0
    assert result["max_probability_delta"] < 1e-12
    assert result["purity"]["target_calls_with_tracers"] == 224
    assert result["purity"]["concrete_target_calls"] == result["purity"]["blocked_compile_calls"] == 0
    assert len({(r["panel"], r["candidate_id"]) for r in result["replayed_rows"]}) == 224
    assert result["public_api_absent"]
    assert result["public_api_enabled"] is False
    assert result["runtime_gpu_validation"] == "NOT_RUN"
    assert result["reference_is_runtime_quota"] is False


@FRESH
def test_missing_active_saved_feature_fails_replay_gate(monkeypatch):
    rows, sources = replay.load_prediction_rows()
    del rows[0]["static_features"]["dtype_bytes"]
    monkeypatch.setattr(replay, "load_prediction_rows", lambda: (rows[:1], sources))
    result = replay.run_replay()
    assert not result["fresh_tracing_and_replay_pass"]
    assert result["completed"] == 0
    assert "dtype_bytes" in result["errors"][0]["error"]
    assert result["public_api_enabled"] is False


def test_purity_guard_rejects_compilation_and_concrete_execution():
    with replay.static_only_guard() as observations:
        with pytest.raises(RuntimeError, match="compilation"):
            jax.jit(lambda x: x)
        fn, _ = adapter.build_workload(adapter.WorkloadSpec("attention", attention(), "float32"))
        with pytest.raises(RuntimeError, match="execution"):
            fn(object())
    assert observations["blocked_compile_calls"] == observations["concrete_target_calls"] == 1


def test_replay_cli_refuses_to_overwrite_existing_evidence(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("KEEP")
    monkeypatch.setattr("sys.argv", ["replay", "--output", str(output)])
    with pytest.raises(FileExistsError):
        replay.main()
    assert output.read_text() == "KEEP"


def test_replay_rejects_changed_prediction_artifacts_even_with_recomputed_hash(tmp_path, monkeypatch):
    rows, _ = replay.load_prediction_rows()
    name, count = replay.PANELS[0]
    payload = json.loads((replay.ROOT / "experiments" / name).read_text())
    payload["predictions"][0]["v2_probability"] = 0.0
    del payload["prediction_hash"]
    payload["prediction_hash"] = adapter.digest(payload)
    folder = tmp_path / "experiments"
    folder.mkdir()
    (folder / name).write_text(json.dumps(payload))
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(replay, "PANELS", ((name, count),))
    with pytest.raises(ValueError, match="artifact bytes"):
        replay.load_prediction_rows()
    # Composite IDs preserve every supplemental row despite H2 ID reuse.
    assert len(rows) == 224


def test_cli_records_failed_gate_and_exits_nonzero(tmp_path, monkeypatch):
    output = tmp_path / "failure.json"
    def failure():
        raise RuntimeError("deliberate test failure")
    monkeypatch.setattr(replay, "run_replay", failure)
    monkeypatch.setattr("sys.argv", ["replay", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        replay.main()
    assert exc.value.code == 1
    result = json.loads(output.read_text())
    assert result["status"] == "FAILED"
    assert result["fresh_tracing_and_replay_pass"] is False
    assert result["public_api_absent"] is None
    assert "deliberate test failure" in result["error"]


def test_public_api_remains_disabled_even_after_a_successful_replay():
    import jaxoom

    assert adapter.PUBLIC_API_ENABLED is False
    assert not hasattr(jaxoom, "assess_oom_risk")
    assert not hasattr(jaxoom, "WorkloadSpec")
    assert "WorkloadSpec" not in jaxoom.__all__
    contract = adapter.pipeline_contract()
    assert contract["reference_is_runtime_quota"] is False
    assert "analysis/peak.py" in contract["static_analysis_source_sha256"]


def test_pipeline_contract_rejects_import_from_another_checkout(tmp_path, monkeypatch):
    import jaxoom

    monkeypatch.setattr(jaxoom, "__file__", str(tmp_path / "jaxoom" / "__init__.py"))
    with pytest.raises(RuntimeError, match="this checkout"):
        adapter.pipeline_contract()
