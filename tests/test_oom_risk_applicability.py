"""Fail-closed applicability checks; modeled runtime fixtures never use a GPU."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from experiments import oom_risk_applicability as policy
from experiments import oom_risk_workload_replay as replay
from experiments import oom_risk_workload_spec as adapter


NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
EXPECTED_ENV = {
    "backend": "gpu", "gpu_vendor": "NVIDIA",
    "device_kind": "NVIDIA GeForce RTX 3050 Laptop GPU", "device_count": 1,
    "physical_memory_bytes": 4 * 1024**3, "driver_version": "566.07",
    "jax_version": "0.11.0", "jaxlib_version": "0.11.0",
    "allocator": "bfc", "preallocate": "false", "memory_fraction": None,
    "tf_gpu_allocator": None, "xla_flags": None, "jax_enable_x64": False,
    "default_matmul_precision": None, "fresh_process": True,
    "donation": False, "sharding": False,
}
METRICS = {
    "free_memory_bytes": 3900 * 1024**2, "used_memory_bytes": 64 * 1024**2,
    "utilization_percent": 5, "compute_process_count": 0,
}


@pytest.fixture
def runtime():
    return dict(EXPECTED_ENV, **METRICS, observed_at_utc=NOW.isoformat())


@pytest.fixture
def evidence():
    path = Path(__file__).resolve().parents[1] / "experiments" / "oom_risk_workload_replay_2026-10-20_final.json"
    return json.loads(path.read_text())


@pytest.fixture
def spec():
    rows, _ = replay.load_prediction_rows()
    row = rows[0]
    return adapter.WorkloadSpec(row["family"], row["configuration"], row["dtype"])


def evaluate(spec, runtime, evidence):
    return policy.evaluate_applicability(spec, runtime, evidence, now=NOW)


def assert_unsupported(result):
    assert result["status"] == "UNSUPPORTED", result
    assert result["reasons"] and all(isinstance(reason, str) for reason in result["reasons"])
    assert result["public_api_enabled"] is False


def test_modeled_matching_environment_only_permits_qualification(spec, runtime, evidence):
    result = evaluate(spec, runtime, evidence)
    assert result["status"] == "ELIGIBLE_FOR_QUALIFICATION", result
    assert result["reasons"] == []
    assert result["public_api_enabled"] is False


def test_allowlist_is_exactly_the_200_primary_frozen_recipes():
    rows, _ = replay.load_prediction_rows()
    expected = {
        candidate.spec_hash: candidate
        for row in rows[:200]
        for candidate in [adapter.WorkloadSpec(row["family"], row["configuration"], row["dtype"])]
    }
    allowed = policy.allowed_specs()
    assert len(expected) == len(allowed) == 200
    assert allowed == expected
    assert all(key == value.spec_hash for key, value in allowed.items())


@pytest.mark.parametrize("key", list(EXPECTED_ENV) + list(METRICS) + ["observed_at_utc"])
def test_missing_runtime_field_is_unsupported(key, spec, runtime, evidence):
    del runtime[key]
    assert_unsupported(evaluate(spec, runtime, evidence))


@pytest.mark.parametrize("key,value", [
    ("backend", "cpu"), ("gpu_vendor", "AMD"),
    ("device_kind", "AMD Radeon(TM) Graphics"), ("device_count", 2),
    ("physical_memory_bytes", 8 * 1024**3), ("driver_version", "566.08"),
    ("jax_version", "0.11.1"), ("jaxlib_version", "0.11.1"),
    ("allocator", "cuda_async"), ("preallocate", "true"),
    ("memory_fraction", "0.75"), ("tf_gpu_allocator", "cuda_malloc_async"),
    ("xla_flags", "--xla_gpu_enable_command_buffer="),
    ("jax_enable_x64", True), ("default_matmul_precision", "highest"),
    ("fresh_process", False), ("donation", True), ("sharding", True),
    ("free_memory_bytes", 3900 * 1024**2 - 1),
    ("used_memory_bytes", 64 * 1024**2 + 1),
    ("utilization_percent", 6), ("compute_process_count", 1),
])
def test_changed_or_busy_runtime_is_unsupported(key, value, spec, runtime, evidence):
    runtime[key] = value
    assert_unsupported(evaluate(spec, runtime, evidence))


@pytest.mark.parametrize("key", ["device_count", "physical_memory_bytes", *METRICS])
@pytest.mark.parametrize("value", [True, False, 0.0, "0", None, -1])
def test_integer_metrics_require_actual_nonnegative_python_integers(key, value, spec, runtime, evidence):
    runtime[key] = value
    assert_unsupported(evaluate(spec, runtime, evidence))


@pytest.mark.parametrize("key", ["jax_enable_x64", "fresh_process", "donation", "sharding"])
def test_boolean_policy_values_do_not_accept_equal_integers(key, spec, runtime, evidence):
    runtime[key] = int(runtime[key])
    assert_unsupported(evaluate(spec, runtime, evidence))


@pytest.mark.parametrize("stamp", [
    (NOW - timedelta(seconds=5, microseconds=1)).isoformat(),
    (NOW + timedelta(microseconds=1)).isoformat(),
    NOW.replace(tzinfo=None).isoformat(), "not-a-date", None, True,
])
def test_stale_future_naive_or_malformed_observation_is_unsupported(stamp, spec, runtime, evidence):
    runtime["observed_at_utc"] = stamp
    assert_unsupported(evaluate(spec, runtime, evidence))


def test_exact_five_second_age_boundary_is_accepted(spec, runtime, evidence):
    runtime["observed_at_utc"] = (NOW - timedelta(seconds=5)).isoformat()
    assert evaluate(spec, runtime, evidence)["status"] == "ELIGIBLE_FOR_QUALIFICATION"


def test_well_formed_unseen_recipe_is_unsupported(runtime, evidence):
    unseen = adapter.WorkloadSpec("matmul", dict(m=7, k=11, n=13, depth=1), "float32")
    assert unseen.spec_hash not in policy.allowed_specs()
    assert_unsupported(evaluate(unseen, runtime, evidence))


def test_missing_replay_evidence_is_unsupported(spec, runtime):
    assert_unsupported(evaluate(spec, runtime, None))


@pytest.mark.parametrize("runtime_value", [None, [], "gpu"])
def test_nonmapping_runtime_is_unsupported(runtime_value, spec, evidence):
    assert_unsupported(evaluate(spec, runtime_value, evidence))


@pytest.mark.parametrize("key,value", [
    ("free_memory_bytes", 4 * 1024**3 + 1),
    ("free_memory_bytes", 4 * 1024**3),
    ("utilization_percent", -1), ("compute_process_count", -1),
])
def test_impossible_or_contradictory_runtime_metrics_are_unsupported(key, value, spec, runtime, evidence):
    runtime[key] = value
    assert_unsupported(evaluate(spec, runtime, evidence))


@pytest.mark.parametrize("key,value", [
    ("source_sha256", []), ("source_sha256", None),
    ("source_sha256", {}), ("contract", None),
    ("max_feature_delta", float("nan")),
    ("max_probability_delta", float("inf")),
    ("max_probability_delta", 2e-12),
    ("purity_pass", False), ("public_api_absent", False),
    ("feature_mismatches", 1), ("score_mismatches", 1),
    ("decision_mismatches", 1), ("rows", True),
    ("errors", ["injected failure"]), ("tolerance", 0.1),
    ("purity", None), ("purity", []), ("purity", {}),
    ("prediction_file_sha256", {}), ("prediction_file_sha256", None),
    ("max_probability_delta", 10**1000),
])
def test_malformed_or_failed_replay_details_fail_closed(key, value, spec, runtime, evidence):
    evidence[key] = value
    assert_unsupported(evaluate(spec, runtime, evidence))


def test_replay_source_hash_mismatch_is_unsupported(spec, runtime, evidence):
    evidence["source_sha256"]["experiments/oom_risk_workload_spec.py"] = "0" * 64
    assert_unsupported(evaluate(spec, runtime, evidence))


@pytest.mark.parametrize("key,value", [
    ("target_calls_with_tracers", 223), ("concrete_target_calls", 1),
    ("blocked_compile_calls", 1), ("concrete_target_calls", False),
])
def test_purity_counts_must_agree_with_pass_flag(key, value, spec, runtime, evidence):
    evidence["purity"][key] = value
    assert_unsupported(evaluate(spec, runtime, evidence))


def test_changed_pinned_prediction_hash_is_unsupported(spec, runtime, evidence):
    first = next(iter(evidence["prediction_file_sha256"]))
    evidence["prediction_file_sha256"][first] = "0" * 64
    assert_unsupported(evaluate(spec, runtime, evidence))


@pytest.mark.parametrize("mutation", ["contract", "missing_contract", "failed", "incomplete", "enabled"])
def test_failed_incomplete_or_stale_replay_is_unsupported(mutation, spec, runtime, evidence):
    if mutation == "contract":
        evidence["contract"]["adapter_source_sha256"] = "0" * 64
    elif mutation == "missing_contract":
        del evidence["contract"]
    elif mutation == "failed":
        evidence["fresh_tracing_and_replay_pass"] = False
    elif mutation == "incomplete":
        evidence["completed"] = 223
    else:
        evidence["public_api_enabled"] = True
    assert_unsupported(evaluate(spec, runtime, evidence))


def test_policy_contract_is_serializable_and_returned_without_mutable_aliases():
    first = policy.policy_contract()
    snapshot = deepcopy(first)
    json.dumps(first, allow_nan=False)
    first["test_injected_field"] = "changed"
    assert policy.policy_contract() == snapshot


def test_public_api_remains_absent():
    import jaxoom
    import jaxoom.api

    assert adapter.PUBLIC_API_ENABLED is False
    for name in ("WorkloadSpec", "evaluate_applicability", "assess_oom_risk", "predict_oom_risk", "oom_risk"):
        assert not hasattr(jaxoom, name)
        assert not hasattr(jaxoom.api, name)
        assert name not in jaxoom.__all__


@pytest.mark.parametrize("flag_owner", [policy, adapter])
def test_accidental_api_enable_flag_blocks_qualification(flag_owner, spec, runtime, evidence, monkeypatch):
    monkeypatch.setattr(flag_owner, "PUBLIC_API_ENABLED", True)
    assert_unsupported(evaluate(spec, runtime, evidence))
