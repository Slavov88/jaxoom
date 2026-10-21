"""Pure qualification helper tests; never initialize or execute a GPU backend."""
from copy import deepcopy
import json

import pytest

from experiments import oom_risk_gpu_qualification as qualification
from experiments import oom_risk_workload_spec as adapter


@pytest.mark.parametrize("phase,expected", [
    ("initialization", "INITIALIZATION_OOM"), ("compile", "COMPILE_OOM"),
    ("execute", "EXECUTION_OOM"), ("preflight", "OTHER_FAILURE"),
])
@pytest.mark.parametrize("error", ["RESOURCE_EXHAUSTED", "out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "CUBLAS_STATUS_ALLOC_FAILED"])
def test_oom_classification_preserves_phase(phase, expected, error):
    assert qualification.classify_error(RuntimeError(error), phase) == expected


@pytest.mark.parametrize("error", ["segmentation fault", "unsupported backend", "bad dtype", "CUDA_ERROR_UNKNOWN"])
def test_unrelated_errors_are_never_oom(error):
    assert qualification.classify_error(error, "execute") == "OTHER_FAILURE"


@pytest.mark.parametrize("phase", ["preflight", "initialization", "compile", "execute"])
def test_timeout_is_never_a_fit_or_oom(phase):
    stdout = 'unstructured diagnostic\n' + json.dumps({"event": "phase", "phase": phase})
    result = qualification.timeout_result(stdout)
    assert result["outcome"] == phase.upper() + "_TIMEOUT"
    assert result["phase"] == phase
    assert result["gpu_runtime_executed"] is (phase == "execute")
    assert qualification.trial_passes(result) is False


@pytest.mark.parametrize("noise", ["null", "[]", "42", '"text"', '{"event":"phase"}'])
def test_timeout_ignores_malformed_or_nonobject_events(noise):
    result = qualification.timeout_result(noise)
    assert result["outcome"] == "PREFLIGHT_TIMEOUT"
    assert result["gpu_runtime_executed"] is False


def test_timeout_uses_last_valid_phase():
    stdout = '\n'.join(json.dumps({"event": "phase", "phase": phase}) for phase in ("initialization", "compile", "execute"))
    assert qualification.timeout_result(stdout)["outcome"] == "EXECUTE_TIMEOUT"


@pytest.mark.parametrize("row", [
    {}, {"outcome": "FIT"}, {"outcome": "FIT", "gpu_runtime_executed": 1},
    {"outcome": "FIT", "gpu_runtime_executed": False},
    {"outcome": "EXECUTION_OOM", "likely_fit": True},
    {"outcome": "COMPILE_OOM", "likely_fit": True},
    {"outcome": "EXECUTION_OOM"},
    {"outcome": "EXECUTION_OOM", "likely_fit": 0},
    {"outcome": "INITIALIZATION_OOM", "likely_fit": False},
    {"outcome": "OTHER_FAILURE", "likely_fit": False},
    {"outcome": "EXECUTE_TIMEOUT", "likely_fit": False},
    {"outcome": "MISSING_OR_DUPLICATE_RESULT", "likely_fit": False},
    {"outcome": "UNSUPPORTED", "likely_fit": False},
])
def test_incomplete_failed_or_false_safe_trial_never_passes(row):
    assert qualification.trial_passes(row) is False


@pytest.mark.parametrize("outcome,phase,executed,likely_fit", [
    ("FIT", "complete", True, True),
    ("EXECUTION_OOM", "execute", True, False),
    ("COMPILE_OOM", "compile", False, False),
])
def test_only_consistent_eligible_trial_outcomes_pass(outcome, phase, executed, likely_fit):
    row = dict(outcome=outcome, phase=phase, gpu_runtime_executed=executed,
               likely_fit=likely_fit, applicability={"status": "ELIGIBLE_FOR_QUALIFICATION"})
    assert qualification.trial_passes(row) is True
    for key, value in (("phase", "preflight"), ("applicability", {"status": "UNSUPPORTED"})):
        assert qualification.trial_passes(dict(row, **{key: value})) is False


def test_result_parser_ignores_noisy_and_nonobject_logs():
    row = {"event": "result", "outcome": "FIT"}
    stdout = 'noise\n{broken\nnull\n[]\n42\n' + json.dumps(row)
    assert qualification.parse_child_result(stdout) == row


@pytest.mark.parametrize("stdout", ["", "null", '{"event":"phase","phase":"execute"}',
    '{"event":"result"}\n{"event":"result"}'])
def test_missing_or_duplicate_result_is_fail_closed(stdout):
    assert qualification.parse_child_result(stdout)["outcome"] == "MISSING_OR_DUPLICATE_RESULT"


@pytest.fixture(scope="module")
def frozen_plan():
    return qualification.make_plan()


@pytest.fixture
def valid_child(frozen_plan):
    candidate = frozen_plan["candidates"][0]
    return dict(spec_hash=candidate["spec_hash"], spec=deepcopy(candidate["spec"]),
                plan_hash=frozen_plan["plan_hash"], policy_hash=frozen_plan["policy_hash"],
                backend_verified=True, platform_version="CUDA 12.4",
                applicability={"status": "ELIGIBLE_FOR_QUALIFICATION"},
                fresh_probability=candidate["expected_probability"],
                fresh_features=deepcopy(candidate["expected_features"]))


def test_consistent_child_provenance_is_valid(valid_child, frozen_plan):
    assert qualification.validate_child_result(valid_child, frozen_plan["candidates"][0], frozen_plan)


@pytest.mark.parametrize("key,value", [
    ("spec_hash", "wrong"), ("spec", {}), ("plan_hash", "wrong"),
    ("policy_hash", "wrong"), ("applicability", None),
    ("applicability", {"status": "UNSUPPORTED"}),
    ("fresh_probability", float("nan")), ("fresh_probability", float("inf")),
    ("fresh_probability", 10**1000), ("fresh_features", {}),
    ("fresh_features", None),
    ("backend_verified", False), ("backend_verified", 1),
    ("platform_version", "cpu"),
])
def test_child_malformed_identity_or_evidence_is_rejected(key, value, valid_child, frozen_plan):
    valid_child[key] = value
    assert qualification.validate_child_result(valid_child, frozen_plan["candidates"][0], frozen_plan) is False


@pytest.mark.parametrize("key", ["spec_hash", "spec", "plan_hash", "policy_hash", "applicability", "fresh_probability", "fresh_features", "backend_verified", "platform_version"])
def test_missing_child_provenance_is_rejected(key, valid_child, frozen_plan):
    del valid_child[key]
    assert qualification.validate_child_result(valid_child, frozen_plan["candidates"][0], frozen_plan) is False


def test_changed_fresh_score_and_features_are_rejected(valid_child, frozen_plan):
    changed = deepcopy(valid_child)
    changed["fresh_probability"] += 0.001
    assert qualification.validate_child_result(changed, frozen_plan["candidates"][0], frozen_plan) is False
    changed = deepcopy(valid_child)
    changed["fresh_features"][adapter.FEATURE_NAMES[0]] += 0.001
    assert qualification.validate_child_result(changed, frozen_plan["candidates"][0], frozen_plan) is False


def test_plan_freezes_200_recipes_18_minimum_demand_pilots_and_hash():
    plan = qualification.make_plan()
    assert plan["status"] == "FROZEN_BEFORE_RUNTIME"
    assert plan["public_api_enabled"] is False
    assert plan["initialization"] == "zeros"
    assert plan["timeout_seconds"] > 0
    assert len(plan["candidates"]) == 200
    candidates = {row["spec_hash"]: row for row in plan["candidates"]}
    assert len(candidates) == 200
    assert len(plan["pilot_spec_hashes"]) == len(set(plan["pilot_spec_hashes"])) == 18
    groups = {}
    for row in candidates.values():
        spec = adapter.WorkloadSpec.from_dict(row["spec"])
        assert row["spec_hash"] == spec.spec_hash
        groups.setdefault((spec.family, spec.dtype), []).append(row)
    expected = {min(group, key=lambda r: (r["structural_bytes"], r["spec_hash"]))["spec_hash"] for group in groups.values()}
    assert set(plan["pilot_spec_hashes"]) == expected
    assert set(groups) == {(family, dtype) for family in adapter.FAMILY_FIELDS for dtype in ("float16", "float32")}
    unsigned = deepcopy(plan)
    saved = unsigned.pop("plan_hash")
    assert saved == adapter.digest(unsigned)
    assert plan["policy_hash"] == adapter.digest(plan["policy_contract"])


def test_plan_changes_are_rejected_before_execution(tmp_path):
    plan = qualification.make_plan()
    plan["timeout_seconds"] += 1
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        qualification.load_plan(path)


def test_result_writer_preserves_existing_evidence(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("KEEP")
    with pytest.raises(FileExistsError):
        qualification.write_new(path, {"outcome": "FIT"})
    assert path.read_text() == "KEEP"


def test_child_environment_is_explicit_and_clears_allocator_overrides(monkeypatch):
    for key in ("JAX_CUDA_VISIBLE_DEVICES", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR", "XLA_FLAGS"):
        monkeypatch.setenv(key, "injected")
    env = qualification.child_env()
    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not any(key in env for key in ("JAX_CUDA_VISIBLE_DEVICES", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR", "XLA_FLAGS"))
