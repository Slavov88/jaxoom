from unittest.mock import Mock
import pytest
from experiments import oom_risk_gpu_resume as resume
import json


def test_unknown_sample_is_retried_with_a_new_full_sample(monkeypatch):
    valid = {"used_memory_bytes": 0, "fresh": True}
    snapshot = Mock(side_effect=[ValueError("invalid literal: '[Unknown Error]'"), valid])
    monkeypatch.setattr(resume, "ORIGINAL_SNAPSHOT", snapshot)
    monkeypatch.setattr(resume.time, "sleep", Mock())
    assert resume.strict_snapshot() is valid
    assert snapshot.call_count == 2


def test_persistent_unknown_fails_closed_after_three_attempts(monkeypatch):
    snapshot = Mock(side_effect=ValueError("invalid literal: '[Unknown Error]'"))
    monkeypatch.setattr(resume, "ORIGINAL_SNAPSHOT", snapshot)
    monkeypatch.setattr(resume.time, "sleep", Mock())
    with pytest.raises(ValueError):
        resume.strict_snapshot()
    assert snapshot.call_count == 3


def test_unrelated_error_is_not_retried_or_converted_to_zero(monkeypatch):
    snapshot = Mock(side_effect=ValueError("invalid literal: bad data"))
    monkeypatch.setattr(resume, "ORIGINAL_SNAPSHOT", snapshot)
    with pytest.raises(ValueError):
        resume.strict_snapshot()
    assert snapshot.call_count == 1


def test_only_proven_preflight_infrastructure_error_can_be_retried():
    candidate = dict(spec_hash="a", spec={})
    plan = dict(plan_hash="p", policy_hash="q")
    child = dict(event="result", spec_hash="a", spec={}, plan_hash="p", policy_hash="q",
                 outcome="OTHER_FAILURE", phase="preflight", gpu_runtime_executed=False,
                 error="ValueError: invalid literal for int() with base 10: '[Unknown Error]'")
    row = dict(outcome="INVALID_CHILD_PROVENANCE_OR_REPLAY", returncode=0)
    assert resume.telemetry_only_failure(row, json.dumps(child), candidate, plan)
    for changes in (dict(phase="initialization"), dict(gpu_runtime_executed=True),
                    dict(spec_hash="wrong"), dict(plan_hash="wrong"), dict(error="different")):
        assert not resume.telemetry_only_failure(row, json.dumps(dict(child, **changes)), candidate, plan)
    # Compact logs must not conceal an initialization phase.
    stdout = '{"event":"phase","phase":"initialization"}\n' + json.dumps(child)
    assert not resume.telemetry_only_failure(row, stdout, candidate, plan)
