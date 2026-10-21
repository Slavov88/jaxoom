from experiments import oom_risk_gpu_continuation as c
import json
import pytest


def test_unknown_never_qualifies_even_with_complete_coverage():
    row = dict(spec_hash="a", outcome="EXECUTE_TIMEOUT", likely_fit=False)
    assert c.can_continue(row)
    result = c.summarize([row], total=1)
    assert result["coverage_complete"]
    assert not result["all_checks_pass"]
    assert result["unresolved"] == 1
    assert not result["public_api_enabled"]


def test_likely_fit_timeout_stops():
    assert not c.can_continue(dict(outcome="EXECUTE_TIMEOUT", likely_fit=True))
    assert not c.can_continue(dict(outcome="EXECUTE_TIMEOUT"))
    assert not c.can_continue(dict(outcome="COMPILE_TIMEOUT", likely_fit=False))
    assert not c.can_continue(dict(outcome="OTHER_FAILURE", likely_fit=False))


def test_only_unattempted_candidates_are_selected():
    plan = dict(candidates=[dict(spec_hash="a"), dict(spec_hash="b")])
    assert c.pending_candidates(plan, [dict(spec_hash="a")]) == [dict(spec_hash="b")]


def test_duplicate_coverage_never_qualifies():
    row = dict(spec_hash="a", outcome="FIT", phase="complete", gpu_runtime_executed=True,
               likely_fit=True, applicability=dict(status="ELIGIBLE_FOR_QUALIFICATION"))
    assert c.summarize([row], total=1)["all_checks_pass"]
    assert not c.summarize([row, row], total=2)["all_checks_pass"]


def test_previous_evidence_corruption_is_rejected(tmp_path):
    (tmp_path / "record.json").write_text("modified")
    (tmp_path / "independent_audit.json").write_text(json.dumps({"artifact_sha256": {"record.json": "wrong"}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        c.audit_previous(tmp_path)


def test_previous_evidence_cannot_escape_directory(tmp_path):
    (tmp_path / "independent_audit.json").write_text(json.dumps({"artifact_sha256": {"../other.json": "wrong"}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        c.audit_previous(tmp_path)
