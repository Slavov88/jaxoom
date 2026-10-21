"""CPU-only checks of diagnostic trust boundaries; all target calls are mocked."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import oom_risk_gpu_diagnostics as diagnostics


@pytest.fixture(scope="module")
def original():
    path = diagnostics.ROOT / "experiments" / (diagnostics.RUN_PREFIX + "run1") / "plan.json"
    return json.loads(path.read_text())


@pytest.fixture
def candidate(original):
    return deepcopy(next(c for c in original["candidates"] if c["candidate_id"] == "HOLDOUT-0012"))


@pytest.fixture
def diagnostic_files(tmp_path, original, candidate):
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    plan = diagnostics.make_plan(original, [candidate], audit)
    path = tmp_path / "plan.json"
    diagnostics.qualification.write_new(path, plan)
    replay = tmp_path / "replay.json"
    replay.write_text('{"fixture": "mocked replay"}')
    return dict(path=path, plan=plan, replay=replay, plan_sha=diagnostics.continuation.sha(path),
                replay_sha=diagnostics.continuation.sha(replay), original_path=tmp_path / "original.json")


@pytest.fixture
def valid_row(original, candidate):
    return dict(spec_hash=candidate["spec_hash"], spec=deepcopy(candidate["spec"]),
                plan_hash=original["plan_hash"], policy_hash=original["policy_hash"],
                backend_verified=True, platform_version="PJRT C API\ncuda 12090",
                applicability={"status": "ELIGIBLE_FOR_QUALIFICATION"},
                fresh_probability=candidate["expected_probability"],
                fresh_features=deepcopy(candidate["expected_features"]),
                likely_fit=False, expected_probability=candidate["expected_probability"],
                outcome="FIT", phase="complete", gpu_runtime_executed=True)


def attestation(files):
    return dict(source_sha256=files["plan"]["source_sha256"],
                diagnostic_plan_sha256=files["plan_sha"], replay_sha256=files["replay_sha"])


def event(value):
    return json.dumps(dict(event="attestation", **value))


def test_source_snapshot_measures_complete_runner_chain():
    sources = diagnostics.source_snapshot()
    assert len(sources) == 8
    assert "experiments/oom_risk_gpu_diagnostics.py" in sources
    assert "experiments/oom_risk_gpu_qualification.py" in sources
    for name, expected in sources.items():
        assert diagnostics.continuation.sha(diagnostics.ROOT / name) == expected
    assert diagnostics.verify_sources(sources) == sources


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed"])
def test_source_attestation_mismatch_is_rejected(mutation):
    expected = diagnostics.source_snapshot()
    first = next(iter(expected))
    if mutation == "missing":
        del expected[first]
    elif mutation == "extra":
        expected["unknown.py"] = "0" * 64
    else:
        expected[first] = "0" * 64
    with pytest.raises(ValueError, match="source attestation mismatch"):
        diagnostics.verify_sources(expected)


def test_plan_roundtrip_freezes_sources_and_preserves_disabled_contract(diagnostic_files):
    f = diagnostic_files
    result = diagnostics.read_plan(f["path"], f["plan_sha"])
    assert result == f["plan"]
    assert result["timeout_seconds"] == diagnostics.TIMEOUT_SECONDS == 600
    assert result["attempts_per_recipe"] == 1
    assert result["original_qualification_pass"] is False
    assert result["public_api_enabled"] is False
    payload = dict(result)
    saved = payload.pop("plan_hash")
    assert saved == diagnostics.qualification.adapter.digest(payload)


def test_plan_file_tampering_is_rejected(diagnostic_files):
    f = diagnostic_files
    f["path"].write_text(f["path"].read_text() + "\n")
    with pytest.raises(ValueError, match="plan file hash mismatch"):
        diagnostics.read_plan(f["path"], f["plan_sha"])


def test_plan_content_tampering_is_rejected_even_with_new_file_hash(diagnostic_files):
    f = diagnostic_files
    plan = deepcopy(f["plan"])
    plan["timeout_seconds"] += 1
    f["path"].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan content hash mismatch"):
        diagnostics.read_plan(f["path"], diagnostics.continuation.sha(f["path"]))


@pytest.mark.parametrize("key,value", [
    ("timeout_seconds", 180), ("public_api_enabled", True),
    ("public_api_enabled", 0), ("original_qualification_pass", True),
    ("original_qualification_pass", 0),
])
def test_rehashed_plan_cannot_enable_api_or_rewrite_original_pass(key, value, diagnostic_files):
    f = diagnostic_files
    plan = deepcopy(f["plan"])
    plan[key] = value
    del plan["plan_hash"]
    plan["plan_hash"] = diagnostics.qualification.adapter.digest(plan)
    f["path"].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="unsupported diagnostic contract"):
        diagnostics.read_plan(f["path"], diagnostics.continuation.sha(f["path"]))


@pytest.mark.parametrize("noise", ["", "null\n[]\n42\ninvalid", '{"event":"result"}',
    '{"event":"attestation"}\n{"event":"attestation"}'])
def test_missing_or_duplicate_attestation_is_rejected(noise):
    assert diagnostics.parse_attestation(noise) is None


def test_parser_measures_single_attestation_amid_noisy_logs(diagnostic_files):
    expected = attestation(diagnostic_files)
    stdout = "diagnostic log\nnull\n{broken\n" + event(expected) + '\n{"event":"phase","phase":"compile"}'
    assert diagnostics.parse_attestation(stdout) == expected


def test_child_measures_and_emits_attestation_before_mocked_target(diagnostic_files, original, candidate, valid_row, monkeypatch, capsys):
    f = diagnostic_files
    expected = attestation(f)
    monkeypatch.setattr(diagnostics.qualification, "load_plan", lambda path: original)
    monkeypatch.setattr(diagnostics.qualification, "gpu_snapshot", lambda: pytest.fail("GPU probe forbidden"))
    calls = []

    def target(plan, spec_hash, evidence):
        calls.append((plan, spec_hash, evidence))
        assert diagnostics.parse_attestation(capsys.readouterr().out) == expected
        return deepcopy(valid_row)

    monkeypatch.setattr(diagnostics.qualification, "run_one", target)
    row = diagnostics.run_child(f["path"], f["plan_sha"], f["original_path"], f["replay"], f["replay_sha"], candidate["spec_hash"])
    assert len(calls) == 1
    assert calls[0] == (original, candidate["spec_hash"], {"fixture": "mocked replay"})
    assert row["child_attestation"] == expected


@pytest.mark.parametrize("failure", ["plan_file", "sources", "replay", "original_plan", "unselected"])
def test_child_refuses_target_and_gpu_probe_before_any_attestation_mismatch(failure, diagnostic_files, original, candidate, monkeypatch, capsys):
    f = diagnostic_files
    selected = candidate["spec_hash"]
    mock_original = deepcopy(original)
    monkeypatch.setattr(diagnostics.qualification, "gpu_snapshot", lambda: pytest.fail("GPU probe forbidden"))
    monkeypatch.setattr(diagnostics.qualification, "run_one", lambda *a: pytest.fail("target forbidden"))
    monkeypatch.setattr(diagnostics.qualification, "load_plan", lambda path: mock_original)
    if failure == "plan_file":
        f["path"].write_text("tampered")
    elif failure == "sources":
        monkeypatch.setattr(diagnostics, "source_snapshot", lambda: {})
    elif failure == "replay":
        f["replay"].write_text("tampered")
    elif failure == "original_plan":
        mock_original["plan_hash"] = "wrong"
    else:
        selected = "unselected"
    with pytest.raises(ValueError):
        diagnostics.run_child(f["path"], f["plan_sha"], f["original_path"], f["replay"], f["replay_sha"], selected)
    assert diagnostics.parse_attestation(capsys.readouterr().out) is None


def test_consistent_result_and_child_measured_attestation_validate(diagnostic_files, valid_row, candidate, original):
    expected = attestation(diagnostic_files)
    valid_row["child_attestation"] = expected
    assert diagnostics.validate_result(valid_row, event(expected), candidate, original, expected)


@pytest.mark.parametrize("mutation", ["missing_event", "duplicate_event", "event_hash", "row_hash", "missing_row", "provenance", "likely_fit", "score"])
def test_result_rejects_missing_duplicate_or_tampered_attestations(mutation, diagnostic_files, valid_row, candidate, original):
    expected = attestation(diagnostic_files)
    valid_row["child_attestation"] = deepcopy(expected)
    stdout = event(expected)
    if mutation == "missing_event": stdout = ""
    elif mutation == "duplicate_event": stdout += "\n" + stdout
    elif mutation == "event_hash": stdout = event(dict(expected, replay_sha256="wrong"))
    elif mutation == "row_hash": valid_row["child_attestation"]["diagnostic_plan_sha256"] = "wrong"
    elif mutation == "missing_row": del valid_row["child_attestation"]
    elif mutation == "provenance": valid_row["spec_hash"] = "wrong"
    elif mutation == "likely_fit": valid_row["likely_fit"] = True
    else: valid_row["expected_probability"] += .001
    assert diagnostics.validate_result(valid_row, stdout, candidate, original, expected) is False


def test_load_acquisition_selects_only_the_five_preserved_unknowns():
    _, selected, aggregate, audit_path = diagnostics.load_acquisition(diagnostics.ROOT / "experiments")
    assert len(selected) == len({c["spec_hash"] for c in selected}) == 5
    assert {c["candidate_id"] for c in selected} == {"HOLDOUT-0012", "HOLDOUT-0028", "HOLDOUT-0029", "HOLDOUT-0031", "H2-0038"}
    assert aggregate["coverage_complete"] is True
    assert aggregate["all_checks_pass"] is False
    assert aggregate["public_api_enabled"] is False
    assert aggregate["unresolved"] == 5
    assert audit_path.is_file()


def test_no_public_diagnostic_export():
    import jaxoom
    assert not hasattr(jaxoom, "run_diagnostics")
    assert not hasattr(jaxoom, "WorkloadSpec")
