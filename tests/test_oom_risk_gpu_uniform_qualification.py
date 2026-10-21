"""CPU-only negative controls for the uniformly bounded qualification protocol."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import oom_risk_gpu_uniform_qualification as uniform


@pytest.fixture(scope="module")
def reference():
    return uniform.q.load_plan(uniform.REFERENCE_PLAN)


@pytest.fixture
def files(tmp_path, monkeypatch):
    # Keep every test mutation outside the repository, including JUnit evidence.
    monkeypatch.setattr(uniform, "ROOT", tmp_path)
    sources = {"runner.py": "source-fixture"}
    tests = {"tests/control.py": "test-fixture"}
    monkeypatch.setattr(uniform, "source_snapshot", lambda: deepcopy(sources))
    monkeypatch.setattr(uniform, "test_source_snapshot", lambda: deepcopy(tests))
    monkeypatch.setattr(uniform.subprocess, "check_output", lambda *a, **kw: "fixture-commit\n")
    auditor = tmp_path / "auditor.py"
    auditor.write_text("# fixture auditor\n")
    monkeypatch.setattr(uniform, "AUDITOR_PATH", auditor)
    junit = {}
    for platform in ("windows", "linux"):
        path = tmp_path / (platform + ".xml")
        path.write_text('<testsuites><testsuite tests="447" errors="0" failures="0" skipped="0"/></testsuites>')
        junit[platform] = dict(path=path.name, sha256=uniform.sha(path), tests=447)
    validation = tmp_path / "validation.json"
    evidence = dict(suite="full", source_sha256=sources, test_source_sha256=tests,
                    auditor_sha256=uniform.sha(auditor), junit=junit)
    validation.write_text(json.dumps(evidence))
    plan = uniform.make_plan(validation)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    replay = tmp_path / "replay.json"
    replay.write_text('{"fixture": "mocked GPU replay"}')
    return dict(root=tmp_path, plan=plan, path=plan_path, validation=validation,
                evidence=evidence, replay=replay, auditor=auditor)


def save_plan(files, plan, rehash=True):
    if rehash:
        plan.pop("plan_hash", None)
        plan["plan_hash"] = uniform.q.adapter.digest(plan)
    files["path"].write_text(json.dumps(plan))
    return uniform.sha(files["path"])


def call_child(files, selected):
    return uniform.run_child(files["path"], uniform.sha(files["path"]), files["replay"],
                             uniform.sha(files["replay"]), selected)


def test_actual_source_snapshot_binds_entry_point_and_all_imported_wrappers():
    actual = uniform.source_snapshot()
    assert len(actual) == 9
    assert actual["experiments/oom_risk_gpu_uniform_qualification.py"] == uniform.sha(uniform.__file__)
    assert set(uniform.diagnostics.source_snapshot()) < set(actual)
    assert all(uniform.sha(uniform.ROOT / p) == digest for p, digest in actual.items())
    assert uniform.verify_sources(actual) == actual
    assert "tests/test_oom_risk_gpu_uniform_qualification.py" in uniform.test_source_snapshot()


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed"])
def test_source_snapshot_rejects_any_difference(mutation):
    sources = uniform.source_snapshot()
    first = next(iter(sources))
    if mutation == "missing": del sources[first]
    elif mutation == "extra": sources["unmeasured.py"] = "wrong"
    else: sources[first] = "wrong"
    with pytest.raises(ValueError, match="source mismatch"):
        uniform.verify_sources(sources)


def test_plan_roundtrip_exact_original_population_and_pilot_first_order(files, reference):
    plan = uniform.read_plan(files["path"], uniform.sha(files["path"]))
    assert plan == files["plan"]
    assert plan["candidates"] == reference["candidates"]
    assert len(plan["order"]) == len(set(plan["order"])) == 200
    assert plan["order"][:18] == reference["pilot_spec_hashes"]
    assert plan["order"][18:] == [c["spec_hash"] for c in reference["candidates"]
                                       if c["spec_hash"] not in reference["pilot_spec_hashes"]]
    assert plan["timeout_seconds"] == 600
    assert plan["max_campaign_seconds"] == 10800
    assert plan["public_api_enabled"] is plan["historical_qualification_pass"] is False
    assert plan["validation_sha256"] == uniform.sha(files["validation"])


def test_plan_rejects_byte_tamper_and_content_tamper(files):
    digest = uniform.sha(files["path"])
    files["path"].write_text(files["path"].read_text() + "\n")
    with pytest.raises(ValueError, match="plan file mismatch"):
        uniform.read_plan(files["path"], digest)
    plan = deepcopy(files["plan"])
    plan["timeout_seconds"] = 601
    save_plan(files, plan, rehash=False)
    with pytest.raises(ValueError, match="plan content mismatch"):
        uniform.read_plan(files["path"], uniform.sha(files["path"]))


@pytest.mark.parametrize("key,value", [
    ("timeout_seconds", 180), ("timeout_seconds", 600.0),
    ("max_campaign_seconds", 10801), ("min_host_free_bytes", 0),
    ("attempts_per_recipe", True), ("initialization", "random"), ("seed", 1),
    ("threshold", .21), ("protocol", "other"),
    ("public_api_enabled", True), ("public_api_enabled", 0),
    ("historical_qualification_pass", True), ("historical_qualification_pass", 0),
])
def test_rehashed_plan_cannot_relax_fixed_contract(files, key, value):
    plan = deepcopy(files["plan"])
    plan[key] = value
    digest = save_plan(files, plan)
    with pytest.raises(ValueError, match="unsupported uniform qualification contract"):
        uniform.read_plan(files["path"], digest)


@pytest.mark.parametrize("mutation", ["reference_hash", "reference_bytes", "policy", "candidate", "pilot", "order", "duplicate", "missing"])
def test_rehashed_plan_cannot_change_population_or_reference(files, mutation):
    plan = deepcopy(files["plan"])
    if mutation == "reference_hash": plan["reference_plan_hash"] = "wrong"
    elif mutation == "reference_bytes": plan["reference_plan_sha256"] = "wrong"
    elif mutation == "policy": plan["policy_hash"] = "wrong"
    elif mutation == "candidate": plan["candidates"][0]["expected_probability"] += .001
    elif mutation == "pilot": plan["pilot_spec_hashes"].reverse()
    elif mutation == "order": plan["order"].reverse()
    elif mutation == "duplicate": plan["order"][-1] = plan["order"][0]
    else: plan["order"].pop()
    with pytest.raises(ValueError, match="reference/catalog mismatch"):
        uniform.read_plan(files["path"], save_plan(files, plan))


def test_full_test_evidence_roundtrip(files):
    assert uniform.validate_test_evidence(files["validation"]) == files["evidence"]


@pytest.mark.parametrize("relative", ["../outside.xml", "/absolute.xml"])
def test_evidence_path_must_stay_repository_relative(files, relative):
    with pytest.raises(ValueError):
        uniform.repo_path(relative)


@pytest.mark.parametrize("mutation", ["partial", "source", "tests", "auditor", "hash", "count", "small", "different_population", "failure", "error", "skip", "duplicate_suite", "escape"])
def test_regression_evidence_rejects_stale_partial_or_failed_suites(files, mutation):
    value = deepcopy(files["evidence"])
    if mutation == "partial": value["suite"] = "targeted"
    elif mutation == "source": value["source_sha256"]["runner.py"] = "wrong"
    elif mutation == "tests": value["test_source_sha256"]["tests/control.py"] = "wrong"
    elif mutation == "auditor": files["auditor"].write_text("changed")
    elif mutation == "hash": value["junit"]["linux"]["sha256"] = "wrong"
    elif mutation == "count": value["junit"]["windows"]["tests"] = 448
    elif mutation == "escape": value["junit"]["linux"]["path"] = "../outside.xml"
    else:
        report = files["root"] / "linux.xml"
        count = 446 if mutation == "small" else 448 if mutation == "different_population" else 447
        failure, error, skip = [int(mutation == name) for name in ("failure", "error", "skip")]
        suite = f'<testsuite tests="{count}" failures="{failure}" errors="{error}" skipped="{skip}"/>'
        report.write_text('<testsuites>' + suite * (2 if mutation == "duplicate_suite" else 1) + '</testsuites>')
        value["junit"]["linux"].update(sha256=uniform.sha(report), tests=count)
    files["validation"].write_text(json.dumps(value))
    with pytest.raises((ValueError, OSError)):
        uniform.validate_test_evidence(files["validation"])


def valid_row(candidate, reference):
    return dict(spec_hash=candidate["spec_hash"], spec=deepcopy(candidate["spec"]),
                plan_hash=reference["plan_hash"], policy_hash=reference["policy_hash"],
                backend_verified=True, platform_version="cuda 12090",
                applicability={"status": "ELIGIBLE_FOR_QUALIFICATION"},
                fresh_probability=candidate["expected_probability"],
                fresh_features=deepcopy(candidate["expected_features"]),
                likely_fit=candidate["expected_probability"] <= .2,
                expected_probability=candidate["expected_probability"],
                outcome="FIT", phase="complete", gpu_runtime_executed=True)


def test_child_emits_measured_attestation_before_any_gpu_target(files, reference, monkeypatch, capsys):
    candidate = reference["candidates"][0]
    expected = uniform.expected_attestation(files["plan"], files["path"], files["replay"])
    sentinel = lambda: pytest.fail("unmocked GPU probe")
    monkeypatch.setattr(uniform.q, "gpu_snapshot", sentinel)
    def target(plan, spec_hash, replay):
        assert uniform.diagnostics.parse_attestation(capsys.readouterr().out) == expected
        assert plan == reference and spec_hash == candidate["spec_hash"]
        assert replay == {"fixture": "mocked GPU replay"}
        assert uniform.q.gpu_snapshot is uniform.diagnostics.resume.strict_snapshot
        return valid_row(candidate, reference)
    monkeypatch.setattr(uniform.q, "run_one", target)
    result = call_child(files, candidate["spec_hash"])
    assert result["child_attestation"] == expected
    assert uniform.q.gpu_snapshot is sentinel


def test_child_restores_snapshot_function_when_target_raises(files, reference, monkeypatch):
    sentinel = lambda: None
    monkeypatch.setattr(uniform.q, "gpu_snapshot", sentinel)
    def target(*args):
        raise RuntimeError("mock target failure")
    monkeypatch.setattr(uniform.q, "run_one", target)
    with pytest.raises(RuntimeError, match="mock target failure"):
        call_child(files, reference["candidates"][0]["spec_hash"])
    assert uniform.q.gpu_snapshot is sentinel


@pytest.mark.parametrize("mutation", ["source", "plan", "validation", "replay", "unselected"])
def test_child_refuses_before_gpu_on_bad_attestation(files, reference, monkeypatch, capsys, mutation):
    selected = reference["candidates"][0]["spec_hash"]
    replay_sha = uniform.sha(files["replay"])
    plan_sha = uniform.sha(files["path"])
    monkeypatch.setattr(uniform.q, "run_one", lambda *a: pytest.fail("GPU target forbidden"))
    monkeypatch.setattr(uniform.q, "gpu_snapshot", lambda: pytest.fail("GPU probe forbidden"))
    if mutation == "source": monkeypatch.setattr(uniform, "source_snapshot", lambda: {})
    elif mutation == "plan": files["path"].write_text("changed")
    elif mutation == "validation": files["validation"].write_text("changed")
    elif mutation == "replay": files["replay"].write_text("changed")
    else: selected = "outside-catalog"
    with pytest.raises(ValueError):
        uniform.run_child(files["path"], plan_sha, files["replay"], replay_sha, selected)
    assert uniform.diagnostics.parse_attestation(capsys.readouterr().out) is None


@pytest.mark.parametrize("safe", [True, False])
def test_valid_result_accepts_safe_and_uncertain_decisions(files, reference, safe):
    candidate = next(c for c in reference["candidates"] if (c["expected_probability"] <= .2) == safe)
    row = valid_row(candidate, reference)
    attestation = uniform.expected_attestation(files["plan"], files["path"], files["replay"])
    row["child_attestation"] = attestation
    stdout = json.dumps(dict(event="attestation", **attestation))
    assert uniform.validate_result(row, stdout, candidate, reference, attestation)


@pytest.mark.parametrize("mutation", ["event", "duplicate", "row", "score", "decision", "identity", "cuda"])
def test_result_rejects_child_attestation_or_provenance_tamper(files, reference, mutation):
    candidate = reference["candidates"][0]
    row = valid_row(candidate, reference)
    attestation = uniform.expected_attestation(files["plan"], files["path"], files["replay"])
    row["child_attestation"] = deepcopy(attestation)
    stdout = json.dumps(dict(event="attestation", **attestation))
    if mutation == "event": stdout = ""
    elif mutation == "duplicate": stdout += "\n" + stdout
    elif mutation == "row": row["child_attestation"]["validation_sha256"] = "wrong"
    elif mutation == "score": row["expected_probability"] += .01
    elif mutation == "decision": row["likely_fit"] = not row["likely_fit"]
    elif mutation == "identity": row["spec_hash"] = "wrong"
    else: row["platform_version"] = "cpu"
    assert not uniform.validate_result(row, stdout, candidate, reference, attestation)


@pytest.fixture
def complete_rows(reference):
    return [dict(valid_row(c, reference), selected_spec_hash=c["spec_hash"])
            for c in reference["candidates"]]


def test_summary_requires_complete_fresh_population_and_preserves_api_gate(files, complete_rows):
    result = uniform.summarize(files["plan"], complete_rows, None, True)
    assert result["runtime_checks_pass"] is True
    assert result["coverage_complete"] is True and result["attempted"] == 200
    assert result["pilots_fit"] == 18 and result["likely_fit_count"] == 157
    assert result["not_run"] == result["likely_fit_failures"] == 0
    assert result["public_api_enabled"] is result["historical_qualification_pass"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "pilot_oom", "safe_oom", "safe_decision_forged", "timeout", "stop", "final"])
def test_summary_fails_closed_on_incomplete_unknown_or_failed_gate(files, complete_rows, mutation):
    rows, stop, final = deepcopy(complete_rows), None, True
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[-1] = deepcopy(rows[0])
    elif mutation == "unknown": rows[-1]["selected_spec_hash"] = "unknown"
    elif mutation == "pilot_oom":
        row = next(r for r in rows if r["selected_spec_hash"] in files["plan"]["pilot_spec_hashes"])
        row.update(outcome="COMPILE_OOM", phase="compile", gpu_runtime_executed=False)
    elif mutation == "safe_oom":
        row = next(r for r in rows if r["likely_fit"])
        row.update(outcome="EXECUTION_OOM", phase="execute")
    elif mutation == "safe_decision_forged":
        row = next(r for r in rows if r["likely_fit"] and r["selected_spec_hash"] not in files["plan"]["pilot_spec_hashes"])
        row.update(likely_fit=False, outcome="EXECUTION_OOM", phase="execute")
    elif mutation == "timeout": rows[-1].update(outcome="EXECUTE_TIMEOUT", phase="execute")
    elif mutation == "stop": stop = "HOST_STORAGE_GATE"
    else: final = False
    result = uniform.summarize(files["plan"], rows, stop, final)
    assert result["runtime_checks_pass"] is False
    assert result["public_api_enabled"] is False


@pytest.mark.parametrize("outcome,phase,executed", [("COMPILE_OOM", "compile", False), ("EXECUTION_OOM", "execute", True)])
def test_phase_correct_oom_acceptable_only_for_uncertain_nonpilot(files, complete_rows, outcome, phase, executed):
    row = next(r for r in complete_rows if not r["likely_fit"] and r["selected_spec_hash"] not in files["plan"]["pilot_spec_hashes"])
    row.update(outcome=outcome, phase=phase, gpu_runtime_executed=executed)
    assert uniform.summarize(files["plan"], complete_rows, None, True)["runtime_checks_pass"]


def test_uniform_runner_is_not_public_api():
    import jaxoom
    assert not hasattr(jaxoom, "WorkloadSpec")
    assert not hasattr(jaxoom, "run_uniform_qualification")
