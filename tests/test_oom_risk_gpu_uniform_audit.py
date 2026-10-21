"""Synthetic source-bound campaigns; all telemetry is saved fixture data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from experiments import oom_risk_gpu_uniform_audit as audit


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def idle(second):
    return dict(device_kind="NVIDIA GeForce RTX 3050 Laptop GPU", driver_version="566.07",
                physical_memory_bytes=4*1024**3, used_memory_bytes=0, free_memory_bytes=3964*1024**2,
                utilization_percent=0, compute_process_count=0,
                observed_at_utc=(datetime(2026, 10, 21, tzinfo=timezone.utc)+timedelta(seconds=second)).isoformat(),
                raw_gpu="NVIDIA GeForce RTX 3050 Laptop GPU, 566.07, 4096, 0, 3964, 0\n", raw_processes="")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    u = audit.uniform
    reference = json.loads(u.REFERENCE_PLAN.read_text())
    model = audit.q.adapter.load_frozen_model()
    original_root = u.ROOT
    # The catalog and frozen model remain real; only host/test/source resources are synthetic.
    source = {f"source_{i}.py": f"hash-{i}" for i in range(9)}
    monkeypatch.setattr(u, "ROOT", tmp_path)
    monkeypatch.setattr(u, "source_snapshot", lambda: deepcopy(source))
    monkeypatch.setattr(u, "test_source_snapshot", lambda: {"tests/synthetic.py": "tested"})
    monkeypatch.setattr(u, "load_reference", lambda: deepcopy(reference))
    monkeypatch.setattr(u.subprocess, "check_output", lambda *args, **kwargs: "synthetic-source-commit\n")
    monkeypatch.setattr(u, "REFERENCE_PLAN", original_root / "experiments/oom_risk_gpu_qualification_2026-10-21_run1/plan.json")
    validation = dict(suite="full", source_sha256=source, test_source_sha256=u.test_source_snapshot(),
                      auditor_sha256=u.sha(audit.__file__), junit={})
    for platform in ("windows", "linux"):
        report = tmp_path / f"{platform}.xml"
        report.write_text('<testsuites><testsuite tests="447" errors="0" failures="0" skipped="0"/></testsuites>')
        validation["junit"][platform] = dict(path=report.name, sha256=u.sha(report), tests=447)
    validation_path = tmp_path / "validation.json"
    save(validation_path, validation)
    plan = u.make_plan(validation_path)
    run = tmp_path / "run"
    run.mkdir()
    save(run / "plan.json", plan)
    replay = json.loads((original_root / "experiments/oom_risk_gpu_qualification_2026-10-21_run1/gpu_replay.json").read_text())
    save(run / "gpu_replay.json", replay)
    (run / "gpu_replay.stdout.txt").write_text("replayed 224\n")
    (run / "gpu_replay.stderr.txt").write_text("")
    save(run / "initial_idle_snapshot.json", idle(0))
    monkeypatch.setattr(audit.q.policy, "evaluate_applicability", lambda spec, runtime, evidence, now:
                        {"status": "ELIGIBLE_FOR_QUALIFICATION" if runtime.get("backend") == "gpu" else "UNSUPPORTED"})
    attestation = u.expected_attestation(plan, run / "plan.json", run / "gpu_replay.json")
    by_id = {c["spec_hash"]: c for c in reference["candidates"]}

    def populate(count=18, stop="CAMPAIGN_TIME_BUDGET", last_outcome=None):
        rows = []
        for index, spec_hash in enumerate(plan["order"][:count]):
            candidate = by_id[spec_hash]
            outcome = last_outcome if index == count-1 and last_outcome else "FIT"
            is_pilot = spec_hash in plan["pilot_spec_hashes"]
            command = ["python", "-m", "experiments.oom_risk_gpu_uniform_qualification", "--child", spec_hash,
                       "--plan-sha256", attestation["qualification_plan_sha256"], "--replay-sha256", attestation["replay_sha256"]]
            raw = [dict(event="attestation", **attestation)]
            phase_names = ["initialization", "initialization", "compile"]
            if outcome != "COMPILE_OOM": phase_names.append("execute")
            raw.extend(dict(event="phase", phase=phase) for phase in phase_names)
            if outcome.endswith("TIMEOUT"):
                row = dict(audit.q.timeout_result("\n".join(json.dumps(e) for e in raw)),
                           child_attestation_event=attestation, returncode=None)
            else:
                child = dict(event="result", spec_hash=spec_hash, spec=candidate["spec"],
                    plan_hash=reference["plan_hash"], policy_hash=reference["policy_hash"],
                    expected_probability=candidate["expected_probability"], likely_fit=candidate["expected_probability"] <= .2,
                    fresh_features=candidate["expected_features"],
                    fresh_probability=audit.q.adapter._score_features(candidate["expected_features"], model),
                    backend_verified=True, platform_version="PJRT C API\ncuda 12090", child_attestation=attestation,
                    applicability={"status": "ELIGIBLE_FOR_QUALIFICATION", "reasons": []},
                    runtime=dict(audit.q.policy.EXACT_ENVIRONMENT, **idle(index*10+2)),
                    outcome=outcome, phase="complete" if outcome == "FIT" else "compile" if outcome == "COMPILE_OOM" else "execute",
                    gpu_runtime_executed=outcome != "COMPILE_OOM", wall_seconds=3.0)
                if outcome.endswith("OOM"): child["error"] = "RESOURCE_EXHAUSTED: out of memory"
                raw.append(child)
                row = dict(child, returncode=0)
            row.update(index=index, selected_spec_hash=spec_hash, selected_probability=candidate["expected_probability"],
                       is_pilot=is_pilot, command=command)
            save(run / f"preflight_{index:03d}.json", dict(idle=idle(index*10+1), host_free_bytes=1024**3))
            save(run / f"trial_{index:03d}.json", row)
            (run / f"trial_{index:03d}.stdout.txt").write_text("\n".join(json.dumps(e) for e in raw))
            (run / f"trial_{index:03d}.stderr.txt").write_text("")
            save(run / f"recovery_{index:03d}.json", idle(index*10+9))
            rows.append(row)
        save(run / "final_idle_snapshot.json", idle(count*10+1))
        save(run / "summary.json", u.summarize(plan, rows, stop, True))
        return rows

    return dict(run=run, root=tmp_path, plan=plan, reference=reference, populate=populate, attestation=attestation)


def mutate(path, action):
    row = json.loads(path.read_text())
    action(row)
    save(path, row)


def test_all_200_unique_fresh_results_qualify_only_new_uniform_campaign(campaign):
    campaign["populate"](200, None)
    result = audit.audit_run(campaign["run"])
    assert result["consistency_pass"] is result["qualification_pass"] is True
    assert result["aggregate"]["coverage_complete"] is True
    assert result["aggregate"]["pilots_fit"] == 18
    assert result["historical_qualification_pass"] is result["public_api_enabled"] is False


def test_consistent_budget_stopped_prefix_does_not_qualify(campaign):
    campaign["populate"]()
    result = audit.audit_run(campaign["run"])
    assert result["consistency_pass"] is True
    assert result["qualification_pass"] is False
    assert result["aggregate"]["not_run"] == 182


def test_consistent_timeout_retains_unknown_and_stops(campaign):
    campaign["populate"](19, "EXECUTE_TIMEOUT", "EXECUTE_TIMEOUT")
    result = audit.audit_run(campaign["run"])
    assert result["consistency_pass"] is True
    assert result["qualification_pass"] is False
    assert result["aggregate"]["outcomes"]["EXECUTE_TIMEOUT"] == 1


def test_consistent_false_safe_oom_is_failed_qualification(campaign):
    # Every pilot must FIT irrespective of score; this non-FIT pilot is a hard stop.
    campaign["populate"](1, "EXECUTION_OOM", "EXECUTION_OOM")
    result = audit.audit_run(campaign["run"])
    assert result["consistency_pass"] is True
    assert result["qualification_pass"] is False


@pytest.mark.parametrize("mode", ["missing", "duplicate", "tampered", "late"])
def test_attestation_tampering_cannot_pass(campaign, mode):
    campaign["populate"](1, "CAMPAIGN_TIME_BUDGET")
    path = campaign["run"] / "trial_000.stdout.txt"
    lines = path.read_text().splitlines()
    if mode == "missing": lines.pop(0)
    elif mode == "duplicate": lines.insert(0, lines[0])
    elif mode == "tampered": lines[0] = lines[0].replace('"hash-0"', '"wrong"')
    else: lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError): audit.audit_run(campaign["run"])


@pytest.mark.parametrize("mode", ["duplicate", "changed", "phase"])
def test_raw_result_and_phases_must_match(campaign, mode):
    campaign["populate"](1)
    path = campaign["run"] / "trial_000.stdout.txt"
    lines = path.read_text().splitlines()
    if mode == "duplicate": lines.append(lines[-1])
    elif mode == "changed": lines[-1] = lines[-1].replace('"outcome": "FIT"', '"outcome": "COMPILE_OOM"')
    else: lines.pop(2)
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError): audit.audit_run(campaign["run"])


@pytest.mark.parametrize("field,value", [("host_free_bytes", 0), ("host_free_bytes", True)])
def test_host_storage_gate_is_independently_checked(campaign, field, value):
    campaign["populate"](1)
    mutate(campaign["run"] / "preflight_000.json", lambda row: row.update({field: value}))
    with pytest.raises(ValueError, match="storage"): audit.audit_run(campaign["run"])


def test_preflight_cannot_predate_previous_recovery(campaign):
    campaign["populate"](2)
    mutate(campaign["run"] / "preflight_001.json", lambda row: row.update(idle=idle(0)))
    with pytest.raises(ValueError, match="preflight precedes"): audit.audit_run(campaign["run"])


def test_continued_launch_after_failed_pilot_is_rejected(campaign):
    rows = campaign["populate"](2)
    # A parent claiming a stop only after later launches cannot conceal a prior failed trial.
    path = campaign["run"] / "trial_000.json"
    mutate(path, lambda row: row.update(returncode=1, outcome="CHILD_PROCESS_FAILURE"))
    mutate(campaign["run"] / "summary.json", lambda row: row.update(stop_reason="CHILD_PROCESS_FAILURE"))
    with pytest.raises(ValueError, match="continued after"): audit.audit_run(campaign["run"])


@pytest.mark.parametrize("key,value", [("selected_spec_hash", "other"), ("index", True), ("is_pilot", False)])
def test_trial_identity_cannot_be_replaced_or_carried_over(campaign, key, value):
    campaign["populate"](1)
    mutate(campaign["run"] / "trial_000.json", lambda row: row.update({key: value}))
    with pytest.raises(ValueError): audit.audit_run(campaign["run"])


@pytest.mark.parametrize("key,value", [("runtime_checks_pass", True), ("coverage_complete", True),
    ("public_api_enabled", True), ("historical_qualification_pass", True), ("attempted", 200)])
def test_aggregate_cannot_promote_incomplete_campaign(campaign, key, value):
    campaign["populate"](1)
    mutate(campaign["run"] / "summary.json", lambda row: row.update({key: value}))
    with pytest.raises(ValueError, match="summary"): audit.audit_run(campaign["run"])


@pytest.mark.parametrize("change", ["junit", "manifest", "test_sources", "sources", "auditor"])
def test_bound_full_test_evidence_cannot_change(campaign, monkeypatch, change):
    campaign["populate"](1)
    root = campaign["root"]
    if change == "junit": (root / "windows.xml").write_text("changed")
    elif change == "manifest": (root / "validation.json").write_text("changed")
    elif change == "test_sources": monkeypatch.setattr(audit.uniform, "test_source_snapshot", lambda: {})
    elif change == "sources": monkeypatch.setattr(audit.uniform, "source_snapshot", lambda: {})
    else: monkeypatch.setattr(audit.uniform, "AUDITOR_PATH", root / "nonexistent.py")
    with pytest.raises((ValueError, OSError)): audit.audit_run(campaign["run"])


def test_failed_recovery_can_be_consistent_but_never_pass(campaign):
    campaign["populate"](1, "RuntimeError: GPU did not return to the frozen idle state")
    (campaign["run"] / "recovery_000.json").unlink()
    mutate(campaign["run"] / "summary.json", lambda row: row.update(final_checks_pass=False))
    result = audit.audit_run(campaign["run"])
    assert result["consistency_pass"] is True and result["qualification_pass"] is False


def test_output_manifest_excludes_itself_and_refuses_overwrite(campaign, monkeypatch):
    campaign["populate"](1)
    output = campaign["run"] / "audit.json"
    monkeypatch.setattr("sys.argv", ["audit", "--run-dir", str(campaign["run"]), "--output", str(output)])
    audit.main()
    assert "audit.json" not in json.loads(output.read_text())["artifact_sha256"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError): audit.main()
    assert output.read_bytes() == before
