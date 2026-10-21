"""Synthetic persisted audit regressions; no GPU/backend initialization."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from experiments import oom_risk_gpu_diagnostic_audit as audit


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def idle(second):
    return dict(device_kind="NVIDIA GeForce RTX 3050 Laptop GPU", driver_version="566.07",
                physical_memory_bytes=4 * 1024**3, used_memory_bytes=0,
                free_memory_bytes=3964 * 1024**2, utilization_percent=0, compute_process_count=0,
                observed_at_utc=(datetime(2026, 10, 21, tzinfo=timezone.utc) + timedelta(seconds=second)).isoformat(),
                raw_gpu="NVIDIA GeForce RTX 3050 Laptop GPU, 566.07, 4096, 0, 3964, 0\n", raw_processes="")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    d = audit.diagnostics
    sources = {"synthetic.py": "frozen-source"}
    monkeypatch.setattr(d, "source_snapshot", lambda: deepcopy(sources))
    selected = []
    for index in range(5):
        spec = audit.q.adapter.WorkloadSpec("matmul", dict(m=index+1, n=1, k=1, depth=1), "float16")
        selected.append(dict(spec_hash=spec.spec_hash, spec=spec.to_dict(), candidate_id=f"UNKNOWN-{index}",
                             expected_probability=.7, expected_features={n: 1.0 for n in audit.q.adapter.FEATURE_NAMES}))
    original = dict(plan_hash="original-plan", policy_hash="policy", candidates=selected,
                    policy_contract={"pipeline_contract_hash": "pipeline"})
    acquisition = dict(all_checks_pass=False, coverage_complete=True, unresolved=5, public_api_enabled=False)
    anchor = tmp_path / "prior-audit.json"
    save(anchor, {"prior": "preserved"})
    monkeypatch.setattr(d, "load_acquisition", lambda base: (deepcopy(original), deepcopy(selected), deepcopy(acquisition), anchor))
    replay = dict(contract_hash="pipeline", public_api_absent=True, public_api_enabled=False,
                  completed=224, fresh_tracing_and_replay_pass=True, provenance={"backend": "gpu"})
    monkeypatch.setattr(audit.q.policy, "_replay_matches", lambda evidence, expected: evidence == replay and expected == "pipeline")
    monkeypatch.setattr(audit.q.policy, "evaluate_applicability", lambda spec, runtime, evidence, now:
                        {"status": "ELIGIBLE_FOR_QUALIFICATION" if runtime.get("backend") == "gpu" else "UNSUPPORTED"})
    folder = tmp_path / "run"
    folder.mkdir()
    plan = d.make_plan(original, selected, anchor)
    save(folder / "plan.json", plan)
    save(folder / "gpu_replay.json", replay)
    (folder / "gpu_replay.stdout.txt").write_text("replay complete\n")
    (folder / "gpu_replay.stderr.txt").write_text("")
    attestation = dict(source_sha256=sources, diagnostic_plan_sha256=d.continuation.sha(folder / "plan.json"),
                       replay_sha256=d.continuation.sha(folder / "gpu_replay.json"))
    save(folder / "initial_idle_snapshot.json", idle(0))
    for index, candidate in enumerate(selected):
        child = dict(event="result", spec_hash=candidate["spec_hash"], spec=candidate["spec"],
                     plan_hash=original["plan_hash"], policy_hash=original["policy_hash"],
                     fresh_probability=.7, expected_probability=.7, likely_fit=False,
                     fresh_features=candidate["expected_features"], backend_verified=True,
                     platform_version="PJRT C API\ncuda 12090", child_attestation=attestation,
                     applicability={"status": "ELIGIBLE_FOR_QUALIFICATION", "reasons": []},
                     runtime=dict(idle(index*10+1), backend="gpu"), outcome="FIT", phase="complete", gpu_runtime_executed=True)
        raw = [dict(event="attestation", **attestation)]
        raw += [dict(event="phase", phase=phase) for phase in ("initialization", "initialization", "compile", "execute")]
        raw.append(child)
        (folder / f"trial_{index:03d}.stdout.txt").write_text("\n".join(json.dumps(row) for row in raw))
        (folder / f"trial_{index:03d}.stderr.txt").write_text("")
        command = ["python", "-m", "experiments.oom_risk_gpu_diagnostics", "--child", candidate["spec_hash"],
                   "--diagnostic-plan-sha256", attestation["diagnostic_plan_sha256"], "--replay-sha256", attestation["replay_sha256"]]
        save(folder / f"trial_{index:03d}.json", dict(child, index=index, returncode=0, command=command,
             selected_spec_hash=candidate["spec_hash"], selected_probability=.7))
        save(folder / f"recovery_{index:03d}.json", idle(index*10+9))
    save(folder / "final_idle_snapshot.json", idle(60))
    save(folder / "summary.json", dict(status="DIAGNOSTICS_COMPLETE", selected=5, attempted=5,
         resolved_in_diagnostics=5, unresolved_in_diagnostics=0, stop_reason=None, outcomes={"FIT": 5},
         original_acquisition=acquisition, original_qualification_pass=False, public_api_enabled=False))
    return folder


def mutate_json(path, transform):
    value = json.loads(path.read_text())
    transform(value)
    save(path, value)


def test_consistent_diagnostics_never_pass_original_qualification(campaign):
    result = audit.audit_run(campaign)
    assert result["verified"] is result["consistency_pass"] is True
    assert result["diagnostics_pass"] is True
    assert result["original_qualification_pass"] is result["public_api_enabled"] is False
    assert result["aggregate"]["original_acquisition"]["all_checks_pass"] is False
    assert len(result["artifact_sha256"]) == len(list(campaign.iterdir()))


@pytest.mark.parametrize("mode", ["missing", "duplicate", "tampered", "late"])
def test_bad_raw_attestation_is_rejected(campaign, mode):
    path = campaign / "trial_000.stdout.txt"
    lines = path.read_text().splitlines()
    if mode == "missing": lines.pop(0)
    elif mode == "duplicate": lines.insert(0, lines[0])
    elif mode == "tampered": lines[0] = lines[0].replace("frozen-source", "changed")
    else: lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="attestation"):
        audit.audit_run(campaign)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "tampered", "phase"])
def test_bad_raw_result_or_phase_is_rejected(campaign, mode):
    path = campaign / "trial_000.stdout.txt"
    lines = path.read_text().splitlines()
    if mode == "missing": lines.pop()
    elif mode == "duplicate": lines.append(lines[-1])
    elif mode == "tampered": lines[-1] = lines[-1].replace('"outcome": "FIT"', '"outcome": "EXECUTION_OOM"')
    else: lines.pop(2)
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError):
        audit.audit_run(campaign)


def test_replay_file_byte_tampering_breaks_child_attestation(campaign):
    path = campaign / "gpu_replay.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="attestation|child argument mismatch"):
        audit.audit_run(campaign)


def test_replay_wrong_backend_is_rejected(campaign):
    mutate_json(campaign / "gpu_replay.json", lambda row: row["provenance"].update(backend="cpu"))
    with pytest.raises(ValueError, match="GPU replay"):
        audit.audit_run(campaign)


def test_sources_changed_after_freeze_are_rejected(campaign, monkeypatch):
    monkeypatch.setattr(audit.diagnostics, "source_snapshot", lambda: {"synthetic.py": "changed"})
    with pytest.raises(ValueError, match="source attestation"):
        audit.audit_run(campaign)


def test_acquisition_anchor_tampering_is_rejected(campaign):
    (campaign.parent / "prior-audit.json").write_text("changed")
    with pytest.raises(ValueError, match="acquisition anchor"):
        audit.audit_run(campaign)


@pytest.mark.parametrize("key,value", [("attempted", 4), ("resolved_in_diagnostics", 4),
    ("original_qualification_pass", True), ("public_api_enabled", True),
    ("status", "GPU_QUALIFICATION_PASS"), ("outcomes", {"FIT": 4, "COMPILE_OOM": 1})])
def test_summary_tampering_is_rejected(campaign, key, value):
    mutate_json(campaign / "summary.json", lambda row: row.update({key: value}))
    with pytest.raises(ValueError, match="summary"):
        audit.audit_run(campaign)


@pytest.mark.parametrize("name", ["trial_000.stderr.txt", "trial_000.stdout.txt", "recovery_000.json", "final_idle_snapshot.json"])
def test_missing_logs_or_recovery_telemetry_fails_closed(campaign, name):
    (campaign / name).unlink()
    with pytest.raises((ValueError, OSError)):
        audit.audit_run(campaign)


def test_busy_final_idle_is_rejected(campaign):
    mutate_json(campaign / "final_idle_snapshot.json", lambda row: row.update(compute_process_count=1))
    with pytest.raises(ValueError, match="idle metric"):
        audit.audit_run(campaign)


@pytest.mark.parametrize("outcome", ["COMPILE_TIMEOUT", "EXECUTE_TIMEOUT"])
def test_consistent_measured_timeout_remains_unknown(campaign, outcome):
    path = campaign / "trial_004.json"
    original = json.loads(path.read_text())
    stdout = campaign / "trial_004.stdout.txt"
    lines = stdout.read_text().splitlines()[:-1]
    if outcome == "COMPILE_TIMEOUT":
        lines.pop()
    stdout.write_text("\n".join(lines))
    measured = audit.q.timeout_result(stdout.read_text())
    assert measured["outcome"] == outcome
    attestation = audit.diagnostics.parse_attestation(stdout.read_text())
    save(path, dict(measured, returncode=None, index=4,
         selected_spec_hash=original["selected_spec_hash"], selected_probability=.7,
         command=original["command"], child_attestation_event=attestation))
    mutate_json(campaign / "summary.json", lambda row: row.update(
        status="DIAGNOSTICS_INCOMPLETE", resolved_in_diagnostics=4, unresolved_in_diagnostics=1,
        outcomes={"FIT": 4, outcome: 1}))
    result = audit.audit_run(campaign)
    assert result["consistency_pass"] is True
    assert result["diagnostics_pass"] is result["original_qualification_pass"] is False


def test_timeout_with_terminal_result_is_rejected(campaign):
    mutate_json(campaign / "trial_004.json", lambda row: row.update(outcome="EXECUTE_TIMEOUT", returncode=None))
    with pytest.raises(ValueError, match="terminal result"):
        audit.audit_run(campaign)


def test_raw_idle_telemetry_must_agree_with_structured_fields(campaign):
    mutate_json(campaign / "final_idle_snapshot.json", lambda row: row.update(raw_processes="123\n"))
    with pytest.raises(ValueError, match="raw idle telemetry"):
        audit.audit_run(campaign)


def test_own_output_is_excluded_from_manifest_and_never_overwritten(campaign, monkeypatch):
    output = campaign / "audit.json"
    monkeypatch.setattr("sys.argv", ["audit", "--run-dir", str(campaign), "--output", str(output)])
    audit.main()
    assert "audit.json" not in json.loads(output.read_text())["artifact_sha256"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.main()
    assert output.read_bytes() == before


def test_cli_reports_consistency_failure_without_claiming_qualification(campaign, monkeypatch, capsys):
    mutate_json(campaign / "summary.json", lambda row: row.update(attempted=0))
    monkeypatch.setattr("sys.argv", ["audit", "--run-dir", str(campaign)])
    with pytest.raises(SystemExit) as exc:
        audit.main()
    result = json.loads(capsys.readouterr().out)
    assert exc.value.code == 1
    assert result["consistency_pass"] is result["diagnostics_pass"] is False
    assert result["original_qualification_pass"] is result["public_api_enabled"] is False
