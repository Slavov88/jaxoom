"""Read-only persisted evidence audit; consistency never qualifies frozen V2."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from experiments import oom_risk_gpu_diagnostics as diagnostics

q = diagnostics.qualification


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def events(stdout):
    result = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def check_idle(snapshot):
    """Historical observation validation; no fresh hardware probing."""
    policy = q.policy
    for name in ("device_kind", "driver_version", "physical_memory_bytes"):
        expected = policy.EXACT_ENVIRONMENT[name]
        require(type(snapshot.get(name)) is type(expected) and snapshot[name] == expected,
                "unsupported idle identity: " + name)
    for name, lo, hi in (
        ("free_memory_bytes", policy.MIN_FREE_BYTES, 4 * 1024**3),
        ("used_memory_bytes", 0, policy.MAX_USED_BYTES),
        ("utilization_percent", 0, policy.MAX_UTILIZATION),
        ("compute_process_count", 0, 0),
    ):
        require(type(snapshot.get(name)) is int and lo <= snapshot[name] <= hi,
                "invalid or busy idle metric: " + name)
    require(snapshot["free_memory_bytes"] + snapshot["used_memory_bytes"] <= 4 * 1024**3,
            "contradictory idle memory")
    observed = datetime.fromisoformat(snapshot["observed_at_utc"])
    require(observed.tzinfo is not None, "idle observation requires timezone")
    require(isinstance(snapshot.get("raw_gpu"), str) and isinstance(snapshot.get("raw_processes"), str),
            "missing raw idle telemetry")
    raw = [line for line in snapshot["raw_gpu"].splitlines() if line.strip()]
    require(len(raw) == 1, "idle raw GPU count mismatch")
    name, driver, total, used, free, utilization = [part.strip() for part in raw[0].split(",")]
    expected = dict(device_kind=name, driver_version=driver, physical_memory_bytes=int(total) * 1024**2,
                    used_memory_bytes=int(used) * 1024**2, free_memory_bytes=int(free) * 1024**2,
                    utilization_percent=int(utilization),
                    compute_process_count=len([line for line in snapshot["raw_processes"].splitlines() if line.strip()]))
    require(all(snapshot[key] == value for key, value in expected.items()), "raw idle telemetry mismatch")
    return observed


def check_command(command, candidate, attestation):
    require(isinstance(command, list) and all(isinstance(part, str) for part in command), "invalid child command")
    for option, expected in (
        ("-m", "experiments.oom_risk_gpu_diagnostics"), ("--child", candidate["spec_hash"]),
        ("--diagnostic-plan-sha256", attestation["diagnostic_plan_sha256"]),
        ("--replay-sha256", attestation["replay_sha256"]),
    ):
        require(command.count(option) == 1, "missing or duplicate child argument: " + option)
        require(command.index(option) + 1 < len(command) and command[command.index(option) + 1] == expected,
                "child argument mismatch: " + option)


def check_trial(row, stdout, candidate, original, attestation, index, replay):
    require(type(row.get("index")) is int and row["index"] == index, "trial index mismatch")
    require(row.get("selected_spec_hash") == candidate["spec_hash"] and
            row.get("selected_probability") == candidate["expected_probability"], "selected recipe mismatch")
    check_command(row.get("command"), candidate, attestation)
    raw = events(stdout)
    require(diagnostics.parse_attestation(stdout) == attestation, "missing duplicate or tampered attestation")
    first_attestation = next(i for i, item in enumerate(raw) if item.get("event") == "attestation")
    require(not any(item.get("event") in ("phase", "result") for item in raw[:first_attestation]),
            "attestation occurred after target events")
    phases = [item.get("phase") for item in raw if item.get("event") == "phase"]
    results = [item for item in raw if item.get("event") == "result"]
    outcome = row.get("outcome")
    if outcome in ("COMPILE_TIMEOUT", "EXECUTE_TIMEOUT"):
        require(row.get("returncode") is None and not results, "timeout has terminal result or exit code")
        require(row.get("child_attestation_event") == attestation, "timeout attestation mismatch")
        measured = q.timeout_result(stdout)
        require(all(row.get(key) == value for key, value in measured.items()), "timeout phase mismatch")
        expected_phases = ["initialization", "initialization", "compile"]
        if outcome == "EXECUTE_TIMEOUT":
            expected_phases.append("execute")
        require(phases == expected_phases, "timeout raw phase sequence mismatch")
        return False
    require(type(row.get("returncode")) is int and row["returncode"] == 0 and len(results) == 1,
            "missing duplicate or failed child result")
    child = results[0]
    require(all(row.get(key) == value for key, value in child.items()), "raw result and saved row differ")
    require(diagnostics.validate_result(child, stdout, candidate, original, attestation),
            "child provenance or measured attestation invalid")
    require(q.trial_passes(child), "unresolved or invalid terminal outcome")
    require(outcome in ("FIT", "COMPILE_OOM", "EXECUTION_OOM"), "unsupported terminal outcome")
    expected_phases = ["initialization", "initialization", "compile"]
    if outcome != "COMPILE_OOM":
        expected_phases.append("execute")
    require(phases == expected_phases, "raw phase sequence mismatch")
    if outcome.endswith("OOM"):
        require(q.classify_error(child.get("error", ""), child["phase"]) == outcome,
                "OOM classification unsupported by terminal error")
    runtime = child["runtime"]
    observed = datetime.fromisoformat(runtime["observed_at_utc"])
    eligibility = q.policy.evaluate_applicability(q.adapter.WorkloadSpec.from_dict(candidate["spec"]),
                                                runtime, replay, now=observed)
    require(eligibility["status"] == "ELIGIBLE_FOR_QUALIFICATION" and not child["applicability"].get("reasons"),
            "persisted runtime is ineligible")
    return True


def audit_run(run_dir, evidence_root=None, *, exclude_paths=()):
    """Raise on inconsistent evidence; report diagnostic resolution separately."""
    run_dir = Path(run_dir)
    evidence_root = Path(evidence_root) if evidence_root is not None else diagnostics.ROOT / "experiments"
    original, selected, aggregate, audit_path = diagnostics.load_acquisition(evidence_root)
    require(len(selected) == len({item["spec_hash"] for item in selected}) == 5, "expected exactly five unknown recipes")
    plan_path = run_dir / "plan.json"
    plan_sha = diagnostics.continuation.sha(plan_path)
    plan = diagnostics.read_plan(plan_path, plan_sha)
    measured_sources = diagnostics.verify_sources(plan["source_sha256"])
    require(plan["original_plan_hash"] == original["plan_hash"] and plan["candidates"] == selected,
            "diagnostic selection or original plan mismatch")
    require(plan["acquisition_audit_sha256"] == diagnostics.continuation.sha(audit_path), "acquisition anchor mismatch")
    require(plan.get("protocol") == "restricted-v2-timeout-diagnostics-1" and
            type(plan.get("attempts_per_recipe")) is int and plan["attempts_per_recipe"] == 1 and
            plan.get("max_campaign_seconds") == 3600 and plan.get("seed") is None,
            "unsupported diagnostic acquisition contract")
    replay_path = run_dir / "gpu_replay.json"
    replay = read_json(replay_path)
    require(q.policy._replay_matches(replay, original["policy_contract"]["pipeline_contract_hash"]) and
            replay["provenance"]["backend"] == "gpu" and replay["public_api_absent"] is True,
            "fresh GPU replay invalid")
    require((run_dir / "gpu_replay.stdout.txt").is_file() and (run_dir / "gpu_replay.stderr.txt").is_file(),
            "replay logs missing")
    expected_attestation = dict(source_sha256=measured_sources, diagnostic_plan_sha256=plan_sha,
                                replay_sha256=diagnostics.continuation.sha(replay_path))
    initial = check_idle(read_json(run_dir / "initial_idle_snapshot.json"))
    paths = sorted(run_dir.glob("trial_*.json"))
    require(len(paths) <= len(selected), "extra diagnostic trials")
    rows = []
    resolved = 0
    last_observed = initial
    for index, path in enumerate(paths):
        require(path.name == f"trial_{index:03d}.json", "missing or noncontiguous trial")
        stdout_path, stderr_path = path.with_suffix(".stdout.txt"), path.with_suffix(".stderr.txt")
        require(stdout_path.is_file() and stderr_path.is_file(), "trial raw logs missing")
        row = read_json(path)
        resolved += check_trial(row, stdout_path.read_text(encoding="utf-8"), selected[index], original,
                                expected_attestation, index, replay)
        recovery = check_idle(read_json(run_dir / f"recovery_{index:03d}.json"))
        require(recovery >= last_observed, "recovery observations out of order")
        last_observed = recovery
        rows.append(row)
    final = check_idle(read_json(run_dir / "final_idle_snapshot.json"))
    require(final >= last_observed, "final idle observation predates trials")
    summary = read_json(run_dir / "summary.json")
    stop = summary.get("stop_reason")
    expected = dict(status="DIAGNOSTICS_COMPLETE" if resolved == len(selected) and stop is None else "DIAGNOSTICS_INCOMPLETE",
                    selected=len(selected), attempted=len(rows), resolved_in_diagnostics=resolved,
                    unresolved_in_diagnostics=len(selected)-resolved, stop_reason=stop,
                    outcomes=dict(Counter(row["outcome"] for row in rows)), original_acquisition=aggregate,
                    original_qualification_pass=False, public_api_enabled=False)
    require(summary == expected, "diagnostic summary mismatch")
    require(summary["original_qualification_pass"] is False and summary["public_api_enabled"] is False,
            "historical qualification or public API changed")
    excluded = {Path(path).resolve() for path in exclude_paths}
    manifest = {path.name: diagnostics.continuation.sha(path) for path in sorted(run_dir.iterdir())
                if path.is_file() and path.resolve() not in excluded}
    return dict(status="CONSISTENT_DIAGNOSTIC_EVIDENCE", verified=True, consistency_pass=True,
                diagnostics_pass=summary["status"] == "DIAGNOSTICS_COMPLETE",
                original_qualification_pass=False, public_api_enabled=False,
                aggregate=expected, artifact_sha256=manifest,
                audit_source_sha256=diagnostics.continuation.sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--evidence-root", type=Path, default=diagnostics.ROOT / "experiments")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = audit_run(args.run_dir, args.evidence_root,
                           exclude_paths=(() if args.output is None else (args.output,)))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, RuntimeError, OverflowError) as exc:
        result = dict(status="INCONSISTENT_DIAGNOSTIC_EVIDENCE", verified=False, consistency_pass=False,
                      diagnostics_pass=False, original_qualification_pass=False, public_api_enabled=False,
                      error=f"{type(exc).__name__}: {exc}")
    if args.output is not None:
        q.write_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "artifact_sha256"}, sort_keys=True))
    if not result["consistency_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
