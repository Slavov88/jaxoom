"""Portable persisted audit of one uniform campaign; never initializes CUDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments import oom_risk_gpu_diagnostic_audit as helpers
from experiments import oom_risk_gpu_uniform_qualification as uniform

q = uniform.q
require = helpers.require


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_command(command, spec_hash, attestation):
    require(isinstance(command, list) and all(isinstance(x, str) for x in command), "invalid child command")
    for key, expected in (("-m", "experiments.oom_risk_gpu_uniform_qualification"),
                          ("--child", spec_hash), ("--plan-sha256", attestation["qualification_plan_sha256"]),
                          ("--replay-sha256", attestation["replay_sha256"])):
        require(command.count(key) == 1 and command.index(key)+1 < len(command), "missing or duplicate child option: " + key)
        require(command[command.index(key)+1] == expected, "child command mismatch: " + key)


def check_trial(row, stdout, candidate, reference, attestation, index, is_pilot, replay):
    """Reconstruct parent rejection as well as validating accepted child evidence."""
    require(type(row.get("index")) is int and row["index"] == index, "trial index mismatch")
    require(row.get("selected_spec_hash") == candidate["spec_hash"] and
            row.get("selected_probability") == candidate["expected_probability"], "selected identity mismatch")
    require(row.get("is_pilot") is is_pilot, "pilot identity mismatch")
    check_command(row.get("command"), candidate["spec_hash"], attestation)
    raw = helpers.events(stdout)
    actual_attestation = uniform.diagnostics.parse_attestation(stdout)
    attestations = [item for item in raw if item.get("event") == "attestation"]
    phases = [item.get("phase") for item in raw if item.get("event") == "phase"]
    returned = [item for item in raw if item.get("event") == "result"]
    code = row.get("returncode")
    if code is None:
        expected = dict(q.timeout_result(stdout), returncode=None, child_attestation_event=actual_attestation)
        if actual_attestation != attestation:
            expected["outcome"] = "INVALID_CHILD_ATTESTATION_OR_PROVENANCE"
        require(all(row.get(key) == value for key, value in expected.items()), "saved timeout classification mismatch")
        return False
    require(type(code) is int, "invalid child return code")
    child = q.parse_child_result(stdout)
    expected = dict(child)
    valid = uniform.validate_result(child, stdout, candidate, reference, attestation)
    if code != 0:
        expected["outcome"] = "CHILD_PROCESS_FAILURE"
    elif not valid:
        expected["outcome"] = "INVALID_CHILD_ATTESTATION_OR_PROVENANCE"
    require(all(row.get(key) == value for key, value in expected.items()), "raw result and saved row differ")
    if code != 0 or not valid:
        return False
    require(len(attestations) == len(returned) == 1, "missing or duplicate measured child events")
    attestation_index = next(i for i, item in enumerate(raw) if item.get("event") == "attestation")
    require(not any(item.get("event") in ("phase", "result") for item in raw[:attestation_index]),
            "attestation follows target events")
    require(child["runtime"]["device_kind"] == q.policy.EXACT_ENVIRONMENT["device_kind"], "CUDA identity mismatch")
    runtime = child["runtime"]
    observed = helpers.datetime.fromisoformat(runtime["observed_at_utc"])
    eligibility = q.policy.evaluate_applicability(q.adapter.WorkloadSpec.from_dict(candidate["spec"]),
                                                runtime, replay, now=observed)
    require(eligibility["status"] == "ELIGIBLE_FOR_QUALIFICATION" and
            child["applicability"].get("reasons") == [], "persisted runtime eligibility mismatch")
    features = child["fresh_features"]
    score = q.adapter._score_features(features, q.adapter.load_frozen_model())
    require(abs(score-child["fresh_probability"]) <= 1e-12, "frozen model score mismatch")
    outcome = child.get("outcome")
    if outcome in ("FIT", "COMPILE_OOM", "EXECUTION_OOM"):
        expected_phases = ["initialization", "initialization", "compile"]
        if outcome != "COMPILE_OOM":
            expected_phases.append("execute")
        require(phases == expected_phases, "terminal raw phase sequence mismatch")
        if outcome.endswith("OOM"):
            require(q.classify_error(child.get("error", ""), child.get("phase")) == outcome,
                    "terminal OOM error mismatch")
    return q.trial_passes(child) and (not is_pilot or outcome == "FIT")


def audit_run(run_dir, *, exclude_paths=()):
    """A consistent stopped campaign is valid evidence, never a qualification pass."""
    run_dir = Path(run_dir)
    plan_path = run_dir / "plan.json"
    plan_sha = uniform.sha(plan_path)
    plan = uniform.read_plan(plan_path, plan_sha)
    measured_sources = uniform.verify_sources(plan["source_sha256"])
    require(len(measured_sources) == 9, "expected nine measured qualification sources")
    reference = uniform.load_reference()
    require(len(plan["candidates"]) == len(plan["order"]) == len(set(plan["order"])) == 200 and
            len(plan["pilot_spec_hashes"]) == len(set(plan["pilot_spec_hashes"])) == 18,
            "invalid full catalog or pilots")
    require(plan["candidates"] == reference["candidates"] and plan["order"] == uniform.launch_order(reference) and
            plan["pilot_spec_hashes"] == reference["pilot_spec_hashes"], "catalog/order differs from frozen reference")
    validation_path = uniform.repo_path(plan["validation_path"])
    require(uniform.sha(validation_path) == plan["validation_sha256"], "test validation manifest hash mismatch")
    validation = uniform.validate_test_evidence(validation_path)
    # Bind the tested population and auditor itself, including the external JUnit bytes.
    require(validation["source_sha256"] == measured_sources and validation["suite"] == "full" and
            validation["auditor_sha256"] == uniform.sha(__file__) and
            validation["test_source_sha256"] == uniform.test_source_snapshot(), "tested source population mismatch")
    test_counts = [validation["junit"][platform]["tests"] for platform in ("windows", "linux")]
    require(all(type(count) is int and count >= 447 for count in test_counts) and test_counts[0] == test_counts[1],
            "regression populations incomplete or unequal")
    initial = helpers.check_idle(read_json(run_dir / "initial_idle_snapshot.json"))
    replay_path = run_dir / "gpu_replay.json"
    replay = read_json(replay_path)
    require((run_dir / "gpu_replay.stdout.txt").is_file() and (run_dir / "gpu_replay.stderr.txt").is_file(),
            "missing replay raw logs")
    require(q.policy._replay_matches(replay, reference["policy_contract"]["pipeline_contract_hash"]) and
            replay["provenance"]["backend"] == "gpu" and replay["public_api_absent"] is True,
            "invalid fresh 224-row GPU replay")
    attestation = uniform.expected_attestation(plan, plan_path, replay_path)
    by_id = {candidate["spec_hash"]: candidate for candidate in plan["candidates"]}
    paths = sorted(run_dir.glob("trial_*.json"))
    require(len(paths) <= 200, "too many trial records")
    rows = []
    last_observation = initial
    invalid_indices = []
    missing_recoveries = []
    preflight_paths = set()
    recovery_paths = set()
    for index, path in enumerate(paths):
        require(path.name == f"trial_{index:03d}.json", "missing or noncontiguous trial")
        preflight_path = run_dir / f"preflight_{index:03d}.json"
        preflight_paths.add(preflight_path.name)
        preflight = read_json(preflight_path)
        require(type(preflight.get("host_free_bytes")) is int and preflight["host_free_bytes"] >= uniform.MIN_HOST_FREE_BYTES,
                "preflight host storage gate failed")
        before = helpers.check_idle(preflight["idle"])
        require(before >= last_observation, "preflight precedes previous recovery")
        stdout_path, stderr_path = path.with_suffix(".stdout.txt"), path.with_suffix(".stderr.txt")
        require(stdout_path.is_file() and stderr_path.is_file(), "trial raw logs missing")
        row = read_json(path)
        spec_hash = plan["order"][index]
        passing = check_trial(row, stdout_path.read_text(encoding="utf-8"), by_id[spec_hash], reference,
                              attestation, index, spec_hash in plan["pilot_spec_hashes"], replay)
        if not passing:
            invalid_indices.append(index)
        recovery_path = run_dir / f"recovery_{index:03d}.json"
        if recovery_path.exists():
            recovery_paths.add(recovery_path.name)
            recovery = helpers.check_idle(read_json(recovery_path))
            require(recovery >= before, "recovery predates preflight")
            last_observation = recovery
        else:
            # A failed idle recovery aborts the runner before this file is written.
            missing_recoveries.append(index)
            last_observation = before
        rows.append(row)
    require(set(path.name for path in run_dir.glob("preflight_*.json")) == preflight_paths,
            "extra preflight evidence without a trial")
    require(set(path.name for path in run_dir.glob("recovery_*.json")) == recovery_paths,
            "extra recovery records")
    require(not invalid_indices or invalid_indices == [len(rows)-1], "campaign continued after a stop outcome")
    require(not missing_recoveries or missing_recoveries == [len(rows)-1], "campaign continued after failed recovery")
    summary = read_json(run_dir / "summary.json")
    require(type(summary.get("final_checks_pass")) is bool, "missing final gate result")
    final_path = run_dir / "final_idle_snapshot.json"
    if final_path.exists():
        final = helpers.check_idle(read_json(final_path))
        require(final >= last_observation, "final idle precedes trial recovery")
    else:
        require(summary["final_checks_pass"] is False and
                str(summary.get("stop_reason", "")).startswith("FINAL_IDLE_GATE:"), "missing final idle evidence")
    require(not missing_recoveries or summary.get("stop_reason") is not None, "unreported failed recovery")
    require(not invalid_indices or summary.get("stop_reason") is not None, "unreported stop outcome")
    expected = uniform.summarize(plan, rows, summary.get("stop_reason"), summary["final_checks_pass"])
    require(summary == expected, "uniform summary differs from recomputed evidence")
    require(summary["public_api_enabled"] is False and summary["historical_qualification_pass"] is False,
            "public or historical qualification changed")
    qualification_pass = bool(expected["runtime_checks_pass"] and not invalid_indices and
                              not missing_recoveries and final_path.exists())
    require(not expected["runtime_checks_pass"] or qualification_pass, "runner claimed pass despite invalid evidence")
    excluded = {Path(path).resolve() for path in exclude_paths}
    manifest = {path.name: uniform.sha(path) for path in sorted(run_dir.iterdir())
                if path.is_file() and path.resolve() not in excluded}
    bound_tests = {plan["validation_path"]: uniform.sha(validation_path)}
    bound_tests.update({record["path"]: uniform.sha(uniform.repo_path(record["path"]))
                        for record in validation["junit"].values()})
    return dict(status="CONSISTENT_UNIFORM_QUALIFICATION_EVIDENCE", consistency_pass=True, verified=True,
                qualification_pass=qualification_pass, public_api_enabled=False, historical_qualification_pass=False,
                aggregate=expected, artifact_sha256=manifest, test_evidence_sha256=bound_tests,
                audit_source_sha256=uniform.sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = audit_run(args.run_dir, exclude_paths=(() if args.output is None else (args.output,)))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, RuntimeError, OverflowError) as exc:
        result = dict(status="INCONSISTENT_UNIFORM_QUALIFICATION_EVIDENCE", consistency_pass=False, verified=False,
                      qualification_pass=False, public_api_enabled=False, historical_qualification_pass=False,
                      error=f"{type(exc).__name__}: {exc}")
    if args.output is not None:
        q.write_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("artifact_sha256", "test_evidence_sha256")}, sort_keys=True))
    if not result["consistency_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
