"""Source-attested, bounded diagnostics for the five unresolved CUDA recipes.

Historical 180-second qualification outcomes are immutable. Diagnostics never
enable an API or retroactively pass that qualification campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import oom_risk_gpu_completion as completion
from experiments import oom_risk_gpu_continuation as continuation
from experiments import oom_risk_gpu_qualification as qualification
from experiments import oom_risk_gpu_resume as resume

ROOT = qualification.ROOT
TIMEOUT_SECONDS = 600
RUN_PREFIX = "oom_risk_gpu_qualification_2026-10-21_"


def source_snapshot():
    paths = [Path(__file__), Path(qualification.__file__), Path(resume.__file__),
             Path(completion.__file__), Path(continuation.__file__),
             Path(qualification.adapter.__file__), Path(qualification.policy.__file__),
             Path(qualification.replay.__file__)]
    return {p.relative_to(ROOT).as_posix(): continuation.sha(p) for p in paths}


def verify_sources(expected):
    actual = source_snapshot()
    if expected != actual:
        raise ValueError("diagnostic source attestation mismatch")
    return actual


def verify_manifest(directory):
    audit_path = directory / "independent_persisted_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("verified") is not True:
        raise ValueError("persisted audit is not verified")
    for name, expected in audit["artifact_sha256"].items():
        path = directory / name
        if path.parent.resolve() != directory.resolve() or continuation.sha(path) != expected:
            raise ValueError("persisted evidence mismatch: " + name)
    return audit


def load_acquisition(base):
    original, first, second, last = [base / (RUN_PREFIX + suffix)
                                    for suffix in ("run1", "continuation1", "completion1", "resume1")]
    plan, previous, telemetry = resume.load_history(original, first, second)
    audit = verify_manifest(last)
    contract = json.loads((last / "resume_plan.json").read_text())
    if contract["source_sha256"] != continuation.sha(resume.__file__):
        raise ValueError("historical telemetry wrapper source mismatch")
    pending = continuation.pending_candidates(plan, previous)
    results = []
    for index, path in enumerate(sorted(last.glob("trial_*.json"))):
        row = json.loads(path.read_text())
        candidate = pending[index]
        stdout = path.with_suffix(".stdout.txt").read_text()
        if row["spec_hash"] != candidate["spec_hash"] or row["index"] != index:
            raise ValueError("acquisition ordering mismatch")
        if row["likely_fit"] != (candidate["expected_probability"] <= .2):
            raise ValueError("acquisition decision mismatch")
        if row["returncode"] is None:
            if any(row[k] != v for k, v in qualification.timeout_result(stdout).items()):
                raise ValueError("timeout phase mismatch")
        else:
            child = qualification.parse_child_result(stdout)
            if (row["returncode"] != 0
                    or not qualification.validate_child_result(child, candidate, plan)
                    or any(row.get(k) != v for k, v in child.items())):
                raise ValueError("acquisition child identity mismatch")
        results.append(row)
    combined = previous + results
    aggregate = continuation.summarize(combined)
    if aggregate != audit["aggregate"] or not aggregate["coverage_complete"]:
        raise ValueError("incomplete or inconsistent acquisition")
    if len(telemetry) != 1 or aggregate["likely_fit_failures"] != 0:
        raise ValueError("historical failure is not eligible for diagnostics")
    unresolved = [r for r in combined if not qualification.trial_passes(r)]
    if len(unresolved) != 5 or any(not completion.recoverable(r) for r in unresolved):
        raise ValueError("unexpected unresolved population")
    unknown_ids = {r["spec_hash"] for r in unresolved}
    selected = [candidate for candidate in plan["candidates"] if candidate["spec_hash"] in unknown_ids]
    return plan, selected, aggregate, last / "independent_persisted_audit.json"


def make_plan(original, selected, audit_path):
    plan = dict(protocol="restricted-v2-timeout-diagnostics-1", timeout_seconds=TIMEOUT_SECONDS,
                original_plan_hash=original["plan_hash"], acquisition_audit_sha256=continuation.sha(audit_path),
                source_sha256=source_snapshot(), candidates=selected,
                original_qualification_pass=False, public_api_enabled=False, seed=None,
                max_campaign_seconds=3600, attempts_per_recipe=1,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["plan_hash"] = qualification.adapter.digest(plan)
    return plan


def read_plan(path, expected_sha256):
    if continuation.sha(path) != expected_sha256:
        raise ValueError("diagnostic plan file hash mismatch")
    plan = json.loads(path.read_text())
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    if qualification.adapter.digest(payload) != plan["plan_hash"]:
        raise ValueError("diagnostic plan content hash mismatch")
    if (plan["timeout_seconds"] != TIMEOUT_SECONDS or plan["public_api_enabled"] is not False
            or plan["original_qualification_pass"] is not False):
        raise ValueError("unsupported diagnostic contract")
    return plan


def run_child(diagnostic_path, diagnostic_sha256, original_path, replay_path, replay_sha256, spec_hash):
    plan = read_plan(diagnostic_path, diagnostic_sha256)
    measured_sources = verify_sources(plan["source_sha256"])
    measured_replay = continuation.sha(replay_path)
    if measured_replay != replay_sha256:
        raise ValueError("diagnostic replay file hash mismatch")
    original = qualification.load_plan(original_path)
    if original["plan_hash"] != plan["original_plan_hash"]:
        raise ValueError("diagnostic original plan mismatch")
    if spec_hash not in {c["spec_hash"] for c in plan["candidates"]}:
        raise ValueError("recipe was not selected for diagnostics")
    attestation = dict(source_sha256=measured_sources, diagnostic_plan_sha256=continuation.sha(diagnostic_path),
                       replay_sha256=measured_replay)
    print(json.dumps(dict(event="attestation", **attestation)), flush=True)
    previous_snapshot = qualification.gpu_snapshot
    qualification.gpu_snapshot = resume.strict_snapshot
    try:
        row = qualification.run_one(original, spec_hash, json.loads(replay_path.read_text()))
    finally:
        qualification.gpu_snapshot = previous_snapshot
    row["child_attestation"] = attestation
    return row


def parse_attestation(stdout):
    found = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and value.get("event") == "attestation":
                found.append({k: v for k, v in value.items() if k != "event"})
        except ValueError:
            pass
    return found[0] if len(found) == 1 else None


def validate_result(row, stdout, candidate, original, expected_attestation):
    return (parse_attestation(stdout) == expected_attestation
            and row.get("child_attestation") == expected_attestation
            and qualification.validate_child_result(row, candidate, original)
            and row.get("likely_fit") is False
            and row.get("expected_probability") == candidate["expected_probability"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child")
    parser.add_argument("--diagnostic-plan", type=Path)
    parser.add_argument("--diagnostic-plan-sha256")
    parser.add_argument("--original-plan", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if args.child:
        row = run_child(args.diagnostic_plan, args.diagnostic_plan_sha256, args.original_plan,
                        args.replay, args.replay_sha256, args.child)
        print(json.dumps(dict(event="result", **row)), flush=True)
        return
    if os.environ.get("JAX_PLATFORMS") != "cpu" or args.output_dir is None:
        parser.error("parent requires JAX_PLATFORMS=cpu and --output-dir")
    original, selected, aggregate, audit_path = load_acquisition(args.evidence_root)
    diagnostic = make_plan(original, selected, audit_path)
    args.output_dir.mkdir(exist_ok=False)
    diagnostic_path = args.output_dir / "plan.json"
    qualification.write_new(diagnostic_path, diagnostic)
    plan_sha = continuation.sha(diagnostic_path)
    replay_path = args.output_dir / "gpu_replay.json"
    results = []
    stop = None
    started = time.monotonic()
    qualification.gpu_snapshot = resume.strict_snapshot
    try:
        qualification.write_new(args.output_dir / "initial_idle_snapshot.json", qualification.wait_for_idle())
        command = [sys.executable, "-m", "experiments.oom_risk_workload_replay", "--output", str(replay_path)]
        try:
            proc = subprocess.run(command, capture_output=True, env=qualification.child_env(), timeout=90, cwd=ROOT)
        except subprocess.TimeoutExpired as exc:
            (args.output_dir / "gpu_replay.stdout.txt").write_bytes(exc.stdout or b"")
            (args.output_dir / "gpu_replay.stderr.txt").write_bytes(exc.stderr or b"")
            raise RuntimeError("diagnostic replay timeout") from exc
        (args.output_dir / "gpu_replay.stdout.txt").write_bytes(proc.stdout)
        (args.output_dir / "gpu_replay.stderr.txt").write_bytes(proc.stderr)
        replay = json.loads(replay_path.read_text())
        if (proc.returncode != 0 or not replay.get("fresh_tracing_and_replay_pass")
                or replay.get("public_api_absent") is not True or replay["provenance"]["backend"] != "gpu"):
            raise RuntimeError("diagnostic GPU replay failed")
        expected_attestation = dict(source_sha256=diagnostic["source_sha256"],
                                    diagnostic_plan_sha256=plan_sha, replay_sha256=continuation.sha(replay_path))
        original_path = args.evidence_root / (RUN_PREFIX + "run1") / "plan.json"
        for index, candidate in enumerate(selected):
            qualification.wait_for_idle()
            verify_sources(diagnostic["source_sha256"])
            if time.monotonic() - started > 3600 - TIMEOUT_SECONDS:
                stop = "CAMPAIGN_TIME_BUDGET"
                break
            command = [sys.executable, "-m", "experiments.oom_risk_gpu_diagnostics", "--child", candidate["spec_hash"],
                       "--diagnostic-plan", str(diagnostic_path), "--diagnostic-plan-sha256", plan_sha,
                       "--original-plan", str(original_path), "--replay", str(replay_path),
                       "--replay-sha256", expected_attestation["replay_sha256"]]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, env=qualification.child_env(),
                                      timeout=TIMEOUT_SECONDS, cwd=ROOT)
                stdout, stderr = proc.stdout, proc.stderr
                row = qualification.parse_child_result(stdout)
                if proc.returncode != 0:
                    row["outcome"] = "CHILD_PROCESS_FAILURE"
                elif not validate_result(row, stdout, candidate, original, expected_attestation):
                    row["outcome"] = "INVALID_CHILD_ATTESTATION_OR_PROVENANCE"
                row["returncode"] = proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout, stderr = exc.stdout or b"", exc.stderr or b""
                stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
                stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
                row = dict(qualification.timeout_result(stdout), returncode=None)
                if parse_attestation(stdout) != expected_attestation:
                    row["outcome"] = "INVALID_CHILD_ATTESTATION_OR_PROVENANCE"
                row["child_attestation_event"] = parse_attestation(stdout)
            row.update(index=index, selected_spec_hash=candidate["spec_hash"], command=command,
                       selected_probability=candidate["expected_probability"])
            qualification.write_new(args.output_dir / f"trial_{index:03d}.json", row)
            (args.output_dir / f"trial_{index:03d}.stdout.txt").write_text(stdout)
            (args.output_dir / f"trial_{index:03d}.stderr.txt").write_text(stderr)
            results.append(row)
            print(index + 1, candidate["candidate_id"], row["outcome"], flush=True)
            if not qualification.trial_passes(row) and row["outcome"] not in ("COMPILE_TIMEOUT", "EXECUTE_TIMEOUT"):
                stop = row["outcome"]
                break
            qualification.write_new(args.output_dir / f"recovery_{index:03d}.json", qualification.wait_for_idle())
    except Exception as exc:
        stop = f"{type(exc).__name__}: {exc}"
    resolved = sum(qualification.trial_passes(row) for row in results)
    summary = dict(status="DIAGNOSTICS_COMPLETE" if resolved == len(selected) and stop is None else "DIAGNOSTICS_INCOMPLETE",
                   selected=len(selected), attempted=len(results), resolved_in_diagnostics=resolved,
                   unresolved_in_diagnostics=len(selected)-resolved, stop_reason=stop,
                   outcomes=dict(Counter(row["outcome"] for row in results)),
                   original_acquisition=aggregate, original_qualification_pass=False, public_api_enabled=False)
    try:
        qualification.write_new(args.output_dir / "final_idle_snapshot.json", qualification.wait_for_idle())
    except Exception as exc:
        summary.update(status="DIAGNOSTICS_INCOMPLETE", recovery_error=str(exc))
    qualification.write_new(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["status"] != "DIAGNOSTICS_COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
