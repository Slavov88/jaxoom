"""Resume after proven telemetry-only failure; preserve all workload unknowns."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import oom_risk_gpu_completion as previous_runner
from experiments import oom_risk_gpu_continuation as c
from experiments import oom_risk_gpu_qualification as q

ORIGINAL_SNAPSHOT = q.gpu_snapshot


def strict_snapshot():
    """Retry only NVML's explicit malformed numeric sample; never substitute zero."""
    for attempt in range(3):
        try:
            return ORIGINAL_SNAPSHOT()
        except ValueError as exc:
            if "[Unknown Error]" not in str(exc) or attempt == 2:
                raise
            print(json.dumps(dict(event="telemetry_retry", attempt=attempt+1, error=str(exc))), flush=True)
            time.sleep(1)


def telemetry_only_failure(row, stdout, candidate, plan):
    child = q.parse_child_result(stdout)
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict) and event.get("event") == "phase":
                return False
        except ValueError:
            pass
    return (row.get("outcome") == "INVALID_CHILD_PROVENANCE_OR_REPLAY"
            and child.get("outcome") == "OTHER_FAILURE" and child.get("phase") == "preflight"
            and child.get("gpu_runtime_executed") is False and row.get("returncode") == 0
            and child.get("error") == "ValueError: invalid literal for int() with base 10: '[Unknown Error]'"
            and all(child.get(k) == v for k,v in (("spec_hash", candidate["spec_hash"]),
                ("spec", candidate["spec"]), ("plan_hash", plan["plan_hash"]), ("policy_hash", plan["policy_hash"]))))


def load_history(original, continuation, completion):
    plan, rows = previous_runner.audit_chain(original, continuation)
    audit = json.loads((completion / "independent_persisted_audit.json").read_text())
    if audit.get("verified") is not True:
        raise ValueError("completion audit missing")
    for name, expected in audit["artifact_sha256"].items():
        path = completion / name
        if path.parent.resolve() != completion.resolve() or c.sha(path) != expected:
            raise ValueError("completion artifact hash mismatch")
    contract = json.loads((completion / "completion_plan.json").read_text())
    if contract["source_sha256"] != c.sha(previous_runner.__file__):
        raise ValueError("previous runner source changed")
    pending = c.pending_candidates(plan, rows)
    prior_new = []
    telemetry = []
    for index, path in enumerate(sorted(completion.glob("trial_*.json"))):
        row = json.loads(path.read_text())
        candidate = pending[index]
        if row["spec_hash"] != candidate["spec_hash"]:
            raise ValueError("completion ordering mismatch")
        prior_new.append(row)
        if telemetry_only_failure(row, path.with_suffix(".stdout.txt").read_text(), candidate, plan):
            telemetry.append(row)
        elif previous_runner.recoverable(row):
            if q.trial_passes(row) and not q.validate_child_result(row, candidate, plan):
                raise ValueError("prior child identity mismatch")
        else:
            raise ValueError("prior runtime failure cannot be retried")
    if c.summarize(rows+prior_new) != audit["aggregate"] or len(telemetry) != 1:
        raise ValueError("unexpected history or telemetry failure count")
    return plan, rows+[r for r in prior_new if r not in telemetry], telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--completion", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    q.gpu_snapshot = strict_snapshot
    if args.child:
        row = q.run_one(q.load_plan(args.plan), args.child, json.loads(args.replay.read_text()))
        print(json.dumps(dict(event="result", **row)), flush=True)
        return
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("parent must select CPU")
    plan, previous, telemetry = load_history(args.original, args.continuation, args.completion)
    pending = c.pending_candidates(plan, previous)
    args.output_dir.mkdir(exist_ok=False)
    contract = dict(protocol="restricted-v2-telemetry-resume-1", source_sha256=c.sha(__file__),
                    previous_audit_sha256=c.sha(args.completion / "independent_persisted_audit.json"),
                    original_plan_hash=plan["plan_hash"], pending_spec_hashes=[x["spec_hash"] for x in pending],
                    telemetry_only_retries=[x["spec_hash"] for x in telemetry],
                    snapshot_attempts=3, timeout_seconds=180, campaign_seconds=3600,
                    max_new_timeouts=12, public_api_enabled=False)
    q.write_new(args.output_dir / "resume_plan.json", contract)
    replay_path = args.output_dir / "gpu_replay.json"
    results = []
    stop = None
    started = time.monotonic()
    try:
        q.write_new(args.output_dir / "initial_idle_snapshot.json", q.wait_for_idle())
        command = [sys.executable, "-m", "experiments.oom_risk_workload_replay", "--output", str(replay_path)]
        try:
            proc = subprocess.run(command, capture_output=True, env=q.child_env(), timeout=90, cwd=q.ROOT)
        except subprocess.TimeoutExpired as exc:
            (args.output_dir / "gpu_replay.stdout.txt").write_bytes(exc.stdout or b"")
            (args.output_dir / "gpu_replay.stderr.txt").write_bytes(exc.stderr or b"")
            raise RuntimeError("fresh replay timeout") from exc
        (args.output_dir / "gpu_replay.stdout.txt").write_bytes(proc.stdout)
        (args.output_dir / "gpu_replay.stderr.txt").write_bytes(proc.stderr)
        evidence = json.loads(replay_path.read_text())
        if proc.returncode != 0 or evidence["provenance"]["backend"] != "gpu" or not evidence["fresh_tracing_and_replay_pass"]:
            raise RuntimeError("fresh replay failed")
        for index, candidate in enumerate(pending):
            q.wait_for_idle()
            if time.monotonic()-started > 3420:
                stop = "CAMPAIGN_TIME_BUDGET"
                break
            command = [sys.executable, "-m", "experiments.oom_risk_gpu_resume", "--child", candidate["spec_hash"],
                       "--plan", str(args.original / "plan.json"), "--replay", str(replay_path)]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, env=q.child_env(), timeout=180, cwd=q.ROOT)
                stdout, stderr = proc.stdout, proc.stderr
                row = q.parse_child_result(stdout)
                if proc.returncode != 0:
                    row["outcome"] = "CHILD_PROCESS_FAILURE"
                elif not q.validate_child_result(row, candidate, plan):
                    row["outcome"] = "INVALID_CHILD_PROVENANCE_OR_REPLAY"
                row["returncode"] = proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout, stderr = exc.stdout or b"", exc.stderr or b""
                stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
                stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
                row = dict(q.timeout_result(stdout), returncode=None)
            row.update(spec_hash=candidate["spec_hash"], likely_fit=candidate["expected_probability"] <= .2,
                       expected_probability=candidate["expected_probability"], command=command, index=index,
                       child_entrypoint_sha256=contract["source_sha256"])
            q.write_new(args.output_dir / f"trial_{index:03d}.json", row)
            (args.output_dir / f"trial_{index:03d}.stdout.txt").write_text(stdout)
            (args.output_dir / f"trial_{index:03d}.stderr.txt").write_text(stderr)
            results.append(row)
            print(index+1, candidate["candidate_id"], row["outcome"], flush=True)
            if not previous_runner.recoverable(row):
                stop = row["outcome"]
                break
            if row["outcome"].endswith("_TIMEOUT"):
                q.write_new(args.output_dir / f"recovery_{index:03d}.json", q.wait_for_idle())
                if sum(r["outcome"].endswith("_TIMEOUT") for r in results) >= 12:
                    stop = "TIMEOUT_BUDGET"
                    break
    except Exception as exc:
        stop = f"{type(exc).__name__}: {exc}"
    summary = c.summarize(previous+results)
    summary.update(new_trials=len(results), stop_reason=stop, prior_telemetry_only_launches=len(telemetry),
                   total_launches=len(previous)+len(telemetry)+len(results))
    try:
        q.write_new(args.output_dir / "final_idle_snapshot.json", q.wait_for_idle())
    except Exception as exc:
        summary.update(all_checks_pass=False, recovery_error=str(exc))
    q.write_new(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
