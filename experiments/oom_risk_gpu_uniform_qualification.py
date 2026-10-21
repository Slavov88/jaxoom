"""Fresh, source-attested 200-recipe qualification with a uniform 600 s bound."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from experiments import oom_risk_gpu_diagnostics as diagnostics

q = diagnostics.qualification
ROOT = q.ROOT
REFERENCE_PLAN = ROOT / "experiments/oom_risk_gpu_qualification_2026-10-21_run1/plan.json"
REFERENCE_PLAN_SHA256 = "f10d325d6b68f4f713cd538aff36e7deb9ce30364070f3abd32b78c7f9d2e2ff"
AUDITOR_PATH = ROOT / "experiments/oom_risk_gpu_uniform_audit.py"
TIMEOUT_SECONDS = 600
MAX_CAMPAIGN_SECONDS = 10800
MIN_HOST_FREE_BYTES = 512 * 1024**2
PROTOCOL = "restricted-v2-uniform-qualification-1"
sha = diagnostics.continuation.sha


def source_snapshot():
    return dict(diagnostics.source_snapshot(), **{Path(__file__).relative_to(ROOT).as_posix(): sha(__file__)})


def test_source_snapshot():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "tests").rglob("*.py"))}


def verify_sources(expected):
    actual = source_snapshot()
    if actual != expected:
        raise ValueError("uniform qualification source mismatch")
    return actual


def repo_path(relative):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    if Path(relative).is_absolute():
        raise ValueError("evidence path must be repository-relative")
    return path


def validate_test_evidence(path):
    value = json.loads(Path(path).read_text())
    if (value.get("suite") != "full" or value.get("source_sha256") != source_snapshot()
            or value.get("test_source_sha256") != test_source_snapshot()
            or value.get("auditor_sha256") != sha(AUDITOR_PATH)):
        raise ValueError("test evidence source binding mismatch")
    counts = []
    for platform_name in ("windows", "linux"):
        record = value["junit"][platform_name]
        report = repo_path(record["path"])
        if sha(report) != record["sha256"]:
            raise ValueError("JUnit hash mismatch")
        suites = ET.parse(report).getroot().findall("testsuite")
        if len(suites) != 1:
            raise ValueError("expected one full regression suite")
        suite = suites[0]
        count = int(suite.attrib["tests"])
        if count < 447 or count != record["tests"]:
            raise ValueError("incomplete regression evidence")
        if any(int(suite.attrib[key]) != 0 for key in ("failures", "errors", "skipped")):
            raise ValueError("regression suite did not pass completely")
        counts.append(count)
    if counts[0] != counts[1]:
        raise ValueError("platform regression populations differ")
    return value


def launch_order(reference):
    pilots = reference["pilot_spec_hashes"]
    return pilots + [c["spec_hash"] for c in reference["candidates"] if c["spec_hash"] not in pilots]


def load_reference():
    if sha(REFERENCE_PLAN) != REFERENCE_PLAN_SHA256:
        raise ValueError("frozen original reference plan bytes changed")
    return q.load_plan(REFERENCE_PLAN)


def make_plan(validation_path):
    reference = load_reference()
    validate_test_evidence(validation_path)
    relative = Path(validation_path).resolve().relative_to(ROOT.resolve()).as_posix()
    plan = dict(protocol=PROTOCOL, timeout_seconds=TIMEOUT_SECONDS,
                max_campaign_seconds=MAX_CAMPAIGN_SECONDS, min_host_free_bytes=MIN_HOST_FREE_BYTES,
                attempts_per_recipe=1, initialization="zeros", seed=None, threshold=q.adapter.FIT_THRESHOLD,
                source_sha256=source_snapshot(), reference_plan_sha256=sha(REFERENCE_PLAN),
                reference_plan_hash=reference["plan_hash"], policy_hash=reference["policy_hash"],
                candidates=reference["candidates"], pilot_spec_hashes=reference["pilot_spec_hashes"],
                order=launch_order(reference), validation_path=relative, validation_sha256=sha(validation_path),
                public_api_enabled=False, historical_qualification_pass=False,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["plan_hash"] = q.adapter.digest(plan)
    return plan


def read_plan(path, expected_sha256):
    if sha(path) != expected_sha256:
        raise ValueError("uniform qualification plan file mismatch")
    plan = json.loads(Path(path).read_text())
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    if q.adapter.digest(payload) != plan["plan_hash"]:
        raise ValueError("uniform qualification plan content mismatch")
    fixed = dict(protocol=PROTOCOL, timeout_seconds=TIMEOUT_SECONDS,
                 max_campaign_seconds=MAX_CAMPAIGN_SECONDS, min_host_free_bytes=MIN_HOST_FREE_BYTES,
                 attempts_per_recipe=1, initialization="zeros", seed=None, threshold=q.adapter.FIT_THRESHOLD,
                 public_api_enabled=False, historical_qualification_pass=False)
    if any(type(plan.get(k)) is not type(v) or plan[k] != v for k, v in fixed.items()):
        raise ValueError("unsupported uniform qualification contract")
    reference = load_reference()
    if (plan["reference_plan_sha256"] != sha(REFERENCE_PLAN)
            or plan["reference_plan_hash"] != reference["plan_hash"]
            or plan["policy_hash"] != reference["policy_hash"]
            or plan["candidates"] != reference["candidates"]
            or plan["pilot_spec_hashes"] != reference["pilot_spec_hashes"]
            or plan["order"] != launch_order(reference)
            or len(plan["order"]) != len(set(plan["order"]))):
        raise ValueError("uniform qualification reference/catalog mismatch")
    if len(plan["order"]) != 200 or len(plan["pilot_spec_hashes"]) != 18:
        raise ValueError("uniform qualification requires 200 recipes and 18 pilots")
    return plan


def expected_attestation(plan, plan_path, replay_path):
    return dict(source_sha256=plan["source_sha256"], qualification_plan_sha256=sha(plan_path),
                replay_sha256=sha(replay_path), validation_sha256=plan["validation_sha256"])


def run_child(plan_path, plan_sha256, replay_path, replay_sha256, spec_hash):
    plan = read_plan(plan_path, plan_sha256)
    measured = verify_sources(plan["source_sha256"])
    validation_path = repo_path(plan["validation_path"])
    if sha(validation_path) != plan["validation_sha256"] or sha(replay_path) != replay_sha256:
        raise ValueError("uniform qualification replay/validation hash mismatch")
    if spec_hash not in plan["order"]:
        raise ValueError("recipe outside uniform qualification catalog")
    attestation = dict(source_sha256=measured, qualification_plan_sha256=sha(plan_path),
                       replay_sha256=sha(replay_path), validation_sha256=sha(validation_path))
    print(json.dumps(dict(event="attestation", **attestation)), flush=True)
    reference = load_reference()
    previous_snapshot = q.gpu_snapshot
    q.gpu_snapshot = diagnostics.resume.strict_snapshot
    try:
        row = q.run_one(reference, spec_hash, json.loads(Path(replay_path).read_text()))
    finally:
        q.gpu_snapshot = previous_snapshot
    row["child_attestation"] = attestation
    return row


def validate_result(row, stdout, candidate, reference, attestation):
    return (diagnostics.parse_attestation(stdout) == attestation
            and row.get("child_attestation") == attestation
            and q.validate_child_result(row, candidate, reference)
            and row.get("likely_fit") is (candidate["expected_probability"] <= q.adapter.FIT_THRESHOLD)
            and row.get("expected_probability") == candidate["expected_probability"])


def summarize(plan, rows, stop_reason, final_checks_pass):
    by_id = {c["spec_hash"]: c for c in plan["candidates"]}
    identities = [r["selected_spec_hash"] for r in rows]
    coverage = len(rows) == 200 and len(set(identities)) == 200 and set(identities) == set(by_id)
    pilot_ids = set(plan["pilot_spec_hashes"])
    pilots_fit = sum(r["selected_spec_hash"] in pilot_ids and r.get("outcome") == "FIT" for r in rows)
    safe_rows = [r for r in rows if by_id.get(r["selected_spec_hash"], {}).get("expected_probability", 1) <= q.adapter.FIT_THRESHOLD]
    passed = (coverage and pilots_fit == 18 and stop_reason is None and final_checks_pass
              and all(r.get("outcome") == "FIT" for r in safe_rows)
              and all(q.trial_passes(row) for row in rows))
    return dict(status="GPU_UNIFORM_RUNTIME_PASS" if passed else "GPU_UNIFORM_QUALIFICATION_STOPPED",
                runtime_checks_pass=passed, public_api_enabled=False, historical_qualification_pass=False,
                attempted=len(rows), not_run=len(set(by_id)-set(identities)), coverage_complete=coverage,
                pilots_fit=pilots_fit, pilot_count=18, likely_fit_count=len(safe_rows),
                likely_fit_failures=sum(r.get("outcome") != "FIT" for r in safe_rows),
                false_safe_execution_ooms=sum(r.get("outcome") == "EXECUTION_OOM" for r in safe_rows),
                outcomes=dict(Counter(r.get("outcome") for r in rows)), stop_reason=stop_reason,
                final_checks_pass=final_checks_pass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if args.child:
        row = run_child(args.plan, args.plan_sha256, args.replay, args.replay_sha256, args.child)
        print(json.dumps(dict(event="result", **row)), flush=True)
        return
    if os.environ.get("JAX_PLATFORMS") != "cpu" or args.validation is None or args.output_dir is None:
        parser.error("parent requires CPU selection, --validation and --output-dir")
    plan = make_plan(args.validation)
    args.output_dir.mkdir(exist_ok=False)
    plan_path = args.output_dir / "plan.json"
    q.write_new(plan_path, plan)
    plan_sha256 = sha(plan_path)
    replay_path = args.output_dir / "gpu_replay.json"
    rows, stop = [], None
    final_checks = False
    started = time.monotonic()
    q.gpu_snapshot = diagnostics.resume.strict_snapshot
    try:
        if shutil.disk_usage(ROOT).free < MIN_HOST_FREE_BYTES:
            raise RuntimeError("HOST_STORAGE_GATE")
        q.write_new(args.output_dir / "initial_idle_snapshot.json", q.wait_for_idle())
        command = [sys.executable, "-m", "experiments.oom_risk_workload_replay", "--output", str(replay_path)]
        try:
            proc = subprocess.run(command, capture_output=True, env=q.child_env(), timeout=90, cwd=ROOT)
        except subprocess.TimeoutExpired as exc:
            (args.output_dir / "gpu_replay.stdout.txt").write_bytes(exc.stdout or b"")
            (args.output_dir / "gpu_replay.stderr.txt").write_bytes(exc.stderr or b"")
            raise RuntimeError("GPU_REPLAY_TIMEOUT") from exc
        (args.output_dir / "gpu_replay.stdout.txt").write_bytes(proc.stdout)
        (args.output_dir / "gpu_replay.stderr.txt").write_bytes(proc.stderr)
        replay = json.loads(replay_path.read_text())
        reference = load_reference()
        if (proc.returncode != 0 or not q.policy._replay_matches(replay, reference["policy_contract"]["pipeline_contract_hash"])
                or replay["provenance"]["backend"] != "gpu" or replay.get("public_api_absent") is not True):
            raise RuntimeError("GPU_REPLAY_FAILED")
        attestation = expected_attestation(plan, plan_path, replay_path)
        by_id = {c["spec_hash"]: c for c in plan["candidates"]}
        for index, spec_hash in enumerate(plan["order"]):
            idle = q.wait_for_idle()
            verify_sources(plan["source_sha256"])
            validate_test_evidence(args.validation)
            if sha(args.validation) != plan["validation_sha256"]:
                raise RuntimeError("REGRESSION_EVIDENCE_CHANGED")
            host_free = shutil.disk_usage(ROOT).free
            if host_free < MIN_HOST_FREE_BYTES:
                stop = "HOST_STORAGE_GATE"
                break
            if time.monotonic()-started > MAX_CAMPAIGN_SECONDS-TIMEOUT_SECONDS:
                stop = "CAMPAIGN_TIME_BUDGET"
                break
            q.write_new(args.output_dir / f"preflight_{index:03d}.json", dict(idle=idle, host_free_bytes=host_free))
            candidate = by_id[spec_hash]
            command = [sys.executable, "-m", "experiments.oom_risk_gpu_uniform_qualification", "--child", spec_hash,
                       "--plan", str(plan_path), "--plan-sha256", plan_sha256, "--replay", str(replay_path),
                       "--replay-sha256", attestation["replay_sha256"]]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, env=q.child_env(), timeout=TIMEOUT_SECONDS, cwd=ROOT)
                stdout, stderr = proc.stdout, proc.stderr
                row = q.parse_child_result(stdout)
                if proc.returncode != 0:
                    row["outcome"] = "CHILD_PROCESS_FAILURE"
                elif not validate_result(row, stdout, candidate, reference, attestation):
                    row["outcome"] = "INVALID_CHILD_ATTESTATION_OR_PROVENANCE"
                row["returncode"] = proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout, stderr = exc.stdout or b"", exc.stderr or b""
                stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
                stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
                row = dict(q.timeout_result(stdout), returncode=None,
                           child_attestation_event=diagnostics.parse_attestation(stdout))
                if row["child_attestation_event"] != attestation:
                    row["outcome"] = "INVALID_CHILD_ATTESTATION_OR_PROVENANCE"
            row.update(index=index, selected_spec_hash=spec_hash, command=command,
                       selected_probability=candidate["expected_probability"], is_pilot=spec_hash in plan["pilot_spec_hashes"])
            q.write_new(args.output_dir / f"trial_{index:03d}.json", row)
            (args.output_dir / f"trial_{index:03d}.stdout.txt").write_text(stdout)
            (args.output_dir / f"trial_{index:03d}.stderr.txt").write_text(stderr)
            rows.append(row)
            print(index+1, candidate["candidate_id"], row["outcome"], flush=True)
            q.write_new(args.output_dir / f"recovery_{index:03d}.json", q.wait_for_idle())
            if not q.trial_passes(row) or (row["is_pilot"] and row.get("outcome") != "FIT"):
                stop = row["outcome"]
                break
        verify_sources(plan["source_sha256"])
        validate_test_evidence(args.validation)
        final_checks = sha(args.validation) == plan["validation_sha256"]
    except Exception as exc:
        stop = f"{type(exc).__name__}: {exc}"
    try:
        q.write_new(args.output_dir / "final_idle_snapshot.json", q.wait_for_idle())
    except Exception as exc:
        stop = f"FINAL_IDLE_GATE: {type(exc).__name__}: {exc}"
        final_checks = False
    summary = summarize(plan, rows, stop, final_checks)
    q.write_new(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["runtime_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
