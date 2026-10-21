"""Fail-closed, internal eligibility policy for qualifying restricted CUDA V2.

Eligibility never means GPU qualification or LIKELY_FIT. There is no public API.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Any

from experiments import oom_risk_workload_spec as adapter
from experiments import oom_risk_workload_replay as replay

POLICY_VERSION = "restricted-v2-primary-exact-1"
PUBLIC_API_ENABLED = False
EXACT_ENVIRONMENT = {
    "backend": "gpu", "gpu_vendor": "NVIDIA",
    "device_kind": "NVIDIA GeForce RTX 3050 Laptop GPU", "device_count": 1,
    "physical_memory_bytes": 4 * 1024**3, "driver_version": "566.07",
    "jax_version": "0.11.0", "jaxlib_version": "0.11.0",
    "allocator": "bfc", "preallocate": "false", "memory_fraction": None,
    "tf_gpu_allocator": None, "xla_flags": None, "jax_enable_x64": False,
    "default_matmul_precision": None, "fresh_process": True,
    "donation": False, "sharding": False,
}
OBSERVATION_MAX_AGE_SECONDS = 5.0
MIN_FREE_BYTES = 3900 * 1024**2
MAX_USED_BYTES = 64 * 1024**2
MAX_UTILIZATION = 5


def allowed_specs() -> dict[str, adapter.WorkloadSpec]:
    """Admit exact primary recipes only; never derive a rectangle from ranges."""
    rows, _ = replay.load_prediction_rows()
    primary_names = {name for name, _ in replay.PANELS[:2]}
    specs = {}
    for row in rows:
        if row["replay_panel"] in primary_names:
            spec = adapter.WorkloadSpec(row["family"], row["configuration"], row["dtype"])
            specs[spec.spec_hash] = spec
    if len(specs) != 200:
        raise RuntimeError("restricted primary allowlist must contain exactly 200 recipes")
    return specs


def policy_contract() -> dict[str, Any]:
    pipeline = adapter.pipeline_contract()
    return {
        "policy_version": POLICY_VERSION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pipeline_contract_hash": adapter.digest(pipeline),
        "model_hash": adapter.MODEL_HASH,
        "domain": "EXACT_PRIMARY_RECIPE_ALLOWLIST_NOT_INTERPOLATION",
        "allowed_spec_hashes": sorted(allowed_specs()),
        "exact_environment": dict(EXACT_ENVIRONMENT),
        "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
        "minimum_free_bytes": MIN_FREE_BYTES, "maximum_used_bytes": MAX_USED_BYTES,
        "maximum_utilization_percent": MAX_UTILIZATION, "other_compute_process_count": 0,
        "reference_bytes": adapter.REFERENCE_BYTES, "reference_is_runtime_quota": False,
        "eligibility_is_qualification": False, "public_api_enabled": False,
    }


def _replay_matches(evidence: Any, contract_hash: str) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    try:
        if evidence["status"] != "COMPUTATIONALLY_VERIFIED":
            return False
        if evidence["fresh_tracing_and_replay_pass"] is not True:
            return False
        if evidence["contract_hash"] != contract_hash or adapter.digest(evidence["contract"]) != contract_hash:
            return False
        if any(type(evidence[k]) is not int or evidence[k] != expected for k, expected in (
            ("rows", 224), ("completed", 224), ("feature_mismatches", 0),
            ("score_mismatches", 0), ("decision_mismatches", 0),
        )):
            return False
        if evidence["errors"] or evidence["purity_pass"] is not True:
            return False
        if not isinstance(evidence["purity"], Mapping) or any(
            type(evidence["purity"].get(k)) is not int or evidence["purity"][k] != expected
            for k, expected in (("target_calls_with_tracers", 224),
                                ("concrete_target_calls", 0), ("blocked_compile_calls", 0))
        ):
            return False
        if evidence["prediction_file_sha256"] != replay.PANEL_SHA256:
            return False
        if evidence["public_api_absent"] is not True or evidence["public_api_enabled"] is not False:
            return False
        if evidence["tolerance"] != replay.TOLERANCE:
            return False
        for name in ("max_feature_delta", "max_probability_delta"):
            if not math.isfinite(evidence[name]) or not 0 <= evidence[name] <= replay.TOLERANCE:
                return False
        if not isinstance(evidence["source_sha256"], Mapping):
            return False
        for name, expected in evidence["source_sha256"].items():
            # Only repository-relative paths recorded by the actual replay.
            source = (replay.ROOT / name).resolve()
            if not source.is_relative_to(replay.ROOT.resolve()):
                return False
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                return False
        required = {"experiments/oom_risk_workload_spec.py", "experiments/oom_risk_workload_replay.py"}
        if not required <= set(evidence["source_sha256"]):
            return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return False


def evaluate_applicability(spec: adapter.WorkloadSpec, runtime: Mapping[str, Any],
                           replay_evidence: Mapping[str, Any] | None,
                           *, now: datetime | None = None) -> dict[str, Any]:
    """Check qualification eligibility. Unsupported inputs never yield a score."""
    reasons = []
    spec_hash = None
    contract_hash = None
    policy_hash = None
    try:
        contract = policy_contract()
        policy_hash = adapter.digest(contract)
        contract_hash = contract["pipeline_contract_hash"]
        adapter._require_spec(spec)
        spec_hash = spec.spec_hash
        if spec_hash not in contract["allowed_spec_hashes"]:
            reasons.append("SPEC_OUTSIDE_EXACT_PRIMARY_ALLOWLIST")
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        reasons.append(f"INVALID_SPEC_OR_PIPELINE: {type(exc).__name__}: {exc}")

    if contract_hash is None or not _replay_matches(replay_evidence, contract_hash):
        reasons.append("FRESH_REPLAY_EVIDENCE_MISSING_STALE_OR_INVALID")
    if not isinstance(runtime, Mapping):
        reasons.append("RUNTIME_STATE_MISSING")
    else:
        for name, expected in EXACT_ENVIRONMENT.items():
            if name not in runtime:
                reasons.append(f"RUNTIME_FIELD_MISSING: {name}")
            elif type(runtime[name]) is not type(expected) or runtime[name] != expected:
                reasons.append(f"UNSUPPORTED_RUNTIME_FIELD: {name}")
        metric_bounds = {
            "free_memory_bytes": (MIN_FREE_BYTES, 4 * 1024**3),
            "used_memory_bytes": (0, MAX_USED_BYTES),
            "utilization_percent": (0, MAX_UTILIZATION),
            "compute_process_count": (0, 0),
        }
        for name, (lo, hi) in metric_bounds.items():
            value = runtime.get(name)
            if type(value) is not int or not lo <= value <= hi:
                reasons.append(f"UNKNOWN_BUSY_OR_INVALID_RUNTIME_METRIC: {name}")
        if all(type(runtime.get(n)) is int for n in ("free_memory_bytes", "used_memory_bytes")):
            if runtime["free_memory_bytes"] + runtime["used_memory_bytes"] > 4 * 1024**3:
                reasons.append("CONTRADICTORY_MEMORY_OBSERVATION")
        try:
            observed = datetime.fromisoformat(runtime["observed_at_utc"])
            current = now if now is not None else datetime.now(timezone.utc)
            if not isinstance(current, datetime) or observed.tzinfo is None or current.tzinfo is None:
                raise ValueError("timezone required")
            age = (current - observed).total_seconds()
            if not 0 <= age <= OBSERVATION_MAX_AGE_SECONDS:
                reasons.append("STALE_OR_FUTURE_RUNTIME_OBSERVATION")
        except (KeyError, TypeError, ValueError):
            reasons.append("RUNTIME_OBSERVATION_TIME_MISSING_OR_INVALID")

    import jaxoom
    if PUBLIC_API_ENABLED or adapter.PUBLIC_API_ENABLED or hasattr(jaxoom, "assess_oom_risk") or hasattr(jaxoom, "WorkloadSpec"):
        reasons.append("PUBLIC_API_MUST_REMAIN_DISABLED")
    return {
        "status": "UNSUPPORTED" if reasons else "ELIGIBLE_FOR_QUALIFICATION",
        "reasons": reasons, "spec_hash": spec_hash,
        "policy_hash": policy_hash, "pipeline_contract_hash": contract_hash,
        "public_api_enabled": False, "gpu_runtime_qualified": False,
        "oom_score": None, "reference_is_runtime_quota": False,
    }
