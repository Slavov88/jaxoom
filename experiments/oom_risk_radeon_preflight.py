"""Probe the user-selected integrated Radeon without touching the CUDA backend.

This is a feasibility/precondition check, not a GPU execution campaign. An
unsupported device/backend records zero GPU trials and cannot pass qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from experiments import oom_risk_applicability as policy
from experiments import oom_risk_workload_spec as adapter

ROOT = Path(__file__).resolve().parents[1]

ROCM_PROBE = r'''
import importlib.metadata, json, os, platform
result = {"target_backend": "rocm", "python": platform.python_version(),
          "environment": {n: os.environ.get(n) for n in
              ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "JAX_CUDA_VISIBLE_DEVICES")},
          "devices": [], "target_compilation": False, "target_execution": False,
          "cpu_fallback_allowed": False}
try:
    import jax, jaxlib
    result.update(jax_version=jax.__version__, jaxlib_version=jaxlib.__version__)
    result["jax_packages"] = sorted(
        {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
         if d.metadata.get("Name", "").lower().startswith("jax")}.items())
    # Explicit backend selection: never enumerate or initialize CUDA as fallback.
    devices = jax.devices("rocm")
    result["devices"] = [{"platform": d.platform, "device_kind": d.device_kind} for d in devices]
    result["backend_available"] = bool(devices)
except Exception as exc:
    result.update(backend_available=False, error=f"{type(exc).__name__}: {exc}")
print(json.dumps(result, sort_keys=True))
'''


def run_command(argv, *, env=None, timeout=20):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout)
        return {"command": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"command": argv, "returncode": None, "stdout": "", "stderr": "",
                "error": f"{type(exc).__name__}: {exc}"}


def probe_selected_radeon():
    # The parent policy/replay checks use CPU only. The child may initialize ROCm
    # explicitly, but must never initialize CUDA or silently use CPU instead.
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("run the parent preflight with JAX_PLATFORMS=cpu")
    hardware = run_command([
        "powershell.exe", "-NoProfile", "-Command",
        "[PSCustomObject]@{video_controllers=@(Get-CimInstance Win32_VideoController | "
        "Select-Object Name,PNPDeviceID,DriverVersion);processors=@(Get-CimInstance "
        "Win32_Processor | Select-Object Name)} | ConvertTo-Json -Depth 4",
    ])
    hardware_data = None
    try:
        if hardware["returncode"] == 0:
            hardware_data = json.loads(hardware["stdout"])
    except (ValueError, TypeError):
        pass
    child_env = dict(os.environ, JAX_PLATFORMS="rocm", CUDA_VISIBLE_DEVICES="-1",
                     JAX_CUDA_VISIBLE_DEVICES="-1")
    backend = run_command([sys.executable, "-c", ROCM_PROBE], env=child_env)
    backend_data = None
    try:
        if backend["returncode"] == 0:
            backend_data = json.loads(backend["stdout"].strip())
    except (ValueError, TypeError):
        pass
    wsl = run_command([
        "wsl.exe", "-d", "Ubuntu", "--", "python3", "-c",
        "import json,pathlib; print(json.dumps({'dev_kfd':pathlib.Path('/dev/kfd').exists(),"
        "'dev_dri':pathlib.Path('/dev/dri').exists(),"
        "'rocm_paths':[str(p) for p in pathlib.Path('/opt').glob('rocm*')]}))",
    ])
    evidence_path = ROOT / "experiments" / "oom_risk_workload_replay_2026-10-20_final.json"
    evidence = json.loads(evidence_path.read_text())
    spec = next(iter(policy.allowed_specs().values()))
    # Missing telemetry stays unknown; shared system RAM is never reported as VRAM.
    runtime = {
        "backend": "rocm", "gpu_vendor": "AMD", "device_kind": "AMD Radeon(TM) Graphics",
        "driver_version": None, "device_count": None, "physical_memory_bytes": None,
        "jax_version": (backend_data or {}).get("jax_version"),
        "jaxlib_version": (backend_data or {}).get("jaxlib_version"),
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    eligibility = policy.evaluate_applicability(spec, runtime, evidence)
    contract = policy.policy_contract()
    available = bool(backend_data and backend_data.get("backend_available"))
    return {
        "status": "BLOCKED_UNSUPPORTED_TARGET_SCOPE" if available else "BLOCKED_UNSUPPORTED_GPU_BACKEND",
        "all_checks_pass": False,
        "public_api_enabled": False, "gpu_runtime_qualification": "NOT_RUN",
        "gpu_runtime_trials": 0, "nvidia_target_execution": False,
        "selected_target": "INTEGRATED_AMD_RADEON", "seed": None,
        "reason": "The selected Radeon/ROCm target is outside frozen CUDA V2 scope. "
                  + ("A ROCm device was discovered, but this does not qualify CUDA V2."
                     if available else "No usable selected JAX GPU backend was discovered; "
                     "no CUDA or CPU fallback qualifies it."),
        "hardware_probe": hardware, "hardware": hardware_data,
        "rocm_backend_probe": backend, "rocm_backend": backend_data,
        "wsl_rocm_nodes_probe": wsl, "applicability": eligibility,
        "policy_contract": contract, "policy_hash": adapter.digest(contract),
        "frozen_model_file_sha256": hashlib.sha256(adapter.MODEL_PATH.read_bytes()).hexdigest(),
        "preflight_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "provenance": {"utc": datetime.now(timezone.utc).isoformat(),
                       "command": [sys.executable, *sys.argv], "platform": platform.platform(),
                       "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "official_support_references": [
            "https://docs.jax.dev/en/latest/installation.html",
            "https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html",
            "https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as output:
        try:
            result = probe_selected_radeon()
        except Exception as exc:
            result = {"status": "PREFLIGHT_ERROR", "all_checks_pass": False,
                      "public_api_enabled": False, "gpu_runtime_qualification": "NOT_RUN",
                      "gpu_runtime_trials": 0, "error": f"{type(exc).__name__}: {exc}"}
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in (
        "policy_contract", "hardware_probe", "rocm_backend_probe", "wsl_rocm_nodes_probe")}, indent=2))
    # A successful probe of an unsupported environment is not qualification PASS.
    if not result["all_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
