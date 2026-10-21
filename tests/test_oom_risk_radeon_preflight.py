"""Backend selection/failure controls; fixtures never execute GPU workloads."""
import json
import subprocess

import pytest

from experiments import oom_risk_radeon_preflight as preflight


def test_preflight_requires_cpu_only_parent_before_subprocesses(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject parent before probing")
    monkeypatch.setattr(preflight, "run_command", forbidden)
    with pytest.raises(RuntimeError, match="JAX_PLATFORMS=cpu"):
        preflight.probe_selected_radeon()


@pytest.mark.parametrize("backend_available", [False, True])
def test_rocm_selection_never_falls_back_to_cuda_or_qualifies_cuda_model(monkeypatch, backend_available):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    calls = []

    def fake_command(argv, *, env=None, timeout=20):
        calls.append((argv, env))
        if argv[0] == "powershell.exe":
            value = {"video_controllers": [{"Name": "AMD Radeon(TM) Graphics"}],
                     "processors": [{"Name": "AMD Ryzen 7 4800H"}]}
        elif argv[0] == "wsl.exe":
            value = {"dev_kfd": False, "dev_dri": False, "rocm_paths": []}
        else:
            assert env["JAX_PLATFORMS"] == "rocm"
            assert env["CUDA_VISIBLE_DEVICES"] == env["JAX_CUDA_VISIBLE_DEVICES"] == "-1"
            assert 'jax.devices("rocm")' in argv[-1]
            value = {"backend_available": backend_available, "jax_version": "0.11.0",
                     "jaxlib_version": "0.11.0", "cpu_fallback_allowed": False}
        return {"command": argv, "returncode": 0, "stdout": json.dumps(value), "stderr": ""}

    monkeypatch.setattr(preflight, "run_command", fake_command)
    result = preflight.probe_selected_radeon()
    assert len(calls) == 3
    assert result["all_checks_pass"] is False
    assert result["gpu_runtime_trials"] == 0
    assert result["gpu_runtime_qualification"] == "NOT_RUN"
    assert result["applicability"]["status"] == "UNSUPPORTED"
    assert result["public_api_enabled"] is False
    assert result["status"] == ("BLOCKED_UNSUPPORTED_TARGET_SCOPE" if backend_available
                                 else "BLOCKED_UNSUPPORTED_GPU_BACKEND")


@pytest.mark.parametrize("error", [OSError("test missing command"), subprocess.TimeoutExpired("test", 1)])
def test_missing_or_timed_out_probe_is_not_a_success(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(preflight.subprocess, "run", fail)
    result = preflight.run_command(["test"])
    assert result["returncode"] is None
    assert result["stdout"] == ""
    assert "error" in result


def test_cli_refuses_to_overwrite_evidence(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("KEEP")
    monkeypatch.setattr("sys.argv", ["preflight", "--output", str(output)])
    with pytest.raises(FileExistsError):
        preflight.main()
    assert output.read_text() == "KEEP"


def test_failed_probe_writes_not_run_and_exits_nonzero(tmp_path, monkeypatch):
    output = tmp_path / "error.json"
    def fail():
        raise RuntimeError("test preflight failure")
    monkeypatch.setattr(preflight, "probe_selected_radeon", fail)
    monkeypatch.setattr("sys.argv", ["preflight", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        preflight.main()
    assert exc.value.code == 2
    result = json.loads(output.read_text())
    assert result["status"] == "PREFLIGHT_ERROR"
    assert result["all_checks_pass"] is False
    assert result["gpu_runtime_trials"] == 0
    assert result["public_api_enabled"] is False
