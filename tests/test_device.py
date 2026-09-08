from types import SimpleNamespace

import jax

import jaxoom
from jaxoom import device as device_module
from jaxoom.types import DeviceBudget, DeviceMemorySnapshot, MemoryRiskLevel


def snapshot(**overrides):
    values = {
        "backend": "gpu",
        "device_kind": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "device_id": 0,
        "device_uuid": "GPU-test",
        "physical_total_bytes": 4 * 1024**3,
        "driver_used_bytes": 1 * 1024**3,
        "driver_free_bytes": 3 * 1024**3,
        "jax_bytes_in_use": 128 * 1024**2,
        "jax_peak_bytes_in_use": 256 * 1024**2,
        "jax_pool_bytes": 512 * 1024**2,
        "external_used_bytes": 512 * 1024**2,
        "allocator_mode": "default",
        "allocator_preallocate": False,
        "allocator_memory_fraction": None,
        "effective_available_bytes": 3 * 1024**3 + 384 * 1024**2,
        "measurement_sources": ("test",),
        "limitations": (),
        "timestamp": "test",
    }
    values.update(overrides)
    return DeviceMemorySnapshot(**values)


def test_device_memory_cpu_fallback_is_partial():
    result = jaxoom.device_memory()
    assert result.backend == jax.default_backend()
    assert result.timestamp
    if result.backend == "cpu":
        assert result.physical_total_bytes is None
        assert result.effective_available_bytes is None


def test_nvidia_query_maps_visible_device(monkeypatch):
    class Completed:
        stdout = "0, GPU-zero, 4096, 1024, 3072\n1, GPU-one, 8192, 2048, 6144\n"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(device_module.subprocess, "run", lambda *args, **kwargs: Completed())
    result = device_module._nvidia_memory(0, "NVIDIA RTX")
    assert result["index"] == 1
    assert result["uuid"] == "GPU-one"
    assert result["free_bytes"] == 6144 * 1024**2


def test_effective_available_accounts_for_free_pool():
    effective = device_module._effective_available(100, 80, 20, 1000, None)
    assert effective == 160


def test_device_budget_reserves_memory():
    budget = device_module.device_budget(snapshot(), reserve_fraction=0.05)
    assert budget.assessment_budget_bytes == budget.effective_available_bytes - budget.safety_reserve_bytes
    assert budget.safety_reserve_bytes >= 64 * 1024**2
    assert "safety reserve" in budget.policy


def test_explicit_memory_limit_remains_authoritative():
    report = jaxoom.estimate(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"))
    assessment = jaxoom.assess(report, "16 GiB")
    assert assessment.memory_limit_bytes == 16 * 1024**3
    assert assessment.device_budget is None


def test_auto_assessment_does_not_compile_target(monkeypatch):
    budget = DeviceBudget(snapshot(effective_available_bytes=4 * 1024**3), 4 * 1024**3, 64 * 1024**2, 4 * 1024**3 - 64 * 1024**2, "test", ())
    monkeypatch.setattr("jaxoom.api._device_budget", lambda: budget)
    def fail_compile(*args, **kwargs):
        raise AssertionError("assess(auto) compiled the target")
    monkeypatch.setattr(jax, "jit", fail_compile)
    assessment = jaxoom.assess(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"), memory_limit="auto")
    assert assessment.device_budget is budget
    assert assessment.risk is not None


def test_high_risk_callable_has_donation_hint(monkeypatch):
    budget = DeviceBudget(snapshot(effective_available_bytes=1), 1, 0, 1, "test", ())
    monkeypatch.setattr("jaxoom.api._device_budget", lambda: budget)
    assessment = jaxoom.assess(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"), memory_limit="auto")
    assert assessment.risk is MemoryRiskLevel.LIKELY_EXCEEDS_BUDGET
    assert assessment.remediation_hint and "analyze_donation" in assessment.remediation_hint
