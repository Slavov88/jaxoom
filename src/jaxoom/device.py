"""Observational device-memory and allocator state."""
from __future__ import annotations

import csv
import io
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

import jax

from .types import DeviceBudget, DeviceMemorySnapshot


_DEFAULT_RESERVE_FRACTION = 0.05
_MIN_RESERVE_BYTES = 64 * 1024**2
_MAX_RESERVE_BYTES = 256 * 1024**2


def device_memory() -> DeviceMemorySnapshot:
    """Return a non-compiling snapshot of the current JAX device memory state."""
    devices = jax.devices()
    device = devices[0] if devices else None
    backend = getattr(device, "platform", None) or jax.default_backend()
    device_id = getattr(device, "id", None) if device is not None else None
    device_kind = getattr(device, "device_kind", None) if device is not None else None
    stats = _jax_stats(device)
    allocator = _allocator_policy()
    driver = _nvidia_memory(device_id, device_kind) if backend == "gpu" else None
    limitations: list[str] = []
    sources: list[str] = []
    if driver is not None:
        sources.append("nvidia-smi query")
    else:
        limitations.append("NVIDIA driver memory query unavailable or backend is not CUDA.")
    if stats:
        sources.append("JAX device.memory_stats")
    else:
        limitations.append("JAX device memory_stats unavailable.")

    physical_total = driver.get("total_bytes") if driver else None
    driver_used = driver.get("used_bytes") if driver else None
    driver_free = driver.get("free_bytes") if driver else None
    jax_in_use = _stat(stats, "bytes_in_use")
    jax_peak = _stat(stats, "peak_bytes_in_use")
    jax_pool = _stat(stats, "pool_bytes")
    if jax_pool is None:
        jax_pool = _stat(stats, "bytes_limit") if allocator["preallocate"] is True else None
    external = None
    if driver_used is not None and jax_pool is not None:
        external = max(0, driver_used - jax_pool)
    elif driver_used is not None and jax_in_use is not None:
        external = max(0, driver_used - jax_in_use)
        limitations.append("external_used_bytes is approximate because JAX pool size was unavailable.")

    effective = _effective_available(driver_free, jax_pool, jax_in_use, physical_total, allocator["fraction"])
    if effective is not None:
        sources.append("derived available-memory policy")
    else:
        limitations.append("effective available memory could not be derived.")
    return DeviceMemorySnapshot(
        backend=backend,
        device_kind=device_kind,
        device_id=device_id,
        device_uuid=driver.get("uuid") if driver else None,
        physical_total_bytes=physical_total,
        driver_used_bytes=driver_used,
        driver_free_bytes=driver_free,
        jax_bytes_in_use=jax_in_use,
        jax_peak_bytes_in_use=jax_peak,
        jax_pool_bytes=jax_pool,
        external_used_bytes=external,
        allocator_mode=allocator["mode"],
        allocator_preallocate=allocator["preallocate"],
        allocator_memory_fraction=allocator["fraction"],
        effective_available_bytes=effective,
        measurement_sources=tuple(sources),
        limitations=tuple(limitations),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def device_budget(snapshot: DeviceMemorySnapshot | None = None, *, reserve_fraction: float = _DEFAULT_RESERVE_FRACTION) -> DeviceBudget:
    """Apply the documented safety-reserve policy to a device snapshot."""
    snapshot = snapshot or device_memory()
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    total = snapshot.physical_total_bytes
    if total is None or snapshot.effective_available_bytes is None:
        return DeviceBudget(snapshot, snapshot.effective_available_bytes, None, None, "current device memory unavailable", snapshot.limitations + ("auto device budget is unavailable; use an explicit memory limit",))
    reserve = max(_MIN_RESERVE_BYTES, min(_MAX_RESERVE_BYTES, round(total * reserve_fraction)))
    budget = max(0, snapshot.effective_available_bytes - reserve)
    return DeviceBudget(snapshot, snapshot.effective_available_bytes, reserve, budget, "effective available memory minus 5% safety reserve, bounded to 64 MiB through 256 MiB", snapshot.limitations + ("current free GPU memory can change after this snapshot",))


def _jax_stats(device: Any) -> dict[str, Any]:
    try:
        values = device.memory_stats() if device is not None and hasattr(device, "memory_stats") else None
        return values or {}
    except Exception:
        return {}


def _stat(stats: dict[str, Any], key: str) -> int | None:
    value = stats.get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _allocator_policy() -> dict[str, Any]:
    preallocate_text = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    preallocate = None if preallocate_text is None else preallocate_text.lower() in {"1", "true", "yes", "on"}
    fraction = None
    for key in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION"):
        value = os.environ.get(key)
        if value is not None:
            try:
                fraction = float(value)
                break
            except ValueError:
                pass
    mode = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") or os.environ.get("TF_GPU_ALLOCATOR")
    return {"preallocate": preallocate, "fraction": fraction, "mode": mode or ("default" if preallocate is not False else "no preallocation")}


def _effective_available(driver_free: int | None, pool: int | None, in_use: int | None, total: int | None, fraction: float | None) -> int | None:
    if driver_free is None:
        return None
    available = driver_free
    if pool is not None and in_use is not None:
        available += max(0, pool - in_use)
    if fraction is not None and total is not None and in_use is not None:
        cap = max(0, round(total * fraction))
        available = min(available, driver_free + max(0, cap - in_use))
    return max(0, available)


def _nvidia_memory(device_id: int | None, device_kind: str | None) -> dict[str, Any] | None:
    if device_kind is not None and "nvidia" not in device_kind.lower() and "cuda" not in device_kind.lower():
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    records = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    if not records:
        return None
    selected = _select_gpu_record(records, device_id)
    if selected is None or len(selected) < 5:
        return None
    try:
        return {
            "index": int(selected[0]),
            "uuid": selected[1],
            "total_bytes": int(float(selected[2]) * 1024**2),
            "used_bytes": int(float(selected[3]) * 1024**2),
            "free_bytes": int(float(selected[4]) * 1024**2),
        }
    except (TypeError, ValueError, OverflowError):
        return None


def _select_gpu_record(records: list[list[str]], device_id: int | None) -> list[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and device_id is not None:
        tokens = [token.strip() for token in visible.split(",")]
        if device_id < len(tokens):
            token = tokens[device_id]
            for record in records:
                if token == record[0] or token == record[1] or token == record[1].removeprefix("GPU-"):
                    return record
    if device_id is not None:
        for record in records:
            if record and record[0] == str(device_id):
                return record
    return records[0] if len(records) == 1 else None
