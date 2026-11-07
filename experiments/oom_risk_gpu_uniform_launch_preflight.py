"""Read-only launch preflight for the uniform qualification campaign.

Records timestamped host measurements (storage, GPU idle state, WSL launch
ability). It never starts a GPU trial and is deliberately not part of the
source-hashed qualification set, so it cannot invalidate frozen evidence.
Output is a snapshot only; recheck immediately before launch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_HOST_FREE_BYTES = 512 * 1024**2
WSL_PYTHON = "/home/slavov/.venvs/jaxoom-cuda-20261021/env/bin/python"
WSL_CONFIG = Path.home() / ".wslconfig"


def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        text = (p.stdout + p.stderr).decode("utf-16-le", "ignore") if cmd[0] == "wsl" and b"\x00" in (p.stdout + p.stderr) \
            else (p.stdout + p.stderr).decode("utf-8", "ignore")
        return p.returncode, text.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, repr(exc)


def wsl_swap_path():
    if not WSL_CONFIG.exists():
        return None
    for line in WSL_CONFIG.read_text().splitlines():
        if line.strip().lower().startswith("swapfile="):
            return line.split("=", 1)[1].strip().replace("\\\\", "\\")
    return None


def gpu_state():
    code, out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"])
    if code != 0:
        return dict(ok=False, detail=out)
    name, driver, used, total, util = [x.strip() for x in out.splitlines()[0].split(",")]
    return dict(ok=True, name=name, driver=driver, used_mib=int(used), total_mib=int(total), util_pct=int(util),
                idle=int(used) == 0 and int(util) == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    free = shutil.disk_usage(ROOT).free
    swap = wsl_swap_path()
    swap_free = shutil.disk_usage(Path(swap).anchor).free if swap and Path(swap).anchor else None
    code, detail = run(["wsl", "-d", "Ubuntu", "--", "test", "-x", WSL_PYTHON])
    gpu = gpu_state()
    checks = dict(
        host_storage=free >= MIN_HOST_FREE_BYTES,
        gpu_idle=bool(gpu.get("ok") and gpu.get("idle")),
        wsl_starts=code == 0,
    )
    record = dict(
        observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        fresh_campaign_started=False, gpu_runtime_trials=0, public_api_enabled=False,
        host_free_bytes=free, min_host_free_bytes=MIN_HOST_FREE_BYTES,
        wsl_swapfile=swap, wsl_swap_drive_free_bytes=swap_free,
        wsl_python=WSL_PYTHON, wsl_exit_code=code, wsl_detail=detail,
        gpu=gpu, checks=checks,
        status="READY" if all(checks.values()) else "BLOCKED",
        blockers=[k for k, v in checks.items() if not v])
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0 if record["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
