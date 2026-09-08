"""Compact JAX 0.11 compiler introspection for selected drift cases.

Run this script in a fresh process on each device. It reuses the committed
accelerator workload definitions and never changes calibration behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def operation_histogram(text: str) -> dict[str, int]:
    names = re.findall(r"(?:stablehlo|mhlo)\.([A-Za-z0-9_]+)", text)
    if not names:
        names = re.findall(r"(?:^|\\n)\\s*%?[^ ]+ = ([A-Za-z0-9_.]+)", text)
    return dict(sorted(Counter(names).items()))


def compact_jaxpr(fn: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    import jax

    closed = jax.make_jaxpr(fn)(*args)
    jaxpr = closed.jaxpr
    text = str(closed)
    primitive_histogram = Counter(str(eqn.primitive) for eqn in jaxpr.eqns)
    return {
        "equation_count": len(jaxpr.eqns),
        "primitive_histogram": dict(sorted(primitive_histogram.items())),
        "input_aval_summaries": [str(var.aval) for var in jaxpr.invars],
        "output_aval_summaries": [str(var.aval) for var in jaxpr.outvars],
        "nested_jaxpr_count": sum(_nested_count(value) for eqn in jaxpr.eqns for value in eqn.params.values()),
        "text_sha256": digest(text),
    }


def _nested_count(value: Any) -> int:
    if hasattr(value, "jaxpr"):
        return 1 + sum(_nested_count(param) for eqn in value.jaxpr.eqns for param in eqn.params.values())
    if hasattr(value, "eqns"):
        return 1 + sum(_nested_count(param) for eqn in value.eqns for param in eqn.params.values())
    if isinstance(value, (tuple, list)):
        return sum(_nested_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_nested_count(item) for item in value.values())
    return 0


def ir_record(lowered: Any, dialect: str) -> dict[str, Any]:
    try:
        text = lowered.as_text(dialect)
        return {
            "available": True,
            "text_sha256": digest(text),
            "text_length": len(text),
            "line_count": text.count("\n") + 1,
            "operation_histogram": operation_histogram(text),
            "custom_call_count": text.count("custom_call"),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def compiler_record(compiled: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"compiled_text_available": False}
    try:
        text = compiled.as_text()
        result.update(
            compiled_text_available=True,
            text_sha256=digest(text),
            text_length=len(text),
            line_count=text.count("\n") + 1,
            instruction_count=len(re.findall(r"^[ ]{2,}[A-Za-z0-9_.-]+[ (]", text, re.MULTILINE)),
            fusion_count=len(re.findall(r"fusion", text, re.IGNORECASE)),
            custom_call_count=text.lower().count("custom-call") + text.lower().count("custom_call"),
            copy_count=len(re.findall(r"(?:^|[ .])copy(?:[ (]|$)", text, re.IGNORECASE)),
            dot_count=text.lower().count("dot("),
            convolution_count=text.lower().count("convolution"),
            transpose_count=text.lower().count("transpose"),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def memory_record(compiled: Any) -> dict[str, Any]:
    analysis = compiled.memory_analysis()
    fields = {
        "argument_size_in_bytes": int(analysis.argument_size_in_bytes),
        "output_size_in_bytes": int(analysis.output_size_in_bytes),
        "temp_size_in_bytes": int(analysis.temp_size_in_bytes),
        "alias_size_in_bytes": int(analysis.alias_size_in_bytes),
    }
    fields["accounted_bytes"] = fields["argument_size_in_bytes"] + fields["output_size_in_bytes"] + fields["temp_size_in_bytes"] - fields["alias_size_in_bytes"]
    return fields


def environment() -> dict[str, Any]:
    import jax
    import jaxlib

    return {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_kind": [getattr(device, "device_kind", None) for device in jax.devices()],
        "allocator_environment": {key: os.environ.get(key) for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")},
    }


def configure_xla(path: Path | None, extra_flags: str) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
    current = os.environ.get("XLA_FLAGS", "").strip()
    additions = [extra_flags.strip()]
    if path is not None:
        additions.extend([f"--xla_dump_to={path}", "--xla_dump_hlo_as_text"])
    os.environ["XLA_FLAGS"] = " ".join(part for part in [current, *additions] if part).strip()


def inventory_dumps(path: Path) -> dict[str, Any]:
    files = []
    if path.exists():
        for item in sorted(path.rglob("*")):
            if not item.is_file():
                continue
            record = {"path": str(item.relative_to(path)), "size_bytes": item.stat().st_size, "kind": _dump_kind(item.name)}
            if record["kind"] == "memory_usage_report":
                text = item.read_text(encoding="utf-8", errors="replace")
                match = re.search(r"Total bytes:\\s*(\\d+)", text)
                record["parser_status"] = "PARSED_TOTAL_BYTES" if match else "UNKNOWN"
                record["total_bytes"] = int(match.group(1)) if match else None
                record["text_sha256"] = digest(text)
            files.append(record)
    return {
        "root": str(path),
        "file_count": len(files),
        "buffer_assignment_file_count": sum(file["kind"] == "buffer_assignment_candidate" for file in files),
        "memory_usage_report_count": sum(file["kind"] == "memory_usage_report" for file in files),
        "files": files,
    }


def _dump_kind(name: str) -> str:
    lower = name.lower()
    if "buffer-assignment" in lower or "buffer_assignment" in lower:
        return "buffer_assignment_candidate"
    if "memory-usage-report" in lower or "memory_usage_report" in lower:
        return "memory_usage_report"
    if "memory" in lower or "allocation" in lower:
        return "memory_candidate"
    if "schedule" in lower or "hlo" in lower:
        return "hlo_or_schedule"
    return "other"


def run(args: argparse.Namespace) -> None:
    configure_xla(args.dump_dir, args.xla_flags)

    import jax
    from accelerator_calibration import cases

    if jax.__version__ != "0.11.0":
        raise RuntimeError(f"expected JAX 0.11.0, found {jax.__version__}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_case_rows = manifest["cases"]
    manifest_rows = {row["case_id"]: row for row in manifest_case_rows}
    if len(manifest_rows) != len(manifest_case_rows):
        raise RuntimeError("manifest contains duplicate case IDs")
    definitions = {case.name: case for case in cases()}
    if set(manifest_rows) - set(definitions):
        raise RuntimeError(f"manifest cases missing from workload definitions: {sorted(set(manifest_rows) - set(definitions))}")

    results: list[dict[str, Any]] = []
    for manifest_row in manifest_case_rows:
        case_id = manifest_row["case_id"]
        case = definitions[case_id]
        row: dict[str, Any] = {"case_id": case_id, "family": case.family, "dtype": case.dtype, "status": "OK"}
        try:
            lowered = jax.jit(case.fn).lower(*case.args)
            compiled = lowered.compile()
            memory = memory_record(compiled)
            expected = manifest_rows[case_id]
            row.update(
                configuration=case.config,
                jaxpr=compact_jaxpr(case.fn, case.args),
                lowered_ir={dialect: ir_record(lowered, dialect) for dialect in ("stablehlo", "hlo")},
                compiled_hlo=compiler_record(compiled),
                memory_analysis=memory,
                reproduction={
                    "expected_accounted_bytes": expected.get("rtx3050_compiler_accounted"),
                    "expected_temp_bytes": expected.get("rtx3050_temp"),
                    "accounted_matches": memory["accounted_bytes"] == expected.get("rtx3050_compiler_accounted"),
                    "temp_matches": memory["temp_size_in_bytes"] == expected.get("rtx3050_temp"),
                },
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            row.update(status="FAIL", error_type=type(exc).__name__, error_message=str(exc)[:2000])
        results.append(row)

    capture_complete = all(row["status"] == "OK" for row in results)
    reproduction_complete = capture_complete and all(row.get("reproduction", {}).get("accounted_matches") and row.get("reproduction", {}).get("temp_matches") for row in results)
    payload = {"status": "COMPUTATIONALLY VERIFIED" if reproduction_complete else ("INCOMPLETE_REPRODUCTION" if capture_complete else "INCOMPLETE"), "capture_status": "COMPLETE" if capture_complete else "INCOMPLETE", "reproduction_status": "MATCHED" if reproduction_complete else "MISMATCHED_OR_UNAVAILABLE", "environment": environment(), "manifest": str(args.manifest), "cases": results, "dump_inventory": inventory_dumps(args.dump_dir) if args.dump_dir else {"status": "NOT_REQUESTED"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "case_count": len(results), "failed": sum(row["status"] != "OK" for row in results), "output": str(args.output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--xla-flags", default="--xla_gpu_autotune_level=0", help="XLA flags set before importing JAX; the default matches the recorded RTX calibration.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
