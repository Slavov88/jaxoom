"""CPU/compiler calibration matrix for the JAXOOM MVP.

Run with an explicit output prefix to preserve prior results, for example:

    PYTHONPATH=src python experiments/compiler_calibration.py \
        --output-prefix experiments/compiler_calibration_2026-09-07

The script uses abstract inputs and does not execute benchmark workloads.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform as host_platform
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

import jaxoom


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    config: dict[str, Any]
    fn: Callable[..., Any]
    args: tuple[Any, ...]


def elementwise_case(depth: int, shape: tuple[int, ...]) -> Case:
    def fn(x):
        y = x
        for _ in range(depth):
            y = jnp.exp(y)
            y = y * 2
            y = jnp.tanh(y)
            y = jnp.sin(y)
            y = y + 1
        return y

    return Case(f"elementwise-d{depth}-{shape[0]}", "elementwise", {"depth": depth, "shape": shape}, fn, (jax.ShapeDtypeStruct(shape, "float32"),))


def matmul_case(shape: tuple[int, int, int]) -> Case:
    m, k, n = shape
    return Case(
        f"matmul-{m}x{k}x{n}", "matmul", {"m": m, "k": k, "n": n},
        lambda x, w: x @ w,
        (jax.ShapeDtypeStruct((m, k), "float32"), jax.ShapeDtypeStruct((k, n), "float32")),
    )


def residual_case(depth: int, shape: tuple[int, ...]) -> Case:
    def fn(x):
        values = []
        y = x
        for _ in range(depth):
            y = jnp.tanh(y)
            values.append(y)
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result + x

    return Case(f"residual-d{depth}-{shape[0]}", "residual", {"depth": depth, "shape": shape}, fn, (jax.ShapeDtypeStruct(shape, "float32"),))


def mlp_case(batch: int, width: int, depth: int) -> Case:
    def fn(x, *weights):
        y = x
        for weight in weights[:-1]:
            y = jnp.tanh(y @ weight)
        return y @ weights[-1]

    args: list[Any] = [jax.ShapeDtypeStruct((batch, width), "float32")]
    args.extend(jax.ShapeDtypeStruct((width, width), "float32") for _ in range(depth - 1))
    args.append(jax.ShapeDtypeStruct((width, width // 2), "float32"))
    return Case(f"mlp-b{batch}-w{width}-d{depth}", "mlp", {"batch": batch, "width": width, "depth": depth}, fn, tuple(args))


def reduction_case(kind: str, shape: tuple[int, ...]) -> Case:
    def fn(x):
        if kind == "sum":
            return jnp.sum(x)
        if kind == "mean":
            return jnp.mean(x)
        if kind == "max":
            return jnp.max(x)
        return jnp.sqrt(jnp.sum(x * x))

    return Case(f"reduction-{kind}-{shape[0]}", "reduction", {"kind": kind, "shape": shape}, fn, (jax.ShapeDtypeStruct(shape, "float32"),))


def nested_cases() -> list[Case]:
    def relu_fn(x):
        return jax.nn.relu(x)

    return [
        Case("nested-relu-16", "nested", {"operation": "jax.nn.relu", "shape": [16, 16]}, relu_fn, (jax.ShapeDtypeStruct((16, 16), "float32"),)),
        Case("nested-relu-64", "nested", {"operation": "jax.nn.relu", "shape": [64, 64]}, relu_fn, (jax.ShapeDtypeStruct((64, 64), "float32"),)),
    ]


def autodiff_cases() -> list[Case]:
    shape_x = (16, 32)
    shape_w = (32, 32)
    x = jax.ShapeDtypeStruct(shape_x, "float32")
    w = jax.ShapeDtypeStruct(shape_w, "float32")
    tx = jax.ShapeDtypeStruct(shape_x, "float32")
    tw = jax.ShapeDtypeStruct(shape_w, "float32")
    cot = jax.ShapeDtypeStruct((), "float32")

    def base(x, w):
        return jnp.sum(jnp.tanh(x @ w))

    def vjp_fn(x, w, cotangent):
        value, pullback = jax.vjp(base, x, w)
        dx, dw = pullback(cotangent)
        return value, dx, dw

    return [
        Case("autodiff-forward", "autodiff", {"mode": "forward"}, base, (x, w)),
        Case("autodiff-grad", "autodiff", {"mode": "grad"}, jax.grad(base, argnums=0), (x, w)),
        Case("autodiff-value-grad", "autodiff", {"mode": "value_and_grad"}, jax.value_and_grad(base, argnums=0), (x, w)),
        Case("autodiff-jvp", "autodiff", {"mode": "jvp"}, lambda x, w, tx, tw: jax.jvp(base, (x, w), (tx, tw)), (x, w, tx, tw)),
        Case("autodiff-vjp", "autodiff", {"mode": "vjp"}, vjp_fn, (x, w, cot)),
    ]


def cases() -> list[Case]:
    result: list[Case] = []
    for depth in (1, 2, 4, 8, 16, 32):
        for shape in ((16, 16), (64, 64), (128, 128)):
            result.append(elementwise_case(depth, shape))
    for shape in ((32, 32, 32), (64, 128, 32), (128, 64, 256)):
        result.append(matmul_case(shape))
    for depth in (1, 2, 4, 8, 16):
        for shape in ((16, 16), (64, 64)):
            result.append(residual_case(depth, shape))
    for batch in (8, 32):
        for depth in (2, 4, 8):
            result.append(mlp_case(batch, 32, depth))
    for kind in ("sum", "mean", "max", "norm"):
        for shape in ((32, 32), (128, 128)):
            result.append(reduction_case(kind, shape))
    result.extend(nested_cases())
    result.extend(autodiff_cases())
    return result


def run_case(case: Case) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": case.name,
        "family": case.family,
        "configuration": json.dumps(case.config, sort_keys=True),
        "backend": jax.default_backend(),
        "platform": getattr(jax.devices()[0], "platform", None),
        "device": str(jax.devices()[0]),
        "python_version": platform_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": _jaxlib_version(),
        "status": "ok",
        "error_message": "",
    }
    try:
        static = jaxoom.estimate(case.fn, *case.args)
        compiler = jaxoom.compile_analyze(case.fn, *case.args)
        comparison = jaxoom.compare_memory(static, compiler)
        row.update(
            static_peak_bytes=static.estimated_peak_bytes,
            equations=static.equations_analyzed,
            confidence=static.confidence,
            nested_constructs=";".join(static.unsupported_constructs),
            compiler_argument_bytes=compiler.argument_bytes,
            compiler_output_bytes=compiler.output_bytes,
            compiler_temp_bytes=compiler.temporary_bytes,
            compiler_alias_bytes=compiler.alias_bytes,
            compiler_accounted_bytes=compiler.compiler_accounted_bytes,
            signed_error_bytes=comparison.signed_difference_bytes,
            absolute_error_bytes=comparison.absolute_difference_bytes,
            relative_error=comparison.relative_difference,
            static_overpredicts=comparison.static_overpredicts,
            static_underpredicts=comparison.static_underpredicts,
        )
        if not compiler.available:
            row["status"] = "compiler_unavailable"
            row["error_message"] = "; ".join(compiler.limitations)
    except Exception as exc:
        row.update(status="failed", error_message=f"{type(exc).__name__}: {exc}")
    return row


def donation_experiment() -> dict[str, Any]:
    def update(params, gradients):
        return params - gradients

    params = jax.ShapeDtypeStruct((1024, 1024), "float32")
    args = (params, params)
    result: dict[str, Any] = {"shape": [1024, 1024], "dtype": "float32"}
    for label, donate in (("without_donation", ()), ("with_donation", (0,))):
        try:
            compiled = jax.jit(update, donate_argnums=donate).lower(*args).compile()
            stats = compiled.memory_analysis()
            result[label] = {
                "argument_bytes": getattr(stats, "argument_size_in_bytes", None),
                "output_bytes": getattr(stats, "output_size_in_bytes", None),
                "temporary_bytes": getattr(stats, "temp_size_in_bytes", None),
                "alias_bytes": getattr(stats, "alias_size_in_bytes", None),
                "compiler_accounted_bytes": _accounted(stats),
            }
        except Exception as exc:
            result[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "ok" and row["relative_error"] is not None]
    return {"overall": summarize(valid), "by_family": {family: summarize([row for row in valid if row["family"] == family]) for family in sorted({row["family"] for row in valid})}}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["relative_error"]) for row in rows]
    signed = [float(row["signed_error_bytes"]) / int(row["compiler_accounted_bytes"]) for row in rows]
    over = [value for value in values if value > 0]
    under = [value for value in values if value < 0]
    return {
        "n": len(rows),
        "mean_relative_error": statistics.mean(values) if values else None,
        "median_relative_error": statistics.median(values) if values else None,
        "p90_absolute_relative_error": percentile([abs(value) for value in values], 0.90),
        "mean_signed_relative_error": statistics.mean(signed) if signed else None,
        "overprediction_fraction": len(over) / len(values) if values else None,
        "underprediction_fraction": len(under) / len(values) if values else None,
        "worst_overprediction": max(over) if over else None,
        "worst_underprediction": min(under) if under else None,
    }


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))]


def write_outputs(prefix: Path, rows: list[dict[str, Any]], summary: dict[str, Any], donation: dict[str, Any]) -> None:
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    for path in (csv_path, json_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing experiment result: {path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "environment": {
            "python_version": platform_version(),
            "jax_version": jax.__version__,
            "jaxlib_version": _jaxlib_version(),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "platform": getattr(jax.devices()[0], "platform", None),
        },
        "summary": summary,
        "donation": donation,
        "rows": len(rows),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def platform_version() -> str:
    return platform_string(f"{host_platform.python_version()}")


def platform_string(value: str) -> str:
    return value


def _jaxlib_version() -> str:
    import jaxlib

    return jaxlib.__version__


def _accounted(stats: Any) -> int | None:
    fields = [getattr(stats, name, None) for name in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")]
    return fields[0] + fields[1] + fields[2] - fields[3] if all(value is not None for value in fields) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = [run_case(case) for case in cases()]
    donation = donation_experiment()
    summary = aggregate(rows)
    write_outputs(args.output_prefix, rows, summary, donation)
    print(json.dumps({"rows": len(rows), "summary": summary, "donation": donation}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
