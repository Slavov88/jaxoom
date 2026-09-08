"""Cross-version JAX compatibility and compiler-memory comparison harness."""
from __future__ import annotations

import argparse
import json
import os
import platform
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
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    dtype: str
    configuration: dict[str, Any]


def abstract(shape: tuple[int, ...], dtype: str) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype)


def cases() -> list[Case]:
    f32 = "float32"
    f16 = "float16"
    cases_: list[Case] = []
    cases_.append(Case("elementwise-f32", "elementwise", lambda x: jnp.sin(x) + x, (abstract((1024, 1024), f32),), f32, {"shape": [1024, 1024]}))
    cases_.append(Case("matmul-f32", "matmul", lambda x, w: x @ w, (abstract((1024, 1024), f32), abstract((1024, 1024), f32)), f32, {"shape": [1024, 1024]}))
    cases_.append(Case("matmul-f16", "matmul", lambda x, w: x @ w, (abstract((1024, 1024), f16), abstract((1024, 1024), f16)), f16, {"shape": [1024, 1024]}))
    cases_.append(Case("residual-f32", "residual", lambda x, w: x @ w + x, (abstract((512, 512), f32), abstract((512, 512), f32)), f32, {"shape": [512, 512]}))
    cases_.append(Case("reduction-f32", "reduction", lambda x: jnp.sum(x, axis=-1), (abstract((2048, 1024), f32),), f32, {"shape": [2048, 1024]}))

    def mlp(x, w1, w2):
        return jnp.tanh(x @ w1) @ w2

    cases_.append(Case("mlp-f32", "mlp", mlp, (abstract((256, 1024), f32), abstract((1024, 1024), f32), abstract((1024, 512), f32)), f32, {"batch": 256, "width": 1024}))
    cases_.append(Case("mlp-f16", "mlp", mlp, (abstract((256, 1024), f16), abstract((1024, 1024), f16), abstract((1024, 512), f16)), f16, {"batch": 256, "width": 1024}))

    def attention(x):
        q = x.reshape(1, 512, 8, 64)
        scores = jnp.einsum("bshd,bthd->bhst", q, q)
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bhst,bthd->bshd", weights, q).reshape(1, 512, 512)

    cases_.append(Case("attention-f32", "attention", attention, (abstract((1, 512, 512), f32),), f32, {"sequence": 512, "heads": 8, "head_dim": 64}))
    cases_.append(Case("attention-f16", "attention", attention, (abstract((1, 512, 512), f16),), f16, {"sequence": 512, "heads": 8, "head_dim": 64}))

    def transformer(x, w1, w2):
        q = x.reshape(1, 256, 8, 32)
        scores = jnp.einsum("bshd,bthd->bhst", q, q)
        attended = jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(scores, axis=-1), q).reshape(1, 256, 256)
        residual = x + attended
        return residual + jnp.tanh(residual @ w1) @ w2

    cases_.append(Case("transformer-f32", "transformer", transformer, (abstract((1, 256, 256), f32), abstract((256, 1024), f32), abstract((1024, 256), f32)), f32, {"sequence": 256, "width": 256, "heads": 8}))

    def grad_case(x, w):
        def loss(weight):
            return jnp.sum(jnp.tanh(x @ weight))
        return jax.grad(loss)(w)

    cases_.append(Case("grad-f32", "autodiff", grad_case, (abstract((128, 256), f32), abstract((256, 256), f32)), f32, {"batch": 128, "width": 256}))

    def value_and_grad_case(x, w):
        def loss(weight):
            return jnp.mean((x @ weight) ** 2)
        return jax.value_and_grad(loss)(w)

    cases_.append(Case("value-and-grad-f32", "autodiff", value_and_grad_case, (abstract((128, 256), f32), abstract((256, 256), f32)), f32, {"batch": 128, "width": 256}))

    def convolution(x, kernel):
        return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))

    cases_.append(Case("convolution-f32", "convolution", convolution, (abstract((1, 128, 128, 16), f32), abstract((3, 3, 16, 32), f32)), f32, {"height": 128, "width": 128, "channels": 16, "features": 32}))
    cases_.append(Case("fft-f32", "fft", lambda x: jnp.fft.rfftn(x, axes=(-2, -1)), (abstract((2, 256, 256), f32),), f32, {"batch": 2, "height": 256, "width": 256}))
    return cases_


def jaxpr_summary(fn: Callable[..., Any], args: tuple[Any, ...]) -> dict[str, Any]:
    traced = jax.make_jaxpr(fn)(*args)
    jaxpr = traced.jaxpr if hasattr(traced, "jaxpr") else traced
    primitive_names = [eqn.primitive.name for eqn in jaxpr.eqns]
    dropvars = sum(type(var).__name__ == "DropVar" for eqn in jaxpr.eqns for var in (*eqn.invars, *eqn.outvars))
    nested = 0
    for eqn in jaxpr.eqns:
        for value in eqn.params.values():
            nested += _nested_jaxprs(value)
    return {
        "equation_count": len(jaxpr.eqns),
        "primitive_names": primitive_names,
        "input_count": len(jaxpr.invars),
        "output_count": len(jaxpr.outvars),
        "dropvar_count": dropvars,
        "nested_jaxpr_count": nested,
    }


def _nested_jaxprs(value: Any) -> int:
    if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        return 1 + sum(_nested_jaxprs(param) for eqn in value.jaxpr.eqns for param in eqn.params.values())
    if hasattr(value, "eqns"):
        return 1 + sum(_nested_jaxprs(param) for eqn in value.eqns for param in eqn.params.values())
    if isinstance(value, (tuple, list)):
        return sum(_nested_jaxprs(item) for item in value)
    if isinstance(value, dict):
        return sum(_nested_jaxprs(item) for item in value.values())
    return 0


def field(value: Any, name: str) -> int | None:
    item = getattr(value, name, None)
    return int(item) if item is not None else None


def environment() -> dict[str, Any]:
    import jaxlib
    import ml_dtypes
    import numpy
    return {
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "numpy_version": numpy.__version__,
        "ml_dtypes_version": ml_dtypes.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_kind": [getattr(device, "device_kind", None) for device in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "allocator_environment": {key: os.environ.get(key) for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")},
    }


def run(output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases():
        row: dict[str, Any] = {"name": case.name, "family": case.family, "dtype": case.dtype, "configuration": case.configuration, "status": "PASS"}
        try:
            static = jaxoom.estimate(case.fn, *case.args)
            compiler = jaxoom.compile_analyze(case.fn, *case.args)
            interval = jaxoom.calibrate(static, backend=jax.default_backend())
            assessment = jaxoom.assess(static, memory_limit="16 GiB", backend=jax.default_backend())
            comparison = jaxoom.compare_memory(static, compiler)
            summary = jaxpr_summary(case.fn, case.args)
            row.update(
                structural_peak_bytes=static.estimated_peak_bytes,
                compiler_available=compiler.available,
                compiler_argument_bytes=compiler.argument_bytes,
                compiler_output_bytes=compiler.output_bytes,
                compiler_temp_bytes=compiler.temporary_bytes,
                compiler_alias_bytes=compiler.alias_bytes,
                compiler_accounted_bytes=compiler.compiler_accounted_bytes,
                compiler_limitations=compiler.limitations,
                calibrated_lower_bytes=interval.lower_bytes,
                calibrated_central_bytes=interval.central_bytes,
                calibrated_upper_bytes=interval.upper_bytes,
                calibration_applicability=interval.applicability,
                calibration_dataset=interval.dataset_version,
                assessment_risk=assessment.risk,
                assessment_calibrated=assessment.calibrated,
                comparison_static_bytes=comparison.static_peak_bytes,
                comparison_compiler_bytes=comparison.compiler_accounted_bytes,
                jaxpr=summary,
            )
            if compiler.compiler_accounted_bytes is not None:
                row["static_compiler_ratio"] = static.estimated_peak_bytes / compiler.compiler_accounted_bytes if compiler.compiler_accounted_bytes else None
        except Exception as exc:
            row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    payload = {"environment": environment(), "cases": rows}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
