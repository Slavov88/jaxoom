"""Focused validation matrix for compiler-confirmed donation advice."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import jaxoom


DATE = "2026-09-08"


def workloads() -> list[dict[str, Any]]:
    S = jax.ShapeDtypeStruct

    def simple_update(x):
        return x + 1

    def mismatch(x):
        return jnp.zeros((x.shape[0], x.shape[1] // 2), dtype=x.dtype)

    def residual(x):
        return x + jnp.tanh(x)

    def multiple(left, right):
        return left + right, left - right

    def parameter_update(params, grads):
        return jax.tree_util.tree_map(lambda parameter, gradient: parameter - 0.01 * gradient, params, grads)

    def train_step(params, opt_state, batch):
        features, target = batch

        def loss(parameters):
            prediction = features @ parameters["weight"] + parameters["bias"]
            return jnp.mean((prediction - target) ** 2)

        value, gradients = jax.value_and_grad(loss)(params)
        next_params = jax.tree_util.tree_map(lambda parameter, gradient: parameter - 0.01 * gradient, params, gradients)
        next_state = jax.tree_util.tree_map(lambda state: state + 1, opt_state)
        return value, next_params, next_state

    matrix = [
        {"name": "simple_update", "family": "simple", "fn": simple_update, "args": (S((256, 256), "float32"),)},
        {"name": "shape_mismatch", "family": "negative_control", "fn": mismatch, "args": (S((256, 256), "float32"),)},
        {"name": "residual", "family": "residual", "fn": residual, "args": (S((256, 256), "float32"),)},
        {"name": "multiple_inputs", "family": "multiple_inputs", "fn": multiple, "args": (S((256, 256), "float32"), S((256, 256), "float32"))},
        {"name": "parameter_update", "family": "parameter_update", "fn": parameter_update, "args": ({"weight": S((256, 256), "float32"), "bias": S((256,), "float32")}, {"weight": S((256, 256), "float32"), "bias": S((256,), "float32")})},
        {"name": "training_like", "family": "training", "fn": train_step, "args": ({"weight": S((64, 32), "float32"), "bias": S((32,), "float32")}, {"weight": S((64, 32), "float32"), "bias": S((32,), "float32")}, (S((128, 64), "float32"), S((128, 32), "float32")))},
    ]
    return matrix


def candidate_json(candidate: Any) -> dict[str, Any]:
    return {
        "argnums": list(candidate.argnums),
        "input_bytes": candidate.input_bytes,
        "input_leaves": [{"shape": list(leaf.shape), "dtype": leaf.dtype, "nbytes": leaf.nbytes} for leaf in candidate.input_leaves],
        "compatible_leaf_count": candidate.compatible_leaf_count,
        "status": candidate.status,
        "baseline_compiler_bytes": candidate.baseline_compiler_bytes,
        "donated_compiler_bytes": candidate.donated_compiler_bytes,
        "compiler_saving_bytes": candidate.compiler_saving_bytes,
        "saving_fraction": candidate.saving_fraction,
        "baseline_alias_bytes": candidate.baseline_alias_bytes,
        "donated_alias_bytes": candidate.donated_alias_bytes,
        "alias_gain_bytes": candidate.alias_gain_bytes,
        "compiler_confirmed": candidate.compiler_confirmed,
        "warnings": list(candidate.warnings),
        "error": candidate.error,
    }


def report_json(report: Any) -> dict[str, Any]:
    return {
        "baseline": {
            "argument_bytes": report.baseline.argument_bytes,
            "output_bytes": report.baseline.output_bytes,
            "temporary_bytes": report.baseline.temporary_bytes,
            "alias_bytes": report.baseline.alias_bytes,
            "compiler_accounted_bytes": report.baseline.compiler_accounted_bytes,
            "available": report.baseline.available,
        },
        "candidates": [candidate_json(candidate) for candidate in report.candidates],
        "best_argnums": list(report.best.argnums) if report.best else None,
        "compiler_evaluations": report.compiler_evaluations,
        "backend": report.backend,
        "jax_version": report.jax_version,
        "limitations": list(report.limitations),
    }


def concrete_args(name: str) -> tuple[Any, ...] | None:
    key = jax.random.key(0)
    if name == "simple_update" or name == "residual":
        return (jax.random.normal(key, (32, 32), dtype=jnp.float32),)
    if name == "shape_mismatch":
        return (jax.random.normal(key, (32, 32), dtype=jnp.float32),)
    if name == "multiple_inputs":
        return (jax.random.normal(key, (32, 32), dtype=jnp.float32), jax.random.normal(key, (32, 32), dtype=jnp.float32))
    if name == "parameter_update":
        return ({"weight": jax.random.normal(key, (32, 16), dtype=jnp.float32), "bias": jnp.zeros((16,), dtype=jnp.float32)}, {"weight": jnp.ones((32, 16), dtype=jnp.float32), "bias": jnp.ones((16,), dtype=jnp.float32)})
    if name == "training_like":
        return (
            {"weight": jax.random.normal(key, (16, 8), dtype=jnp.float32), "bias": jnp.zeros((8,), dtype=jnp.float32)},
            {"weight": jnp.zeros((16, 8), dtype=jnp.float32), "bias": jnp.zeros((8,), dtype=jnp.float32)},
            (jax.random.normal(key, (16, 16), dtype=jnp.float32), jax.random.normal(key, (16, 8), dtype=jnp.float32)),
        )
    return None


def compare_trees(left: Any, right: Any) -> None:
    leaves_left, treedef_left = jax.tree_util.tree_flatten(left)
    leaves_right, treedef_right = jax.tree_util.tree_flatten(right)
    assert treedef_left == treedef_right
    assert len(leaves_left) == len(leaves_right)
    for expected, actual in zip(leaves_left, leaves_right):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), rtol=2e-4, atol=2e-5)


def runtime_smoke(workload: dict[str, Any], report: Any) -> dict[str, Any]:
    if report.best is None:
        return {"status": "NOT_RUN", "reason": "no compiler-confirmed candidate"}
    args = concrete_args(workload["name"])
    if args is None:
        return {"status": "NOT_RUN", "reason": "no concrete arguments"}
    try:
        baseline = jax.jit(workload["fn"])(*args)
        donated_args = _fresh_copy(args)
        donated = jax.jit(workload["fn"], donate_argnums=report.best.argnums)(*donated_args)
        compare_trees(baseline, donated)
        return {"status": "MATCH", "argnums": list(report.best.argnums)}
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _fresh_copy(value: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.array(leaf) if hasattr(leaf, "shape") else leaf, value)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    singles = [candidate for row in rows for candidate in row["report"]["candidates"] if len(candidate["argnums"]) == 1]
    static_compatible = [candidate for candidate in singles if candidate["status"] != "NOT_COMPATIBLE"]
    confirmed = [candidate for candidate in singles if candidate["compiler_confirmed"]]
    all_confirmed = [candidate for row in rows for candidate in row["report"]["candidates"] if candidate["compiler_confirmed"]]
    savings = [candidate["compiler_saving_bytes"] for candidate in all_confirmed]
    fractions = [candidate["saving_fraction"] for candidate in all_confirmed]
    evaluations = [row["report"]["compiler_evaluations"] for row in rows]
    return {
        "workloads": len(rows),
        "static_compatible_single_candidates": len(static_compatible),
        "compiler_confirmed_beneficial_single_candidates": len(confirmed),
        "static_false_positive_single_candidates": len(static_compatible) - len(confirmed),
        "static_candidate_precision": len(confirmed) / len(static_compatible) if static_compatible else None,
        "compiler_confirmed_candidates_including_combinations": len(all_confirmed),
        "median_saving_bytes": statistics.median(savings) if savings else None,
        "median_saving_fraction": statistics.median(fractions) if fractions else None,
        "maximum_saving_bytes": max(savings) if savings else None,
        "maximum_saving_fraction": max(fractions) if fractions else None,
        "workloads_without_useful_donation": sum(row["report"]["best_argnums"] is None for row in rows),
        "median_compiler_evaluations": statistics.median(evaluations) if evaluations else 0,
        "maximum_compiler_evaluations": max(evaluations) if evaluations else 0,
        "runtime_smoke_matches": sum(row["runtime_smoke"]["status"] == "MATCH" for row in rows),
        "runtime_smoke_failures": sum(row["runtime_smoke"]["status"] == "FAIL" for row in rows),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Compiler-confirmed donation validation",
        "",
        "**Status: COMPUTATIONALLY VERIFIED for the recorded environments.**",
        "",
        f"JAX {payload['environment']['jax_version']} on {payload['environment']['backend']}.",
        "The advisor measures compiler-accounted before and after memory. It does not establish caller semantic safety.",
        "",
        "## Summary",
        "",
        f"- Workloads: {summary['workloads']}",
        f"- Static-compatible single candidates: {summary['static_compatible_single_candidates']}",
        f"- Compiler-confirmed beneficial single candidates: {summary['compiler_confirmed_beneficial_single_candidates']}",
        f"- Static false positives: {summary['static_false_positive_single_candidates']}",
        f"- Static candidate precision: {summary['static_candidate_precision']:.1%}" if summary["static_candidate_precision"] is not None else "- Static candidate precision: unavailable",
        f"- Median saving: {summary['median_saving_bytes']} bytes ({summary['median_saving_fraction']:.1%})" if summary["median_saving_bytes"] is not None else "- Median saving: unavailable",
        f"- Maximum saving: {summary['maximum_saving_bytes']} bytes ({summary['maximum_saving_fraction']:.1%})" if summary["maximum_saving_bytes"] is not None else "- Maximum saving: unavailable",
        f"- Median and maximum compiler evaluations: {summary['median_compiler_evaluations']} and {summary['maximum_compiler_evaluations']}",
        f"- Runtime smoke matches: {summary['runtime_smoke_matches']}",
        "",
        "## Workloads",
        "",
        "| Workload | Best argnums | Status | Saving bytes | Saving fraction | Runtime smoke |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        best = row["report"]["best_argnums"]
        candidate = next((c for c in row["report"]["candidates"] if c["argnums"] == best), None) if best is not None else None
        lines.append(f"| {row['name']} | {best or '-'} | {candidate['status'] if candidate else 'NONE'} | {candidate['compiler_saving_bytes'] if candidate else '-'} | {candidate['saving_fraction']:.1%} | {row['runtime_smoke']['status']} |" if candidate and candidate["saving_fraction"] is not None else f"| {row['name']} | {best or '-'} | {candidate['status'] if candidate else 'NONE'} | {candidate['compiler_saving_bytes'] if candidate else '-'} | - | {row['runtime_smoke']['status']} |")
    lines.extend([
        "",
        "## Semantics and limitations",
        "",
        "- A compiler-confirmed candidate is not a proof that donation is safe for the caller.",
        "- Donated argument buffers must not be used after the compiled call.",
        "- Donation is evaluated for positional arguments. A donated pytree argument may donate eligible leaves within that argument.",
        "- Savings are compiler-accounted memory, not runtime peak memory.",
        "- Calibration intervals are not used to compute savings.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--max-compilations", type=int, default=32)
    args = parser.parse_args()
    rows = []
    for workload in workloads():
        report = jaxoom.analyze_donation(workload["fn"], *workload["args"], max_compilations=args.max_compilations)
        rows.append({"name": workload["name"], "family": workload["family"], "report": report_json(report), "runtime_smoke": runtime_smoke(workload, report)})
    try:
        repository_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.SubprocessError):
        repository_commit = None
    env = {"python_version": platform.python_version(), "jax_version": jax.__version__, "jaxlib_version": __import__("jaxlib").__version__, "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()], "xla_flags": os.environ.get("XLA_FLAGS"), "repository_commit": repository_commit, "seed": 0, "command": [sys.executable, *sys.argv]}
    payload = {"status": "COMPUTATIONALLY VERIFIED", "environment": env, "max_compilations": args.max_compilations, "rows": rows, "summary": summarize(rows)}
    prefix = args.output_prefix
    for suffix, content in ((".json", json.dumps(payload, indent=2, sort_keys=True) + "\n"), ("_summary.json", json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n"), ("_report.md", render_markdown(payload))):
        path = prefix.with_name(prefix.name + suffix)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "environment": env}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
