"""Analyze the committed RTX 3050 and returned Tesla T4 compiler results.

The numerical analysis is backend-independent. Optional StableHLO summaries are
collected on the active JAX installation and are labeled with that device. The
returned T4 archive did not contain T4 IR, so this script never treats local IR
as a cross-device IR comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
from pathlib import Path
from typing import Any, Callable


FROZEN_LOWER = 0.6734660838594787
FROZEN_CENTRAL = 1.0
FROZEN_UPPER = 1.6676505193119118
FAMILIES = ("attention", "autodiff", "convolution", "elementwise", "matmul", "mlp", "residual", "training", "transformer")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def q(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))]


def ratio(row: dict[str, Any], device: str) -> float:
    if device == "reference":
        return row["reference_compiler_bytes"] / row["static_bytes_reference"]
    return row["current_compiler_bytes"] / row["static_bytes_current"]


def compiler(row: dict[str, Any], device: str) -> int:
    return row["reference_compiler_bytes"] if device == "reference" else row["current_compiler_bytes"]


def static(row: dict[str, Any], device: str) -> int:
    return row["static_bytes_reference"] if device == "reference" else row["static_bytes_current"]


def quantile_bounds(rows: list[dict[str, Any]], device: str, upper_probability: float = 0.95) -> dict[str, float]:
    values = [ratio(row, device) for row in rows]
    return {"lower": q(values, 0.10), "central": q(values, 0.50), "upper": q(values, upper_probability)}


def additive_bounds(rows: list[dict[str, Any]], device: str, upper_probability: float = 0.95) -> dict[str, float]:
    residuals = [compiler(row, device) - static(row, device) for row in rows]
    return {"lower": q(residuals, 0.10), "central": q(residuals, 0.50), "upper": q(residuals, upper_probability)}


def score(rows: list[dict[str, Any]], bound: Callable[[int], tuple[float, float, float]]) -> dict[str, Any]:
    inside = []
    upper = []
    widths = []
    for row in rows:
        lo, _, up = bound(static(row, "current"))
        value = compiler(row, "current")
        inside.append(lo <= value <= up)
        upper.append(value <= up)
        widths.append((up - lo) / static(row, "current"))
    return {
        "count": len(rows),
        "interval_coverage_count": sum(inside),
        "interval_coverage": sum(inside) / len(rows),
        "upper_coverage_count": sum(upper),
        "upper_coverage": sum(upper) / len(rows),
        "upper_miss_rate": 1 - sum(upper) / len(rows),
        "median_width_over_static": statistics.median(widths),
        "p90_width_over_static": q(widths, 0.90),
        "maximum_width_over_static": max(widths),
    }


def cross_score(train: list[dict[str, Any]], test: list[dict[str, Any]], train_device: str, test_device: str, method: str) -> dict[str, Any]:
    rb = quantile_bounds(train, train_device)
    ab = additive_bounds(train, train_device)

    def bound(static_bytes: int) -> tuple[float, float, float]:
        if method == "ratio":
            return static_bytes * rb["lower"], static_bytes * rb["central"], static_bytes * rb["upper"]
        additive = lambda key: static_bytes + ab[key]
        if method == "additive":
            return additive("lower"), additive("central"), additive("upper")
        if method == "hybrid":
            return (
                min(static_bytes * rb["lower"], additive("lower")),
                static_bytes * rb["central"],
                max(static_bytes * rb["upper"], additive("upper")),
            )
        raise ValueError(method)

    values = []
    for row in test:
        lo, _, up = bound(static(row, test_device))
        value = compiler(row, test_device)
        values.append({"inside": lo <= value <= up, "upper": value <= up, "width": (up - lo) / static(row, test_device)})
    return {
        "fit_device": train_device,
        "test_device": test_device,
        "method": method,
        "fit_bounds": {"ratio": rb, "additive_bytes": ab},
        "count": len(values),
        "interval_coverage_count": sum(x["inside"] for x in values),
        "interval_coverage": sum(x["inside"] for x in values) / len(values),
        "upper_coverage_count": sum(x["upper"] for x in values),
        "upper_coverage": sum(x["upper"] for x in values) / len(values),
        "upper_miss_rate": sum(not x["upper"] for x in values) / len(values),
        "median_width_over_static": statistics.median(x["width"] for x in values),
        "p90_width_over_static": q([x["width"] for x in values], 0.90),
        "maximum_width_over_static": max(x["width"] for x in values),
    }


def ir_summaries(repo: Path, names: set[str]) -> dict[str, Any]:
    try:
        import jax
        from accelerator_calibration import cases
    except Exception as exc:
        return {"status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}"}

    rows = []
    for case in cases():
        if case.name not in names:
            continue
        try:
            lowered = jax.jit(case.fn).lower(*case.args)
            module = lowered.compiler_ir(dialect="stablehlo")
            text = str(module)
            ops: dict[str, int] = {}
            for op in re.findall(r"stablehlo\.([A-Za-z0-9_]+)", text):
                ops[op] = ops.get(op, 0) + 1
            rows.append({"name": case.name, "family": case.family, "dtype": case.dtype, "backend": jax.default_backend(), "device": [str(d) for d in jax.devices()], "stablehlo_text_bytes": len(text.encode()), "stablehlo_operation_counts": dict(sorted(ops.items())), "custom_call_count": text.count("stablehlo.custom_call") + text.count("mhlo.custom_call")})
        except Exception as exc:
            rows.append({"name": case.name, "status": "IR_FAILURE", "error": f"{type(exc).__name__}: {exc}"})
    return {"status": "REFERENCE_DEVICE_ONLY", "device": [str(d) for d in jax.devices()], "jax_version": jax.__version__, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("device_temporary_drift_2026-09-08.json"))
    parser.add_argument("--with-ir", action="store_true")
    args = parser.parse_args()
    exp = args.repo / "experiments"
    t4 = load(exp / "device_transfer_validation_t4_2026-09-08.json")
    summary_artifact = load(exp / "device_transfer_validation_t4_summary_2026-09-08.json")
    baseline = load(exp / "jax_0_11_gpu_calibration_2026-09-08_v2.json")["rows"]
    rows = t4["cases"]
    baseline_by_name = {row["name"]: row for row in baseline}
    assert len(rows) == 46 and len({row["name"] for row in rows}) == 46
    assert set(row["name"] for row in rows) == set(baseline_by_name)
    assert all(row["static_bytes_current"] == row["static_bytes_reference"] for row in rows)
    for row in rows:
        for device in ("reference", "current"):
            components = (row[f"reference_argument_bytes"], row[f"reference_output_bytes"], row[f"reference_temp_bytes"], row[f"reference_alias_bytes"]) if device == "reference" else (row[f"current_argument_bytes"], row[f"current_output_bytes"], row[f"current_temp_bytes"], row[f"current_alias_bytes"])
            assert all(value is not None and value >= 0 for value in components)
            assert compiler(row, device) == components[0] + components[1] + components[2] - components[3]

    misses = []
    for row in rows:
        if not row["current_under_upper"]:
            misses.append({
                "name": row["name"], "family": row["family"], "dtype": row["dtype"], "configuration": row["configuration"],
                "static_bytes": static(row, "current"), "reference_compiler_bytes": compiler(row, "reference"), "current_compiler_bytes": compiler(row, "current"),
                "frozen_upper_bytes": row["frozen_upper_bytes"], "shortfall_bytes": compiler(row, "current") - row["frozen_upper_bytes"],
                "shortfall_over_static": (compiler(row, "current") - row["frozen_upper_bytes"]) / static(row, "current"),
                "reference_temp_bytes": row["reference_temp_bytes"], "current_temp_bytes": row["current_temp_bytes"], "temp_delta": row["temp_delta"],
                "reference_alias_bytes": row["reference_alias_bytes"], "current_alias_bytes": row["current_alias_bytes"],
            })

    controls = {family: [{"name": row["name"], "upper_covered": row["current_under_upper"], "ratio": ratio(row, "current")} for row in rows if row["family"] == family and row["current_under_upper"]] for family in FAMILIES}
    methods = {}
    for method in ("ratio", "additive", "hybrid"):
        methods[f"rtx_to_t4_{method}"] = cross_score(rows, rows, "reference", "current", method)
        methods[f"t4_to_rtx_{method}"] = cross_score(rows, rows, "current", "reference", method)

    # In the cross_score implementation, current/reference are fixed by the
    # paired artifact. Re-score the reverse direction explicitly below.
    def reverse_score(method: str) -> dict[str, Any]:
        train_device, test_device = "current", "reference"
        rb = quantile_bounds(rows, train_device); ab = additive_bounds(rows, train_device)
        vals=[]
        for row in rows:
            s=static(row,test_device); value=compiler(row,test_device)
            if method == "ratio": lo,up=s*rb["lower"],s*rb["upper"]
            elif method == "additive": lo,up=s+ab["lower"],s+ab["upper"]
            else: lo,up=min(s*rb["lower"],s+ab["lower"]),max(s*rb["upper"],s+ab["upper"])
            vals.append(((lo<=value<=up),value<=up,(up-lo)/s))
        return {"fit_device":"t4","test_device":"rtx3050","method":method,"fit_bounds":{"ratio":rb,"additive_bytes":ab},"count":len(vals),"interval_coverage_count":sum(x[0] for x in vals),"interval_coverage":sum(x[0] for x in vals)/len(vals),"upper_coverage_count":sum(x[1] for x in vals),"upper_coverage":sum(x[1] for x in vals)/len(vals),"upper_miss_rate":sum(not x[1] for x in vals)/len(vals),"median_width_over_static":statistics.median(x[2] for x in vals),"p90_width_over_static":q([x[2] for x in vals],.9),"maximum_width_over_static":max(x[2] for x in vals)}
    # Replace the reverse and forward candidate records with paired-direction results.
    for method in ("ratio", "additive", "hybrid"):
        rb=quantile_bounds(rows,"reference");ab=additive_bounds(rows,"reference"); vals=[]
        for row in rows:
            s=static(row,"current");value=compiler(row,"current")
            if method=="ratio":lo,up=s*rb["lower"],s*rb["upper"]
            elif method=="additive":lo,up=s+ab["lower"],s+ab["upper"]
            else:lo,up=min(s*rb["lower"],s+ab["lower"]),max(s*rb["upper"],s+ab["upper"])
            vals.append((lo<=value<=up,value<=up,(up-lo)/s))
        methods[f"rtx_to_t4_{method}"]={"fit_device":"rtx3050","test_device":"t4","method":method,"fit_bounds":{"ratio":rb,"additive_bytes":ab},"count":len(vals),"interval_coverage_count":sum(x[0] for x in vals),"interval_coverage":sum(x[0] for x in vals)/len(vals),"upper_coverage_count":sum(x[1] for x in vals),"upper_coverage":sum(x[1] for x in vals)/len(vals),"upper_miss_rate":sum(not x[1] for x in vals)/len(vals),"median_width_over_static":statistics.median(x[2] for x in vals),"p90_width_over_static":q([x[2] for x in vals],.9),"maximum_width_over_static":max(x[2] for x in vals)}
        methods[f"t4_to_rtx_{method}"]=reverse_score(method)

    output = {"status":"COMPUTATIONALLY VERIFIED","source":"device_transfer_validation_t4_2026-09-08.json","t4_environment":load(exp/"device_transfer_validation_t4_environment_2026-09-08.json"),"validation":{"case_count":len(rows),"static_matches":sum(row["static_bytes_current"]==row["static_bytes_reference"] for row in rows),"miss_count":len(misses),"misses":misses,"controls":controls,"frozen_summary":summary_artifact,"candidate_methods":methods,"ir":ir_summaries(args.repo,set(row["name"] for row in rows if row["family"] in {"autodiff","convolution"})) if args.with_ir else {"status":"NOT_RUN","reason":"T4 archive contains no T4 IR; rerun optional IR collection on an available device."}},"limitations":["T4 StableHLO was not captured in the returned archive, so IR summaries are reference-device-only when requested.","Two devices and one JAX release do not justify shipped alternative calibration."]}
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    summ={"status":output["status"],"misses":misses,"candidate_methods":methods,"ir_status":output["validation"]["ir"]["status"]}
    args.output.with_name("device_temporary_drift_summary_2026-09-08.json").write_text(json.dumps(summ,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(args.output)

if __name__ == "__main__": main()
