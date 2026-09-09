"""Scan-based missing-family T4 campaign infrastructure.

Scale search is deliberately separate from allocator thresholding: it finds
fixed moderate-compute configurations that cross a low T4 capacity, then the
existing fresh-process threshold runner labels allocator capacity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def configuration_id(family: str, config: dict[str, Any], dtype: str) -> str:
    payload = json.dumps({"family": family, "config": config, "dtype": dtype}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def candidates() -> list[dict[str, Any]]:
    rows=[]
    for dtype in ("float32", "float16"):
        for width in (512, 768, 1024, 1536):
            for layers in (8, 16, 32, 64):
                for batch in (64, 128, 256):
                    for family, config in (
                        ("mlp_scan", {"batch": batch, "width": width, "layers": layers, "retain": False}),
                        ("mlp_scan", {"batch": batch, "width": width, "layers": layers, "retain": True}),
                        ("training_scan", {"batch": batch, "width": width, "layers": layers}),
                        ("autodiff_scan", {"batch": batch, "width": width, "layers": layers}),
                    ):
                        rows.append({"configuration_id": configuration_id(family, config, dtype), "family": family, "configuration": config, "dtype": dtype})
    return rows


def static_screen(candidate: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jaxoom
    from runtime_validation import workload_from_config
    workload = workload_from_config(candidate["family"], candidate["configuration"], candidate["dtype"])
    report = jaxoom.estimate(workload.fn, *workload.abstract_args)
    interval = jaxoom.calibrate(report, backend="gpu", jax_version="0.11.0")
    jaxpr = jax.make_jaxpr(workload.fn)(*workload.abstract_args).jaxpr
    scans = [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "scan"]
    body_equations = 0
    for eqn in scans:
        body = eqn.params.get("jaxpr")
        if body is not None:
            body_equations += len(getattr(body, "eqns", getattr(getattr(body, "jaxpr", None), "eqns", ())))
    return {**candidate, "structural_peak_bytes": report.estimated_peak_bytes, "largest_buffer_bytes": max((getattr(item, "size_bytes", 0) for item in report.largest_buffers), default=0), "equation_count": report.equations_analyzed, "primitive_counts": {name: sum(str(e.primitive) == name for e in jaxpr.eqns) for name in sorted({str(e.primitive) for e in jaxpr.eqns})}, "scan_count": len(scans), "scan_body_equation_count": body_equations, "calibrated_lower_bytes": interval.lower_bytes, "calibrated_upper_bytes": interval.upper_bytes}


def screen(output: Path) -> None:
    rows=[]
    for candidate in candidates():
        try: rows.append(static_screen(candidate))
        except Exception as exc: rows.append({**candidate, "status":"OTHER_FAILURE", "failure":f"{type(exc).__name__}: {exc}"})
    output.write_text(json.dumps({"status":"OBSERVED","rows":rows},indent=2,sort_keys=True)+"\n",encoding="utf-8")


def probe(candidate: dict[str, Any], fraction: float, timeout: int) -> dict[str, Any]:
    env=os.environ.copy(); env["XLA_CLIENT_MEM_FRACTION"]=str(fraction); env["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"; env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION",None)
    command=[sys.executable,str(Path(__file__).with_name("execution_oom_diagnosis.py")),"--trial",candidate["family"],"--config",json.dumps(candidate["configuration"],sort_keys=True),"--dtype",candidate["dtype"],"--repetitions","1"]
    try: completed=subprocess.run(command,capture_output=True,text=True,env=env,timeout=timeout)
    except subprocess.TimeoutExpired as exc: return {**candidate,"requested_fraction":fraction,"outcome":"EXECUTION_TIMEOUT","message":str(exc)}
    for line in reversed(completed.stdout.splitlines()):
        try:
            row=json.loads(line); row.update({**candidate,"requested_fraction":fraction,"outcome":(row.get("execution_statuses") or [row.get("compile_status") or "OTHER_FAILURE"])[0]}); return row
        except json.JSONDecodeError: pass
    return {**candidate,"requested_fraction":fraction,"outcome":"OTHER_FAILURE","stderr":completed.stderr[-2000:]}


def scale_search(screening: Path, output: Path, fraction: float, timeout: int, max_probes: int, max_per_family: int) -> None:
    candidates_data=json.loads(screening.read_text())["rows"]
    candidates_data=[row for row in candidates_data if row.get("scan_count",0)>0]
    # Probe high-demand configurations first so the bounded budget reaches a
    # scale boundary instead of spending all probes on obvious FIT rows.
    candidates_data.sort(key=lambda row: row.get("calibrated_upper_bytes", 0), reverse=True)
    checkpoint=json.loads(output.read_text()) if output.exists() else {"status":"OBSERVED","rows":[]}
    rows=checkpoint.get("rows",[]); seen={r.get("configuration_id") for r in rows}
    for candidate in candidates_data:
        if len(rows)>=max_probes or candidate["configuration_id"] in seen: continue
        row=probe(candidate,fraction,timeout); rows.append(row); seen.add(candidate["configuration_id"])
        output.write_text(json.dumps({"status":"OBSERVED","stage":"scale_search","allocator_fraction":fraction,"rows":rows},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    grouped={}
    for row in rows:
        if row.get("outcome") in {"COMPILE_OOM","EXECUTION_OOM"}: grouped.setdefault(row.get("family"),[]).append(row)
    selected=[]
    for family, values in grouped.items():
        values.sort(key=lambda row:(row.get("dtype")!="float16",json.dumps(row.get("configuration",{}),sort_keys=True)))
        selected.extend(values[:max_per_family])
    Path(str(output)+".manifest.json").write_text(json.dumps({"status":"MANIFEST","selected":selected},indent=2,sort_keys=True)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--screening",type=Path,required=True); parser.add_argument("--scale-output",type=Path,required=True); parser.add_argument("--screen-only",action="store_true"); parser.add_argument("--fraction",type=float,default=0.06); parser.add_argument("--timeout",type=int,default=60); parser.add_argument("--max-probes",type=int,default=100); parser.add_argument("--max-per-family",type=int,default=2)
    args=parser.parse_args();
    if args.screen_only: screen(args.screening)
    else: scale_search(args.screening,args.scale_output,args.fraction,args.timeout,args.max_probes,args.max_per_family)

if __name__ == "__main__": main()
