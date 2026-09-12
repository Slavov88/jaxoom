"""Fresh-process runtime campaign for the frozen V2 expansion plan."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "oom_risk_expansion_plan_v2_2026-09-12.json"
DEFAULT_OUTPUT = ROOT / "oom_risk_expansion_results_v2_2026-09-12.json"


def nvidia_snapshot() -> dict[str, Any]:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return {"gpu_query": q.stdout.strip(), "gpu_query_returncode": q.returncode, "process_query": p.stdout.strip(), "process_query_returncode": p.returncode}
    except Exception as exc:
        return {"status": "NOT_AVAILABLE", "error": f"{type(exc).__name__}: {exc}"}


def classify_error(text: str, phase: str) -> str:
    lower=text.lower()
    if "timed out" in lower: return "COMPILE_TIMEOUT" if phase == "compile" else "EXECUTION_TIMEOUT"
    if any(s in lower for s in ("resource_exhausted", "out of memory", "cuda_error_out_of_memory", "cublas_status_alloc_failed")):
        return "COMPILE_OOM" if phase == "compile" else "EXECUTION_OOM"
    return "OTHER_FAILURE"


def run_one(candidate: dict[str, Any]) -> dict[str, Any]:
    # Importing JAX and constructing abstract inputs happens inside the fresh child.
    import jax
    import jax.numpy as jnp
    import jaxoom
    from oom_risk_expansion_v2 import candidate_functions
    started=time.time(); phase="initialization"; before=nvidia_snapshot()
    row={"candidate_id":candidate["candidate_id"],"workload_group_id":candidate["workload_group_id"],"family":candidate["family"],"configuration":candidate["configuration"],"dtype":candidate["dtype"],"graph_fingerprint":candidate["graph_fingerprint"],"v1_predicted_probability":candidate["v1_predicted_probability"],"aggregate_score":candidate["aggregate_score"],"phase":"initialization","status":"OTHER_FAILURE","outcome":"OTHER_FAILURE","environment":{"python_version":platform.python_version(),"jax_version":jax.__version__,"jaxlib_version":jax.lib.__version__,"backend":jax.default_backend(),"devices":[str(d) for d in jax.devices()],"allocator_environment":{k:os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_PREALLOCATE","XLA_CLIENT_MEM_FRACTION","XLA_PYTHON_CLIENT_ALLOCATOR","CUDA_VISIBLE_DEVICES")},"nvidia_before":before}}
    try:
        fn, abstract_args=candidate_functions(candidate["family"],candidate["configuration"],candidate["dtype"])
        args=tuple(jnp.zeros(a.shape,dtype=a.dtype) for a in abstract_args)
        phase="compile"; lowered=jax.jit(fn).lower(*args); compiled=lowered.compile()
        phase="execute"; result=compiled(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(),result)
        row.update({"phase":"complete","status":"FIT","outcome":"FIT","compile_seconds":None,"execute_seconds":time.time()-started,"nvidia_after":nvidia_snapshot()})
    except Exception as exc:
        text=f"{type(exc).__name__}: {exc}"
        row.update({"phase":phase,"status":classify_error(text,phase),"outcome":classify_error(text,phase),"error":text,"nvidia_after":nvidia_snapshot()})
    row["wall_seconds"]=time.time()-started
    return row


def child_main(candidate_id: str) -> None:
    plan=json.loads(PLAN_PATH.read_text()); candidate=next(r for r in plan["selected_candidates"] if r["candidate_id"]==candidate_id)
    print(json.dumps(run_one(candidate),sort_keys=True))


def campaign_main(args: argparse.Namespace) -> None:
    if not PLAN_PATH.exists(): raise SystemExit(f"missing frozen plan: {PLAN_PATH}")
    if args.output.exists() and not args.resume: raise SystemExit(f"refusing to overwrite existing results: {args.output}; use --resume")
    plan=json.loads(PLAN_PATH.read_text()); candidates=plan["selected_candidates"]
    results=[]
    if args.resume and args.output.exists(): results=json.loads(args.output.read_text())["results"]
    done={r["candidate_id"] for r in results}; pending=[r for r in candidates if r["candidate_id"] not in done]
    if args.limit is not None: pending=pending[:args.limit]
    for candidate in pending:
        env=os.environ.copy(); env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); env.pop("XLA_CLIENT_MEM_FRACTION",None); env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION",None)
        command=[sys.executable,str(Path(__file__).resolve()),"--run-one",candidate["candidate_id"]]
        started=time.time()
        try:
            completed=subprocess.run(command,cwd=ROOT.parent,capture_output=True,text=True,timeout=args.timeout,env=env)
            lines=[line for line in completed.stdout.splitlines() if line.strip()]
            row=json.loads(lines[-1]) if lines else {"candidate_id":candidate["candidate_id"],"outcome":"OTHER_FAILURE","status":"OTHER_FAILURE","error":completed.stderr[-4000:]}
            row["subprocess_returncode"]=completed.returncode; row["subprocess_stderr_tail"]=completed.stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            row={"candidate_id":candidate["candidate_id"],"workload_group_id":candidate["workload_group_id"],"family":candidate["family"],"configuration":candidate["configuration"],"dtype":candidate["dtype"],"v1_predicted_probability":candidate["v1_predicted_probability"],"aggregate_score":candidate["aggregate_score"],"status":"COMPILE_TIMEOUT","outcome":"COMPILE_TIMEOUT","phase":"compile","error":f"subprocess timeout after {args.timeout}s","stdout_tail":((exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout,bytes) else (exc.stdout or ""))[-4000:],"stderr_tail":((exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr,bytes) else (exc.stderr or ""))[-4000:]}
        row["campaign_wall_seconds"]=time.time()-started; row["campaign_environment"]={"primary_allocator":"XLA_PYTHON_CLIENT_PREALLOCATE=false","capacity_sweep":False}
        results.append(row); args.output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","plan_hash":__import__("hashlib").sha256(PLAN_PATH.read_bytes()).hexdigest(),"results":results},indent=2,sort_keys=True)+"\n")
        print(row["candidate_id"],row.get("outcome"),row.get("phase"),round(row["campaign_wall_seconds"],2),flush=True)
    print(json.dumps({"completed_this_call":len(pending),"total_results":len(results),"remaining":len(candidates)-len({r["candidate_id"] for r in results})},sort_keys=True))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--run-one"); ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); ap.add_argument("--limit",type=int); ap.add_argument("--timeout",type=int,default=180); ap.add_argument("--resume",action="store_true"); args=ap.parse_args()
    if args.run_one: child_main(args.run_one)
    else: campaign_main(args)

if __name__=="__main__": main()
