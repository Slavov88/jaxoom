"""Fresh-process execution for the frozen, model-independent held-out panel."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=ROOT/"oom_risk_heldout_plan_2026-09-13.json"; PRED=ROOT/"oom_risk_heldout_predictions_2026-09-13.json"; DEFAULT=ROOT/"oom_risk_heldout_results_2026-09-13.json"

def gpu_snapshot():
 try:
  g=subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
  p=subprocess.run(["nvidia-smi","--query-compute-apps=pid,name,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
  return {"gpu":g.stdout.strip(),"processes":p.stdout.strip(),"returncode":g.returncode}
 except Exception as e:return {"status":"NOT_AVAILABLE","error":f"{type(e).__name__}: {e}"}

def classify(text,phase):
 s=text.lower()
 if any(x in s for x in ("resource_exhausted","out of memory","cuda_error_out_of_memory","cublas_status_alloc_failed")):
  return {"initialization":"INITIALIZATION_OOM","compile":"COMPILE_OOM","execute":"EXECUTION_OOM"}.get(phase,"OTHER_FAILURE")
 if "timed out" in s:return "COMPILE_TIMEOUT" if phase=="compile" else "EXECUTION_TIMEOUT"
 return "OTHER_FAILURE"

def run_one(c,repeat_id=0):
 import jax,jax.numpy as jnp
 from oom_risk_expansion_v2 import candidate_functions
 started=time.time(); phase="initialization"; before=gpu_snapshot()
 pred=json.loads(PRED.read_text()); p=next(x for x in pred["predictions"] if x["candidate_id"]==c["candidate_id"])
 row={"candidate_id":c["candidate_id"],"repeat_id":repeat_id,"workload_group_id":c["workload_group_id"],"graph_fingerprint":c["graph_fingerprint"],"family":c["family"],"configuration":c["configuration"],"dtype":c["dtype"],"v2_probability":p["v2_probability"],"v3_probability":p["v3_probability"],"aggregate_score":p["aggregate_score"],"sampling_stratum":c["sampling_stratum"],"phase":phase,"outcome":"OTHER_FAILURE","environment":{"python_version":platform.python_version(),"jax_version":jax.__version__,"jaxlib_version":jax.lib.__version__,"backend":jax.default_backend(),"devices":[str(d) for d in jax.devices()],"allocator_environment":{k:os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_PREALLOCATE","XLA_CLIENT_MEM_FRACTION","XLA_PYTHON_CLIENT_ALLOCATOR")},"nvidia_before":before}}
 try:
  fn,abstract=candidate_functions(c["family"],c["configuration"],c["dtype"]); args=tuple(jnp.zeros(a.shape,dtype=a.dtype) for a in abstract)
  phase="compile"; executable=jax.jit(fn).lower(*args).compile(); phase="execute"; out=executable(*args); jax.tree_util.tree_map(lambda x:x.block_until_ready(),out); outcome="FIT"
 except Exception as e: outcome=classify(f"{type(e).__name__}: {e}",phase); row["error"]=f"{type(e).__name__}: {e}"
 row.update({"phase":"complete" if outcome=="FIT" else phase,"outcome":outcome,"status":outcome,"wall_seconds":time.time()-started,"nvidia_after":gpu_snapshot()}); return row

def child(cid,repeat):
 plan=json.loads(PLAN.read_text()); c=next(x for x in plan["selected_candidates"] if x["candidate_id"]==cid); print(json.dumps(run_one(c,repeat),sort_keys=True))

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run-one");ap.add_argument("--output",type=Path,default=DEFAULT);ap.add_argument("--resume",action="store_true");ap.add_argument("--timeout",type=int,default=180);ap.add_argument("--repeat",type=int,default=0);ap.add_argument("--candidate-ids",nargs="*");ap.add_argument("--limit",type=int);a=ap.parse_args()
 if a.run_one:child(a.run_one,a.repeat);return
 plan=json.loads(PLAN.read_text()); candidates=plan["selected_candidates"]
 if a.output.exists() and not a.resume:raise SystemExit(f"refusing to overwrite {a.output}; use --resume")
 results=json.loads(a.output.read_text())["results"] if a.resume and a.output.exists() else []
 done={(r["candidate_id"],r.get("repeat_id",0)) for r in results}; selected=set(a.candidate_ids) if a.candidate_ids else None
 pending=[c for c in candidates if (c["candidate_id"],a.repeat) not in done and (selected is None or c["candidate_id"] in selected)]
 if a.limit is not None:pending=pending[:a.limit]
 for c in pending:
  env=os.environ.copy();env["XLA_PYTHON_CLIENT_PREALLOCATE"]="false";env.pop("XLA_CLIENT_MEM_FRACTION",None);env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION",None);start=time.time()
  try:
   q=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--run-one",c["candidate_id"],"--repeat",str(a.repeat)],cwd=ROOT.parent,capture_output=True,text=True,timeout=a.timeout,env=env);lines=[x for x in q.stdout.splitlines() if x.strip()]
   r=json.loads(lines[-1]) if lines else {"candidate_id":c["candidate_id"],"repeat_id":a.repeat,"outcome":"OTHER_FAILURE","status":"OTHER_FAILURE","error":q.stderr[-4000:]};r["subprocess_returncode"]=q.returncode;r["subprocess_stderr_tail"]=q.stderr[-4000:]
  except subprocess.TimeoutExpired:
   r={"candidate_id":c["candidate_id"],"repeat_id":a.repeat,"workload_group_id":c["workload_group_id"],"graph_fingerprint":c["graph_fingerprint"],"family":c["family"],"configuration":c["configuration"],"dtype":c["dtype"],"phase":"compile","outcome":"COMPILE_TIMEOUT","status":"COMPILE_TIMEOUT","error":f"subprocess timeout after {a.timeout}s"}
  r["campaign_wall_seconds"]=time.time()-start;results.append(r);a.output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","plan_hash":hashlib.sha256(PLAN.read_bytes()).hexdigest(),"prediction_hash":json.loads(PRED.read_text())["prediction_hash"],"results":results},indent=2,sort_keys=True)+"\n");print(c["candidate_id"],a.repeat,r["outcome"],round(r["campaign_wall_seconds"],2),flush=True)
 print(json.dumps({"completed":len(pending),"total":len(results),"remaining":len(candidates)-len({r["candidate_id"] for r in results if r.get("repeat_id",0)==0})},sort_keys=True))
if __name__=="__main__":main()
