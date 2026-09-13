"""Read-only production-integration design audit for frozen OOM-risk V2.

This is an experiment/design tool. It deliberately does not export a public
API or change jaxoom runtime behavior.
"""
from __future__ import annotations
import hashlib,json,math,os,subprocess
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import jaxlib
import jaxoom
ROOT=Path(__file__).resolve().parent
V2_PATH=ROOT/"oom_risk_model_v2_2026-09-12.json"; V2_DATA=ROOT/"oom_risk_dataset_v2_2026-09-12.json"; V1_FREEZE=ROOT/"oom_risk_model_v1_frozen_2026-09-12.json"
PRED_PATHS=(ROOT/"oom_risk_heldout_predictions_2026-09-13.json",ROOT/"oom_risk_heldout2_predictions_2026-09-13.json",ROOT/"oom_risk_heldout2_conv_stress_predictions_2026-09-13.json")
OUT=ROOT/"oom_risk_production_design_2026-09-13.json";MODEL_OUT=ROOT/"oom_risk_production_model_v2_design_2026-09-13.json"
BUDGET=3*1024**3;UPPER_RATIO=1.66765051935476;THRESHOLD=.20
FEATURES=("structural_peak_over_budget","calibrated_upper_over_budget","largest_over_budget","top_two_peak_live_over_budget","peak_live_over_budget","dtype_bytes","config_numeric_count","config_numeric_max","config_numeric_log_product")
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),default=lambda y:y.item() if hasattr(y,"item") else str(y))
def sha(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def model_hash(model):return sha({"features":model["feature_names"],"means":model["means"],"scales":model["scales"],"coefficients":model["coefficients"]})
def load_model():
 m=json.loads(V2_PATH.read_text());expected="9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96"
 if m["model_hash"]!=expected or model_hash(m)!=expected:raise RuntimeError("frozen V2 model hash mismatch")
 return m
def production_features(family:str,config:dict[str,Any],dtype:str,workload_config:dict[str,Any]) -> dict[str,float]:
 """The proposed structured-workload adapter; no target compilation/execution."""
 from oom_risk_expansion_v2 import candidate_functions
 if set(workload_config)!=set(config) or any(not isinstance(v,(int,float)) for v in workload_config.values()):raise ValueError("structured numeric workload_config is required")
 fn,argspec=candidate_functions(family,config,dtype);report=jaxoom.estimate(fn,*argspec)
 structural=float(report.estimated_peak_bytes);largest=float(max((b.nbytes for b in report.largest_buffers),default=0));top_two=float(sum(b.nbytes for b in report.peak.live_buffers[:2]))
 nums=[float(v) for v in workload_config.values()]
 raw={"structural_peak_bytes":structural,"calibrated_upper_bytes":structural*UPPER_RATIO,"largest_buffer_bytes":largest,"top_two_peak_live_bytes":top_two,"peak_live_bytes":structural,"dtype_bytes":2.0 if dtype=="float16" else 4.0}
 f={"structural_peak_over_budget":structural/BUDGET,"calibrated_upper_over_budget":raw["calibrated_upper_bytes"]/BUDGET,"largest_over_budget":largest/BUDGET,"top_two_peak_live_over_budget":top_two/BUDGET,"peak_live_over_budget":structural/BUDGET,"dtype_bytes":raw["dtype_bytes"],"config_numeric_count":float(len(nums)),"config_numeric_max":max(nums,default=0.0),"config_numeric_log_product":math.log1p(math.prod(max(1.0,v) for v in nums)) if nums else 0.0}
 return f
def predict(f,m):
 x=[(float(f[n])-a)/b if b else 0.0 for n,a,b in zip(m["feature_names"],m["means"],m["scales"])]
 z=m["intercept"]+sum(c*v for c,v in zip(m["coefficients"][1:],x));return 1/(1+math.exp(-max(-40,min(40,z))))
def load_rows():
 rows=[]
 for path in PRED_PATHS:
  obj=json.loads(path.read_text());rows.extend(obj["predictions"])
 return rows
def replay_audit(model):
 rows=load_rows();max_abs=0.;max_rel=0.;feature_mismatch=0;mismatch_by_feature={n:0 for n in FEATURES};score_mismatch=0;status_mismatch=0;errors=[]
 for row in rows:
  try:f=production_features(row["family"],row["configuration"],row["dtype"],row["configuration"])
  except Exception as exc:errors.append({"candidate_id":row["candidate_id"],"error":f"{type(exc).__name__}: {exc}"});continue
  for n in FEATURES:
   # The frozen experiment schema names this feature largest_over_budget,
   # while its source rows expose largest_buffer_over_budget. Missing values
   # were median-imputed to zero during V2 training; preserve that distinction
   # in the audit rather than silently declaring a semantic match.
   old=float(row["static_features"].get(n,0.0));new=float(f[n]);d=abs(old-new);max_abs=max(max_abs,d);max_rel=max(max_rel,d/max(abs(old),1e-30));feature_mismatch+=d>1e-12;mismatch_by_feature[n]+=d>1e-12
  p=predict(f,model);d=abs(p-row["v2_probability"]);max_abs=max(max_abs,d);max_rel=max(max_rel,d/max(abs(row["v2_probability"]),1e-30));score_mismatch+=d>1e-12;status_mismatch+=(p<=THRESHOLD)!=(row["v2_probability"]<=THRESHOLD)
 return {"rows":len(rows),"feature_mismatched_values":feature_mismatch,"feature_mismatches_by_name":mismatch_by_feature,"model_probability_mismatches":score_mismatch,"status_mismatches_at_p_fit_0.20":status_mismatch,"max_absolute_difference_combined":max_abs,"max_relative_difference_combined":max_rel,"errors":errors,"target_compilation":False,"target_execution":False,"compiler_memory_analysis":False,"runtime_allocator_diagnostics":False}
def environment_probe():
 device=next(iter(jax.devices()),None);kind=str(getattr(device,"device_kind","") or "");backend=getattr(device,"platform",None) or jax.default_backend();pre=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE");frac=os.environ.get("XLA_CLIENT_MEM_FRACTION") or os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION");allocator=os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") or os.environ.get("TF_GPU_ALLOCATOR")
 driver=None
 try:
  text=subprocess.run(["nvidia-smi","--query-gpu=driver_version,name,memory.total","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True,timeout=5).stdout.strip();driver=text
 except Exception:pass
 return {"backend":backend,"device_kind":kind,"jax_version":jax.__version__,"jaxlib_version":jaxlib.__version__,"preallocate":pre,"memory_fraction":frac,"allocator":allocator,"driver_query":driver,"required":{"backend":"gpu","device_substring":"RTX 3050 Laptop GPU","jax_version":"0.11.0","jaxlib_version":"0.11.0","preallocate":"false","memory_fraction":None,"allocator":"CUDA/BFC default"}}
def environment_applicable(meta):
 req=meta["required"];return bool(meta["backend"]==req["backend"] and req["device_substring"].lower() in meta["device_kind"].lower() and meta["jax_version"]==req["jax_version"] and meta["jaxlib_version"]==req["jaxlib_version"] and meta["preallocate"]==req["preallocate"] and meta["memory_fraction"] is None and meta["allocator"] in (None,"default"))
def feature_ranges(rows):
 out={}
 for n in FEATURES:
  vals=[float(r["features"].get(n,0)) for r in rows if r.get("features") and r["features"].get(n) is not None]
  if vals:out[n]={"min":min(vals),"max":max(vals),"count":len(vals)}
 return out
def main():
 model=load_model();replay=replay_audit(model);env=environment_probe();training=json.loads(V2_DATA.read_text())["rows"];ranges=feature_ranges(training);model_artifact={"schema_version":"oom-risk-v2-design-1","model_name":"jaxoom_frozen_v2","model_version":"v2","model_hash":model["model_hash"],"dataset_hash":model["dataset_hash"],"feature_names":model["feature_names"],"means":model["means"],"scales":model["scales"],"coefficients":model["coefficients"],"intercept":model["intercept"],"regularization":model["regularization"],"primary_threshold":THRESHOLD,"validated_environment":{"device":"NVIDIA RTX 3050 Laptop GPU","driver":"566.07","jax":"0.11.0","jaxlib":"0.11.0","backend":"gpu/CUDA","allocator":"CUDA/BFC","preallocate":False,"budget_bytes":BUDGET},"validation":{"primary_heldout_likely_fit_n":157,"primary_false_safe_n":0,"one_sided_95_upper_bound":0.01890020551196819,"strict_sensitivity_likely_fit_n":126,"strict_sensitivity_upper_bound":0.023495238866670053},"status":"DESIGN_ONLY_NOT_PUBLIC"};MODEL_OUT.write_text(json.dumps(model_artifact,indent=2,sort_keys=True)+"\n")
 design={"status":"DESIGN_ONLY_NO_PUBLIC_API_CHANGE","selected_model":"V2","model_hash":model["model_hash"],"threshold":THRESHOLD,"budget_bytes":BUDGET,"calibration_upper_ratio":UPPER_RATIO,"feature_schema":list(FEATURES),"feature_sources":{"structural_peak_over_budget":"MemoryReport.estimated_peak_bytes / fixed validated 3 GiB budget; generic static feature","calibrated_upper_over_budget":"structural peak * frozen V1 static upper ratio / fixed validated budget; exact RTX 3050/JAX 0.11 risk calibration","largest_over_budget":"max MemoryReport.largest_buffers nbytes / fixed budget; generic static feature","top_two_peak_live_over_budget":"sum of two largest MemoryReport.peak.live_buffers / fixed budget; generic static feature","peak_live_over_budget":"same structural peak / fixed budget; generic static feature","dtype_bytes":"experiment workload metadata; not uniquely defined for arbitrary multi-dtype callables","config_numeric_count":"experiment workload configuration metadata; no canonical source in current public API","config_numeric_max":"experiment workload configuration metadata; no canonical source in current public API","config_numeric_log_product":"experiment workload configuration metadata; no canonical source in current public API"},"feature_ranges_from_v2_training":ranges,"proposed_api":{"name":"assess_oom_risk","public_status":"NOT_IMPLEMENTED","signature":"assess_oom_risk(fn, *args, workload_config=..., memory_limit=validated_3GiB)","result_statuses":["LIKELY_FIT","UNCERTAIN","UNSUPPORTED"],"rule":"applicable and p_oom <= 0.20 => LIKELY_FIT; applicable otherwise => UNCERTAIN; unknown/out-of-scope => UNSUPPORTED"},"applicability":{"required":env["required"],"feature_domain_guard":"future design only; inclusive ranges from frozen V2 training artifact, not outcome-tuned","unsupported_overrides_score":True},"replay_audit":replay,"environment_probe":env,"environment_applicable_on_current_runtime":environment_applicable(env),"readiness_gates":{"A_frozen_artifact_reproducibility":"PASS","B_feature_reproducibility_structured_adapter":"PASS" if replay["feature_mismatched_values"]==0 else "FAIL","B_arbitrary_callable_feature_reproducibility":"BLOCKED","C_logistic_inference_equivalence":"PASS" if replay["model_probability_mismatches"]==0 else "FAIL","D_strict_environment_guard":"PASS","E_no_hidden_compilation":"PASS","F_non_guarantee_semantics":"PASS","G_backward_compatibility":"PASS_BY_NO_PUBLIC_CHANGE"},"blocking_issues":["config_numeric_count/max/log_product require structured workload metadata absent from the current generic public callable API","dtype_bytes is not canonically defined for arbitrary multi-dtype callables","fixed 3 GiB validation budget cannot be silently replaced by dynamic device_budget","the existing calibrate() compiler-calibration registry is not the frozen OOM-risk calibration artifact"],"compile_risk":"KEEP_SEPARATE; the experiment-only compile filter is not a production gate","decision":"V2_FEATURE_PIPELINE_NOT_PRODUCTION_REPRODUCIBLE","production_decision":"NO PUBLIC PRODUCTION CHANGE"};OUT.write_text(json.dumps(design,indent=2,sort_keys=True)+"\n");print(json.dumps(design,indent=2,sort_keys=True))
if __name__=="__main__":main()
