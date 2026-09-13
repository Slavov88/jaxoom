"""Static V3 candidate pool and compile-feasibility filter.

Selection is frozen before V3 runtime outcomes. All candidate analysis uses
abstract JAX tracing; compile-feasibility is an experiment-only campaign gate.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, platform
from collections import Counter
from pathlib import Path
from typing import Any
import jax
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parent
V1_DATA=ROOT/"oom_risk_dataset_v1_2026-09-12.json"; V2_DATA=ROOT/"oom_risk_dataset_v2_2026-09-12.json"
V1_FREEZE=ROOT/"oom_risk_model_v1_frozen_2026-09-12.json"; V2_MODEL=ROOT/"oom_risk_model_v2_2026-09-12.json"

def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
E=load_module("oom_risk_expansion_v2",ROOT/"oom_risk_expansion_v2.py")

def dump(v): return json.dumps(v,sort_keys=True,separators=(",",":"),default=lambda x:x.item() if hasattr(x,"item") else str(x))
def sha(v): return hashlib.sha256(dump(v).encode()).hexdigest()

def model_score(features,model):
 vals=np.array([float(features.get(n,0) or 0) for n in model["feature_names"]]); z=float(model["intercept"]+np.dot(np.asarray(model["coefficients"][1:]),(vals-np.asarray(model["means"]))/np.asarray(model["scales"])))
 return float(1/(1+math.exp(-np.clip(z,-40,40))))

def complexity(family,config,dtype):
 fn,args=E.candidate_functions(family,config,dtype); closed=jax.make_jaxpr(fn)(*args).jaxpr
 primitives=Counter(); nested=0; eq_count=0; scan_count=0; while_count=0
 def visit(jaxpr):
  nonlocal nested,eq_count,scan_count,while_count
  for eqn in jaxpr.eqns:
   eq_count+=1; name=str(eqn.primitive); primitives[name]+=1
   if name=="scan": scan_count+=1
   if name in {"while","while_loop"}: while_count+=1
   for value in eqn.params.values():
    if hasattr(value,"jaxpr"):
     nested+=1; visit(value.jaxpr)
    elif isinstance(value,(tuple,list)):
     for item in value:
      if hasattr(item,"jaxpr"): nested+=1; visit(item.jaxpr)
 visit(closed)
 def nbytes(value):
  return int(np.prod(value.shape))*np.dtype(value.dtype).itemsize if hasattr(value,"shape") and hasattr(value,"dtype") else 0
 out=jax.eval_shape(fn,*args); output_bytes=sum(nbytes(x) for x in jax.tree_util.tree_leaves(out)); input_bytes=sum(nbytes(x) for x in args)
 return {"equation_count":eq_count,"nested_jaxpr_count":nested,"scan_count":scan_count,"while_count":while_count,"primitive_counts":dict(primitives),"input_bytes":input_bytes,"output_bytes":output_bytes,"max_input_numel":max((int(np.prod(x.shape)) for x in args),default=0)}

COMPILE_FEATURES=["structural_peak_over_budget","calibrated_upper_over_budget","largest_over_budget","top_two_peak_live_over_budget","peak_live_over_budget","dtype_bytes","config_numeric_count","config_numeric_max","config_numeric_log_product"]
FAMILIES=["attention","convolution","mlp","training","autodiff","transformer","scan","matmul","reduction"]

def feature_vector(row): return np.array([float(row["features"].get(n) or 0) for n in COMPILE_FEATURES],float)
def fit_logistic(x,y,l2=1):
 def fun(b):
  z=np.clip(x@b,-40,40); return np.mean(np.logaddexp(0,z)-y*z)+l2*np.sum(b[1:]**2)/2
 def jac(b):
  p=1/(1+np.exp(-np.clip(x@b,-40,40))); return x.T@(p-y)/len(y)+l2*np.r_[0,b[1:]]
 return minimize(fun,np.zeros(x.shape[1]),jac=jac,method="L-BFGS-B").x

def compile_shape_proxy(row):
 c=row.get("configuration") or row.get("config") or {}; family=row.get("family") or "other"
 def product(keys):
  v=1
  for k in keys: v*=int(c.get(k,1) or 1)
  return v
 if family=="attention": return {"max_input_numel":product(("batch","sequence","heads","head_dim")),"workspace_numel":product(("batch","heads","sequence","sequence"))}
 if family=="convolution": return {"max_input_numel":product(("batch","height","width","channels")),"workspace_numel":product(("batch","height","width","out_channels"))}
 if family in {"reduction","scan"}: return {"max_input_numel":product(("batch","length","width")),"workspace_numel":0}
 if family=="transformer": return {"max_input_numel":product(("sequence","width")),"workspace_numel":product(("sequence","sequence","heads"))}
 if family=="matmul": return {"max_input_numel":max(product(("m","k")),product(("k","n"))),"workspace_numel":product(("m","n"))}
 if family in {"mlp","training","autodiff"}: return {"max_input_numel":max(product(("batch","width")),product(("width","width"))),"workspace_numel":product(("batch","width"))}
 return {"max_input_numel":0,"workspace_numel":0}


def compile_rule(proxy,family):
 # Conservative campaign-only rule; not a production prediction model.
 return int(family=="other" or proxy["max_input_numel"]>=300_000_000 or proxy["workspace_numel"]>=300_000_000 or (family=="convolution" and proxy["workspace_numel"]>=100_000_000))


def compile_filter_audit(rows):
 use=[r for r in rows if r["outcome_class"] in {"FIT","EXECUTION_OOM","COMPILE_OOM","COMPILE_TIMEOUT"}]
 x=np.c_[np.ones(len(use)),np.array([feature_vector(r) for r in use])]; y=np.array([int(r["outcome_class"] in {"COMPILE_OOM","COMPILE_TIMEOUT"}) for r in use]); b=fit_logistic(x,y)
 rule=np.array([compile_rule(compile_shape_proxy(r),r["family"]) for r in use]); failures=y==1
 recall=float(np.sum(rule&failures)/max(np.sum(failures),1)); retention=float(np.sum((~rule)&(~failures))/max(np.sum(~failures),1))
 return {"feature_names":COMPILE_FEATURES,"logistic_coefficients":b.tolist(),"selected_method":"static_shape_rule","threshold_description":"max_input_numel or workspace_numel >= 300,000,000; convolution workspace >= 100,000,000","historical_grouped_audit":{"N":len(use),"groups":len({r["workload_group_id"] for r in use}),"compile_failures":int(y.sum()),"compile_failure_recall_at_rule":recall,"compile_success_retention_at_rule":retention,"rule_rejected_rows":int(rule.sum())},"model_hash":sha({"features":COMPILE_FEATURES,"rule":"shape_v1"})}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--pool-size",type=int,default=4000); ap.add_argument("--selected-count",type=int,default=60); ap.add_argument("--seed",type=int,default=20260913); ap.add_argument("--pool-output",type=Path,default=ROOT/"oom_risk_candidate_pool_v3_2026-09-13.json"); ap.add_argument("--plan-output",type=Path,default=ROOT/"oom_risk_expansion_plan_v3_2026-09-13.json"); ap.add_argument("--filter-output",type=Path,default=ROOT/"compile_feasibility_filter_v1_2026-09-13.json"); args=ap.parse_args()
 v1=json.loads(V1_FREEZE.read_text()); v2=json.loads(V2_MODEL.read_text()); old=json.loads(V1_DATA.read_text())["rows"]+json.loads(V2_DATA.read_text())["rows"]; existing={r["workload_group_id"] for r in old}; filter_model=compile_filter_audit(json.loads(V1_DATA.read_text())["rows"])
 pool=[]; seen=set(existing)
 for spec in E.candidate_specs(args.pool_size,args.seed):
  group=E.group_id(spec["family"],spec["configuration"],spec["dtype"])
  if group in seen: continue
  try:
   features=E.static_features(spec["family"],spec["configuration"],spec["dtype"],v1)
   comp=complexity(spec["family"],spec["configuration"],spec["dtype"]); features["equation_count"]=comp["equation_count"]; features["nested_jaxpr_count"]=comp["nested_jaxpr_count"]; features["scan_count"]=comp["scan_count"]; features["while_count"]=comp["while_count"]; features["input_bytes"]=comp["input_bytes"]; features["output_bytes"]=comp["output_bytes"]; features["max_input_numel"]=comp["max_input_numel"]
   p1=model_score(features,v1["model"]); p2=model_score(features,v2); proxy=compile_shape_proxy({"family":spec["family"],"configuration":spec["configuration"]}); pc=float(compile_rule(proxy,spec["family"]))
   pool.append({**spec,"status":"STATIC_OK","workload_group_id":group,"graph_fingerprint":E.graph_fingerprint(spec["family"],spec["configuration"],spec["dtype"]),"shape_summary":spec["configuration"],"v1_predicted_probability":p1,"v2_predicted_probability":p2,"aggregate_score":features["calibrated_upper_over_budget"],"compile_feasibility_score":pc,"compile_filter_reject":bool(pc),"v1_features":features,"compile_features":comp,"compile_shape_proxy":proxy}); seen.add(group)
  except Exception as exc: pool.append({**spec,"status":"STATIC_ERROR","error":f"{type(exc).__name__}: {exc}"})
 # The filter removes obvious compile-risk cases. A small, explicit audit
 # slice from the rejected high-risk tail is retained to measure prospective
 # filter recall without allowing the runtime set to become trivially easy.
 retained=[r for r in pool if r.get("status")=="STATIC_OK" and not r["compile_filter_reject"]]; rejected=[r for r in pool if r.get("status")=="STATIC_OK" and r["compile_filter_reject"]]
 rng=np.random.default_rng(args.seed+1)
 for r in retained:
  p=r["v2_predicted_probability"]; a=r["aggregate_score"]; r["selection_coarse_bucket"]="LOW_RISK_CHALLENGER" if p<=.30 and a>=.30 else "MODERATE_BOUNDARY"; r["selection_category"]="MODEL_DISAGREEMENT" if ((p>=.5)!=(a>1)) else r["selection_coarse_bucket"]; r["selection_reason"]="V2 low-risk challenger with nontrivial aggregate demand" if r["selection_category"]=="LOW_RISK_CHALLENGER" else ("V1/V2 logistic versus aggregate disagreement" if r["selection_category"]=="MODEL_DISAGREEMENT" else "V2 moderate-risk boundary candidate")
 for r in rejected:
  r["selection_coarse_bucket"]="HIGH_RISK_COMPILE_AUDIT"; r["selection_category"]="HIGH_RISK_COMPILE_AUDIT"; r["selection_reason"]="small audit slice retained despite compile filter rejection"
 selected=[]; family=Counter(); targets={"LOW_RISK_CHALLENGER":20,"MODERATE_BOUNDARY":34,"HIGH_RISK_COMPILE_AUDIT":6}
 for cat in targets:
  source=retained if cat!="HIGH_RISK_COMPILE_AUDIT" else rejected; choices=[r for r in source if r["selection_coarse_bucket"]==cat]; rng.shuffle(choices); choices.sort(key=lambda r:abs(r["v2_predicted_probability"]-(.15 if cat=="LOW_RISK_CHALLENGER" else .35 if cat=="MODERATE_BOUNDARY" else .8)))
  for r in choices:
   if len(selected)>=args.selected_count or sum(z["selection_coarse_bucket"]==cat for z in selected)>=targets[cat]: break
   if family[r["family"]]>=max(1,args.selected_count//3): continue
   selected.append(r); family[r["family"]]+=1
 for r in sorted(retained,key=lambda z:z["compile_feasibility_score"]):
  if len(selected)>=args.selected_count: break
  if r not in selected and family[r["family"]]<args.selected_count//3: selected.append(r); family[r["family"]]+=1
 # Preserve meaningful representation for every generated family when possible.
 for fam in sorted({r["family"] for r in retained}):
  while family[fam]<2:
   replacement=next((r for r in retained if r not in selected and r["family"]==fam),None)
   donor=next((r for r in reversed(selected) if family[r["family"]]>2 and r["selection_coarse_bucket"]==replacement["selection_coarse_bucket"]),None) if replacement else None
   if donor is None: donor=next((r for r in reversed(selected) if family[r["family"]]>2),None) if replacement else None
   if replacement is None or donor is None: break
   selected.remove(donor); family[donor["family"]]-=1; selected.append(replacement); family[fam]+=1
 for r in selected: r["selected_for_runtime"]=True
 plan={"status":"FROZEN_BEFORE_RUNTIME","selection_date":"2026-09-13","seed":args.seed,"v1_model_hash":v1["model_hash"],"v2_model_hash":v2["model_hash"],"compile_filter_hash":filter_model["model_hash"],"candidate_pool_size":len(pool),"static_error_count":sum(r.get("status")!="STATIC_OK" for r in pool),"filter_rejected_count":len(rejected),"filter_retained_count":len(retained),"selected_count":len(selected),"selection_distribution":dict(Counter(r["selection_coarse_bucket"] for r in selected)),"selection_category_distribution":dict(Counter(r["selection_category"] for r in selected)),"family_distribution":dict(family),"selected_candidates":[{k:r[k] for k in ("candidate_id","workload_group_id","family","configuration","dtype","graph_fingerprint","shape_summary","v1_predicted_probability","v2_predicted_probability","aggregate_score","compile_feasibility_score","selection_coarse_bucket","selection_category","selection_reason")} for r in selected],"outcomes_excluded":True,"capacity_sweeps":False,"filter_is_experiment_only":True}
 args.pool_output.write_text(json.dumps({"status":"STATIC_CANDIDATE_POOL","seed":args.seed,"v1_model_hash":v1["model_hash"],"v2_model_hash":v2["model_hash"],"compile_filter":filter_model,"candidates":pool},indent=2,sort_keys=True)+"\n"); args.plan_output.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n"); args.filter_output.write_text(json.dumps(filter_model,indent=2,sort_keys=True)+"\n"); print(json.dumps(plan,indent=2,sort_keys=True))
if __name__=="__main__": main()
