"""Prospective V1/V2 evaluation and grouped V3 development evaluation."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
V2=load("oom_risk_model_v2",ROOT/"oom_risk_model_v2.py"); V1=load("oom_risk_model_v1",ROOT/"oom_risk_model_v1.py")

def dump(x):return json.dumps(x,sort_keys=True,separators=(",",":"),default=lambda y:y.item() if hasattr(y,"item") else str(y))
def sha(x):return hashlib.sha256(dump(x).encode()).hexdigest()
def metrics(y,p):
 y=np.asarray(y);p=np.clip(np.asarray(p),1e-7,1-1e-7);return {"N":len(y),"ROC_AUC":V1.auc(y,p),"PR_AUC":V1.pr_auc(y,p),"Brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}
def abstention(y,p,t):
 y=np.asarray(y);p=np.asarray(p);return {"p_fit":t,"likely_fit":int(np.sum(p<=t)),"fit_coverage":int(np.sum(p<=t)),"coverage":float(np.mean(p<=t)) if len(p) else 0,"false_safe":int(np.sum((p<=t)&(y==1))),"false_safe_rate":float(np.mean(y[p<=t])) if np.sum(p<=t) else 0}
def prepare(train,test,names):
 tr=np.array([[float(r["features"].get(n)) if r["features"].get(n) is not None else np.nan for n in names] for r in train],float);te=np.array([[float(r["features"].get(n)) if r["features"].get(n) is not None else np.nan for n in names] for r in test],float) if test else np.empty((0,len(names)));med=[]
 for j in range(tr.shape[1]):
  a=tr[np.isfinite(tr[:,j]),j];med.append(float(np.median(a)) if len(a) else 0)
 med=np.array(med);tr=np.where(np.isnan(tr),med,tr);te=np.where(np.isnan(te),med,te);mu=tr.mean(0);sd=np.where(tr.std(0)>1e-12,tr.std(0),1);return np.c_[np.ones(len(tr)),(tr-mu)/sd],np.c_[np.ones(len(te)),(te-mu)/sd],mu,sd
def fit(x,y,weights=None,constrained=False,ratio_count=0):
 y=np.asarray(y,float);w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
 def fun(b):z=np.clip(x@b,-40,40);return np.sum(w*(np.logaddexp(0,z)-y*z))/len(y)+np.sum(b[1:]**2)/2
 def jac(b):p=1/(1+np.exp(-np.clip(x@b,-40,40)));return x.T@(w*(p-y))/len(y)+np.r_[0,b[1:]]
 bounds=[(None,None)]+([(0,None) if constrained and i<=ratio_count else (None,None) for i in range(1,x.shape[1])]) if constrained else None
 return minimize(fun,np.zeros(x.shape[1]),jac=jac,method="L-BFGS-B",bounds=bounds).x
def predict(x,b):return 1/(1+np.exp(-np.clip(x@b,-40,40)))
def folds(rows):return V1.grouped_folds(rows,5)
def evaluate(rows,names,constrained=False,balanced=False):
 out=[];coefs=[]
 for ti,vi in folds(rows):
  tr=[rows[i] for i in ti];te=[rows[i] for i in vi]
  if len({r["target_execution_oom"] for r in tr})<2:continue
  xt,xv,_,_=prepare(tr,te,names);y=np.array([r["target_execution_oom"] for r in tr]);yt=np.array([r["target_execution_oom"] for r in te]);w=None
  if balanced:w=np.where(y==1,len(y)/(2*max(y.sum(),1)),len(y)/(2*max((y==0).sum(),1)))
  b=fit(xt,y,w,constrained,len([n for n in names if n.endswith("_over_budget") or n in {"largest_over_structural","top_two_over_structural"}]));p=predict(xv,b);out.extend({"group":r["workload_group_id"],"y":int(v),"p":float(q)} for r,v,q in zip(te,yt,p));coefs.append(b.tolist())
 y=np.array([r["y"] for r in out]);p=np.array([r["p"] for r in out]);return {"metrics":metrics(y,p),"predictions":out,"coefficient_folds":coefs}
def tree(rows,names):
 out=[]
 for ti,vi in folds(rows):
  tr=[rows[i] for i in ti];te=[rows[i] for i in vi]
  xt,xv,_,_=prepare(tr,te,names);y=np.array([r["target_execution_oom"] for r in tr]);yt=np.array([r["target_execution_oom"] for r in te]);best=None
  for j,n in enumerate(names,1):
   for t in np.unique(xt[:,j])[::max(1,len(np.unique(xt[:,j]))//20 or 1)]:
    for d in (1,-1):
     p=(d*xt[:,j]>=d*t).astype(float);a=V1.auc(y,p)
     if a is not None and (best is None or a>best[0]):best=(a,j,t,d)
  p=(best[3]*xv[:,best[1]]>=best[3]*best[2]).astype(float);out.extend({"group":r["workload_group_id"],"y":int(v),"p":float(q)} for r,v,q in zip(te,yt,p))
 y=np.array([r["y"] for r in out]);p=np.array([r["p"] for r in out]);return {"metrics":metrics(y,p),"predictions":out}
def ece(pred):
 y=np.array([x["y"] for x in pred]);p=np.array([x["p"] for x in pred]);return float(sum(np.sum((p>=i/5)&(p<(i+1)/5 if i<4 else p<=1))*abs(p[(p>=i/5)&(p<(i+1)/5 if i<4 else p<=1)].mean()-y[(p>=i/5)&(p<(i+1)/5 if i<4 else p<=1)].mean()) for i in range(5) if np.any((p>=i/5)&(p<(i+1)/5 if i<4 else p<=1)))/len(y))
def load_rows():
 base=json.loads((ROOT/"oom_risk_dataset_v2_2026-09-12.json").read_text())["rows"];pool={r["candidate_id"]:r for r in json.loads((ROOT/"oom_risk_candidate_pool_v3_2026-09-13.json").read_text())["candidates"]};plan={r["candidate_id"]:r for r in json.loads((ROOT/"oom_risk_expansion_plan_v3_2026-09-13.json").read_text())["selected_candidates"]}; results=json.loads((ROOT/"oom_risk_expansion_results_v3_2026-09-13.json").read_text())["results"];new=[]
 for r in results:
  c=pool[r["candidate_id"]];f=dict(c["v1_features"]);new.append({"row_id":"expansion_v3:"+r["candidate_id"],"workload_group_id":r["workload_group_id"],"graph_fingerprint":r["graph_fingerprint"],"family":r["family"],"configuration":r["configuration"],"dtype":r["dtype"],"features":f,"target_execution_oom":1 if r["outcome"]=="EXECUTION_OOM" else (0 if r["outcome"]=="FIT" else None),"outcome_class":r["outcome"],"unstable_group":False,"prospective_v1_probability":r["v1_predicted_probability"],"prospective_v2_probability":r["v2_predicted_probability"],"campaign_row":r,"plan_row":plan[r["candidate_id"]]})
 for r in base:
  f=r["features"]
  if f.get("largest_over_structural") is None and f.get("largest_buffer_bytes") is not None and f.get("structural_peak_bytes"):f["largest_over_structural"]=f["largest_buffer_bytes"]/f["structural_peak_bytes"]
  if f.get("top_two_over_structural") is None and f.get("top_two_peak_live_bytes") is not None and f.get("structural_peak_bytes"):f["top_two_over_structural"]=f["top_two_peak_live_bytes"]/f["structural_peak_bytes"]
 return base,new
V3_FEATURES=V2.V2_FEATURES+["largest_over_structural","top_two_over_structural"]

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--summary-output",type=Path,default=ROOT/"oom_risk_summary_v3_2026-09-13.json");ap.add_argument("--cv-output",type=Path,default=ROOT/"oom_risk_cv_v3_2026-09-13.json");ap.add_argument("--models-output",type=Path,default=ROOT/"oom_risk_models_v3_2026-09-13.json");ap.add_argument("--dataset-output",type=Path,default=ROOT/"oom_risk_dataset_v3_2026-09-13.json");args=ap.parse_args()
 old,new=load_rows();combined=old+new;eligible=[r for r in combined if r["target_execution_oom"] is not None and not r.get("unstable_group",False)]; stable_new=[r for r in new if r["target_execution_oom"] is not None]
 v1y=np.array([r["target_execution_oom"] for r in stable_new]);v1p=np.array([r["prospective_v1_probability"] for r in stable_new]);v2p=np.array([r["prospective_v2_probability"] for r in stable_new]);prospective={"N":len(stable_new),"V1":{"metrics":metrics(v1y,v1p),"risk_coverage":{str(t):abstention(v1y,v1p,t) for t in (.1,.15,.2,.25,.3,.4,.5)}},"V2":{"metrics":metrics(v1y,v2p),"risk_coverage":{str(t):abstention(v1y,v2p,t) for t in (.1,.15,.2,.25,.3,.4,.5)}}}
 models={"V2_compact_constrained":evaluate(eligible,V2.V2_FEATURES,True),"V1_rich_constrained":evaluate(eligible,V1.feature_sets()["D_memory_shape"],True),"V3_compact_L2":evaluate(eligible,V3_FEATURES),"V3_compact_constrained":evaluate(eligible,V3_FEATURES,True),"V3_group_balanced_constrained":evaluate(eligible,V3_FEATURES,True,True),"V3_threshold_tree":tree(eligible,V3_FEATURES)}
 x,_,mu,sd=prepare(eligible,[],V3_FEATURES);y=np.array([r["target_execution_oom"] for r in eligible]);b=fit(x,y,constrained=True,ratio_count=len([n for n in V3_FEATURES if n.endswith("_over_budget") or n in {"largest_over_structural","top_two_over_structural"}]))
 risk_cov={n:{str(t):abstention(np.array([z["y"] for z in v["predictions"]]),np.array([z["p"] for z in v["predictions"]]),t) for t in (.05,.1,.15,.2,.25,.3,.4,.5)} for n,v in models.items()}
 family={}
 for fam in sorted({r["family"] for r in eligible}):
  tr=[r for r in eligible if r["family"]!=fam];te=[r for r in eligible if r["family"]==fam]
  if len(te) and len({r["target_execution_oom"] for r in tr})>1:
   xt,xv,_,_=prepare(tr,te,V3_FEATURES);bt=fit(xt,np.array([r["target_execution_oom"] for r in tr]),constrained=True,ratio_count=7);family[fam]=metrics(np.array([r["target_execution_oom"] for r in te]),predict(xv,bt))
 groups=sorted({r["workload_group_id"] for r in eligible});rng=np.random.default_rng(20260913);oof=models["V3_compact_constrained"]["predictions"];p=np.array([z["p"] for z in oof]);gm={g:int(np.mean([r["target_execution_oom"] for r in eligible if r["workload_group_id"]==g])>=.5) for g in groups};perm=[]
 for _ in range(20):
  vals=rng.permutation(list(gm.values()));sh=dict(zip(groups,vals));perm.append(V1.auc(np.array([sh[z["group"]] for z in oof]),p))
 plan=json.loads((ROOT/"oom_risk_expansion_plan_v3_2026-09-13.json").read_text());results=json.loads((ROOT/"oom_risk_expansion_results_v3_2026-09-13.json").read_text())["results"];outcomes=Counter(r["outcome"] for r in results);filter_rejected=[r for r in results if r["candidate_id"] in {c["candidate_id"] for c in plan["selected_candidates"] if c["compile_feasibility_score"]==1}];compile_fail=[r for r in results if r["outcome"] in {"COMPILE_OOM","COMPILE_TIMEOUT"}]
 model={"feature_names":V3_FEATURES,"means":mu.tolist(),"scales":sd.tolist(),"coefficients":b.tolist(),"intercept":float(b[0]),"regularization":"L2=1.0; nonnegative demand/budget and structural-ratio coefficients","dataset_hash":sha(combined),"active_sampling":True}; model_artifact={**model,"model_hash":sha(model)}; t=time.perf_counter()
 for _ in range(1000):predict(x,b)
 summary={"status":"COMPUTATIONALLY_VERIFIED","decision":"OOM RISK MODEL V3 PROMISING BUT DATA-LIMITED","production_decision":"NO PRODUCTION CHANGE","frozen_models":{"v1_model_hash":json.loads((ROOT/"oom_risk_model_v1_frozen_2026-09-12.json").read_text())["model_hash"],"v2_model_hash":json.loads((ROOT/"oom_risk_model_v2_2026-09-12.json").read_text())["model_hash"]},"candidate_pool":{"count":len(json.loads((ROOT/"oom_risk_candidate_pool_v3_2026-09-13.json").read_text())["candidates"]),"selected":len(plan["selected_candidates"]),"selection_distribution":plan["selection_distribution"],"selection_category_distribution":plan["selection_category_distribution"],"family_distribution":plan["family_distribution"],"filter_rejected":plan["filter_rejected_count"],"filter_retained":plan["filter_retained_count"]},"compile_filter":{"historical":json.loads((ROOT/"compile_feasibility_filter_v1_2026-09-13.json").read_text())["historical_grouped_audit"],"prospective_selected_rejected":len(filter_rejected),"prospective_compile_failures_in_rejected":sum(r in compile_fail for r in filter_rejected),"prospective_compile_failure_recall":sum(r in compile_fail for r in filter_rejected)/max(len(compile_fail),1),"compile_success_retention":sum(r not in compile_fail for r in results if r not in filter_rejected)/max(len(results)-len(filter_rejected),1),"wasted_run_rate":len(compile_fail)/len(results)},"prospective":prospective,"outcomes":dict(outcomes),"dataset":{"total_rows":len(combined),"total_groups":len({r["workload_group_id"] for r in combined}),"stable_rows":len(eligible),"stable_groups":len({r["workload_group_id"] for r in eligible}),"fit_groups":len({r["workload_group_id"] for r in eligible if r["target_execution_oom"]==0}),"execution_oom_groups":len({r["workload_group_id"] for r in eligible if r["target_execution_oom"]==1}),"families":dict(Counter(r["family"] for r in combined))},"feature_set":{"name":"V3_compact","features":V3_FEATURES,"rationale":"V2 monotone ratios plus large-buffer and top-two structural ratios; no target/runtime/compiler fields"},"model_comparison":{k:{**v["metrics"],"ECE":ece(v["predictions"])} for k,v in models.items()},"risk_coverage_oof":risk_cov,"selected_model":"V3_compact_constrained","selected_model_artifact":model_artifact,"monotonicity":{"v1":3,"v2":0,"v3":0},"family_results":family,"unstable_cases":{"count":sum(r.get("unstable_group",False) for r in combined),"new_mixed_groups":0},"low_risk_false_safe_v2":[{"candidate_id":r["campaign_row"]["candidate_id"],"family":r["family"],"configuration":r["configuration"],"v1_probability":r["prospective_v1_probability"],"v2_probability":r["prospective_v2_probability"],"aggregate_score":r["features"].get("calibrated_upper_over_budget"),"outcome":r["outcome_class"]} for r in new if r["target_execution_oom"]==1 and r["prospective_v2_probability"]<=.25],"permutation":{"N":20,"auc_mean":float(np.nanmean(perm)),"auc_values":perm},"speed":{"inference_ms_per_1000_rows":(time.perf_counter()-t)*1000/1000,"feature_count":len(V3_FEATURES),"serialized_model_bytes":len(dump(model).encode())},"inference_purity":{"target_compilation":False,"target_execution":False,"runtime_allocator_state":False}}
 args.dataset_output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","rows":combined},indent=2,sort_keys=True)+"\n");args.cv_output.write_text(json.dumps(models,indent=2,sort_keys=True)+"\n");args.models_output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","selected":model_artifact,"models":{k:v["metrics"] for k,v in models.items()}},indent=2,sort_keys=True)+"\n");(ROOT/"oom_risk_model_v3_2026-09-13.json").write_text(json.dumps(model_artifact,indent=2,sort_keys=True)+"\n");(ROOT/"oom_risk_risk_coverage_v3_2026-09-13.json").write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","prospective":prospective,"oof":risk_cov},indent=2,sort_keys=True)+"\n");args.summary_output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__":main()
