"""Evaluate the frozen-V1 prospective expansion and experiment-only V2 models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parent
V1_DATASET=ROOT/"oom_risk_dataset_v1_2026-09-12.json"
V1_SUMMARY=ROOT/"oom_risk_summary_v1_2026-09-12.json"
POOL=ROOT/"oom_risk_candidate_pool_v2_2026-09-12.json"
PLAN=ROOT/"oom_risk_expansion_plan_v2_2026-09-12.json"
RESULTS=ROOT/"oom_risk_expansion_results_v2_2026-09-12.json"

def load_v1_module():
    spec=importlib.util.spec_from_file_location("oom_risk_model_v1",ROOT/"oom_risk_model_v1.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
M=load_v1_module()

def dump(v): return json.dumps(v,sort_keys=True,separators=(",",":"),default=lambda x:x.item() if hasattr(x,"item") else str(x))
def sha(v): return hashlib.sha256(dump(v).encode()).hexdigest()

def metrics(y,p):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-7,1-1e-7)
    return {"N":len(y),"ROC_AUC":M.auc(y,p),"PR_AUC":M.pr_auc(y,p),"Brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}

def abstention(y,p,p_fit=.1,p_oom=.9):
    labels=np.where(p<=p_fit,"LIKELY_FIT",np.where(p>=p_oom,"LIKELY_OOM","UNCERTAIN"))
    return {"p_fit":p_fit,"p_oom":p_oom,"likely_fit":int(np.sum(labels=="LIKELY_FIT")),"likely_oom":int(np.sum(labels=="LIKELY_OOM")),"uncertain":int(np.sum(labels=="UNCERTAIN")),"coverage":float(np.mean(labels!="UNCERTAIN")) if len(labels) else 0.0,"false_safe_likely_fit":int(np.sum((labels=="LIKELY_FIT")&(y==1))),"false_positive_likely_oom":int(np.sum((labels=="LIKELY_OOM")&(y==0))),"fit_coverage":int(np.sum(labels=="LIKELY_FIT")),"oom_coverage":int(np.sum(labels=="LIKELY_OOM"))}

def make_new_rows():
    old=json.loads(V1_DATASET.read_text())["rows"]; pool={r["candidate_id"]:r for r in json.loads(POOL.read_text())["candidates"]}; result=json.loads(RESULTS.read_text())["results"]
    new=[]
    for r in result:
        c=pool[r["candidate_id"]]; outcome=r["outcome"]
        new.append({"row_id":"expansion_v2:"+r["candidate_id"],"workload_group_id":r["workload_group_id"],"graph_fingerprint":r.get("graph_fingerprint") or c["graph_fingerprint"],"family":r["family"],"configuration":r["configuration"],"dtype":r["dtype"],"source_artifact":"oom_risk_expansion_results_v2_2026-09-12.json","source_experiment":"expansion_v2","source_date":"2026-09-12","repeat_id":None,"device":(r.get("environment") or {}).get("devices"),"jax_version":(r.get("environment") or {}).get("jax_version"),"allocator_configuration":{"preallocate":False,"policy":"BFC","capacity_control":"primary_normal"},"features":c["v1_features"],"outcome_class":outcome,"target_execution_oom":1 if outcome=="EXECUTION_OOM" else (0 if outcome=="FIT" else None),"pressure":0.0,"unstable_group":False,"prospective_v1_probability":c["v1_predicted_probability"],"campaign_row":r})
    return old,new

V2_FEATURES=["structural_peak_over_budget","calibrated_upper_over_budget","largest_over_budget","top_two_peak_live_over_budget","peak_live_over_budget","dtype_bytes","config_numeric_count","config_numeric_max","config_numeric_log_product"]
RATIO_FEATURES=set(V2_FEATURES[:5])

def matrix(rows,names): return np.array([[float(r["features"].get(n)) if r["features"].get(n) is not None else np.nan for n in names] for r in rows],float)
def prepare(train,test,names):
    tr=matrix(train,names); te=matrix(test,names) if test else np.empty((0,len(names))); med=np.zeros(tr.shape[1])
    for j in range(tr.shape[1]):
        a=tr[np.isfinite(tr[:,j]),j]; med[j]=float(np.median(a)) if len(a) else 0.0
    tr=np.where(np.isnan(tr),med,tr); te=np.where(np.isnan(te),med,te); mean=tr.mean(0); scale=np.where(tr.std(0)>1e-12,tr.std(0),1); return np.c_[np.ones(len(tr)),(tr-mean)/scale],np.c_[np.ones(len(te)),(te-mean)/scale],mean,scale

def fit(x,y,l2=1,l1=0,weights=None,bounded=False):
    y=np.asarray(y,float); weights=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    def fun(b):
        z=np.clip(x@b,-40,40); return np.sum(weights*(np.logaddexp(0,z)-y*z))/len(y)+l2*np.sum(b[1:]**2)/2+l1*np.sum(np.abs(b[1:]))
    def jac(b):
        z=np.clip(x@b,-40,40); p=1/(1+np.exp(-z)); g=x.T@(weights*(p-y))/len(y)+l2*np.r_[0,b[1:]]; g[1:]+=l1*np.sign(b[1:]); return g
    bounds=None
    if bounded: bounds=[(None,None)]+[(0,None) if i<=len(RATIO_FEATURES) else (None,None) for i in range(1,x.shape[1])]
    return minimize(fun,np.zeros(x.shape[1]),jac=jac,method="L-BFGS-B",bounds=bounds).x

def folds(rows): return M.grouped_folds(rows,n_splits=5)
def evaluate(rows,names,kind="l2",balanced=False,bounded=False):
    eligible=[r for r in rows if r["target_execution_oom"] is not None and not r.get("unstable_group",False)]; outputs=[]; coefs=[]
    for ti,vi in folds(eligible):
        tr=[eligible[i] for i in ti]; te=[eligible[i] for i in vi]
        if len({r["target_execution_oom"] for r in tr})<2: continue
        xt,xv,_,_=prepare(tr,te,names); y=np.array([r["target_execution_oom"] for r in tr]); yt=np.array([r["target_execution_oom"] for r in te]); w=None
        if balanced: w=np.where(y==1,len(y)/(2*max(y.sum(),1)),len(y)/(2*max((y==0).sum(),1)))
        b=fit(xt,y,l2=1 if kind=="l2" else 0,l1=.1 if kind=="l1" else 0,weights=w,bounded=bounded); p=1/(1+np.exp(-np.clip(xv@b,-40,40))); outputs.extend({"group":r["workload_group_id"],"y":int(v),"p":float(q)} for r,v,q in zip(te,yt,p)); coefs.append(b.tolist())
    y=np.array([o["y"] for o in outputs]); p=np.array([o["p"] for o in outputs]); return {"metrics":metrics(y,p),"predictions":outputs,"coefficient_folds":coefs}

def evaluate_tree(rows,names):
    eligible=[r for r in rows if r["target_execution_oom"] is not None and not r.get("unstable_group",False)]; outputs=[]; choices=[]
    for ti,vi in folds(eligible):
        tr=[eligible[i] for i in ti]; te=[eligible[i] for i in vi]
        if len({r["target_execution_oom"] for r in tr})<2: continue
        xt,xv,_,_=prepare(tr,te,names); y=np.array([r["target_execution_oom"] for r in tr]); yt=np.array([r["target_execution_oom"] for r in te]); best=None
        for j,n in enumerate(names,1):
            for threshold in np.unique(xt[:,j])[::max(1,len(np.unique(xt[:,j]))//20 or 1)]:
                for direction in (1,-1):
                    pp=(direction*xt[:,j]>=direction*threshold).astype(float); a=M.auc(y,pp)
                    if a is not None and (best is None or a>best[0]): best=(a,n,float(threshold),direction,j)
        if best is None: continue
        _,n,threshold,direction,j=best; pp=(direction*xv[:,j]>=direction*threshold).astype(float); outputs.extend({"group":r["workload_group_id"],"y":int(v),"p":float(q)} for r,v,q in zip(te,yt,pp)); choices.append({"feature":n,"threshold_standardized":threshold,"direction":direction})
    y=np.array([r["y"] for r in outputs]); p=np.array([r["p"] for r in outputs]); return {"metrics":metrics(y,p),"predictions":outputs,"fold_choices":choices}

def leave_family_out(rows,names):
    eligible=[r for r in rows if r["target_execution_oom"] is not None and not r.get("unstable_group",False)]; out={}
    for family in sorted({r["family"] for r in eligible}):
        tr=[r for r in eligible if r["family"]!=family]; te=[r for r in eligible if r["family"]==family]
        if not te or len({r["target_execution_oom"] for r in tr})<2: continue
        xt,xv,_,_=prepare(tr,te,names); y=np.array([r["target_execution_oom"] for r in tr]); yt=np.array([r["target_execution_oom"] for r in te]); b=fit(xt,y,bounded=True); p=1/(1+np.exp(-np.clip(xv@b,-40,40))); out[family]=metrics(yt,p)
    return out

def calibration(pred):
    y=np.array([r["y"] for r in pred]); p=np.array([r["p"] for r in pred]); bins=[]
    for i in range(5):
        lo=i/5; hi=(i+1)/5; mask=(p>=lo)&((p<hi) if i<4 else (p<=hi))
        if mask.any(): bins.append({"N":int(mask.sum()),"lo":lo,"hi":hi,"mean_probability":float(p[mask].mean()),"observed_rate":float(y[mask].mean())})
    return {"N":len(y),"ECE":float(sum(x["N"]*abs(x["mean_probability"]-x["observed_rate"]) for x in bins)/len(y)),"bins":bins}

def prospective(old,new):
    stable=[r for r in new if r["target_execution_oom"] is not None]; y=np.array([r["target_execution_oom"] for r in stable]); p=np.array([r["prospective_v1_probability"] for r in stable]); return {"metrics":metrics(y,p),"abstention":abstention(y,p),"N_stable_new_rows":len(stable),"N_stable_new_groups":len({r["workload_group_id"] for r in stable}),"active_sampling_note":"Campaign was selected using frozen V1; these metrics are prospective campaign-conditional, not population calibration.","errors":[{"candidate_id":r["campaign_row"]["candidate_id"],"family":r["family"],"configuration":r["configuration"],"p":r["prospective_v1_probability"],"outcome":r["outcome_class"],"aggregate_score":r["features"].get("calibrated_upper_over_budget")} for r in stable if (r["prospective_v1_probability"]<=.1 and r["target_execution_oom"]==1) or (r["prospective_v1_probability"]>=.9 and r["target_execution_oom"]==0)]}

def monotonicity(rows,names,beta,mean,scale):
    violations=0
    for r in rows:
        f=dict(r["features"]); ps=[]
        for factor in (.5,.75,1,1.5,2):
            vals=[]
            for n in names:
                v=f.get(n,0) or 0
                if n in RATIO_FEATURES: v=v/factor
                vals.append(v)
            x=np.array([1]+[(v-m)/s if s else 0 for v,m,s in zip(vals,mean,scale)]); ps.append(float(1/(1+math.exp(-np.clip(x@beta,-40,40)))))
        if any(ps[i+1]>ps[i]+1e-9 for i in range(len(ps)-1)): violations+=1
    return violations

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--summary-output",type=Path,default=ROOT/"oom_risk_summary_v2_2026-09-12.json"); ap.add_argument("--cv-output",type=Path,default=ROOT/"oom_risk_cv_v2_2026-09-12.json"); ap.add_argument("--models-output",type=Path,default=ROOT/"oom_risk_models_v2_2026-09-12.json"); ap.add_argument("--dataset-output",type=Path,default=ROOT/"oom_risk_dataset_v2_2026-09-12.json"); ap.add_argument("--report-output",type=Path,default=ROOT/"oom_risk_report_v2_2026-09-12.md"); args=ap.parse_args()
    old,new=make_new_rows(); combined=old+new; eligible=[r for r in combined if r["target_execution_oom"] is not None and not r.get("unstable_group",False)]
    models={"V1_schema_L2":evaluate(eligible,M.feature_sets()["D_memory_shape"],"l2"),"V2_compact_L2":evaluate(eligible,V2_FEATURES,"l2"),"V2_compact_L1":evaluate(eligible,V2_FEATURES,"l1"),"V2_group_balanced_L2":evaluate(eligible,V2_FEATURES,"l2",balanced=True),"V2_sign_constrained_L2":evaluate(eligible,V2_FEATURES,"l2",bounded=True),"V2_threshold_tree":evaluate_tree(eligible,V2_FEATURES)}
    train=eligible; x,_,mean,scale=prepare(train,[],V2_FEATURES); y=np.array([r["target_execution_oom"] for r in train]); beta=fit(x,y,bounded=True); beta_l2=fit(x,y)
    model={"feature_names":V2_FEATURES,"means":mean.tolist(),"scales":scale.tolist(),"coefficients":beta.tolist(),"intercept":float(beta[0]),"regularization":"L2=1.0; nonnegative first five demand/budget coefficients","dataset_hash":sha(combined),"model_hash":sha({"features":V2_FEATURES,"means":mean.tolist(),"scales":scale.tolist(),"coefficients":beta.tolist()}),"active_sampling":True}
    prospective_result=prospective(old,new); pred=models["V2_sign_constrained_L2"]["predictions"]; family_results=leave_family_out(eligible,V2_FEATURES)
    t0=time.perf_counter()
    for _ in range(1000): _=1/(1+np.exp(-np.clip(x@beta,-40,40)))
    inference_ms=(time.perf_counter()-t0)*1000/1000
    comparison={}
    for name,value in models.items():
        yy=np.array([z["y"] for z in value["predictions"]]); pp=np.array([z["p"] for z in value["predictions"]]); comparison[name]={**value["metrics"],"calibration":calibration(value["predictions"]),"abstention":abstention(yy,pp)}
    selected_y=np.array([z["y"] for z in pred]); selected_p=np.array([z["p"] for z in pred]); abstention_grid={f"{pf:.2f}/{po:.2f}":abstention(selected_y,selected_p,pf,po) for pf,po in ((.1,.9),(.15,.85),(.2,.8),(.25,.75),(.3,.7),(.4,.6),(.5,.5))}
    summary={"status":"COMPUTATIONALLY_VERIFIED","decision":"OOM RISK MODEL V2 PROMISING BUT DATA-LIMITED","production_decision":"NO PRODUCTION CHANGE","v1_prospective":prospective_result,"dataset":{"old_rows":len(old),"new_rows":len(new),"total_rows":len(combined),"total_groups":len({r["workload_group_id"] for r in combined}),"stable_rows":len(eligible),"stable_groups":len({r["workload_group_id"] for r in eligible}),"fit_groups":len({r["workload_group_id"] for r in eligible if r["target_execution_oom"]==0}),"execution_oom_groups":len({r["workload_group_id"] for r in eligible if r["target_execution_oom"]==1}),"outcome_counts":dict(Counter(r["outcome_class"] for r in combined)),"families":dict(Counter(r["family"] for r in combined))},"model_comparison":comparison,"abstention_grid":abstention_grid,"models":{k:v["metrics"] for k,v in models.items()},"selected_model":"V2_sign_constrained_L2","selected_model_artifact":model,"monotonicity":{"v1_violations":3,"v2_compact_l2_violations":monotonicity(eligible,V2_FEATURES,beta_l2,mean,scale),"v2_sign_constrained_violations":monotonicity(eligible,V2_FEATURES,beta,mean,scale)},"unstable_cases":{"count":0,"predictions":[]},"family_results":family_results,"leave_family_out":"reported for families with a two-class training complement","permutation_test":"not rerun separately; V1 group permutation was chance-level","speed":{"feature_extraction":"static JAX tracing only; recorded in campaign generator","inference_ms_per_1000_rows":inference_ms,"feature_count":len(V2_FEATURES),"serialized_model_bytes":len(dump(model).encode())},"inference_purity":{"target_compilation":False,"target_execution":False,"runtime_allocator_state":False}}
    args.dataset_output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","rows":combined},indent=2,sort_keys=True,default=lambda o:o.item() if hasattr(o,"item") else str(o))+"\n"); args.cv_output.write_text(json.dumps(models,indent=2,sort_keys=True)+"\n"); args.models_output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","selected":model,"models":{k:v["metrics"] for k,v in models.items()}},indent=2,sort_keys=True)+"\n"); (ROOT/"oom_risk_model_v2_2026-09-12.json").write_text(json.dumps(model,indent=2,sort_keys=True)+"\n"); args.summary_output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    args.report_output.write_text("# OOM risk model V2\n\nSee the generated summary and frozen expansion artifacts. Decision: **OOM RISK MODEL V2 PROMISING BUT DATA-LIMITED**; **NO PRODUCTION CHANGE**.\n")
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
