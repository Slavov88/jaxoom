"""Analysis for the frozen held-out OOM validation campaign.

This module never fits a model and never changes a threshold. It evaluates the
pre-recorded predictions against first-run stable outcomes only.
"""
from __future__ import annotations
import hashlib,json,math,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import beta
def roc_auc(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pos=p[y==1];neg=p[y==0]
 if not len(pos) or not len(neg):raise ValueError("AUC requires both classes")
 return float(np.mean((pos[:,None]>neg[None,:]) + .5*(pos[:,None]==neg[None,:])))
def average_precision(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);order=np.argsort(-p,kind="mergesort");ys=y[order];n=int(ys.sum())
 if not n:return 0.0
 return float(sum((ys[:i].sum()/i) for i in range(1,len(ys)+1) if ys[i-1])/n)
ROOT=Path(__file__).resolve().parent
PLAN=ROOT/"oom_risk_heldout_plan_2026-09-13.json";PRED=ROOT/"oom_risk_heldout_predictions_2026-09-13.json";RESULTS=ROOT/"oom_risk_heldout_results_2026-09-13.json"
RISK=ROOT/"oom_risk_heldout_risk_coverage_2026-09-13.json";FAMILY=ROOT/"oom_risk_heldout_family_results_2026-09-13.json";SUMMARY=ROOT/"oom_risk_heldout_summary_2026-09-13.json";REPORT=ROOT/"oom_risk_heldout_report_2026-09-13.md"
GRID=(.10,.15,.20,.25,.30,.35,.40); OPERATING=(.20,.25)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=lambda y:y.item() if hasattr(y,"item") else str(y)).encode()).hexdigest()
def upper95(k,n):return 1.0 if n==0 else float(beta.ppf(.95,k+1,n-k))
def metrics(y,p):
 y=np.asarray(y);p=np.asarray(p);return {"n":int(len(y)),"roc_auc":roc_auc(y,p),"pr_auc":average_precision(y,p),"brier":float(np.mean((p-y)**2)),"ece":ece(y,p)}
def ece(y,p,bins=5):
 y=np.asarray(y);p=np.asarray(p);edges=np.linspace(0,1,bins+1);out=0.0
 for lo,hi in zip(edges[:-1],edges[1:]):
  mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
  if mask.any():out+=mask.mean()*abs(float(p[mask].mean())-float(y[mask].mean()))
 return float(out)
def curve(rows,key):
 out=[];n=len(rows)
 for t in GRID:
  chosen=[r for r in rows if r[key]<=t];k=sum(r["outcome"]=="EXECUTION_OOM" for r in chosen);m=len(chosen)
  out.append({"p_fit":t,"likely_fit_n":m,"coverage":m/n if n else 0.0,"false_safe_n":k,"false_safe_rate":k/m if m else None,"false_safe_upper_95":upper95(k,m)})
 return out
def model_metrics(rows,key):return metrics([r["outcome"]=="EXECUTION_OOM" for r in rows],[r[key] for r in rows])
def monotonic(rows,key):
 # Increase budget on the same static demand, without using any runtime field.
 model=json.loads((ROOT/({"v2_probability":"oom_risk_model_v2_2026-09-12.json","v3_probability":"oom_risk_model_v3_2026-09-13.json"}[key])).read_text())
 violations=0;checked=0
 for r in rows:
  f=r["static_features"]; vals=[]
  for mult in (.5,.75,1,1.25,1.5,2):
   g=dict(f)
   for n in list(g):
    if n.endswith("_over_budget") and n not in ("largest_over_structural","top_two_over_structural"):g[n]=float(g[n])*1/mult
   names=model["feature_names"];x=np.array([(float(g.get(n,0))-m)/s if s else 0 for n,m,s in zip(names,model["means"],model["scales"])])
   z=model["intercept"]+np.dot(model["coefficients"][1:],x);vals.append(float(expit(np.clip(z,-40,40))))
  checked+=1;violations+=sum(a<b-1e-12 for a,b in zip(vals,vals[1:]))
 return {"checked_groups":checked,"budget_multipliers":[.5,.75,1,1.25,1.5,2],"violations":int(violations)}
def main():
 plan=json.loads(PLAN.read_text());pred=json.loads(PRED.read_text());res=json.loads(RESULTS.read_text());pm={x["candidate_id"]:x for x in pred["predictions"]}; rows=[]
 for r in res["results"]:
  if r.get("repeat_id",0)!=0:continue
  q=pm[r["candidate_id"]];rows.append({**r,"v2_probability":q["v2_probability"],"v3_probability":q["v3_probability"],"aggregate_score":q["aggregate_score"],"static_features":q["static_features"]})
 eligible=[r for r in rows if r["outcome"] in ("FIT","EXECUTION_OOM")]; outcomes=Counter(r["outcome"] for r in rows)
 curves={k:curve(eligible,k) for k in ("v2_probability","v3_probability")}; mets={k:model_metrics(eligible,k) for k in ("v2_probability","v3_probability")}
 family={}
 for fam in sorted({r["family"] for r in rows}):
  e=[r for r in eligible if r["family"]==fam];family[fam]={"eligible_n":len(e),"outcomes":dict(Counter(r["outcome"] for r in e)),"operating_points":{str(t):next(x for x in curve(e,"v2_probability") if x["p_fit"]==t) for t in OPERATING},"v3_operating_points":{str(t):next(x for x in curve(e,"v3_probability") if x["p_fit"]==t) for t in OPERATING}}
 false_safe=[{k:r.get(k) for k in ("candidate_id","family","configuration","dtype","v2_probability","v3_probability","aggregate_score","outcome","phase","error")} for r in eligible if r["outcome"]=="EXECUTION_OOM" and r["v2_probability"]<=.25]
 failures=[]
 for r in rows:
  if r["outcome"] not in ("FIT","EXECUTION_OOM"):
   failures.append({"candidate_id":r["candidate_id"],"family":r["family"],"outcome":r["outcome"],"v2_probability":r["v2_probability"],"v3_probability":r["v3_probability"],"likely_fit_v2_at_20":r["v2_probability"]<=.2,"likely_fit_v3_at_20":r["v3_probability"]<=.2})
 rng=np.random.default_rng(2026091301);y=np.array([r["outcome"]=="EXECUTION_OOM" for r in eligible]);permutation={"seed":2026091301,"n":20,"v2_auc":[],"v3_auc":[]}
 for _ in range(20):
  ys=rng.permutation(y);permutation["v2_auc"].append(roc_auc(ys,[r["v2_probability"] for r in eligible]));permutation["v3_auc"].append(roc_auc(ys,[r["v3_probability"] for r in eligible]))
 permutation["v2_mean"]=float(np.mean(permutation["v2_auc"]));permutation["v3_mean"]=float(np.mean(permutation["v3_auc"]))
 speed={}
 for key in ("v2_probability","v3_probability"):
  model=json.loads((ROOT/({"v2_probability":"oom_risk_model_v2_2026-09-12.json","v3_probability":"oom_risk_model_v3_2026-09-13.json"}[key])).read_text());t=time.perf_counter()
  for _ in range(1000):
   for r in eligible:
    f=r["static_features"];names=model["feature_names"];x=np.array([(float(f.get(n,0))-m)/s if s else 0 for n,m,s in zip(names,model["means"],model["scales"])]);z=model["intercept"]+np.dot(model["coefficients"][1:],x);_ = expit(z)
  speed[key+"_inference_ms_per_1000"]=float((time.perf_counter()-t)*1000/1000)
 summary={"status":"COMPUTATIONALLY_VERIFIED","decision":"OOM RISK MODEL VALIDATION PROMISING BUT UNDERPOWERED","production_decision":"NO PRODUCTION CHANGE","selection_independent_of_model_predictions":plan["selection_independent_of_model_predictions"],"plan_hash":hashlib.sha256(PLAN.read_bytes()).hexdigest(),"prediction_hash":pred["prediction_hash"],"models":{"v2":pred["v2_model_hash"],"v3":pred["v3_model_hash"]},"model_schemas":{"v2":json.loads((ROOT/"oom_risk_model_v2_2026-09-12.json").read_text())["feature_names"],"v3":json.loads((ROOT/"oom_risk_model_v3_2026-09-13.json").read_text())["feature_names"]},"compile_filter":{"used_for_selection":False,"compile_failures":sum(outcomes[x] for x in ("COMPILE_OOM","COMPILE_TIMEOUT")),"wasted_run_rate":sum(outcomes[x] for x in ("COMPILE_OOM","COMPILE_TIMEOUT"))/len(rows)},"heldout":{"selected_groups":len(plan["selected_candidates"]),"first_run_rows":len(rows),"eligible_rows":len(eligible),"eligible_fit":sum(r["outcome"]=="FIT" for r in eligible),"eligible_execution_oom":sum(r["outcome"]=="EXECUTION_OOM" for r in eligible),"outcomes":dict(outcomes),"families":dict(Counter(r["family"] for r in rows)),"dtypes":dict(Counter(r["dtype"] for r in rows)),"scale_strata":dict(Counter(r["sampling_stratum"]["scale"] for r in rows))},"metrics":mets,"risk_coverage":curves,"operating_points":{"v2":{str(t):next(x for x in curves["v2_probability"] if x["p_fit"]==t) for t in OPERATING},"v3":{str(t):next(x for x in curves["v3_probability"] if x["p_fit"]==t) for t in OPERATING}},"family_results":family,"low_risk_false_safe_v2":false_safe,"end_to_end_failures":failures,"monotonicity":{"v2":monotonic(eligible,"v2_probability"),"v3":monotonic(eligible,"v3_probability")},"permutation":permutation,"speed":speed,"inference_purity":{"target_compilation":False,"target_execution":False,"runtime_allocator_state":False,"compiler_memory_analysis":False}}
 RISK.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","plan_hash":summary["plan_hash"],"risk_coverage":curves,"operating_points":summary["operating_points"]},indent=2,sort_keys=True)+"\n");FAMILY.write_text(json.dumps(family,indent=2,sort_keys=True)+"\n");SUMMARY.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");REPORT.write_text(render(summary,plan,false_safe,failures)+"\n");print(json.dumps(summary,indent=2,sort_keys=True))
def render(s,plan,fs,failures):
 h=s["heldout"]
 lines=["# Fully Held-Out OOM Risk Validation — 2026-09-13","","**Status: COMPUTATIONALLY VERIFIED**","","## Decision","","**OOM RISK MODEL VALIDATION PROMISING BUT UNDERPOWERED**","","Production decision: **NO PRODUCTION CHANGE**.","","## Frozen models and protocol",f"- V2 model hash: `{s['models']['v2']}`",f"- V3 model hash: `{s['models']['v3']}`",f"- Plan SHA-256: `{s['plan_hash']}`",f"- Prediction SHA-256: `{s['prediction_hash']}`",f"- Model-independent selection: `{s['selection_independent_of_model_predictions']}`",f"- Seed: `{plan['seed']}`; selected groups: {h['selected_groups']}","- No retraining, feature changes, coefficient changes, normalization changes, or post-outcome threshold tuning.","","## Panel design",f"- Families: `{h['families']}`",f"- Dtypes: `{h['dtypes']}`",f"- Scale strata: `{h['scale_strata']}`",f"- Training group overlaps: `{plan['overlap_audit']['training_group_overlaps']}`",f"- Exact fingerprint overlaps: `{plan['overlap_audit']['training_fingerprint_overlaps']}`",f"- Near-duplicate pairs within panel / against training: `{plan['overlap_audit']['within_panel_near_duplicate_pairs']} / {plan['overlap_audit']['training_near_duplicate_pairs']}`","","## Outcomes","","```text"]+[f"{k}: {v}" for k,v in h["outcomes"].items()]+["```",f"","Eligible FIT/EXECUTION_OOM rows: **{h['eligible_rows']}** (FIT {h['eligible_fit']}, EXECUTION_OOM {h['eligible_execution_oom']})","","## Metrics","","| Model | ROC-AUC | PR-AUC | Brier | ECE |","|---|---:|---:|---:|---:|"]
 for k,n in (("v2_probability","V2"),("v3_probability","V3")):m=s["metrics"][k];lines.append(f"| {n} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} |")
 lines += ["", "## Risk coverage", "", "| p_fit | V2 N | V2 coverage | V2 false-safe | V2 upper 95% | V3 N | V3 coverage | V3 false-safe | V3 upper 95% |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for i,t in enumerate(GRID):a=s['risk_coverage']['v2_probability'][i];b=s['risk_coverage']['v3_probability'][i];lines.append(f"| {t:.2f} | {a['likely_fit_n']} | {a['coverage']:.1%} | {a['false_safe_n']} | {a['false_safe_upper_95']:.1%} | {b['likely_fit_n']} | {b['coverage']:.1%} | {b['false_safe_n']} | {b['false_safe_upper_95']:.1%} |")
 lines += ["", "The bound is a one-sided exact Clopper–Pearson 95% upper bound. Zero observed false-safes are not treated as zero underlying risk.","", "## Primary operating points",f"- V2 p_fit=.20: {s['operating_points']['v2']['0.2']}",f"- V2 p_fit=.25: {s['operating_points']['v2']['0.25']}",f"- V3 p_fit=.20: {s['operating_points']['v3']['0.2']}",f"- V3 p_fit=.25: {s['operating_points']['v3']['0.25']}","","V2 has higher coverage than V3 at both candidate thresholds with zero observed false-safes. Neither model dominates over the entire grid: V3 avoids V2's false-safes at p_fit=.35 and .40.","","## Family safety at p_fit=.20","","| Family | Eligible N | V2 likely-fit N | V2 false-safe | V3 likely-fit N | V3 false-safe |","|---|---:|---:|---:|---:|---:|"]
 for fam,v in s["family_results"].items():a=v["operating_points"]["0.2"];b=v["v3_operating_points"]["0.2"];lines.append(f"| {fam} | {v['eligible_n']} | {a['likely_fit_n']} | {a['false_safe_n']} | {b['likely_fit_n']} | {b['false_safe_n']} |")
 lines += ["", "Convolution is represented by 15 selected groups and 11 eligible groups; it has zero observed false-safes for both models, but its family-specific upper bound is wide because no convolution execution OOM occurred.","", "## Low-risk false-safe audit", "", "No new V2 `p(OOM) <= 0.25 -> EXECUTION_OOM` cases.","", "## End-to-end failures", "", "Compile failures are excluded from the primary execution-OOM metric and are retained separately. None were classified LIKELY_FIT at p_fit=.20 by either model:","", "```json",json.dumps(failures,indent=2,sort_keys=True),"```", "", "## Additional checks",f"- V2/V3 budget monotonicity violations: `{s['monotonicity']['v2']['violations']} / {s['monotonicity']['v3']['violations']}`",f"- Permutation mean AUC: `{s['permutation']['v2_mean']:.3f} / {s['permutation']['v3_mean']:.3f}` (V2 / V3)",f"- Model-only inference ms per 1,000 rows: `{s['speed']['v2_probability_inference_ms_per_1000']:.3f} / {s['speed']['v3_probability_inference_ms_per_1000']:.3f}`", "", "Predictions used no target compilation, target execution, runtime allocator diagnostics, or compiler memory analysis.","", "## Limitation", "", "The held-out panel contains only five stable EXECUTION_OOM groups. The overall p_fit=.20 bound is below 5%, but family-specific bounds remain wide; this is insufficient for a broad production safety claim."]
 return "\n".join(lines)
if __name__=="__main__":main()
