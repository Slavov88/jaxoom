"""Construct and score confirmatory held-out batch 2.

Panel construction never imports or evaluates model probabilities. Scoring is a
separate stage, after the outcome-free panel files have been frozen.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT/"oom_risk_heldout_validation.py"; V1=ROOT/"oom_risk_model_v1_frozen_2026-09-12.json";V2=ROOT/"oom_risk_model_v2_2026-09-12.json";V3=ROOT/"oom_risk_model_v3_2026-09-13.json"
B1PLAN=ROOT/"oom_risk_heldout_plan_2026-09-13.json";B1RESULTS=ROOT/"oom_risk_heldout_results_2026-09-13.json"
PLAN=ROOT/"oom_risk_heldout2_plan_2026-09-13.json";CONVPLAN=ROOT/"oom_risk_heldout2_conv_stress_plan_2026-09-13.json";PRED=ROOT/"oom_risk_heldout2_predictions_2026-09-13.json";CONVPRED=ROOT/"oom_risk_heldout2_conv_stress_predictions_2026-09-13.json";MANIFEST=ROOT/"oom_risk_heldout2_validation_manifest_2026-09-13.json"
spec=importlib.util.spec_from_file_location("heldout_base",BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SEED=20260914
PRIMARY_QUOTAS={"attention":{"small":6,"medium":6,"large":6,"near_device":4},"convolution":{"small":5,"medium":5,"large":5,"near_device":3},"mlp":{"small":3,"medium":3,"large":3,"near_device":2},"training":{"small":3,"medium":3,"large":3,"near_device":2},"transformer":{"small":4,"medium":4,"large":4,"near_device":2},"autodiff":{"small":3,"medium":4,"large":3,"near_device":2},"scan":{"small":2,"medium":3,"large":2,"near_device":1},"matmul":{"small":2,"medium":2,"large":2,"near_device":1},"reduction":{"small":2,"medium":2,"large":2,"near_device":1}}
CONV_QUOTAS={"convolution":{x:6 for x in ("small","medium","large","near_device")}}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),default=lambda y:y.item() if hasattr(y,"item") else str(y))
def sha(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prior_rows():
 rows=[]
 for path in (base.V1_DATA,base.V2_DATA,base.V3_DATA,B1PLAN):
  obj=json.loads(path.read_text());rows.extend(obj.get("rows",obj.get("selected_candidates",[])))
 return rows
def prior_groups():return {r["workload_group_id"] for r in prior_rows()}
def prior_fps():return {r.get("graph_fingerprint") for r in prior_rows() if r.get("graph_fingerprint")}
def make_population(seed, families, per_cell=240):
 rng=np.random.default_rng(seed);out=[];used=prior_groups();
 for fam in families:
  for scale in ("small","medium","large","near_device"):
   for _ in range(per_cell):
    dtype="float16" if rng.random()<.5 else "float32";c=base.make_config(fam,scale,rng);g=base.group(fam,c,dtype)
    if g in used:continue
    row={"candidate_id":f"H2-POP-{len(out):06d}","workload_group_id":g,"family":fam,"configuration":c,"dtype":dtype,"graph_fingerprint":base.fp(fam,c,dtype),"sampling_stratum":{"family":fam,"scale":scale},"size_proxy":base.size_proxy(fam,c),"generator_seed":seed}
    out.append(row);used.add(g)
 return out
def select(pop,quotas,seed,used):
 rng=np.random.default_rng(seed);selected=[]
 for fam,scales in quotas.items():
  if isinstance(scales,int):scales={scale:scales for scale in ("small","medium","large","near_device")}
  for scale,n in scales.items():
   choices=[r for r in pop if r["family"]==fam and r["sampling_stratum"]["scale"]==scale and r["workload_group_id"] not in used];rng.shuffle(choices)
   if len(choices)<n:raise RuntimeError(f"insufficient {fam}/{scale}")
   selected.extend(choices[:n]);used.update(x["workload_group_id"] for x in choices[:n])
 for i,r in enumerate(selected):r["candidate_id"]=f"H2-{i:04d}"
 return selected
def audit(selected):
 old=prior_rows();oldfp=prior_fps();return {"prior_group_overlaps":sum(x["workload_group_id"] in prior_groups() for x in selected),"prior_exact_fingerprint_overlaps":sum(x["graph_fingerprint"] in oldfp for x in selected),"near_duplicate_definition":"same family and dtype with all but one configuration scalar equal","within_panel_near_duplicate_pairs":sum(base.near_duplicate(a,b) for i,a in enumerate(selected) for b in selected[i+1:]),"vs_prior_near_duplicate_pairs":sum(base.near_duplicate(a,b) for a in selected for b in old)}
def manifest():
 models={}
 for name,path in (("v2",V2),("v3",V3)):
  x=json.loads(path.read_text());models[name]={"model_hash":x["model_hash"],"artifact_sha256":file_sha(path),"feature_names":x["feature_names"],"means":x["means"],"scales":x["scales"],"coefficients":x["coefficients"],"intercept":x["intercept"],"regularization":x["regularization"]}
 x={"status":"FROZEN_BEFORE_BATCH2_PANEL","date":"2026-09-13","primary_model":"V2","models":models,"thresholds":{"primary":.20,"secondary":[.10,.15,.20,.25,.30,.35,.40]},"primary_seed":SEED,"supplemental_conv_seed":SEED+1,"no_retraining":True,"selection_independent_of_model_predictions":True,"primary_population_replicates_batch1_design":True,"primary_compile_filter_used":False}
 if MANIFEST.exists() and json.loads(MANIFEST.read_text())!=x:raise RuntimeError("refusing to overwrite frozen batch2 manifest")
 if not MANIFEST.exists():MANIFEST.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
 return x
def build():
 manifest();used=prior_groups();pop=make_population(SEED,list(PRIMARY_QUOTAS));primary=select(pop,PRIMARY_QUOTAS,SEED+100,used);convpop=make_population(SEED+1,["convolution"]);conv=select(convpop,CONV_QUOTAS,SEED+101,used)
 common={"status":"FROZEN_BEFORE_SCORING","date":"2026-09-13","generator":"oom_risk_heldout2_validation.py","generator_sha256":file_sha(Path(__file__))}
 p={**common,"panel":"PRIMARY_CONFIRMATORY","seed":SEED,"population_size":len(pop),"selected_count":len(primary),"sampling_quotas":PRIMARY_QUOTAS,"selection_independent_of_model_predictions":True,"compile_filter_used":False,"selected_candidates":primary,"overlap_audit":audit(primary),"outcomes_excluded":True}
 c={**common,"panel":"SUPPLEMENTAL_CONVOLUTION_STRESS","seed":SEED+1,"population_size":len(convpop),"selected_count":len(conv),"sampling_quotas":CONV_QUOTAS,"selection_independent_of_model_predictions":True,"compile_filter_used":False,"selected_candidates":conv,"overlap_audit":audit(conv),"outcomes_excluded":True}
 return p,c
def score_model(f,m):
 vals=np.array([float(f.get(n,0) or 0) for n in m["feature_names"]]);x=(vals-np.array(m["means"]))/np.array(m["scales"]);z=m["intercept"]+np.dot(m["coefficients"][1:],x);return float(1/(1+np.exp(-np.clip(z,-40,40))))
def score_panel(path,out):
 p=json.loads(path.read_text());v1=json.loads(V1.read_text());v2=json.loads(V2.read_text());v3=json.loads(V3.read_text());items=[]
 for c in p["selected_candidates"]:
  f=base.expansion().static_features(c["family"],c["configuration"],c["dtype"],v1);items.append({"candidate_id":c["candidate_id"],"workload_group_id":c["workload_group_id"],"family":c["family"],"configuration":c["configuration"],"dtype":c["dtype"],"graph_fingerprint":c["graph_fingerprint"],"sampling_stratum":c["sampling_stratum"],"v2_probability":score_model(f,v2),"v3_probability":score_model(f,v3),"aggregate_score":f.get("calibrated_upper_over_budget"),"static_features":f})
 payload={"status":"FROZEN_BEFORE_RUNTIME","panel":p["panel"],"plan_hash":hashlib.sha256(path.read_bytes()).hexdigest(),"v2_model_hash":v2["model_hash"],"v3_model_hash":v3["model_hash"],"predictions":items,"outcomes_excluded":True};payload["prediction_hash"]=sha(payload);out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");return payload
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("manifest","plan","score"),required=True);a=ap.parse_args()
 if a.stage=="manifest":print(json.dumps(manifest(),indent=2,sort_keys=True))
 elif a.stage=="plan":
  p,c=build();PLAN.write_text(json.dumps(p,indent=2,sort_keys=True)+"\n");CONVPLAN.write_text(json.dumps(c,indent=2,sort_keys=True)+"\n");print(json.dumps({"primary":len(p["selected_candidates"]),"conv_stress":len(c["selected_candidates"]),"primary_hash":hashlib.sha256(PLAN.read_bytes()).hexdigest(),"conv_hash":hashlib.sha256(CONVPLAN.read_bytes()).hexdigest(),"primary_families":dict(Counter(x["family"] for x in p["selected_candidates"]))},indent=2))
 else:
  a=score_panel(PLAN,PRED);b=score_panel(CONVPLAN,CONVPRED);print(json.dumps({"primary_prediction_hash":a["prediction_hash"],"conv_prediction_hash":b["prediction_hash"]},indent=2))
if __name__=="__main__":main()
