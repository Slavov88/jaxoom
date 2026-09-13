"""Model-independent held-out panel construction and frozen-model scoring.

Stages are intentionally separate: --plan writes the outcome-free panel,
--score writes predictions only after the panel is immutable, and the runtime
campaign is a separate fresh-process program.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
V1_FREEZE=ROOT/"oom_risk_model_v1_frozen_2026-09-12.json"; V2_MODEL=ROOT/"oom_risk_model_v2_2026-09-12.json"; V3_MODEL=ROOT/"oom_risk_model_v3_2026-09-13.json"
V1_DATA=ROOT/"oom_risk_dataset_v1_2026-09-12.json"; V2_DATA=ROOT/"oom_risk_dataset_v2_2026-09-12.json"; V3_DATA=ROOT/"oom_risk_dataset_v3_2026-09-13.json"
PLAN=ROOT/"oom_risk_heldout_plan_2026-09-13.json"; PRED=ROOT/"oom_risk_heldout_predictions_2026-09-13.json"; MANIFEST=ROOT/"oom_risk_heldout_validation_manifest_2026-09-13.json"

def load_expansion():
 s=importlib.util.spec_from_file_location("oom_risk_expansion_v2",ROOT/"oom_risk_expansion_v2.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
E=None
def expansion():
 global E
 if E is None: E=load_expansion()
 return E
def dump(x):return json.dumps(x,sort_keys=True,separators=(",",":"),default=lambda y:y.item() if hasattr(y,"item") else str(y))
def digest(x):return hashlib.sha256(dump(x).encode()).hexdigest()
def group(f,c,d):return f+"|"+dump(c)+"|"+d
def fp(f,c,d):return hashlib.sha256(group(f,c,d).encode()).hexdigest()[:16]

def freeze_manifest():
 files={"v1":V1_FREEZE,"v2":V2_MODEL,"v3":V3_MODEL}
 models={}
 for name,path in files.items():
  obj=json.loads(path.read_text()); models[name]={"path":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"model_hash":obj.get("model_hash") or obj.get("selected",{}).get("model_hash"),"dataset_hash":obj.get("dataset_hash") or obj.get("selected",{}).get("dataset_hash"),"feature_names":(obj.get("model") or obj.get("selected") or obj).get("feature_names"),"means":(obj.get("model") or obj.get("selected") or obj).get("means"),"scales":(obj.get("model") or obj.get("selected") or obj).get("scales"),"coefficients":(obj.get("model") or obj.get("selected") or obj).get("coefficients"),"intercept":(obj.get("model") or obj.get("selected") or obj).get("intercept"),"regularization":(obj.get("model") or obj.get("selected") or obj).get("regularization")}
 manifest={"status":"FROZEN_BEFORE_HELDOUT_VALIDATION","date":"2026-09-13","models":models,"operating_points":{"p_fit":[.10,.15,.20,.25,.30],"risk_coverage_grid":[.10,.15,.20,.25,.30,.35,.40]},"selection_independent_of_model_predictions":True,"no_retraining":True,"primary_environment_scope":"RTX 3050 Laptop GPU / CUDA-BFC / JAX 0.11.x / PREALLOCATE=false"}
 if MANIFEST.exists():
  old=json.loads(MANIFEST.read_text())
  if digest(old)!=digest(manifest):raise RuntimeError("refusing to overwrite frozen validation manifest")
 else:MANIFEST.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 return manifest

def ranges(fam,scale):
 if fam=="attention":
  return {"small":([1,2],[2,4,8],[256,512,768,1024],[32,64]),"medium":([1,2,4],[4,8,12,16],[1280,1536,2048,2560,3072],[32,48,64,80,96]),"large":([1,2,4],[4,8,12,16],[3072,3584,4096,4608,5120],[32,48,64,80,96,128]),"near_device":([1,2],[4,8,12,16],[5376,5632,5888,6144,6656,7168],[32,48,64,80,96])}[scale]
 if fam=="convolution":
  return {"small":([1,2,4,8],[32,48,64,80,96],[32,48,64,80,96],[3,8,16,32],[8,16,32,64]),"medium":([8,16,32,64],[64,80,96,112,128],[64,80,96,112,128],[3,8,16,32],[16,32,64,96]),"large":([64,128,256],[96,112,128,160,192],[96,112,128,160,192],[3,8,16,32],[16,32,64,96]),"near_device":([256,512,1024],[96,128,160,192],[96,128,160,192],[3,8,16],[16,32,64])}[scale]
 if fam in {"mlp","training","autodiff"}:
  return {"small":([8,16,32,64],[128,256,384,512],[1,2,3,4]),"medium":([64,128,256,512],[512,768,1024,1536],[2,3,4,6]),"large":([256,512,1024,2048],[1536,2048,3072,4096],[2,3,4,6]),"near_device":([512,1024,2048],[3072,4096,6144,8192],[2,3,4,6])}[scale]
 if fam=="transformer":
  return {"small":([256,384,512,768],[256,384,512,768],[2,4,8],[1,2]),"medium":([768,1024,1280,1536],[512,768,1024,1536],[4,8,16],[1,2,3]),"large":([1536,2048,2560],[1024,1536,2048],[8,16],[1,2,3,4]),"near_device":([2560,3072,3584,4096],[2048,2560,3072,4096],[8,16],[1,2,3,4])}[scale]
 if fam in {"scan","reduction"}:
  return {"small":([1,2,4],[128,256,512],[32,64,128]),"medium":([4,8,16],[512,1024,2048],[64,128,256]),"large":([8,16,32],[2048,4096],[128,256,512]),"near_device":([16,32,64],[4096,8192],[256,512,1024])}[scale]
 if fam=="matmul":
  return {"small":([128,256,512],[128,256,512],[128,256,512]),"medium":([512,768,1024],[512,1024,1536],[512,1024,1536]),"large":([1024,1536,2048],[1024,2048,3072],[1024,2048,3072]),"near_device":([2048,3072,4096],[2048,3072,4096],[2048,3072,4096])}[scale]
 raise ValueError(fam)

def make_config(fam,scale,rng):
 z=ranges(fam,scale)
 if fam=="attention":
  b,h,s,hd=[int(rng.choice(x)) for x in z];return {"batch":b,"heads":h,"sequence":s,"head_dim":hd}
 if fam=="convolution":
  b,h,w,c,oc=[int(rng.choice(x)) for x in z];return {"batch":b,"height":h,"width":w,"channels":c,"out_channels":oc}
 if fam in {"mlp","training","autodiff"}:
  b,w,d=[int(rng.choice(x)) for x in z];return {"batch":b,"width":w,"depth":d}
 if fam=="transformer":
  s,w,h,l=[int(rng.choice(x)) for x in z];w-=w%h;return {"sequence":s,"width":w,"heads":h,"layers":l}
 if fam in {"scan","reduction"}:
  b,l,w=[int(rng.choice(x)) for x in z];return {"batch":b,"length":l,"width":w,"depth":int(rng.choice([1,2,4,8]))}
 if fam=="matmul":
  m,k,n=[int(rng.choice(x)) for x in z];return {"m":m,"k":k,"n":n,"depth":int(rng.choice([1,2,3]))}

def size_proxy(fam,c):
 if fam=="attention":return c["batch"]*c["heads"]*c["sequence"]**2
 if fam=="convolution":return c["batch"]*c["height"]*c["width"]*c["out_channels"]
 if fam in {"mlp","training","autodiff"}:return max(c["batch"]*c["width"],c["width"]**2)
 if fam=="transformer":return max(c["sequence"]*c["width"],c["sequence"]**2*c["heads"])
 if fam in {"scan","reduction"}:return c["batch"]*c["length"]*c["width"]
 return max(c["m"]*c["k"],c["k"]*c["n"],c["m"]*c["n"])

def quotas():return {"attention":{"small":5,"medium":5,"large":5,"near_device":3},"convolution":{"small":4,"medium":4,"large":4,"near_device":3},"mlp":{"small":2,"medium":3,"large":3,"near_device":1},"training":{"small":2,"medium":3,"large":3,"near_device":1},"transformer":{"small":3,"medium":3,"large":4,"near_device":2},"autodiff":{"small":3,"medium":3,"large":3,"near_device":1},"scan":{"small":2,"medium":2,"large":2,"near_device":1},"matmul":{"small":1,"medium":2,"large":1,"near_device":1},"reduction":{"small":1,"medium":2,"large":1,"near_device":1}}

def existing_rows():return json.loads(V1_DATA.read_text())["rows"]+json.loads(V2_DATA.read_text())["rows"]+json.loads(V3_DATA.read_text())["rows"]
def existing():return {r["workload_group_id"] for r in existing_rows()}
def near_duplicate(a,b):
 if a["family"]!=b["family"] or a["dtype"]!=b["dtype"]: return False
 ka=set(a["configuration"]);kb=set(b["configuration"])
 if ka!=kb:return False
 return sum(a["configuration"][k]!=b["configuration"][k] for k in ka)==1
def build_plan(seed=20260913,pop_per_cell=180):
 rng=np.random.default_rng(seed);seen=existing();population=[];families=list(quotas())
 for fam in families:
  for scale in ("small","medium","large","near_device"):
   for i in range(pop_per_cell):
    dtype="float16" if rng.random()<.5 else "float32";c=make_config(fam,scale,rng);g=group(fam,c,dtype)
    if g in seen:continue
    population.append({"candidate_id":f"HOLDOUT-POP-{len(population):06d}","workload_group_id":g,"family":fam,"configuration":c,"dtype":dtype,"graph_fingerprint":fp(fam,c,dtype),"sampling_stratum":{"family":fam,"scale":scale},"size_proxy":size_proxy(fam,c),"generator_seed":seed});seen.add(g)
 selected=[];q=quotas()
 for fam,scales in q.items():
  for scale,n in scales.items():
   choices=[r for r in population if r["family"]==fam and r["sampling_stratum"]["scale"]==scale];rng.shuffle(choices)
   if len(choices)<n:raise RuntimeError(f"insufficient candidates for {fam}/{scale}: {len(choices)}")
   selected.extend(choices[:n])
 for i,r in enumerate(selected):r["candidate_id"]=f"HOLDOUT-{i:04d}"
 old=existing_rows();old_fps={r.get("graph_fingerprint") for r in old}; exact_fp=sum(r["graph_fingerprint"] in old_fps for r in selected)
 near=sum(near_duplicate(a,b) for i,a in enumerate(selected) for b in selected[i+1:])
 near_old=sum(near_duplicate(a,b) for a in selected for b in old)
 return {"status":"FROZEN_BEFORE_SCORING","date":"2026-09-13","seed":seed,"population_size":len(population),"selected_count":len(selected),"generator":"oom_risk_heldout_validation.py","sampling_quotas":q,"selection_independent_of_model_predictions":True,"selected_candidates":selected,"overlap_audit":{"training_group_overlaps":0,"training_fingerprint_overlaps":exact_fp,"near_duplicate_definition":"same family and dtype with all but one configuration scalar equal","within_panel_near_duplicate_pairs":near,"training_near_duplicate_pairs":near_old},"outcomes_excluded":True}

def score(features,model):
 names=model["feature_names"];vals=[]
 for n in names:
  if n=="largest_over_structural" and features.get(n) is None: features[n]=(features.get("largest_buffer_bytes") or 0)/(features.get("structural_peak_bytes") or 1)
  if n=="top_two_over_structural" and features.get(n) is None: features[n]=(features.get("top_two_peak_live_bytes") or 0)/(features.get("structural_peak_bytes") or 1)
  vals.append(float(features.get(n) or 0))
 z=model["intercept"]+float(np.dot(np.asarray(model["coefficients"])[1:],(np.asarray(vals)-np.asarray(model["means"]))/np.asarray(model["scales"])))
 return float(1/(1+math.exp(-np.clip(z,-40,40))) )
def score_plan(plan):
 v1=json.loads(V1_FREEZE.read_text())["model"];v2=json.loads(V2_MODEL.read_text());v3=json.loads(V3_MODEL.read_text());pred=[]
 for c in plan["selected_candidates"]:
  f=expansion().static_features(c["family"],c["configuration"],c["dtype"],json.loads(V1_FREEZE.read_text()));p1=score(f,v1);p2=score(f,v2);p3=score(f,v3);pred.append({"candidate_id":c["candidate_id"],"workload_group_id":c["workload_group_id"],"family":c["family"],"configuration":c["configuration"],"dtype":c["dtype"],"graph_fingerprint":c["graph_fingerprint"],"sampling_stratum":c["sampling_stratum"],"v1_probability":p1,"v2_probability":p2,"v3_probability":p3,"aggregate_score":f.get("calibrated_upper_over_budget"),"aggregate_fit":(f.get("calibrated_upper_over_budget") or 0)<=1,"static_features":f})
 payload={"status":"FROZEN_BEFORE_RUNTIME","plan_hash":hashlib.sha256(PLAN.read_bytes()).hexdigest(),"v1_model_hash":json.loads(V1_FREEZE.read_text())["model_hash"],"v2_model_hash":v2["model_hash"],"v3_model_hash":v3["model_hash"],"predictions":pred,"outcomes_excluded":True};payload["prediction_hash"]=digest(payload);return payload

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("manifest","plan","score"),required=True);ap.add_argument("--seed",type=int,default=20260913);a=ap.parse_args()
 if a.stage=="manifest":print(json.dumps(freeze_manifest(),indent=2,sort_keys=True))
 elif a.stage=="plan":
  freeze_manifest();p=build_plan(a.seed);PLAN.write_text(json.dumps(p,indent=2,sort_keys=True)+"\n");print(json.dumps({"plan_hash":digest(p),"selected":len(p["selected_candidates"]),"families":dict(Counter(r["family"] for r in p["selected_candidates"]))},indent=2))
 else:
  p=json.loads(PLAN.read_text());payload=score_plan(p);PRED.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps({"plan_hash":payload["plan_hash"],"prediction_hash":payload["prediction_hash"],"count":len(payload["predictions"])},indent=2))
if __name__=="__main__":main()
