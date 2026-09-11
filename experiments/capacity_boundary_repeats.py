"""Targeted fresh-process repeats for capacity-boundary contradictions."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPEATS = [
    ("CAP-ATT-S4096-F32", [0.85, 0.90, 0.925, 0.95], 1),
    ("CAP-ATT-S4608-F32", [0.85, 0.90, 0.925, 0.95], 1),
    ("CAP-ATT-S5120-F32", [0.85, 0.90, 0.95], 1),
    ("CAP-ATT-B2-H12-S2560-D64-F32", [0.45, 0.65, 0.75], 2),
    ("CAP-ATT-B2-H16-S2560-D96-F16", [0.25, 0.30, 0.35], 2),
    ("CAP-CONV2-B1024", [0.85, 0.90, 0.95], 2),
    ("CAP-ATT-S4096-F32", [0.875, 0.8875], 1),
    ("CAP-ATT-S4608-F32", [0.875, 0.8875], 1),
    ("CAP-ATT-S5120-F32", [0.925], 1),
    ("CAP-ATT-B2-H16-S2560-D96-F16", [0.275, 0.2875], 1),
    ("CAP-CONV2-B1024", [0.925], 1),
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--plan",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--timeout",type=int,default=90); ap.add_argument("--refine-only",action="store_true"); a=ap.parse_args()
    plan=json.loads(a.plan.read_text()); panels={p["configuration_id"]:p for p in plan["panel"]}; rows=[]
    selected = REPEATS[-5:] if a.refine_only else REPEATS
    for ident, fractions, count in selected:
        panel=panels[ident]
        for fraction in fractions:
            for rep in range(count):
                env=os.environ.copy(); env["XLA_CLIENT_MEM_FRACTION"]=str(fraction); env["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"; env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION",None)
                try:
                    p=subprocess.run([sys.executable,str(ROOT/"capacity_boundary_study.py"),"--child-json",json.dumps(panel,sort_keys=True)],capture_output=True,text=True,timeout=a.timeout,env=env)
                    parsed=None
                    for line in reversed(p.stdout.splitlines()):
                        try: parsed=json.loads(line); break
                        except json.JSONDecodeError: pass
                    row=parsed or {"outcome":"OTHER_FAILURE","stderr":p.stderr[-3000:]}
                except subprocess.TimeoutExpired: row={"outcome":"COMPILE_TIMEOUT"}
                if row.get("compile_status") == "COMPILE_OOM": outcome = "COMPILE_OOM"
                elif row.get("execution_status") == "EXECUTION_OOM": outcome = "EXECUTION_OOM"
                elif row.get("execution_status") == "FIT": outcome = "FIT"
                else: outcome = "OTHER_FAILURE"
                row.update({"configuration_id":ident,"fraction":fraction,"repeat":rep+1,"outcome":outcome,"family":panel["family"],"config":panel["config"],"dtype":panel["dtype"]})
                rows.append(row); print(json.dumps({"id":ident,"fraction":fraction,"repeat":rep+1,"outcome":row.get("outcome"),"limit":row.get("allocator_limit_bytes")},sort_keys=True),flush=True)
    a.output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","rows":rows},indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
