"""Post-prediction diagnostics for selected discriminator rows."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent

def attention(q,k,v):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), v)

def args_for(c):
    dtype=getattr(jnp,c["dtype"]); shape=(c["B"],c["H"],c["S"],c["D"])
    return tuple(jnp.ones(shape,dtype) for _ in range(3))

def child(c):
    result={"configuration_id":c["configuration_id"]}
    try:
        args=args_for(c); compiled=jax.jit(attention).lower(*args).compile()
        analysis=compiled.memory_analysis()
        result["compiler_diagnostics"]={k:getattr(analysis,k,None) for k in ("generated_code_size_in_bytes","argument_size_in_bytes","output_size_in_bytes","alias_size_in_bytes","temp_size_in_bytes")}
        try:
            result["pre_execute_memory_stats"]=jax.devices()[0].memory_stats()
        except Exception: result["pre_execute_memory_stats"]=None
        try:
            out=compiled(*args); jax.tree_util.tree_map(lambda x:x.block_until_ready(),out)
            result["outcome"]="FIT"
        except Exception as exc:
            result["outcome"]="EXECUTION_OOM" if "out of memory" in str(exc).lower() or "resource_exhausted" in str(exc).lower() else "OTHER_FAILURE"
            result["error"]=f"{type(exc).__name__}: {exc}"
        try: result["post_execute_memory_stats"]=jax.devices()[0].memory_stats()
        except Exception: result["post_execute_memory_stats"]=None
    except Exception as exc:
        result["outcome"]="COMPILE_OOM" if "out of memory" in str(exc).lower() or "resource_exhausted" in str(exc).lower() else "OTHER_FAILURE"
        result["error"]=f"{type(exc).__name__}: {exc}"
    print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--plan",type=Path,required=True); ap.add_argument("--ids",nargs="+",required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--timeout",type=int,default=90); a=ap.parse_args()
    plan=json.loads(a.plan.read_text()); rows={r["configuration_id"]:r for r in plan["selected_rows"]}; results=[]
    for ident in a.ids:
        c=rows[ident]; env=os.environ.copy(); env["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
        try: p=subprocess.run([sys.executable,str(Path(__file__)),"--child",json.dumps({k:c[k] for k in ("configuration_id","B","H","S","D","dtype")})],capture_output=True,text=True,timeout=a.timeout,env=env)
        except subprocess.TimeoutExpired: results.append({"configuration_id":ident,"outcome":"EXECUTION_TIMEOUT"}); continue
        parsed=None
        for line in reversed(p.stdout.splitlines()):
            try: parsed=json.loads(line); break
            except json.JSONDecodeError: pass
        results.append(parsed or {"configuration_id":ident,"outcome":"OTHER_FAILURE","stderr":p.stderr[-2000:]})
    a.output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","rows":results},indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    if "--child" in sys.argv:
        child(json.loads(sys.argv[sys.argv.index("--child")+1]))
    else: main()
