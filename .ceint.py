import sys, math, torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server as S
def finite(x):
    if x is None: return True
    if isinstance(x,float): return math.isfinite(x)
    if isinstance(x,(int,bool,str)): return True
    if isinstance(x,list): return all(finite(v) for v in x)
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    return True
q={"loss":["ce"],"dataset":["cifar10"],"arch":["mlp"],"act":["tanh"],"width":["12"],"depth":["2"],
   "nsamp":["3"],"lr":["0.05"],"init":["0.4"],"seed":["1"],"steps":["5"],"eigevery":["1"],"opt":["gd"],
   "s36":["1"],"s37":["1"],"s38":["1"],"s39":["1"],"s41":["1"],"s42":["1"],"s29":["1"],
   "prthr":["0.4"],"edsmrk":["12"],"edsangk":["4"]}
import time; t0=time.time()
P=S._parse_params(q)
keys=["g_pred3","g_pred4","g_ray","g_trace","g_pred6","g_eds","g21"]
seen={k:0 for k in keys}; bad={k:0 for k in keys}; err=None
for ch in S.run_stream(P):
    if not isinstance(ch,dict): continue
    if ch.get("error"): err=ch.get("error")
    for k in keys:
        v=ch.get(k)
        if v is not None:
            seen[k]+=1
            if not finite(v): bad[k]+=1
print(f"CE cifar10 run ({time.time()-t0:.0f}s, cuda={torch.cuda.is_available()}):", "ERR="+str(err) if err else "completed")
for k in keys: print(f"  {k:9s}: seen {seen[k]:3d} nonfinite {bad[k]}")
print("CE-INT", "PASS" if (not err and sum(bad.values())==0 and sum(seen.values())>0) else "CHECK")
