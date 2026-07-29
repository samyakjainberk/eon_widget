# Deep validation of the 4 page-1 intervention panels (gw5-8) for one config. Emits PASS/FAIL + sanity summary.
import sys,json,math,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
cfg=json.loads(sys.argv[1])
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir="/nas/ucb/samsj/data" if cfg.get("dataset") in ("cifar10","mnist") else None
torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
BASE=dict(arch="mlp",act="tanh",initscheme="kaiming_normal",init="0.6",seed="0",steps="2",eigevery="1",bias="1",
          gw5="1",gw6="1",gw7="1",gw8="1",gw_n="4",gw_steps="60",gw_ev="12",gw_nte="48")
BASE.update(OFF); BASE["s19"]="1"; BASE.update(cfg)
pp=capture_run.default_params(); pp.update(BASE); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
tag=cfg.get("_tag","?")
issues=[]; got={"g_gw5":None,"g_gw6":None,"g_gw7":None,"g_gw8":None}; err=None
try:
    for msg in server.run_stream(P):
        if msg.get("type")=="error": err=msg; break
        if msg.get("type")=="step":
            for k in got:
                if msg.get(k) is not None and got[k] is None: got[k]=msg[k]
except Exception as e:
    import traceback; err={"exc":traceback.format_exc()}
if err: print(json.dumps({"tag":tag,"PASS":False,"error":str(err)[:400]})); sys.exit(0)
def chk(name,v):
    if v is None: issues.append(f"{name}:NOT_EMITTED"); return
    runs=v.get("runs",[])
    if not runs: issues.append(f"{name}:EMPTY_RUNS"); return
    for r in runs:
        for key in ("ps","align","ratio"):
            arr=[x for x in (r.get(key) or []) if x is not None]
            if any((x!=x or math.isinf(x)) for x in arr): issues.append(f"{name}:{key}:NAN/INF")
        al=[x for x in (r.get("align") or []) if x is not None]
        if any(x<-1e-6 or x>1.001 for x in al): issues.append(f"{name}:align_out_of_[0,1]")
        ps=[x for x in (r.get("ps") or []) if x is not None]
        if any(x<-1e-3 for x in ps): issues.append(f"{name}:PS_negative")
        ra=[x for x in (r.get("ratio") or []) if x is not None]
        if any(x<-1e-9 for x in ra): issues.append(f"{name}:ratio_negative")
        it=r.get("it") or []
        if it!=sorted(it): issues.append(f"{name}:it_not_monotone")
        if any((x!=x) for x in (r.get("ltr") or []) if x is not None): issues.append(f"{name}:ltr_NAN")
for k in got: chk(k,got[k])
# structural: gw7 carries rule/sign, gw8 runs carry names, gw5 carries layer
if got["g_gw7"] and "rule" not in got["g_gw7"]: issues.append("gw7:no_rule")
if got["g_gw8"] and not all("name" in r for r in got["g_gw8"]["runs"]): issues.append("gw8:no_names")
summ={k:(len(v["runs"]) if v else 0) for k,v in got.items()}
print(json.dumps({"tag":tag,"PASS":len(issues)==0,"issues":issues,"nruns":summ,
   "gw8_acts":[r.get("name") for r in got["g_gw8"]["runs"]] if got["g_gw8"] else None}))
