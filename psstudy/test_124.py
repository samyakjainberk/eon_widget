import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
B=dict(dataset="modadd",arch="mlp",loss="ce",act="tanh",width="20",depth="2",nsamp="24",indim="7",outdim="7",
       lr="0.3",initscheme="kaiming_normal",init="0.6",seed="0",steps="12",eigevery="2",bias="1",
       gw1="1",gw1n="20",gw1steps="3",gw2="1",gw2tau="0.7",gw4="1",gw4K="5",gw4thr="0.5",gw4t0s="0,4,8"); B.update(OFF); B["s19"]="1"
pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
g1=None; g2=[]; g4=[]
for msg in server.run_stream(P):
    if msg.get("type")=="error": print("ERR",msg); break
    if msg.get("type")=="step":
        if msg.get("g_gw1"): g1=msg["g_gw1"]
        if msg.get("g_gw2"): g2.append(msg["g_gw2"])
        if msg.get("g_gw4"): g4.append(msg["g_gw4"])
print("=== grok-① (eig-every=2, wider σ, lr-capped) ===")
if g1: 
    print("  #points:",len(g1["pts"]),"σ range:",round(g1["pts"][0]["s"],3),"→",round(g1["pts"][-1]["s"],2))
    print("  sample pt keys:",sorted(g1["pts"][0].keys()),"| lreff<=lr capped:",g1["pts"][-1]["lreff"]<=0.3+1e-9)
print("=== grok-② (eig-every=2 → lag-1 & lag-5 MUST be populated now) ===")
if g2:
    last=g2[-1]; print("  cos[k=0]:",["%.2f"%c if c is not None else None for c in last["cos"][0]])
    nonnull=sum(1 for e in g2 for c in e["cos"][0] if c is not None)
    l1=sum(1 for e in g2 if e["cos"][0][0] is not None); l5=sum(1 for e in g2 if e["cos"][0][2] is not None)
    print("  ticks with lag-1 populated:",l1,"| lag-5 populated:",l5,"(both >0 = BUG FIXED)")
    print("  count[k=0] (3 lags):",last["count"][0])
print("=== grok-④ (persistence, T0={0,4,8}) ===")
if g4:
    last=g4[-1]; print("  t=",last["t"],"k=",last["k"],"refs captured:",sorted(last["refs"].keys()))
    for T0,R in sorted(last["refs"].items(),key=lambda kv:int(kv[0])):
        print(f"    T0={T0}: top_in={R['top_in']} top_out={R['top_out']} bot_in={R['bot_in']} bot_out={R['bot_out']} (in+out={R['top_in']+R['top_out']} should=k)")
