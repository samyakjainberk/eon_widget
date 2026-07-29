import sys,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
BASE=dict(dataset="maxfind",arch="mlp",loss="ce",act="tanh",width="14",depth="2",nsamp="14",indim="10",outdim="10",
          lr="0.2",initscheme="kaiming_normal",init="0.6",seed="0",steps="6",eigevery="1",bias="1",
          gw1="1",gw1n="6",gw1steps="2",gw2="1",gw2tau="0.7",gw3="1",gw3rank="4",gw4="1",gw4K="8",gw3specevery="3")
BASE.update(OFF); BASE["s19"]="1"
pp=capture_run.default_params(); pp.update(BASE); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
seen={"g_gw1":0,"g_gw2":0,"g_gw3":0,"g_gw4":0}; spec=0; err=None
for msg in server.run_stream(P):
    if msg.get("type")=="error": err=msg; break
    if msg.get("type")=="step":
        for k in seen:
            if msg.get(k) is not None: seen[k]+=1
        if msg.get("g_gw3") and "spec" in msg["g_gw3"]: spec+=1
print("FINAL EMISSION CHECK (fixed code, CE maxfind):")
print("  error:",err)
for k in seen: print("  %s emitted on %d ticks"%(k,seen[k]))
print("  gw3 SLQ-spectrum snapshots:",spec)
print("  ALL FOUR PANELS EMIT:", all(v>0 for v in seen.values()))
