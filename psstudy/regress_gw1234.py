import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
B=dict(dataset="maxfind",arch="mlp",loss="ce",act="tanh",width="16",depth="2",nsamp="16",indim="10",outdim="10",
       lr="0.2",initscheme="kaiming_normal",init="0.6",seed="0",steps="5",eigevery="1",bias="1",
       gw1="1",gw1n="6",gw1steps="2",gw2="1",gw3="1",gw3rank="4",gw4="1",gw4K="8"); B.update(OFF); B["s19"]="1"; B["s6"]="1"
pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
seen={"g_gw1":0,"g_gw2":0,"g_gw3":0,"g_gw4":0}; err=None
for msg in server.run_stream(P):
    if msg.get("type")=="error": err=msg; break
    if msg.get("type")=="step":
        for k in seen:
            if msg.get(k) is not None: seen[k]+=1
print(json.dumps({"REGRESSION_old_panels":seen,"error":err,"all_ok":all(v>0 for v in seen.values()) and err is None}))
