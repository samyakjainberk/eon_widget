import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
for ee in [1,2]:
    B=dict(dataset="modadd",arch="mlp",loss="ce",act="tanh",width="16",depth="2",nsamp="24",indim="7",outdim="7",
           lr="0.3",initscheme="kaiming_normal",init="0.6",seed="0",steps="12",eigevery=str(ee),bias="1",gw2="1",gw2tau="0.7")
    B.update(OFF); B["s19"]="1"
    pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    g2=[]
    for msg in server.run_stream(P):
        if msg.get("type")=="step" and msg.get("g_gw2"): g2.append(msg["g_gw2"])
    l1=sum(1 for e in g2 if e["cos"][0][0] is not None)
    l2=sum(1 for e in g2 if e["cos"][0][1] is not None)
    l5=sum(1 for e in g2 if e["cos"][0][2] is not None)
    print(f"eig-every={ee}: g_gw2 ticks={len(g2)} | lag-1 populated={l1} lag-2={l2} lag-5={l5} | last count[k0]={g2[-1]['count'][0] if g2 else None}")
