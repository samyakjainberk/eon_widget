import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
for ds,loss,inD,outD,lr,ex in [("ksparse","mse",10,1,0.4,{}),("chebyshev","mse",1,1,0.1,{"degree":3}),("modadd","ce",11,11,0.3,{})]:
    B=dict(dataset=ds,arch="mlp",loss=loss,act="tanh",width="20",depth="2",nsamp="20",indim=str(inD),outdim=str(outD),
           lr=str(lr),initscheme="kaiming_normal",init="0.6",seed="0",steps="2",eigevery="1",bias="1",
           gw5="1",gw1="1",gw1n="12",gw1steps="3",gw_n="3",gw_steps="30",gw_ev="15",gw_nsamp="60",gw_nte="60")
    B.update(OFF); B["s19"]="1"; B.update({k:str(v) for k,v in ex.items()})
    pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    g5=g1=None
    for msg in server.run_stream(P):
        if msg.get("type")=="step":
            if msg.get("g_gw5"): g5=msg["g_gw5"]
            if msg.get("g_gw1"): g1=msg["g_gw1"]
    r0=g5["runs"][0] if g5 else None
    accok = r0 and r0["ate"][-1] is not None and r0["atr"][-1] is not None
    # grok-1: lr_eff should be << lr (0.2/sharp)
    lreff = g1["pts"][len(g1["pts"])//2].get("lreff") if g1 else None
    print(f"{ds:10s} ({loss}): gw5 accuracy populated (ate,atr non-null)={accok}  ate={r0['ate'][-1] if r0 else None}  | grok-① lreff(mid σ)={lreff:.2e} (<< lr={lr}: {lreff<lr if lreff else '?'})")
