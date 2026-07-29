import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
# bspline derivative vs FD
z=torch.linspace(-4,4,60,dtype=torch.float64)
fd=(server.actf("bspline",z+1e-6)-server.actf("bspline",z-1e-6))/2e-6
print("bspline max|actd-FD| = %.2e"%float((server.actd("bspline",z)-fd).abs().max()))
# gw9 + bspline(gw8) + test-set-on-chebyshev(MSE) emission with LARGER samples
OFF={f"s{i}":"0" for i in range(1,43)}
def run(ds,loss,inD,outD,lr,**ex):
    B=dict(dataset=ds,arch="mlp",loss=loss,act="tanh",width="20",depth="2",nsamp="20",indim=str(inD),outdim=str(outD),
           lr=str(lr),initscheme="kaiming_normal",init="0.6",seed="0",steps="2",eigevery="1",bias="1",
           gw8="1",gw9="1",gw9K="6",gw9refresh="10",gw9try="6",gw_n="3",gw_steps="40",gw_ev="20",gw_nsamp="120",gw_nte="120")
    B.update(OFF); B["s19"]="1"; B.update({k:str(v) for k,v in ex.items()})
    pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    g8=g9=None
    for msg in server.run_stream(P):
        if msg.get("type")=="error": return {"err":msg}
        if msg.get("type")=="step":
            if msg.get("g_gw8"): g8=msg["g_gw8"]
            if msg.get("g_gw9"): g9=msg["g_gw9"]
    return g8,g9
for ds,loss,inD,outD,lr,ex in [("modadd","ce",7,7,0.3,{}),("chebyshev","mse",1,1,0.1,{"degree":3})]:
    g8,g9=run(ds,loss,inD,outD,lr,**ex)
    print(f"\n=== {ds} ({loss}) — LARGER samples (train=120, test=120) ===")
    if g8:
        acts=[r["name"] for r in g8["runs"]]; print("  gw8 activations:",acts,"(bspline present:", "bspline" in acts,")")
        r0=g8["runs"][0]; print(f"    test-set POPULATED? lte non-null: {r0['lte'][-1] is not None} | ltr non-null: {r0['ltr'][-1] is not None} | Ntr used gives pts={len(r0['it'])}")
    if g9:
        print("  gw9 runs:",[r["name"] for r in g9["runs"]])
        for r in g9["runs"]:
            print(f"    {r['name']}: final ltr={r['ltr'][-1]:.3f} lte={r['lte'][-1] if r['lte'][-1] is not None else None} pts={len(r['it'])}")
