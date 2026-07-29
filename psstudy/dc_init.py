import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
def run(extra):
    B=dict(dataset="modadd",arch="mlp",loss="ce",act="tanh",width="16",depth="2",nsamp="22",indim="11",outdim="11",
           lr="0.3",initscheme="kaiming_normal",init="0.6",seed="0",steps="4",eigevery="1",bias="1",
           gw2="1",gw3="1",gw3k="5",gw4="1",gw4K="6",gw4t0s="0,2",gw6="1",gw_n="3",gw_steps="20",gw_ev="10",gw_nsamp="50",gw_nte="50")
    B.update(OFF); B["s19"]="1"; B.update(extra)
    pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    got={f"g_gw{i}":None for i in [2,3,4,6]}
    for msg in server.run_stream(P):
        if msg.get("type")=="error": return {"err":msg}
        if msg.get("type")=="step":
            for k in got:
                if msg.get(k) is not None: got[k]=msg[k]
    return got
a=run({"gwdiaginit":"0.3","gw6init":"0.3"})   # shadow ON, small interv init
print("shadow ON: all emit:", all(a.get(k) is not None for k in ["g_gw2","g_gw3","g_gw4","g_gw6"]), "err:", a.get("err"))
b=run({"gwdiaginit":"1.0","gw6init":"1.0"})   # main run, large init
def ntk0(g): return g["g_gw3"]["ntk"]["n1"][0] if g and g.get("g_gw3") else None
print(f"grok-3 NTK n1[0]: shadow(0.3)={ntk0(a):.5f}  main(1.0)={ntk0(b):.5f}  differ(shadow works)={abs(ntk0(a)-ntk0(b))>1e-6}")
ps_s=a["g_gw6"]["runs"][0]["ps"][0]; ps_l=b["g_gw6"]["runs"][0]["ps"][0]
print(f"grok-6 PS[0]: gw6init=0.3 ⇒ {ps_s:.4f}  vs gw6init=1.0 ⇒ {ps_l:.4f}  differ(init-scale works)={abs(ps_s-ps_l)>1e-4}")
# sanity: main-run th not corrupted by shadow — grok-6 with shadow-on vs shadow-off gives SAME interv result (interv uses θ₀, not shadow)
c=run({"gwdiaginit":"1.0","gw6init":"0.3"})  # shadow OFF but same interv init
ps_c=c["g_gw6"]["runs"][0]["ps"][0]
print(f"grok-6 PS[0] shadow-ON(0.3)={ps_s:.4f} vs shadow-OFF(0.3)={ps_c:.4f}  same(interv NOT affected by shadow)={abs(ps_s-ps_c)<1e-6}")
