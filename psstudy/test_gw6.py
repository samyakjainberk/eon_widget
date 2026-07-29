import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
def run(mode):
    B=dict(dataset="modadd",arch="mlp",loss="ce",act="tanh",width="20",depth="2",nsamp="24",indim="7",outdim="7",
           lr="0.3",initscheme="kaiming_normal",init="0.6",seed="0",steps="2",eigevery="1",bias="1",
           gw6="1",gw6mode=mode,gw_n="4",gw_steps="60",gw_ev="15",gw_nte="40"); B.update(OFF); B["s19"]="1"
    pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    for msg in server.run_stream(P):
        if msg.get("type")=="error": return {"err":msg}
        if msg.get("type")=="step" and msg.get("g_gw6"): return msg["g_gw6"]
    return None
for mode in ["init","interval"]:
    g=run(mode)
    print(f"=== gw6 mode={mode} (target-label scaling) ===")
    if not g or "err" in (g or {}): print("  FAIL",g); continue
    print("  axis:",g["axis"],"| mode:",g["mode"],"| #runs:",len(g["runs"]))
    for r in g["runs"]:
        print(f"    a={r['x']:.3f}: pts={len(r['it'])} final ltr={r['ltr'][-1]:.3f} lte={r['lte'][-1]:.3f} ate={r['ate'][-1]:.3f} atr={r['atr'][-1]:.3f} keys_ok={all(k in r for k in ('ltr','lte','atr','ate','ps','align','ratio'))}")
