import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
BASE=dict(dataset="modadd",arch="mlp",loss="ce",act="tanh",width="24",depth="2",nsamp="30",indim="11",outdim="11",
          lr="0.3",initscheme="kaiming_normal",init="0.6",seed="0",steps="2",eigevery="1",bias="1",
          gw5="1",gw5layer="all",gw6="1",gw6mode="init",gw7="1",gw7rule="star",gw7sign="plus",gw8="1",
          gw_n="4",gw_steps="40",gw_ev="10",gw_nte="48"); BASE.update(OFF); BASE["s19"]="1"
pp=capture_run.default_params(); pp.update(BASE); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
got={"g_gw5":None,"g_gw6":None,"g_gw7":None,"g_gw8":None}; err=None
for msg in server.run_stream(P):
    if msg.get("type")=="error": err=msg; break
    if msg.get("type")=="step":
        for k in got:
            if msg.get(k) is not None and got[k] is None: got[k]=msg[k]
print("error:",err)
for k,v in got.items():
    if v is None: print(f"{k}: NOT EMITTED"); continue
    runs=v["runs"]; print(f"\n{k}  axis={v.get('axis')}  extra={ {kk:v[kk] for kk in v if kk not in ('t','axis','runs')} }  #runs={len(runs)}")
    for r in runs:
        xlab=r.get('name',round(r['x'],3))
        nrec=len(r['it']); lte=r['lte'][-1] if r['lte'] else None; ate=r['ate'][-1] if r['ate'] else None
        ps=r['ps'][-1] if r['ps'] else None; al=r['align'][-1] if r['align'] else None; ra=r['ratio'][-1] if r['ratio'] else None
        print(f"   x={xlab:>6}  it={nrec:2d}  final: lte={lte}  ate={ate}  PS={ps and round(ps,3)}  align={al and round(al,3)}  ratio={ra and round(ra,4)}  grokIt={r['grokIt']}")
