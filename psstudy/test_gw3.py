import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
B=dict(dataset="ksparse",arch="mlp",loss="mse",act="tanh",width="16",depth="2",nsamp="30",indim="10",outdim="1",
       lr="0.4",initscheme="kaiming_normal",init="0.6",seed="0",steps="6",eigevery="1",bias="1",
       gw3="1",gw3rank="4",gw3k="6",gw3specevery="3"); B.update(OFF); B["s19"]="1"
pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
g3=[]
for msg in server.run_stream(P):
    if msg.get("type")=="error": print("ERR",msg);break
    if msg.get("type")=="step" and msg.get("g_gw3"): g3.append(msg["g_gw3"])
print("=== grok-③ new structure (ksparse MSE) ===")
print("  ticks emitting:",len(g3))
if g3:
    s=g3[-1]
    print("  keys:",sorted(s.keys()))
    print("  ntk keys:",sorted(s['ntk'].keys()),"| nt=",s['ntk']['nt'],"| len(n1)=",len(s['ntk']['n1']))
    print("  gn keys:",sorted(s['gn'].keys()),"| ng=",s['gn']['ng'],"| len(gp1)=",len(s['gn']['gp1']))
    print("  mr modes:",sorted(s['mr'].keys()))
    for m in s['mr']: print(f"    mr[{m}]: len(p1)={len(s['mr'][m]['p1'])} lam[0]={s['mr'][m]['lam'][0]:.4f} p1[0]={s['mr'][m]['p1'][0]:.4f}")
    print("  spec present:", 'spec' in s, "| spec modes:", sorted(s.get('spec',{}).keys()) if 'spec' in s else None)
    # sanity: ev vs fx differ (different M_r), ntk same regardless
    print("  ev vs lr p1[0] differ:", abs(s['mr']['ev']['p1'][0]-s['mr']['lr']['p1'][0])>1e-6)
