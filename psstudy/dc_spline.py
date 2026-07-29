import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
# 1) SplineMlpModel basic: build, forward, param count, train
P={"width":"24","depth":"2","bias":"1","init":"0.6","seed":"0","act":"bspline"}
m=server.SplineMlpModel(10,24,2,10,True,nc=50,srange=5.0); m.init_scheme="kaiming_normal"
server._TL.model=m; server._TL.loss=server.build_loss("ce")
th=m.init_theta(1,0.6)
print("SplineMlp p =",m.p,"(incl psi:",sum(n.startswith('psi') for n,*_ in m._specs),"splines × 50)")
X=torch.randn(20,10,dtype=torch.float64,device=dev); idx=X.argmax(1); Y=torch.zeros(20,10,dtype=torch.float64,device=dev); Y[torch.arange(20),idx]=1.0
o=m.forward(th,X); print("forward out shape:",tuple(o.shape))
L0=float(server._TL.loss.value(o,Y,20))
for _ in range(80): th=th-0.3*server.gradL(th,X,Y)[0]
L1=float(server._TL.loss.value(m.forward(th,X),Y,20))
print(f"train (spline net): loss {L0:.4f} -> {L1:.4f}  learns={L1<L0}")
# hvpS works (exact)?
o=m.forward(th,X); rc=(-20*server._TL.loss.resid_cotangent(o,Y,20)).reshape(20,10)
v=torch.randn(m.p,dtype=torch.float64,device=dev); hv=server.hvpS(th,X,v,rc)
print("hvpS works, ‖hvpS·v‖=%.4f"%float(hv.norm()))

# 2) grok-⑧ via run_stream — bspline run uses the spline model + emits
OFF={f"s{i}":"0" for i in range(1,43)}
B=dict(dataset="maxfind",arch="mlp",loss="ce",act="tanh",width="24",depth="2",nsamp="16",indim="10",outdim="10",
       lr="0.1",initscheme="kaiming_normal",init="0.6",seed="0",steps="3",eigevery="1",bias="1",
       gw8="1",gw_n="3",gw_steps="60",gw_ev="20",gw_nsamp="60",gw_nte="60",gw8init="0.5")
B.update(OFF); B["s19"]="1"
pp=capture_run.default_params(); pp.update(B); P2=server._parse_params({k:[str(v)] for k,v in pp.items()})
g8=None
for msg in server.run_stream(P2):
    if msg.get("type")=="error": print("ERR",msg);break
    if msg.get("type")=="step" and msg.get("g_gw8"): g8=msg["g_gw8"]
if g8:
    for r in g8["runs"]:
        ltr=r["ltr"]; print(f"  gw8 [{r['name']:8s}]: train loss {ltr[0]:.3f}->{ltr[-1]:.3f} test_acc {r['ate'][-1]:.3f} pts={len(r['it'])}")
    print("  bspline present:", any(r["name"]=="bspline" for r in g8["runs"]))
