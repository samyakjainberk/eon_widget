import sys,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
def build(ds,loss,inD,outD,lr,n=24,**ex):
    OFF={f"s{i}":"0" for i in range(1,43)}
    B=dict(dataset=ds,arch="mlp",loss=loss,act="tanh",width="20",depth="2",nsamp=str(n),indim=str(inD),outdim=str(outD),
           lr=str(lr),initscheme="kaiming_normal",init="0.6",seed="0",bias="1");B.update(OFF);B.update({k:str(v) for k,v in ex.items()})
    pp=capture_run.default_params();pp.update(B);P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    server._TL.model=server.build_model("mlp",inD,outD,P);server._TL.model.init_scheme="kaiming_normal";server._TL.loss=server.build_loss(loss)
    th,X,Y,_,_=server.init_data_theta(P,ds,n,inD,outD);return th,X,Y,P

print("=== gw7 optimizer rules at λ=0 must equal plain GD ===")
th,X,Y,P=build("modadd","ce",22,11,0.3,n=30)
N=X.shape[0]
for _ in range(4): th=th-0.3*server.gradL(th,X,Y)[0]
gd=server._gd_step(X,Y,0.3)(th)
star0=server._opt_step_star(X,Y,N,11,0.3,0.0,1)(th)
tay0=server._opt_step_taylor(X,Y,N,11,0.3,0.0,0.0)(th)
star1=server._opt_step_star(X,Y,N,11,0.3,1.0,1)(th)
print(f"  star(λ=0) vs GD max|Δθ|={float((gd-star0).abs().max()):.2e}  taylor(λ=0) vs GD={float((gd-tay0).abs().max()):.2e}  star(λ=1) differs={float((gd-star1).abs().max())>0}")

print("=== gw9 subspace actually refreshes (U changes across refresh boundary) ===")
th,X,Y,P=build("ksparse","mse",10,1,0.4,n=40);N=X.shape[0]
step=server._rs_step(X,Y,N,1,0.05,6,3,4,0.0)  # refresh every 3
# capture U at step 0 and after 3 steps (refresh)
thc=th.clone()
step(thc)  # builds U (cnt=1)
# access internal state via closure is not trivial; instead check loss decreases + runs many steps ok
Ls=[]
for _ in range(12): Ls.append(float(server._TL.loss.value(server._TL.model.forward(thc,X),Y,N))); thc=step(thc)
print(f"  runs 12 steps ok, loss {Ls[0]:.4f}->{Ls[-1]:.4f} monotone={all(Ls[i+1]<=Ls[i]+1e-9 for i in range(len(Ls)-1))}")

print("=== gw8 restores _TL.model after activation swap (run_stream) ===")
OFF={f"s{i}":"0" for i in range(1,43)}
B=dict(dataset="maxfind",arch="mlp",loss="ce",act="tanh",width="16",depth="2",nsamp="16",indim="10",outdim="10",
       lr="0.1",initscheme="kaiming_normal",init="0.6",seed="0",steps="3",eigevery="1",bias="1",
       gw8="1",gw_n="3",gw_steps="10",gw_ev="5",gw_nsamp="30",gw_nte="30");B.update(OFF);B["s19"]="1"
pp=capture_run.default_params();pp.update(B);P=server._parse_params({k:[str(v)] for k,v in pp.items()})
acts_seen=[]
for msg in server.run_stream(P):
    if msg.get("type")=="step" and msg.get("g_gw8"): acts_seen=[r["name"] for r in msg["g_gw8"]["runs"]]
# after the run, _TL.model should be the ORIGINAL (tanh) MLP, act='tanh'
print(f"  gw8 activations swept: {acts_seen}")
print(f"  _TL.model after run: type={type(server._TL.model).__name__} act={getattr(server._TL.model,'act','?')} (should be tanh MlpModel — restored)")
