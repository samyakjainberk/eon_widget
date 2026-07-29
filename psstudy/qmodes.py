#!/usr/bin/env python
"""Deep three-way Q-mode analysis. For each dataset, along the (identical) GD trajectory, compute the
residual-weighted curvature M_r = (1/N) Σ_k r_k Q_k under five Q modes and record how they diverge:
  evolve   : Q_k = ∇²f_k(θ_t)          — the true, evolving function Hessian
  initfix  : Q_k = ∇²f_k(θ_0)          — true Q frozen at initialization
  gauss/bern/unif : random symmetric Q drawn at θ_0, Frobenius-matched to the true M_r (random STRUCTURE)
Records per frame: λ_max(M_r), λ_min(M_r), and |cos(Jᵀr, u₁)| (residual ↔ top-M_r-eigenvector alignment).
"""
import os,sys,json,argparse,time
import numpy as np, torch
DIR="/nas/ucb/samsj/TestingPSTheory/eos_widget"; sys.path.insert(0,DIR)
import server, capture_run

ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--every",type=int,default=5)
ap.add_argument("--out",default="psstudy/qmodes.json"); A=ap.parse_args()
dev=torch.device(A.device); dtype=torch.float32 if dev.type=="cuda" else torch.float64
server.DTYPE=dtype; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None
if dev.type=="cuda": torch.cuda.set_device(dev)

RUNS=[dict(key="chebyshev",P=dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.1,indim=1,outdim=1,degree=3,steps=400)),
      dict(key="ksparse",  P=dict(dataset="ksparse",width=16,depth=2,nsamp=100,lr=0.5,indim=10,outdim=1,steps=400)),
      dict(key="saddle",   P=dict(dataset="saddle",width=16,depth=3,nsamp=64,lr=0.3,indim=4,outdim=4,steps=400))]
MODES=["evolve","initfix","gauss","bern","unif"]

def build(cfg):
    params=capture_run.default_params()
    params.update({**{k:str(v) for k,v in cfg["P"].items()},"arch":"mlp","loss":"mse","act":"tanh","bias":"1","init":"0.5","seed":"0","initscheme":"kaiming_normal"})
    P=server._parse_params({k:[str(v)] for k,v in params.items()})
    inD=int(cfg["P"]["indim"]); outD=int(cfg["P"]["outdim"])
    server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"
    server._TL.loss=server.build_loss("mse")
    th,X,Y,_,_=server.init_data_theta(P,cfg["P"]["dataset"],int(cfg["P"]["nsamp"]),inD,outD)
    return th,X,Y,outD

def lanczos(hvp,dim,m=48,seed=7):
    Qb,T,k=server._lanczos_core(hvp,dim,min(dim,m),seed)
    mu,Sv=np.linalg.eigh(T.cpu().numpy()); Qm=torch.stack(Qb)
    u1=torch.tensor(Sv[:,-1],dtype=Qm.dtype,device=Qm.device)@Qm
    return float(mu[-1]),float(mu[0]),u1

out={}
for cfg in RUNS:
    th,X,Y,outD=build(cfg); N=X.shape[0]; p=server._TL.model.p; lr=float(cfg["P"]["lr"]); steps=int(cfg["P"]["steps"]); M=N*outD
    th0=th.clone(); t0=time.time()
    QR={m:{} for m in ("gauss","bern","unif")}
    for m in QR: QR[m]=server._build_randQ(th0,X,M,m,12345,4)
    modecfg={"evolve":None,"initfix":{"mode":"fix","theta_t":th0,"Qrand":None},
             "gauss":{"mode":"gauss","theta_t":th0,"M":M,"seed":12345,"probes":4,"Qrand":QR["gauss"]},
             "bern": {"mode":"bern", "theta_t":th0,"M":M,"seed":12345,"probes":4,"Qrand":QR["bern"]},
             "unif": {"mode":"unif", "theta_t":th0,"M":M,"seed":12345,"probes":4,"Qrand":QR["unif"]}}
    rec={"frames":[],"lr":lr,"twoOverLr":2/lr,"P":p,
         "top":{m:[] for m in MODES},"bot":{m:[] for m in MODES},"align":{m:[] for m in MODES},"loss":[],"sharp":[]}
    for t in range(steps+1):
        if t%A.every==0:
            o=server._TL.model.forward(th,X); r=(Y-o); rc=r.reshape(N,outD); rv=r.reshape(-1)[:M]
            Jc,_=server.jac_cols(th,X); Jg=Jc[:M]; Jr=Jg.t()@rv; Jrn=float(Jr.norm())+1e-30
            rec["frames"].append(t); rec["loss"].append(float(server._TL.loss.value(o,Y,N)))
            rec["sharp"].append(lanczos(lambda v:server.hvpL(th,X,Y,v),p)[0])
            for m in MODES:
                server._TL.qcfg=modecfg[m]
                tv,bv,u1=lanczos(lambda v:server.hvpS(th,X,v,rc)/N,p)
                rec["top"][m].append(tv); rec["bot"][m].append(bv); rec["align"][m].append(float(abs(float(u1@Jr))/Jrn))
            server._TL.qcfg=None
        if t<steps:
            g,_=server.gradL(th,X,Y); th=th-lr*g
    out[cfg["key"]]=rec
    print("  %-10s p=%d frames=%d (%.0fs)"%(cfg["key"],p,len(rec["frames"]),time.time()-t0),flush=True)
json.dump(out,open(A.out,"w")); print("wrote",A.out,flush=True)
