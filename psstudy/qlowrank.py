#!/usr/bin/env python
"""EXPERIMENT: is it low-RANK that makes the true function-Hessian special, or the specific DIRECTION of its
low-rank subspace? At several checkpoints along the (identical) GD run, for each dataset we compare, as the
operator M_r that the theory uses:
  true      : the real evolving M_r = (1/N)Σ_k r_k ∇²f_k(θ_t)         [right rank, right directions]
  initfix   : real M_r frozen at θ_0                                   [right directions, drifting scale]
  randfull  : random symmetric Q, Frobenius-matched (my earlier test) [full rank, spread out]
  randlr-R  : RANDOM but LOW-RANK operator of rank R, built with the TRUE top-R eigenvalues but RANDOM
              orthonormal eigenvectors — i.e. identical spectrum/rank/concentration, wrong directions.
For each we report λ_max and the alignment |cos(Jᵀr, u₁)| of its top eigenvector with the GN-gradient.
If randlr recovers a large λ_max but NOT the alignment, then low-rank explains the eigenvalue but the
DIRECTION (which subspace) is what actually matters.
"""
import os,sys,json,time
import numpy as np, torch
DIR="/nas/ucb/samsj/TestingPSTheory/eos_widget"; sys.path.insert(0,DIR)
import server, capture_run
dev=torch.device("cuda:0"); dtype=torch.float32
server.DTYPE=dtype; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
RUNS=[dict(key="ksparse",  P=dict(dataset="ksparse", width=16,depth=2,nsamp=100,lr=0.5,indim=10,outdim=1,steps=400)),
      dict(key="saddle",   P=dict(dataset="saddle",  width=16,depth=3,nsamp=64, lr=0.3,indim=4, outdim=4,steps=400)),
      dict(key="chebyshev",P=dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.1,indim=1,outdim=1,degree=3,steps=400))]
RANKS=[1,2,4,8,16]
def build(cfg):
    params=capture_run.default_params()
    params.update({**{k:str(v) for k,v in cfg["P"].items()},"arch":"mlp","loss":"mse","act":"tanh","bias":"1","init":"0.5","seed":"0","initscheme":"kaiming_normal"})
    P=server._parse_params({k:[str(v)] for k,v in params.items()}); inD=int(cfg["P"]["indim"]); outD=int(cfg["P"]["outdim"])
    server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"; server._TL.loss=server.build_loss("mse")
    th,X,Y,_,_=server.init_data_theta(P,cfg["P"]["dataset"],int(cfg["P"]["nsamp"]),inD,outD); return th,X,Y,outD
def topk_eig(hvp,dim,k=16,m=60,seed=7):
    Qb,T,kk=server._lanczos_core(hvp,dim,min(dim,m),seed); mu,Sv=np.linalg.eigh(T.cpu().numpy())
    Qm=torch.stack(Qb)  # (kk, dim)
    order=np.argsort(mu)[::-1][:k]
    vecs=torch.tensor(Sv[:,order].T,dtype=Qm.dtype,device=Qm.device)@Qm  # (k, dim) top-k Ritz vectors
    return mu[order].copy(), vecs   # eigenvalues (desc), eigenvectors rows
def frob(hvp,p,probes=32,seed=0):
    g=torch.Generator(device=dev).manual_seed(seed); tot=0.0
    for i in range(probes):
        v=(torch.randint(0,2,(p,),generator=g,device=dev,dtype=torch.float32)*2-1); Av=hvp(v); tot+=float((Av*Av).sum())
    return (tot/probes)**0.5
out={}
for cfg in RUNS:
    th,X,Y,outD=build(cfg); N=X.shape[0]; p=server._TL.model.p; lr=float(cfg["P"]["lr"]); steps=int(cfg["P"]["steps"]); M=N*outD; th0=th.clone(); t0=time.time()
    QRAND=server._build_randQ(th0,X,M,"gauss",12345,4)
    ckpts=[int(f*steps) for f in (0.3,0.5,0.7,0.9)]
    rec={"lr":lr,"twoOverLr":2/lr,"P":p,"ckpts":[],"ranks":RANKS}
    gen=torch.Generator(device=dev).manual_seed(2024)
    for t in range(steps+1):
        if t in ckpts:
            o=server._TL.model.forward(th,X); r=(Y-o); rc=r.reshape(N,outD); rv=r.reshape(-1)[:M]
            Jc,_=server.jac_cols(th,X); Jg=Jc[:M]; Jr=Jg.t()@rv; Jrn=float(Jr.norm())+1e-30
            def align(u): return float(abs(float(u@Jr))/Jrn)
            entry={"t":t}
            # true evolving M_r: top-k eigenpairs + Frobenius
            server._TL.qcfg=None; hvpT=lambda v: server.hvpS(th,X,v,rc)/N
            muT,vecsT=topk_eig(hvpT,p,k=max(RANKS)); frT=frob(hvpT,p)
            entry["true"]={"lmax":float(muT[0]),"align":align(vecsT[0]),"frob":float(frT)}
            # init-fixed
            server._TL.qcfg={"mode":"fix","theta_t":th0,"Qrand":None}; hvpF=lambda v: server.hvpS(th,X,v,rc)/N
            muF,vecsF=topk_eig(hvpF,p,k=1); entry["initfix"]={"lmax":float(muF[0]),"align":align(vecsF[0])}
            # random full-rank (gauss)
            server._TL.qcfg={"mode":"gauss","theta_t":th0,"M":M,"seed":12345,"probes":4,"Qrand":QRAND}; hvpR=lambda v: server.hvpS(th,X,v,rc)/N
            muR,vecsR=topk_eig(hvpR,p,k=1); entry["randfull"]={"lmax":float(muR[0]),"align":align(vecsR[0])}
            server._TL.qcfg=None
            # RANDOM LOW-RANK: true top-R eigenvalues, random orthonormal eigenvectors (matched spectrum, wrong directions)
            entry["randlr"]={}
            for R in RANKS:
                lam=torch.tensor(muT[:R],dtype=dtype,device=dev)                     # true top-R eigenvalues
                G=torch.randn(p,R,generator=gen,device=dev,dtype=dtype); U,_=torch.linalg.qr(G)  # random orthonormal p×R
                # top eigenvector = column of U paired with largest |lambda|; lam is descending so U[:,0]
                u1=U[:,0]
                # M_r_rlr λ_max = max lam (=muT[0]); alignment of its top eigvec with Jr:
                entry["randlr"][str(R)]={"lmax":float(muT[0]),"align":align(u1),
                    "frob":float(torch.linalg.vector_norm(lam))}
            rec["ckpts"].append(entry)
        if t<steps:
            g,_=server.gradL(th,X,Y); th=th-lr*g
    out[cfg["key"]]=rec; print("  %-10s p=%d ckpts=%d (%.0fs)"%(cfg["key"],p,len(rec["ckpts"]),time.time()-t0),flush=True)
json.dump(out,open("psstudy/qlowrank.json","w")); print("wrote psstudy/qlowrank.json",flush=True)
