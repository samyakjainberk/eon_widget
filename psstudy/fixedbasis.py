#!/usr/bin/env python
"""§6, re-plotted in a FIXED basis fixed at initialization.
At t=0 fix the eigenvectors u_1..u_K of M_r(θ0)=Σ_a r_a Q_a(θ0), ranked by their init eigenvalues (this ranking
never changes). Precompute the per-direction, per-output init curvature  Q[a,i] = u_iᵀ Q_a(θ0) u_i.
Then over training the curvature along each FIXED direction is just  d_i(t) = Σ_a r_a(t) · Q[a,i]  — fixed Q,
fixed basis, only the residual r(t) evolves. This shows how the (residual-weighted) curvature flows OUT of each
fixed init-direction as it gets learned — unlike the usual scree, rank i always means the SAME direction.
Also records: (a) the TRUE evolving curvature u_iᵀ M_r(θ_t) u_i for comparison, and (b) how much of M_r(t) lives
in the fixed top-K basis vs leaks to new directions."""
import sys,json,time
import numpy as np, torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server, capture_run
dev=torch.device("cuda:0"); dtype=torch.float64  # fp64 CPU-like precision on GPU for clean curvature
server.DTYPE=dtype; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
RUNS=[("ksparse", dict(dataset="ksparse", width=16,depth=2,nsamp=100,lr=0.5,indim=10,outdim=1,steps=400)),
      ("saddle",  dict(dataset="saddle",  width=16,depth=3,nsamp=64, lr=0.3,indim=4, outdim=4,steps=400)),
      ("chebyshev",dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.1,indim=1,outdim=1,degree=3,steps=400))]
K=20; EVERY=4
def build(P0):
    params=capture_run.default_params()
    params.update({**{k:str(v) for k,v in P0.items()},"arch":"mlp","loss":"mse","act":"tanh","bias":"1","init":"0.5","seed":"0","initscheme":"kaiming_normal"})
    P=server._parse_params({k:[str(v)] for k,v in params.items()}); inD=int(P0["indim"]); outD=int(P0["outdim"])
    server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"; server._TL.loss=server.build_loss("mse")
    th,X,Y,_,_=server.init_data_theta(P,P0["dataset"],int(P0["nsamp"]),inD,outD); return th,X,Y,outD
def topK_eig(hvp,dim,k,m=60,seed=7):
    Qb,T,kk=server._lanczos_core(hvp,dim,min(dim,m),seed); mu,Sv=np.linalg.eigh(T.cpu().numpy()); Qm=torch.stack(Qb)
    order=np.argsort(mu)[::-1][:k]
    vecs=torch.tensor(Sv[:,order].T,dtype=Qm.dtype,device=Qm.device)@Qm
    return mu[order].copy(), vecs
out={}
for key,P0 in RUNS:
    th,X,Y,outD=build(P0); N=X.shape[0]; p=server._TL.model.p; lr=float(P0["lr"]); steps=int(P0["steps"]); M=N*outD; th0=th.clone(); t0=time.time()
    # residual at init
    o=server._TL.model.forward(th,X); r0=(Y-o).reshape(-1)[:M]; rc0=(Y-o).reshape(N,outD)
    server._TL.qcfg=None
    lam0,U=topK_eig(lambda v:server.hvpS(th,X,v,rc0),p,K)   # top-K eigpairs of M_r(θ0) (NOT /N here; absolute)
    # precompute Q[a,i] = u_iᵀ Q_a(θ0) u_i  (K × M)
    Qmat=np.zeros((K,M))
    for i in range(K):
        ui=U[i]; Qui=server.jac_hvp(th0,X,ui)[:M]          # M×p, row a = Q_a·u_i
        Qmat[i]=(Qui@ui).detach().cpu().numpy()             # u_iᵀ Q_a u_i for each output a
    frames=[];loss=[];dfix=[];dtrue=[];captured=[];lammax=[]
    for t in range(steps+1):
        if t%EVERY==0:
            o=server._TL.model.forward(th,X); rr=(Y-o); r=rr.reshape(-1)[:M].detach().cpu().numpy(); rc=rr.reshape(N,outD)
            frames.append(t); loss.append(float(server._TL.loss.value(o,Y,N)))
            dfix.append((Qmat@r).tolist())                  # d_i(t)=Σ_a r_a(t)·Q[a,i]  (fixed Q, fixed basis)
            # true evolving curvature along each fixed u_i: u_iᵀ M_r(θ_t) u_i
            server._TL.qcfg=None
            dt=[float(U[i]@server.hvpS(th,X,U[i],rc)) for i in range(K)]; dtrue.append(dt)
            # how much of M_r(θ_t) energy is in the fixed basis: ||P_U M_r u|| type — use λmax and captured ratio
            lm,_=topK_eig(lambda v:server.hvpS(th,X,v,rc),p,1); lammax.append(float(lm[0]))
        if t<steps:
            g,_=server.gradL(th,X,Y); th=th-lr*g
    out[key]=dict(lr=lr,twoOverLr=2/lr,steps=steps,K=K,frames=frames,loss=loss,
                  lam0=lam0.tolist(), dfix=dfix, dtrue=dtrue, lammax=lammax)
    print("  %-10s p=%d K=%d frames=%d λ0[top3]=%s (%.0fs)"%(key,p,K,len(frames),[round(x,3) for x in lam0[:3]],time.time()-t0),flush=True)
json.dump(out,open("psstudy/fixedbasis.json","w")); print("wrote psstudy/fixedbasis.json",flush=True)
