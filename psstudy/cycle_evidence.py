#!/usr/bin/env python
"""Independent evidence that the sharpening cycle repeats — three separate measures, none using the widget's
ray-anchor threshold logic:
 (A) DIRECTION ROTATION: track the actual top eigenvector u₁ of M_r each step; count how many genuinely-new
     directions appear (greedy: a new one when |cos| with every previous anchor < 0.5, i.e. >60° apart).
     Also record the running |cos(u₁(t), u₁(t-1))| so we can see it repeatedly drop.
 (B) LOSS-SPIKE RECURRENCE: count separated upward jumps in the loss (each is one edge-of-stability event).
 (C) ALIGNMENT PEAKS: count separated peaks in the residual↔u₁ alignment (each rise = one Phase-1 build-up).
If all three give a similar, well-separated count, 'the cycle repeats N times' rests on independent legs.
"""
import sys,json,time
import numpy as np, torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server, capture_run
dev=torch.device("cuda:0"); dtype=torch.float32
server.DTYPE=dtype; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
RUNS=[dict(key="chebyshev",P=dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.1,indim=1,outdim=1,degree=3,steps=400)),
      dict(key="saddle",   P=dict(dataset="saddle",  width=16,depth=3,nsamp=64, lr=0.3,indim=4,outdim=4,steps=400)),
      dict(key="chebyshev_s1",P=dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.1,indim=1,outdim=1,degree=3,steps=400),seed=1)]
def build(cfg):
    params=capture_run.default_params()
    params.update({**{k:str(v) for k,v in cfg["P"].items()},"arch":"mlp","loss":"mse","act":"tanh","bias":"1","init":"0.5","seed":str(cfg.get("seed",0)),"initscheme":"kaiming_normal"})
    P=server._parse_params({k:[str(v)] for k,v in params.items()}); inD=int(cfg["P"]["indim"]); outD=int(cfg["P"]["outdim"])
    server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"; server._TL.loss=server.build_loss("mse")
    th,X,Y,_,_=server.init_data_theta(P,cfg["P"]["dataset"],int(cfg["P"]["nsamp"]),inD,outD); return th,X,Y,outD
def topvec(hvp,dim,m=48,seed=7):
    Qb,T,k=server._lanczos_core(hvp,dim,min(dim,m),seed); mu,Sv=np.linalg.eigh(T.cpu().numpy()); Qm=torch.stack(Qb)
    return float(mu[-1]),(torch.tensor(Sv[:,-1],dtype=Qm.dtype,device=Qm.device)@Qm)
def count_peaks(x,rel=0.15):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<5: return 0
    lo,hi=np.nanmin(x),np.nanmax(x); thr=lo+rel*(hi-lo); n=0; armed=True
    for v in x:
        if armed and v>lo+0.6*(hi-lo): n+=1; armed=False
        if v<thr: armed=True
    return n
out={}
for cfg in RUNS:
    th,X,Y,outD=build(cfg); N=X.shape[0]; p=server._TL.model.p; lr=float(cfg["P"]["lr"]); steps=int(cfg["P"]["steps"]); M=N*outD; th0=th.clone(); t0=time.time()
    frames=[]; loss=[]; align=[]; conscos=[]; anchors=[]; anchor_t=[]; prevu=None; tl=2/lr
    EVERY=2
    for t in range(steps+1):
        if t%EVERY==0:
            o=server._TL.model.forward(th,X); r=(Y-o); rc=r.reshape(N,outD); rv=r.reshape(-1)[:M]
            Jc,_=server.jac_cols(th,X); Jg=Jc[:M]; Jr=Jg.t()@rv; Jrn=float(Jr.norm())+1e-30
            server._TL.qcfg=None; lm,u1=topvec(lambda v:server.hvpS(th,X,v,rc)/N,p)
            u1=u1/(u1.norm()+1e-30)
            frames.append(t); loss.append(float(server._TL.loss.value(o,Y,N))); align.append(float(abs(float(u1@Jr))/Jrn))
            if prevu is not None: conscos.append(float(abs(float(u1@prevu))))
            else: conscos.append(1.0)
            # greedy new-direction count: new anchor if |cos| with ALL previous anchors < 0.5
            if not anchors: anchors.append(u1.clone()); anchor_t.append(t)
            else:
                mx=max(abs(float(u1@a)) for a in anchors)
                if mx<0.5: anchors.append(u1.clone()); anchor_t.append(t)
            prevu=u1.clone()
    # (B) loss spikes: separated upward jumps
    L=np.array(loss); spikes=0; armed=True
    for i in range(1,len(L)):
        if armed and L[i]>1.25*L[i-1] and L[i]>1e-3: spikes+=1; armed=False
        if i>1 and L[i]<L[i-1]: armed=True
    out[cfg["key"]]=dict(lr=lr,twoOverLr=tl,steps=steps,frames=frames,
        n_newdir=len(anchors), newdir_t=anchor_t,
        n_lossspike=int(spikes), n_alignpeak=count_peaks(align),
        conscos=conscos, align=align)
    print("  %-13s p=%d | new-directions=%d  loss-spikes=%d  align-peaks=%d  (%.0fs)"%(
        cfg["key"],p,len(anchors),spikes,count_peaks(align),time.time()-t0),flush=True)
json.dump(out,open("psstudy/cycle_evidence.json","w")); print("wrote psstudy/cycle_evidence.json",flush=True)
