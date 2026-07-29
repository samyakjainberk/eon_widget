#!/usr/bin/env python
"""Independent cycle evidence, clean version. Track the true top eigenvector u1 of M_r each step; save it so
we can measure genuine direction rotation offline. Also save loss, sharpness, lambda_max, alignment."""
import sys,json,time
import numpy as np, torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server, capture_run
dev=torch.device("cuda:0"); dtype=torch.float32
server.DTYPE=dtype; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
RUNS=[("cheb_lr0.05",dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.05,indim=1,outdim=1,degree=3,steps=500),0),
      ("cheb_lr0.1", dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.1, indim=1,outdim=1,degree=3,steps=500),0),
      ("saddle_lr0.3",dict(dataset="saddle",  width=16,depth=3,nsamp=64, lr=0.3, indim=4,outdim=4,steps=600),0),
      ("cheb_lr0.05_s1",dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.05,indim=1,outdim=1,degree=3,steps=500),1)]
def build(P0,seed):
    params=capture_run.default_params()
    params.update({**{k:str(v) for k,v in P0.items()},"arch":"mlp","loss":"mse","act":"tanh","bias":"1","init":"0.5","seed":str(seed),"initscheme":"kaiming_normal"})
    P=server._parse_params({k:[str(v)] for k,v in params.items()}); inD=int(P0["indim"]); outD=int(P0["outdim"])
    server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"; server._TL.loss=server.build_loss("mse")
    th,X,Y,_,_=server.init_data_theta(P,P0["dataset"],int(P0["nsamp"]),inD,outD); return th,X,Y,outD
def topvec(hvp,dim,m=48,seed=7):
    Qb,T,k=server._lanczos_core(hvp,dim,min(dim,m),seed); mu,Sv=np.linalg.eigh(T.cpu().numpy()); Qm=torch.stack(Qb)
    return float(mu[-1]),(torch.tensor(Sv[:,-1],dtype=Qm.dtype,device=Qm.device)@Qm)
out={}
for key,P0,seed in RUNS:
    th,X,Y,outD=build(P0,seed); N=X.shape[0]; p=server._TL.model.p; lr=float(P0["lr"]); steps=int(P0["steps"]); M=N*outD; t0=time.time()
    EVERY=2
    frames=[];loss=[];sharp=[];lmax=[];align=[];U=[]
    for t in range(steps+1):
        if t%EVERY==0:
            o=server._TL.model.forward(th,X); r=(Y-o); rc=r.reshape(N,outD); rv=r.reshape(-1)[:M]
            Jc,_=server.jac_cols(th,X); Jg=Jc[:M]; Jr=Jg.t()@rv; Jrn=float(Jr.norm())+1e-30
            sh,_=topvec(lambda v:server.hvpL(th,X,Y,v),p)
            server._TL.qcfg=None; lm,u1=topvec(lambda v:server.hvpS(th,X,v,rc)/N,p); u1=(u1/(u1.norm()+1e-30))
            frames.append(t); loss.append(float(server._TL.loss.value(o,Y,N))); sharp.append(sh); lmax.append(lm)
            align.append(float(abs(float(u1@Jr))/Jrn)); U.append(u1.detach().cpu().numpy().astype(np.float32))
        if t<steps:
            g,_=server.gradL(th,X,Y); th=th-lr*g
    U=np.stack(U)  # (F, p)
    # consecutive rotation and greedy new-direction count offline
    cons=[1.0]+[float(abs(U[i]@U[i-1])) for i in range(1,len(U))]
    def newdirs(thr):
        anch=[0]
        for i in range(1,len(U)):
            if max(abs(U[i]@U[j]) for j in anch)<thr: anch.append(i)
        return [int(frames[a]) for a in anch]
    out[key]=dict(lr=lr,twoOverLr=2/lr,steps=steps,frames=frames,loss=loss,sharp=sharp,lmax=lmax,align=align,
                  conscos=cons, newdir_050=newdirs(0.5), newdir_070=newdirs(0.7), newdir_085=newdirs(0.85))
    print("  %-16s p=%d frames=%d | consCos min=%.2f mean=%.2f | new-dirs @0.5=%d @0.7=%d @0.85=%d | align[min,max]=[%.2f,%.2f] (%.0fs)"%(
        key,p,len(frames),min(cons),float(np.mean(cons)),len(out[key]["newdir_050"]),len(out[key]["newdir_070"]),len(out[key]["newdir_085"]),min(align),max(align),time.time()-t0),flush=True)
json.dump(out,open("psstudy/cycle_ev2.json","w")); print("wrote psstudy/cycle_ev2.json",flush=True)
