import sys,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
def build(ds,loss,inD,outD,lr,w=20,d=2,n=24,**ex):
    OFF={f"s{i}":"0" for i in range(1,43)}
    B=dict(dataset=ds,arch="mlp",loss=loss,act="tanh",width=str(w),depth=str(d),nsamp=str(n),indim=str(inD),outdim=str(outD),
           lr=str(lr),initscheme="kaiming_normal",init="0.6",seed="0",bias="1");B.update(OFF);B.update({k:str(v) for k,v in ex.items()})
    pp=capture_run.default_params();pp.update(B);P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    server._TL.model=server.build_model("mlp",inD,outD,P);server._TL.model.init_scheme="kaiming_normal";server._TL.loss=server.build_loss(loss)
    th,X,Y,_,_=server.init_data_theta(P,ds,n,inD,outD);return th,X,Y,P

print("===== grok-③ per-mode §6: evolving M_r Panel-2 (J·r projections) vs EXACT eigendecomp =====")
th,X,Y,P=build("maxfind","ce",10,10,0.2)
N=X.shape[0];M=N*10;p=server._TL.model.p
for _ in range(5): th=th-0.2*server.gradL(th,X,Y)[0]
server._TL.qcfg=None
o=server._TL.model.forward(th,X);rr=(-N*server._TL.loss.resid_cotangent(o,Y,N)).reshape(-1)
Jm=server.jac_cols(th,X)[0][:M];r=rr[:M];rc=r.reshape(N,10);Jr=Jm.t()@r;Jrn=float(Jr.norm())+1e-30
I=torch.eye(p,dtype=torch.float64,device=dev);Mr=torch.stack([server.hvpS(th,X,I[:,i],rc) for i in range(p)]).t();Mr=(Mr+Mr.t())/2
w_,V_=torch.linalg.eigh(Mr);ei=torch.argsort(w_,descending=True);K=6;scN=1.0/N
p1_ex=[abs(float(V_[:,ei[i]]@Jr))/Jrn for i in range(K)];lam_ex=[float(w_[ei[i]])*scN for i in range(K)]
q0=server._randvec16(p,server.SEC21_SEED);Qb,T,k=server._lanczos_core(lambda v:server.hvpS(th,X,v,rc),p,min(p,max(5*K,64)),0,dt=torch.float64,q0=q0)
mu,Sv=server._safe_eigh(T);desc=torch.argsort(mu,descending=True);Qm=torch.stack(Qb)
p1_lz=[abs(float((Sv[:,int(desc[i])].to(device=dev,dtype=torch.float64)@Qm)@Jr))/Jrn for i in range(K)]
lam_lz=[float(mu[int(desc[i])])*scN for i in range(K)]
print("  bare-cos p1 top-3 exact",[round(x,5) for x in p1_ex[:3]],"lanczos",[round(x,5) for x in p1_lz[:3]])
print("  max|Δp1|=%.2e  max rel-Δλ=%.2e"%(max(abs(a-b) for a,b in zip(p1_ex,p1_lz)),max(abs(a-b)/(abs(a)+1e-12) for a,b in zip(lam_ex,lam_lz))))

print("\n===== grok-④ persistence self-consistency (T0=t ⇒ all top/bot in span; cross ⇒ ~0) =====")
th,X,Y,P=build("ksparse","mse",10,1,0.4,n=30)
N=X.shape[0];M=N;p=server._TL.model.p
for _ in range(4): th=th-0.4*server.gradL(th,X,Y)[0]
o=server._TL.model.forward(th,X);rr=(-N*server._TL.loss.resid_cotangent(o,Y,N)).reshape(-1);rc=rr[:M].reshape(N,1)
Qb,T,k=server._lanczos_core(lambda v:server.hvpS(th,X,v,rc),p,min(p,48),0,dt=torch.float64,q0=server._randvec16(p,server.SEC21_SEED))
mu,Sv=server._safe_eigh(T);asc=torch.argsort(mu);Qm=torch.stack(Qb);K=5
def vecs(idx):
    Vv=torch.stack([Sv[:,int(j)].to(device=dev,dtype=torch.float64)@Qm for j in idx]);Q,_=torch.linalg.qr(Vv.t());return Q
topV=vecs([int(asc[-1-i]) for i in range(K)]);botV=vecs([int(asc[i]) for i in range(K)])
def cin(cur,ref,thr=0.5):
    pr=ref@(ref.t()@cur);e=(pr*pr).sum(0)/(cur*cur).sum(0).clamp_min(1e-30);return int((e>thr).sum())
print(f"  self: top-in(top)={cin(topV,topV)}/{K} bot-in(bot)={cin(botV,botV)}/{K} (want K)  cross: top-in(bot)={cin(topV,botV)}/{K} (want ~0)")

print("\n===== grok-⑥ target-scaling: 2Y raises MSE =====")
th,X,Y,P=build("ksparse","mse",10,1,0.4,n=30);N=X.shape[0]
l1=float(server._TL.loss.value(server._TL.model.forward(th,X),Y,N));l2=float(server._TL.loss.value(server._TL.model.forward(th,X),2.0*Y,N))
print(f"  MSE(Y)={l1:.4f} MSE(2Y)={l2:.4f} larger={l2>l1}")

print("\n===== grok-⑨ random-search: only decreases loss (never increases) =====")
th,X,Y,P=build("ksparse","mse",10,1,0.4,n=40);N=X.shape[0];M=N
step=server._rs_step(X,Y,N,1,0.05,6,10,8,0.0)
L=[];thc=th.clone()
for _ in range(15):
    L.append(float(server._TL.loss.value(server._TL.model.forward(thc,X),Y,N)));thc=step(thc)
mono=all(L[i+1]<=L[i]+1e-9 for i in range(len(L)-1))
print(f"  loss sequence monotone non-increasing: {mono}  (start {L[0]:.4f} -> end {L[-1]:.4f})")
