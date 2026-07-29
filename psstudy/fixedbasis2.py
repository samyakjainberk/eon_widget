import sys,json,time
import numpy as np, torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server, capture_run
dev=torch.device("cuda:0"); dtype=torch.float64
server.DTYPE=dtype; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
RUNS=[("ksparse", dict(dataset="ksparse", width=16,depth=2,nsamp=100,lr=0.5,indim=10,outdim=1,steps=400)),
      ("saddle",  dict(dataset="saddle",  width=16,depth=3,nsamp=64, lr=0.3,indim=4, outdim=4,steps=400)),
      ("chebyshev",dict(dataset="chebyshev",width=16,depth=3,nsamp=100,lr=0.1,indim=1,outdim=1,degree=3,steps=400))]
K=16; EVERY=4
def build(P0):
    params=capture_run.default_params()
    params.update({**{k:str(v) for k,v in P0.items()},"arch":"mlp","loss":"mse","act":"tanh","bias":"1","init":"0.5","seed":"0","initscheme":"kaiming_normal"})
    P=server._parse_params({k:[str(v)] for k,v in params.items()}); inD=int(P0["indim"]); outD=int(P0["outdim"])
    server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"; server._TL.loss=server.build_loss("mse")
    th,X,Y,_,_=server.init_data_theta(P,P0["dataset"],int(P0["nsamp"]),inD,outD); return th,X,Y,outD
def Jmat(th,X,M):
    Jc,_=server.jac_cols(th,X); return Jc[:M]
out={}
for key,P0 in RUNS:
    th,X,Y,outD=build(P0); N=X.shape[0]; p=server._TL.model.p; lr=float(P0["lr"]); steps=int(P0["steps"]); M=N*outD; th0=th.clone(); t0=time.time()
    J0=Jmat(th0,X,M)
    NTK=(J0@J0.t()).cpu().numpy(); w,V=np.linalg.eigh(NTK); idx=np.argsort(w)[::-1][:K]; lam0=w[idx].copy()
    U=[]
    for i in idx:
        vi=torch.tensor(V[:,i],dtype=dtype,device=dev); ui=J0.t()@vi; ui=ui/(ui.norm()+1e-30); U.append(ui)
    U=torch.stack(U)
    frames=[];loss=[];energy=[];captured=[];lammax=[]
    for t in range(steps+1):
        if t%EVERY==0:
            o=server._TL.model.forward(th,X); frames.append(t); loss.append(float(server._TL.loss.value(o,Y,N)))
            Jt=Jmat(th,X,M); JU=Jt@U.t(); e=(JU*JU).sum(0).detach().cpu().numpy(); energy.append(e.tolist())
            NTKt=(Jt@Jt.t()).cpu().numpy(); wt,Vt=np.linalg.eigh(NTKt); lammax.append(float(wt[-1]))
            vtop=torch.tensor(Vt[:,-1],dtype=dtype,device=dev); gtop=Jt.t()@vtop; gtop=gtop/(gtop.norm()+1e-30)
            captured.append(float(((U@gtop)**2).sum()))
        if t<steps:
            g,_=server.gradL(th,X,Y); th=th-lr*g
    out[key]=dict(lr=lr,twoOverLr=2/lr,steps=steps,K=K,frames=frames,loss=loss,lam0=lam0.tolist(),energy=energy,captured=captured,lammax=lammax)
    e=np.array(energy)
    print("  %-10s p=%d K=%d | top-dir e end/init=%.2f | captured init=%.2f end=%.2f | lammax %.0f->%.0f (%.0fs)"%(
        key,p,K,e[-1,0]/(e[0,0]+1e-12),captured[0],captured[-1],lammax[0],lammax[-1],time.time()-t0),flush=True)
json.dump(out,open("psstudy/fixedbasis2.json","w")); print("wrote fixedbasis2.json",flush=True)
