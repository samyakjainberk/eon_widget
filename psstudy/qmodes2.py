#!/usr/bin/env python
"""Q-mode analysis for the multi-output datasets (mnist, cifar10). Same as qmodes.py, plus an inline
Frobenius-concentration measurement (λ_max / ‖M_r‖_F) at a mid-training checkpoint for each mode."""
import os,sys,json,time
import numpy as np, torch
DIR="/nas/ucb/samsj/TestingPSTheory/eos_widget"; sys.path.insert(0,DIR)
import server, capture_run
dev=torch.device("cuda:0"); dtype=torch.float32
server.DTYPE=dtype; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
RUNS=[dict(key="mnist",  P=dict(dataset="mnist",  width=32,depth=2,nsamp=100,lr=0.05,indim=10,outdim=10,steps=300)),
      dict(key="cifar10",P=dict(dataset="cifar10",width=32,depth=2,nsamp=100,lr=0.12,indim=10,outdim=10,steps=300))]
MODES=["evolve","initfix","gauss","bern","unif"]; EVERY=6
def build(cfg):
    params=capture_run.default_params()
    params.update({**{k:str(v) for k,v in cfg["P"].items()},"arch":"mlp","loss":"mse","act":"tanh","bias":"1","init":"0.5","seed":"0","initscheme":"kaiming_normal"})
    P=server._parse_params({k:[str(v)] for k,v in params.items()}); inD=int(cfg["P"]["indim"]); outD=int(cfg["P"]["outdim"])
    server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"; server._TL.loss=server.build_loss("mse")
    th,X,Y,_,_=server.init_data_theta(P,cfg["P"]["dataset"],int(cfg["P"]["nsamp"]),inD,outD); return th,X,Y,outD
def lanczos(hvp,dim,m=40,seed=7):
    Qb,T,k=server._lanczos_core(hvp,dim,min(dim,m),seed); mu,Sv=np.linalg.eigh(T.cpu().numpy()); Qm=torch.stack(Qb)
    u1=torch.tensor(Sv[:,-1],dtype=Qm.dtype,device=Qm.device)@Qm; return float(mu[-1]),float(mu[0]),u1
def frob(hvp,p,probes=32,seed=0):
    g=torch.Generator(device=dev).manual_seed(seed); tot=0.0
    for i in range(probes):
        v=(torch.randint(0,2,(p,),generator=g,device=dev,dtype=torch.float32)*2-1); Av=hvp(v); tot+=float((Av*Av).sum())
    return (tot/probes)**0.5
out={}
for cfg in RUNS:
    th,X,Y,outD=build(cfg); N=X.shape[0]; p=server._TL.model.p; lr=float(cfg["P"]["lr"]); steps=int(cfg["P"]["steps"]); M=N*outD; th0=th.clone(); t0=time.time()
    QR={m:server._build_randQ(th0,X,M,m,12345,4) for m in ("gauss","bern","unif")}
    mc={"evolve":None,"initfix":{"mode":"fix","theta_t":th0,"Qrand":None},
        "gauss":{"mode":"gauss","theta_t":th0,"M":M,"seed":12345,"probes":4,"Qrand":QR["gauss"]},
        "bern": {"mode":"bern","theta_t":th0,"M":M,"seed":12345,"probes":4,"Qrand":QR["bern"]},
        "unif": {"mode":"unif","theta_t":th0,"M":M,"seed":12345,"probes":4,"Qrand":QR["unif"]}}
    rec={"frames":[],"lr":lr,"twoOverLr":2/lr,"P":p,"top":{m:[] for m in MODES},"bot":{m:[] for m in MODES},"align":{m:[] for m in MODES},"loss":[],"sharp":[],"conc":{}}
    ckpt=steps//2
    for t in range(steps+1):
        if t%EVERY==0:
            o=server._TL.model.forward(th,X); r=(Y-o); rc=r.reshape(N,outD); rv=r.reshape(-1)[:M]
            Jc,_=server.jac_cols(th,X); Jg=Jc[:M]; Jr=Jg.t()@rv; Jrn=float(Jr.norm())+1e-30
            rec["frames"].append(t); rec["loss"].append(float(server._TL.loss.value(o,Y,N))); rec["sharp"].append(lanczos(lambda v:server.hvpL(th,X,Y,v),p)[0])
            do_conc = abs(t-ckpt)<EVERY
            for m in MODES:
                server._TL.qcfg=mc[m]; hvp=lambda v:server.hvpS(th,X,v,rc)/N
                tv,bv,u1=lanczos(hvp,p); rec["top"][m].append(tv); rec["bot"][m].append(bv); rec["align"][m].append(float(abs(float(u1@Jr))/Jrn))
                if do_conc and m in ("evolve","initfix","gauss"):
                    fr=frob(hvp,p); rec["conc"][m]=round(tv/(fr+1e-30),3)
            server._TL.qcfg=None
        if t<steps:
            g,_=server.gradL(th,X,Y); th=th-lr*g
    out[cfg["key"]]=rec; print("  %-8s p=%d M=%d frames=%d conc=%s (%.0fs)"%(cfg["key"],p,M,len(rec["frames"]),rec["conc"],time.time()-t0),flush=True)
json.dump(out,open("psstudy/qmodes2.json","w")); print("wrote psstudy/qmodes2.json",flush=True)
