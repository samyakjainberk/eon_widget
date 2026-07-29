import sys,time,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
def build(ds,loss,inD,outD,lr,arch="mlp",n=24,**ex):
    OFF={f"s{i}":"0" for i in range(1,43)}
    B=dict(dataset=ds,arch=arch,loss=loss,act="tanh",width="20",depth="2",nsamp=str(n),indim=str(inD),outdim=str(outD),
           lr=str(lr),initscheme="kaiming_normal",init="0.6",seed="0",bias="1");B.update(OFF);B.update({k:str(v) for k,v in ex.items()})
    pp=capture_run.default_params();pp.update(B);P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    server._TL.model=server.build_model(arch,inD,outD,P);server._TL.model.init_scheme="kaiming_normal";server._TL.loss=server.build_loss(loss)
    th,X,Y,_,_=server.init_data_theta(P,ds,n,inD,outD);return th,X,Y,P

print("=== matrix-free _grok_diag vs EXACT (MLP maxfind CE) ===")
th,X,Y,P=build("maxfind","ce",10,10,0.2,n=24)
N=X.shape[0];M=N*10;p=server._TL.model.p
for _ in range(5): th=th-0.2*server.gradL(th,X,Y)[0]
d=server._grok_diag(th,X,Y,None,None,N,0,10,ce=True)   # matrix-free
# exact reference via jac_cols + full NTK eigh
o=server._TL.model.forward(th,X);rr=(-N*server._TL.loss.resid_cotangent(o,Y,N)).reshape(-1)
Jm=server.jac_cols(th,X)[0][:M];r=rr[:M];rc=r.reshape(N,10);rn=float(r.norm())+1e-30
w_,V_=torch.linalg.eigh(Jm@Jm.t());u1=V_[:,torch.argmax(w_)];align_ex=abs(float(u1@r))/rn
g=Jm.t()@r;gMrg=abs(float(g@server.hvpS(th,X,g,rc)));gJJg=float((Jm@g).pow(2).sum());ratio_ex=gMrg/(gJJg+1e-30)
ps_ex=float(torch.linalg.eigvalsh(torch.stack([server.hvpL(th,X,Y,torch.eye(p,dtype=torch.float64,device=dev)[:,i]) for i in range(p)]).t()).max())
print(f"  align: mf={d['align']:.6f} exact={align_ex:.6f}  |Δ|={abs(d['align']-align_ex):.2e}")
print(f"  ratio: mf={d['ratio']:.6f} exact={ratio_ex:.6f}  rel={abs(d['ratio']-ratio_ex)/abs(ratio_ex):.2e}")
print(f"  PS:    mf={d['ps']:.6f} exact={ps_ex:.6f}  rel={abs(d['ps']-ps_ex)/abs(ps_ex):.2e}")

print("=== speed: _grok_diag on gpt-transformer (was the bottleneck) ===")
th,X,Y,P=build("maxfind","ce",10,10,0.02,arch="gpt",n=24,dmodel=32,nhead=2,nlayer=2,seqlen=10)
N=X.shape[0]
t0=time.time(); d=server._grok_diag(th,X,Y,None,None,N,0,10,ce=True); dt=time.time()-t0
print(f"  gpt _grok_diag: {dt*1000:.0f} ms/call  (align={d['align']:.4f} ratio={d['ratio']:.4f} ps={d['ps']:.4f}) — no jac_cols")
