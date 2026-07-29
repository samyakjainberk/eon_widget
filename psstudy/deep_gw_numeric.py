# NUMERICAL correctness of the intervention machinery:
#  (A) _grok_diag's ratio/PS/align match an independent exact recompute.
#  (B) star/taylor optimizer steps at λ=0 reduce EXACTLY to plain GD (baseline sanity).
#  (C) star/taylor at λ>0 differ from GD in the intended direction.
import sys,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
def build(ds,loss,w,d,n,inD,outD,lr,act="tanh"):
    OFF={f"s{i}":"0" for i in range(1,43)}
    B=dict(dataset=ds,arch="mlp",loss=loss,act=act,width=str(w),depth=str(d),nsamp=str(n),indim=str(inD),outdim=str(outD),
           lr=str(lr),initscheme="kaiming_normal",init="0.6",seed="0",bias="1"); B.update(OFF)
    pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
    server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"; server._TL.loss=server.build_loss(loss)
    th,X,Y,_,_=server.init_data_theta(P,ds,n,inD,outD); return th,X,Y,P

for ds,loss,inD,outD,lr in [("maxfind","ce",10,10,0.2),("ksparse","mse",10,1,0.5)]:
    print(f"\n===== {ds} ({loss}) =====")
    th,X,Y,P=build(ds,loss,24,2,30,inD,outD,lr)
    N=X.shape[0]; M=N*outD; p=server._TL.model.p
    for _ in range(6): th=th-lr*server.gradL(th,X,Y)[0]   # advance a bit
    # ---- (A) exact recompute of _grok_diag quantities ----
    d=server._grok_diag(th,X,Y,None,None,N,0,outD,ce=(loss=="ce"))
    o=server._TL.model.forward(th,X); rr=(-N*server._TL.loss.resid_cotangent(o,Y,N)).reshape(-1)
    Jc,_=server.jac_cols(th,X); Jm=Jc[:M]; r=rr[:M]; rc=r.reshape(N,outD); rn=float(r.norm())+1e-30
    # exact align via full NTK eigh
    w_,V_=torch.linalg.eigh(Jm@Jm.t()); u1=V_[:,torch.argmax(w_)]; align_ex=abs(float(u1@r))/rn
    g=Jm.t()@r; gMrg=abs(float(g@server.hvpS(th,X,g,rc))); gJJg=float((Jm@g).pow(2).sum()); ratio_ex=gMrg/(gJJg+1e-30)
    # exact PS via materialized loss Hessian
    I=torch.eye(p,dtype=torch.float64,device=dev); HL=torch.stack([server.hvpL(th,X,Y,I[:,i]) for i in range(p)]).t(); HL=(HL+HL.t())/2
    ps_ex=float(torch.linalg.eigvalsh(HL).max())
    print(f"  align: diag={d['align']:.6f} exact={align_ex:.6f}  |Δ|={abs(d['align']-align_ex):.2e}")
    print(f"  ratio: diag={d['ratio']:.6f} exact={ratio_ex:.6f}  rel={abs(d['ratio']-ratio_ex)/abs(ratio_ex):.2e}")
    print(f"  PS   : diag={d['ps']:.6f} exact={ps_ex:.6f}  rel={abs(d['ps']-ps_ex)/abs(ps_ex):.2e}")
    # ---- (B) λ=0 optimizer steps == plain GD ----
    gd=server._gd_step(X,Y,lr); star0=server._opt_step_star(X,Y,N,outD,lr,0.0,1); tay0=server._opt_step_taylor(X,Y,N,outD,lr,0.0,0.0)
    a=gd(th); bstar=star0(th); btay=tay0(th)
    print(f"  star(λ=0) vs GD: max|Δθ|={float((a-bstar).abs().max()):.2e}   taylor(λ=0) vs GD: max|Δθ|={float((a-btay).abs().max()):.2e}")
    # ---- (C) λ>0 differs from GD ----
    star1=server._opt_step_star(X,Y,N,outD,lr,1.0,1)(th); tay1=server._opt_step_taylor(X,Y,N,outD,lr,1.0,1.0)(th)
    print(f"  star(λ=1) vs GD: max|Δθ|={float((a-star1).abs().max()):.2e}   taylor(λ=1) vs GD: max|Δθ|={float((a-tay1).abs().max()):.2e}  (should be >0)")
