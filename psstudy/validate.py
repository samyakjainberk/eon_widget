# DEEP VALIDATION: exact eigendecomposition vs the widget's Lanczos, for the grok panels' operators.
import sys,numpy as np,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
p_={};p_.update(capture_run.default_params())
p_.update(dict(dataset="ksparse",arch="mlp",loss="mse",act="tanh",width="16",depth="2",nsamp="100",indim="10",outdim="1",lr="0.5",initscheme="kaiming_normal",init="0.5",seed="0"))
P=server._parse_params({k:[str(v)] for k,v in p_.items()})
server._TL.model=server.build_model("mlp",10,1,P);server._TL.model.init_scheme="kaiming_normal";server._TL.loss=server.build_loss("mse")
th,X,Y,_,_=server.init_data_theta(P,"ksparse",100,10,1);N=X.shape[0];p=server._TL.model.p;M=N*1;th0=th.clone()
for _ in range(10):  # advance to a checkpoint
    g,_=server.gradL(th,X,Y);th=th-0.5*g
o=server._TL.model.forward(th,X);r=(Y-o).reshape(-1)[:M];rc=(Y-o).reshape(N,1)
Jc,_=server.jac_cols(th,X);Jg=Jc[:M];Jr=Jg.t()@r;Jrn=float(Jr.norm())+1e-30

def materialize(hvp):  # exact p×p operator
    I=torch.eye(p,dtype=torch.float64,device=dev);return torch.stack([hvp(I[:,i]) for i in range(p)]).t()
def lanczos_top(hvp):  # the widget's method
    Qb,T,k=server._lanczos_core(hvp,p,min(p,48),0,dt=torch.float64,q0=server._randvec16(p,server.SEC21_SEED))
    mu,Sv=server._safe_eigh(T);o=torch.argsort(mu,descending=True);Qm=torch.stack(Qb)
    return float(mu[int(o[0])]),(Sv[:,int(o[0])].double().to(Qm.device)@Qm)

print("=== EXACT vs LANCZOS (the widget's method) — evolving M_r ===")
server._TL.qcfg=None
Mr_exact=materialize(lambda v:server.hvpS(th,X,v,rc))
w,V=torch.linalg.eigh((Mr_exact+Mr_exact.t())/2);eidx=torch.argsort(w,descending=True)
lam1_exact=float(w[eidx[0]]);u1_exact=V[:,eidx[0]]
align_exact=abs(float(u1_exact@Jr))/Jrn
lam1_lanc,u1_lanc=lanczos_top(lambda v:server.hvpS(th,X,v,rc))
align_lanc=abs(float(u1_lanc@Jr))/Jrn
print("  λ₁(M_r):  exact=%.6f  lanczos=%.6f  rel-err=%.2e"%(lam1_exact,lam1_lanc,abs(lam1_exact-lam1_lanc)/abs(lam1_exact)))
print("  alignment: exact=%.6f  lanczos=%.6f  |diff|=%.2e"%(align_exact,align_lanc,abs(align_exact-align_lanc)))
print("  eigvec agreement |cos(u1_exact,u1_lanc)|=%.6f (should be ~1)"%abs(float(u1_exact@u1_lanc)/(u1_exact.norm()*u1_lanc.norm())))

print("=== init-fixed vs evolving are DIFFERENT operators (diverge over training)? ===")
server._TL.qcfg={"mode":"fix","theta_t":th0,"Qrand":None}
Mrfix=materialize(lambda v:server.hvpS(th,X,v,rc))
server._TL.qcfg=None
opnorm_diff=float((Mrfix-Mr_exact).norm())/float(Mr_exact.norm())
print("  ‖M_r(fix)−M_r(evolve)‖/‖M_r(evolve)‖ = %.3f (>0 ⇒ genuinely different operators)"%opnorm_diff)
wf=torch.linalg.eigvalsh((Mrfix+Mrfix.t())/2)
print("  λ₁ evolve=%.4f  init-fix=%.4f (differ as θ left θ₀)"%(lam1_exact,float(wf.max())))

print("=== random-low-rank alignment is chance-level (independent check) ===")
lam=torch.tensor([lam1_exact]+[float(w[eidx[i]]) for i in range(1,4)],dtype=torch.float64,device=dev)
gen=torch.Generator(device=dev).manual_seed(123);U,_=torch.linalg.qr(torch.randn(p,4,generator=gen,dtype=torch.float64,device=dev))
print("  random-low-rank align=%.4f  vs true align=%.4f  vs chance sqrt(1/p)=%.4f"%(abs(float(U[:,0]@Jr))/Jrn,align_exact,1/np.sqrt(p)))
print("  (random ≈ chance ≪ true ⇒ the direction, not rank, matters — the item-③/experiment conclusion)")

print("=== item ④: exact recompute of counts ===")
K=8;Uk=V[:,eidx[:K]].t();gL=server.gradL(th,X,Y)[0];ngL=float(gL.norm())+1e-30
gcos=(Uk@gL).abs()/ngL;JU=Jg@Uk.t();lam1ntk=float(torch.linalg.eigvalsh(Jg@Jg.t()).max())+1e-30
nrg=(JU*JU).sum(0)/lam1ntk
print("  exact: can-lower-loss(|cos(∇L,u)|>0.1)=%d  learned(NTK-energy>0.1)=%d  (K=%d)"%(int((gcos>0.1).sum()),int((nrg>0.1).sum()),K))
