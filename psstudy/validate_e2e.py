# END-TO-END + EXACT validation (CE path, maxfind): drive ACTUAL run_stream, independently replay + recompute exact.
import sys,math,numpy as np,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
STEPS=10
# All heavy sections OFF except s19 (light; needed so run_stream computes Jc/rr that the grok blocks consume).
OFF={f"s{i}":"0" for i in range(1,43)}
BASE=dict(dataset="maxfind",arch="mlp",loss="ce",act="tanh",width="14",depth="2",nsamp="14",indim="10",outdim="10",
          lr="0.2",initscheme="kaiming_normal",init="0.6",seed="0",steps=str(STEPS),eigevery="1",bias="1",
          gw2="1",gw2tau="0.7",gw3="1",gw3rank="4",gw3full="0",gw4="1",gw4K="8",gw4tau="0.1",gw1="1",gw1n="8",gw1steps="3",
          gw3specevery="4")
BASE.update(OFF); BASE["s19"]="1"
def parse(d):
    pp=capture_run.default_params(); pp.update(d); return server._parse_params({k:[str(v)] for k,v in pp.items()})
P=parse(BASE)
emitted={}; losses={}
for msg in server.run_stream(P):
    if msg.get("type")=="step":
        t=msg["t"]; losses[t]=msg.get("loss"); emitted[t]={k:msg.get(k) for k in ("g_gw1","g_gw2","g_gw3","g_gw4")}
    if msg.get("type")=="error": print("STREAM ERROR:",msg); sys.exit(1)
ne=[t for t in emitted if emitted[t]["g_gw3"] is not None]
print("run_stream ticks=%d, g_gw3-emitting ticks=%d %s"%(len(emitted),len(ne),ne[:8]),flush=True)

# INDEPENDENT REPLAY (same P, deterministic)
server._TL.qcfg=None
inD,outD=10,10; N=14; M=N*outD; lr=0.2
server._TL.model=server.build_model("mlp",inD,outD,P); server._TL.model.init_scheme="kaiming_normal"; server._TL.loss=server.build_loss("ce")
th,X,Y,_,_=server.init_data_theta(P,"maxfind",N,inD,outD); p=server._TL.model.p
print("independent p=%d M=%d"%(p,M),flush=True)
def materialize_Mr(th_,rc_):
    I=torch.eye(p,dtype=torch.float64,device=dev)
    return torch.stack([server.hvpS(th_,X,I[:,i],rc_) for i in range(p)]).t()
mx=dict(lam1=0.0,align=0.0,dbal=0.0,loss=0.0); gw4ok=True; last=None
for t in range(STEPS+1):
    o=server._TL.model.forward(th,X); loss=float(server._TL.loss.value(o,Y,N))
    if t in losses and losses[t] is not None: mx["loss"]=max(mx["loss"],abs(loss-losses[t])/(abs(losses[t])+1e-30))
    cS=server._TL.loss.resid_cotangent(o,Y,N); rr=(-N*cS).reshape(-1)
    Jc,_=server.jac_cols(th,X); Jm=Jc[:M]; r=rr[:M]; rc=r.reshape(N,outD)
    Jr=Jm.t()@r; Jrn=float(Jr.norm())+1e-30; gL=server.gradL(th,X,Y)[0]
    if t in emitted and (emitted[t]["g_gw3"] is not None or emitted[t]["g_gw4"] is not None):
        Mr=materialize_Mr(th,rc); Mr=(Mr+Mr.t())/2
        w,V=torch.linalg.eigh(Mr); ei=torch.argsort(w,descending=True)
        if emitted[t]["g_gw3"] is not None:
            g3=emitted[t]["g_gw3"]; lam1_ex=float(w[ei[0]]); u1=V[:,ei[0]]; align_ex=abs(float(u1@Jr))/Jrn
            JJg=Jm.t()@(Jm@gL); dbal_ex=float((Mr@gL).norm())-float(JJg.norm())
            mx["lam1"]=max(mx["lam1"],abs(lam1_ex-g3["lam1"]["ev"])/(abs(lam1_ex)+1e-30))
            mx["align"]=max(mx["align"],abs(align_ex-g3["align"]["ev"]))
            mx["dbal"]=max(mx["dbal"],abs(dbal_ex-g3["dbal"]["ev"])/(abs(dbal_ex)+1e-9))
        if emitted[t]["g_gw4"] is not None:
            g4=emitted[t]["g_gw4"]; K4=g4["K"]; ei2=torch.argsort(w.abs(),descending=True)
            U=V[:,ei2[:K4]].t(); ngL=float(gL.norm())+1e-30
            gcos=(U@gL).abs()/ngL; lam1ntk=float(torch.linalg.eigvalsh(Jm@Jm.t()).max())+1e-30
            JU=Jm@U.t(); nrg=(JU*JU).sum(0)/lam1ntk
            nd,nl=int((gcos>0.1).sum()),int((nrg>0.1).sum())
            if not(nd==g4["ndec"] and nl==g4["nlearn"]): gw4ok=False
            last=(nd,nl,g4["ndec"],g4["nlearn"],g4["jrel"])
    if t<STEPS: th=th-lr*server._opt_dir(server._TL.model,server.gradL(th,X,Y)[0],"gd")
print("\n=== END-TO-END (run_stream EMITTED) vs INDEPENDENT EXACT — CE maxfind ===")
print("  trajectory loss match (max rel-err): %.2e  %s"%(mx["loss"],"OK" if mx["loss"]<1e-9 else "MISMATCH!"))
print("  item③ λ₁(M_r,ev) max rel-err vs exact eigendecomp: %.2e"%mx["lam1"])
print("  item③ alignment  max |diff| vs exact:              %.2e"%mx["align"])
print("  item③ dbal       max rel-err vs exact:             %.2e"%mx["dbal"])
if last: print("  item④ @last exact(ndec=%d,nl=%d) vs emitted(ndec=%d,nl=%d) jrel=%.3f"%last)
print("  item④ counts exact-match every tick:", "OK" if gw4ok else "MISMATCH!")
