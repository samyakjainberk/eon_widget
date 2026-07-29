# Diagnose item④ integer count differences: exact eigendecomp vs run_stream-emitted, per tick, with margins.
import sys,numpy as np,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
STEPS=10
OFF={f"s{i}":"0" for i in range(1,43)}
BASE=dict(dataset="maxfind",arch="mlp",loss="ce",act="tanh",width="14",depth="2",nsamp="14",indim="10",outdim="10",
          lr="0.2",initscheme="kaiming_normal",init="0.6",seed="0",steps=str(STEPS),eigevery="1",bias="1",
          gw4="1",gw4K="8",gw4tau="0.1",gw3specevery="4"); BASE.update(OFF); BASE["s19"]="1"
def parse(d):
    pp=capture_run.default_params(); pp.update(d); return server._parse_params({k:[str(v)] for k,v in pp.items()})
P=parse(BASE)
emit={}
for msg in server.run_stream(P):
    if msg.get("type")=="step" and msg.get("g_gw4") is not None: emit[msg["t"]]=msg["g_gw4"]
server._TL.qcfg=None
inD,outD=10,10;N=14;M=N*outD;lr=0.2
server._TL.model=server.build_model("mlp",inD,outD,P);server._TL.model.init_scheme="kaiming_normal";server._TL.loss=server.build_loss("ce")
th,X,Y,_,_=server.init_data_theta(P,"maxfind",N,inD,outD);p=server._TL.model.p
I=torch.eye(p,dtype=torch.float64,device=dev)
print("tick | exact(nd,nl) emit(nd,nl) | top-K |λ|-set match Lanczos↔exact | nrg values near 0.1 threshold")
for t in range(STEPS+1):
    o=server._TL.model.forward(th,X); cS=server._TL.loss.resid_cotangent(o,Y,N); rr=(-N*cS).reshape(-1)
    Jc,_=server.jac_cols(th,X);Jm=Jc[:M];r=rr[:M];rc=r.reshape(N,outD);gL=server.gradL(th,X,Y)[0];ngL=float(gL.norm())+1e-30
    if t in emit:
        g4=emit[t];K4=g4["K"]
        Mr=torch.stack([server.hvpS(th,X,I[:,i],rc) for i in range(p)]).t();Mr=(Mr+Mr.t())/2
        w,V=torch.linalg.eigh(Mr);ei=torch.argsort(w.abs(),descending=True)
        U=V[:,ei[:K4]].t()
        gcos=(U@gL).abs()/ngL;lam1ntk=float(torch.linalg.eigvalsh(Jm@Jm.t()).max())+1e-30
        JU=Jm@U.t();nrg=(JU*JU).sum(0)/lam1ntk
        nd,nl=int((gcos>0.1).sum()),int((nrg>0.1).sum())
        # how close are nrg values to the 0.1 boundary?
        near=sorted([float(x) for x in nrg],key=lambda v:abs(v-0.1))[:2]
        exact_topK=set(int(x) for x in ei[:K4]); exact_lamK=sorted([abs(float(w[i])) for i in ei[:K4]])
        flag="" if (nd==g4["ndec"] and nl==g4["nlearn"]) else "  <-- DIFF"
        print("%4d | exact(%d,%d) emit(%d,%d) | nrg-near-0.1: %s%s"%(t,nd,nl,g4["ndec"],g4["nlearn"],[round(x,4) for x in near],flag))
    if t<STEPS: th=th-lr*server._opt_dir(server._TL.model,server.gradL(th,X,Y)[0],"gd")
