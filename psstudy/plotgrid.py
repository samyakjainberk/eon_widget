#!/usr/bin/env python
"""Render a full diagnostic grid for one captured run — every phase / prediction signal — to a PNG."""
import json,sys,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    d=json.load(open(path)); pr=d["params"]; lr=float(pr["lr"])
    st=[r for r in d["records"] if r.get("type")=="step"]
    return d,pr,lr,st
def arr(st,f):
    return np.array([ (r.get(f) if isinstance(r.get(f),(int,float)) else np.nan) for r in st],float)
def sub(st,k,s,idx=None):
    out=[]
    for r in st:
        v=r.get(k); v=v.get(s) if isinstance(v,dict) else None
        if idx is not None and isinstance(v,list):
            v=v[idx] if idx<len(v) else None
        out.append(float(v) if isinstance(v,(int,float)) else np.nan)
    return np.array(out)
def subvec(st,k,s,n):
    return [sub(st,k,s,i) for i in range(n)]

def plot(path,outpng):
    d,pr,lr,st=load(path); T=arr(st,"t"); tl=2/lr
    lab=pr.get("dataset","?")+" "+pr.get("arch","")+" lr"+str(lr)+" "+pr.get("initscheme","")
    fig,ax=plt.subplots(4,3,figsize=(16,13)); fig.suptitle(lab+"   (2/η=%.1f, %d steps)"%(tl,len(st)),fontsize=13,y=0.995)
    A=ax.ravel()
    # 1 loss
    A[0].semilogy(T,arr(st,"loss"),lw=1.2,color="#3b6fd4"); A[0].set_title("1. training loss (log)")
    # 2 sharpness vs 2/lr
    A[1].plot(T,arr(st,"sharp"),lw=1,color="#3b6fd4"); A[1].axhline(tl,ls="--",color="#d98a17",lw=1)
    A[1].set_title("2. sharpness λ₁(∇²L) vs 2/η")
    # 3 phase-1: qr top3+bot3 alignment
    for i,c in enumerate(["#0f9d8f","#12a150","#3b6fd4","#e0559a","#d98a17","#8b5cf6"]):
        A[2].plot(T,sub(st,"g_phase","qr",i),lw=0.9,color=c,label="u%d"%(i+1))
    A[2].set_title("3. Phase-1 Qr→Jr align (top3⊕bot3)"); A[2].axhline(0,color="#999",lw=0.5)
    # 4 power iteration: Jrn + ntk top3
    A[3].semilogy(T,sub(st,"g_phase","Jrn"),lw=1.1,color="#d98a17",label="‖Jᵀr‖")
    for i,c in enumerate(["#8b5cf6","#a78bfa","#c4b5fd"]):
        A[3].semilogy(T,sub(st,"g_phase","ntk",i),lw=0.8,color=c)
    A[3].set_title("4. Phase-2 power: ‖Jᵀr‖ & top-3 NTK"); A[3].legend(fontsize=7)
    # 5 pred3: jrd, jdr, diff
    A[4].plot(T,sub(st,"g_pred3","jrd"),lw=1,color="#3b6fd4",label="‖J·ṙ‖ (NTK)")
    A[4].plot(T,sub(st,"g_pred3","jdr"),lw=1,color="#e0559a",label="‖J̇·r‖ (Mr)")
    A[4].axhline(0,color="#999",lw=0.5); A[4].set_title("5. Phase-3 residual-dominance terms"); A[4].legend(fontsize=7)
    # 6 pred4: NTK spectrum actual vs predicted (top3)
    for i,c in enumerate(["#0f9d8f","#3b6fd4","#8b5cf6"]):
        A[5].plot(T,sub(st,"g_pred4","kAct",i),lw=1.1,color=c)
        A[5].plot(T,sub(st,"g_pred4","kPred",i),lw=0.8,ls="--",color=c)
    A[5].set_yscale("log"); A[5].set_title("6. Pred-4 NTK eig: solid=act dash=pred")
    # 7 pred4 cos5 eigenvector match
    for i,c in enumerate(["#0f9d8f","#3b6fd4","#8b5cf6"]):
        A[6].plot(T,np.abs(sub(st,"g_pred4","cos5",i)),lw=1,color=c,label="v%d"%(i+1))
    A[6].set_ylim(0,1.05); A[6].set_title("7. Pred-4 eigenvector |cos| act vs pred"); A[6].legend(fontsize=7)
    # 8 trace pred5.1
    A[7].plot(T,sub(st,"g_trace","trGN"),lw=1.1,color="#0f9d8f",label="Tr GN")
    A[7].plot(T,sub(st,"g_trace","trHess"),lw=1,color="#3b6fd4",label="Tr ∇²L")
    A[7].plot(T,sub(st,"g_trace","qLive"),lw=0.8,ls="--",color="#e0559a",label="pred qLive")
    A[7].set_title("8. Pred-5.1 trace(NTK)"); A[7].legend(fontsize=7)
    # 9 eds posSum/negSum
    A[8].plot(T,sub(st,"g_eds","posSum"),lw=1,color="#0f9d8f",label="posSum")
    A[8].plot(T,sub(st,"g_eds","negSum"),lw=1,color="#e0559a",label="negSum")
    A[8].set_title("9. early-dyn σ-weighted ± Mr proj"); A[8].legend(fontsize=7)
    # 10 eds principal angles H vs Mr
    A[9].plot(T,sub(st,"g_eds","angH"),lw=1,color="#3b6fd4",label="ang(H)")
    A[9].plot(T,sub(st,"g_eds","angMr"),lw=1,color="#8b5cf6",label="ang(Mr)")
    A[9].set_title("10. eds principal-angle drift"); A[9].legend(fontsize=7)
    # 11 self-stab top cosines
    for i in range(3):
        A[10].plot(T,sub(st,"g_ss","top",i),lw=0.9,label="top%d"%(i+1))
    A[10].plot(T,sub(st,"g_ss","p3"),lw=1,color="#000",alpha=0.5,label="p3")
    A[10].set_title("11. self-stab cos(∇S, H_P eigvecs)"); A[10].legend(fontsize=6); A[10].set_ylim(-1.05,1.05)
    # 12 cumulative ray anchors
    ridx=sub(st,"g_ray","idx"); cum=np.maximum.accumulate(np.nan_to_num(ridx,nan=0))
    A[11].plot(T,cum,lw=1.4,color="#8b5cf6"); A[11].set_title("12. cumulative direction-switches (g_ray.idx)")
    for a in A: a.tick_params(labelsize=8); a.grid(alpha=0.15)
    plt.tight_layout(rect=[0,0,1,0.985]); plt.savefig(outpng,dpi=85); plt.close()
    print("wrote",outpng)

if __name__=="__main__":
    plot(sys.argv[1],sys.argv[2])
