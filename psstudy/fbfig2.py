import json,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
D=json.load(open("psstudy/fixedbasis2.json")); DS=["ksparse","saddle","chebyshev"]
fig,ax=plt.subplots(2,3,figsize=(16,8))
for j,ds in enumerate(DS):
    r=D[ds]; t=np.array(r["frames"]); E=np.array(r["energy"]); K=r["K"]; cap=np.array(r["captured"])
    # top: heatmap of NTK energy per FIXED init-direction over time (log)
    a=ax[0,j]
    im=a.imshow(E.T,aspect="auto",origin="lower",extent=[t[0],t[-1],0.5,K+0.5],cmap="magma",
                norm=LogNorm(vmin=max(1e-1,E.max()/1e4),vmax=E.max()))
    a.set_title(f"{ds}: NTK energy per FIXED init-direction (rank)"); a.set_xlabel("iteration"); a.set_ylabel("fixed init-rank i")
    plt.colorbar(im,ax=a,fraction=0.045)
    # bottom: 'captured' = how much of the CURRENT top sharpening direction is still in the fixed init top-K basis
    b=ax[1,j]
    b.plot(t,cap,color="#dc2626",lw=2)
    b.fill_between(t,cap,1,color="#dc2626",alpha=0.12)
    b.set_ylim(0,1.02); b.set_title(f"{ds}: fraction of the current TOP direction still in the init top-{K} basis"); b.set_xlabel("iteration")
    b.axhline(1,color="#999",lw=.5,ls="--"); b.grid(alpha=.15)
    b.text(t[-1]*0.5, cap.min()+0.08, "the gap below 1 = NEW\ndirections being learned", fontsize=8, color="#dc2626")
plt.suptitle("Section-6 in a FIXED init basis (NTK energy): energy drains from the init-dominant directions into NEW directions the network learns",fontsize=12,y=1.0)
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig("psstudy/figs/g_fixedbasis2.png",dpi=92); plt.close(); print("wrote g_fixedbasis2.png")
