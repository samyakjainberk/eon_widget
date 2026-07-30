import os, sys, math
import numpy as np, torch
sys.path.insert(0, "/nas/ucb/samsj/TestingPSTheory/eos_widget")
import server as S

torch.set_grad_enabled(True)
dev = S._dev()
dt = torch.float64
print("device", dev, "DTYPE", S.DTYPE)

# ---- Build small symmetric operators with KNOWN spectra ----
torch.manual_seed(0)
p = 40

def sym(A): return 0.5*(A+A.t())

# (1) PSD operator G = J^T J with a big nullspace (rank << p)
r_rank = 6
J = torch.randn(r_rank, p, dtype=dt, device=dev)   # (r,p): G=J^T J rank 6
G = J.t() @ J
def hvpG(v): return G @ v
egG = torch.linalg.eigvalsh(G).cpu().numpy()   # ascending

# (2) Indefinite operator H = symmetric with real negatives
Araw = torch.randn(p, p, dtype=dt, device=dev)
H = sym(Araw)
def hvpH(v): return H @ v
egH = torch.linalg.eigvalsh(H).cpu().numpy()

print("\n=== exact spectra ===")
print("G (PSD) min/max eig:", egG.min(), egG.max(), " #>1e-9:", int((egG>1e-9).sum()))
print("H (indef) min/max eig:", egH.min(), egH.max(), " #neg:", int((egH<0).sum()))

# ---- SLQ density with nonneg for PSD ----
for block in (1,4):
    dGc = S.slq_density(hvpG, p, 6, min(p,24), 80, 0x22, block=block, nonneg=True)
    dGn = S.slq_density(hvpG, p, 6, min(p,24), 80, 0x22, block=block, nonneg=False)
    dH  = S.slq_density(hvpH, p, 6, min(p,24), 80, 0x44, block=block, nonneg=False)
    xGc = np.array(dGc['x']); xGn=np.array(dGn['x']); xH=np.array(dH['x'])
    print(f"\n--- block={block} ---")
    print("  G nonneg=True : grid x min =", xGc.min(), " (must be >=0)")
    print("  G nonneg=False: grid x min =", xGn.min(), " (may dip <0, quadrature artifact)")
    print("  H nonneg=False: grid x min =", xH.min(), " max =", xH.max(), " (must keep <0)")
    # density mass below 0 for H should be nonzero (real negatives preserved)
    yH = np.array(dH['y']); negmass = np.trapz(yH[xH<0], xH[xH<0]) if (xH<0).any() else 0.0
    print("  H density mass at x<0 (approx):", negmass)

# ---- SLQ scree vs exact eigvalsh: compare sorted spectra (single block) ----
def dens_to_sorted(d, p, nsamp=200):
    # mimic client slqScree: density -> CDF -> rank->eigval, then sample
    x=np.array(d['x']); y=np.array(d['y'])
    seg=np.clip(0.5*(y[:-1]+y[1:])*np.diff(x),0,None)
    tot=seg.sum()
    if tot<=0: return None
    # cumulative mass from top
    ranks=[0.0]; vals=[x[-1]]
    acc=0.0
    for i in range(len(x)-2,-1,-1):
        acc+=seg[i]; ranks.append(acc/tot*p); vals.append(x[i])
    return np.array(ranks), np.array(vals)

rk,vl = dens_to_sorted(S.slq_density(hvpH,p,8,min(p,30),120,0x44,block=1),p)
# interpolate exact spectrum onto same ranks (rank 0 = largest)
egH_desc = egH[::-1]
exact_at = np.interp(rk, np.arange(p), egH_desc)
print("\n=== H: SLQ-scree vs exact (indef) ===")
print("  SLQ top 3 eig ~", vl[:3], " exact top 3:", egH_desc[:3])
print("  SLQ bot 3 eig ~", vl[-3:], " exact bot 3:", egH_desc[-3:])
print("  max|SLQ-exact| over ranks:", np.max(np.abs(vl-exact_at)))

# ---- lanczos_extreme_vals check ----
top,bot = S.lanczos_extreme_vals(hvpH, p, 1, min(p,30), 0x44)
print("\nlanczos_extreme_vals H: top", top[0], "vs exact", egH.max(), " bot", bot[0], "vs exact", egH.min())
tG,bG = S.lanczos_extreme_vals(hvpG, p, 1, min(p,30), 0x22)
print("lanczos_extreme_vals G: top", tG[0], "vs exact", egG.max(), " bot", bG[0], "vs exact", egG.min())

# ---- _blockslq_extremes ----
mx,mn = S._blockslq_extremes(hvpH, p)
print("_blockslq_extremes H: max", mx, "vs", egH.max(), " min", mn, "vs", egH.min())
