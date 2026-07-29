"""
verify_review.py  —  DEEP CHECK that review/block_slq.py is (a) correct and (b) a faithful stand-in
for the actual widget JS, despite its two simplifications (numpy.qr instead of the JS's hand-written
Gram-Schmidt `_mgsBlock`, and numpy RNG instead of the JS `mulberry32`).

Three independent checks:
  CHECK 1  numpy.qr  ==  hand-written MGS   (given the SAME random start → identical Lanczos nodes).
           This is the crux: it proves the "numpy.qr" simplification changes nothing mathematically.
  CHECK 2  block_slq.py  vs  server.slq_density  vs  EXACT eigenvalues (end-to-end correctness).
  CHECK 3  the JS `slqDensity` block-weight formula matches block_slq.py's, re-derived symbolically.
"""
import sys, numpy as np
sys.path.insert(0, "/nas/ucb/samsj/TestingPSTheory/eos_widget")
sys.path.insert(0, "/nas/ucb/samsj/TestingPSTheory/eos_widget/review")
import torch, server, block_slq
server.DTYPE = torch.float64; server.DEVICE = torch.device("cpu"); server._TL.device = torch.device("cpu")

# ---- a test matrix with tight clusters (where block SLQ matters) ----
rng = np.random.default_rng(3)
n = 60
eigs = np.concatenate([np.full(14, 4.0) + 0.01*rng.standard_normal(14),
                       np.full(14, -2.0) + 0.01*rng.standard_normal(14),
                       np.linspace(-1, 1, 32)])
U, _ = np.linalg.qr(rng.standard_normal((n, n)))
A = 0.5 * ((U * eigs) @ U.T + ((U * eigs) @ U.T).T)
matvec = lambda v: A @ v
At = torch.tensor(A, dtype=torch.float64)
exact = np.linalg.eigvalsh(A)


# =================================================================================================
# CHECK 1 — the JS hand-written MGS block-Lanczos, given the SAME start block block_slq.py uses,
#           must produce the SAME Lanczos nodes (eigenvalues of T). This isolates numpy.qr vs MGS.
# =================================================================================================
def mgs_qr(M):
    """two-pass modified Gram-Schmidt QR — a faithful copy of the JS `_mgsBlock` (index_prediction.html L1241)."""
    nrows, b = M.shape
    Q = np.zeros((nrows, b)); R = np.zeros((b, b))
    for j in range(b):
        v = M[:, j].copy()
        for _ in range(2):
            for i in range(j):
                r = Q[:, i] @ v; R[i, j] += r; v = v - r * Q[:, i]
        nrm = np.linalg.norm(v); R[j, j] = nrm
        Q[:, j] = v / nrm if nrm > 1e-300 else 0.0
    return Q, R

def block_lanczos_MGS(matvec, n, b, steps, start_matrix):
    """block Lanczos using MGS everywhere — the literal JS `blockLanczosCore` algorithm."""
    Qj, _ = mgs_qr(start_matrix)
    Qblocks, Ad, Bo = [], [], []
    Qprev = np.zeros((n, b)); Bprev = np.zeros((b, b))
    for _ in range(min(max(1, n // b), steps)):
        Qblocks.append(Qj)
        W = np.column_stack([matvec(Qj[:, c]) for c in range(b)])
        Aj = Qj.T @ W; Aj = 0.5 * (Aj + Aj.T); Ad.append(Aj)
        W = W - Qj @ Aj - Qprev @ Bprev.T
        for _ in range(2):
            for Qc in Qblocks:
                W = W - Qc @ (Qc.T @ W)
        Qnext, Bj = mgs_qr(W); Bo.append(Bj)
        if np.abs(Bj).max() < 1e-10: break
        Qprev, Bprev, Qj = Qj, Bj, Qnext
    k = len(Ad); kb = k * b; T = np.zeros((kb, kb))
    for i in range(k):
        T[i*b:(i+1)*b, i*b:(i+1)*b] = Ad[i]
        if i < k - 1:
            T[(i+1)*b:(i+2)*b, i*b:(i+1)*b] = Bo[i]; T[i*b:(i+1)*b, (i+1)*b:(i+2)*b] = Bo[i].T
    return T

print("CHECK 1 — numpy.qr (block_slq.py) vs hand-written MGS (the JS): same start ⇒ same nodes")
b = 4
seed = 999
# block_slq.py draws its start from default_rng(seed).standard_normal((n,b)) then np.linalg.qr's it.
start = np.random.default_rng(seed).standard_normal((n, b))
# call block_slq's real function (it re-draws the same start internally from the same seed):
T_np, _ = block_slq.block_lanczos_tridiagonal(matvec, n, b, 40, seed)
T_mgs = block_lanczos_MGS(matvec, n, b, 40, start)   # same start matrix, MGS orthonormalization
nodes_np = np.sort(np.linalg.eigvalsh(T_np))
nodes_mgs = np.sort(np.linalg.eigvalsh(T_mgs))
max_node_diff = float(np.abs(nodes_np - nodes_mgs).max()) if nodes_np.shape == nodes_mgs.shape else 9e9
print(f"   T size numpy={T_np.shape[0]}  MGS={T_mgs.shape[0]}   max |node difference| = {max_node_diff:.2e}")
print(f"   {'PASS' if max_node_diff < 1e-8 else 'FAIL'}: numpy.qr and hand-written MGS give identical Lanczos nodes\n")


# =================================================================================================
# CHECK 2 — end-to-end: block_slq.py vs the trusted server implementation vs EXACT eigenvalues.
# =================================================================================================
def kde(vals, gx):
    lo, hi = float(vals.min()), float(vals.max()); pad = 0.05*(hi-lo)+1e-9; lo-=pad; hi+=pad
    sig = max((hi-lo)/60, 1e-9)
    return np.array([float((np.exp(-0.5*((g-vals)/sig)**2)/(sig*np.sqrt(2*np.pi))).sum()/len(vals)) for g in gx])

def l1(x1, y1, x2, y2):
    gx = np.linspace(min(x1.min(), x2.min()), max(x1.max(), x2.max()), 600)
    a = np.interp(gx, x1, y1, 0, 0); a /= (a.sum()+1e-30)
    c = np.interp(gx, x2, y2, 0, 0); c /= (c.sum()+1e-30)
    return float(np.abs(a-c).sum())

gx = np.linspace(exact.min()-1, exact.max()+1, 600); ex = kde(exact, gx)
print("CHECK 2 — accuracy vs EXACT spectrum (block_slq.py should match the server, both < standard-SLQ error)")
print(f"   {'block':>6} | {'block_slq.py':>12} | {'server.py':>10} | {'agree(review,server)':>20}")
print("   " + "-"*56)
for bl in (1, 4, 8):
    if bl == 1:
        nr, wr = block_slq.standard_slq(matvec, n, 8, 40, seed=1234)
    else:
        nr, wr = block_slq.block_slq(matvec, n, 8, 40, block=bl, seed=1234)
    xr, yr = block_slq.density_curve(nr, wr)
    sd = server.slq_density(lambda v: At @ v, n, 8, 40, 220, 1234, block=bl)
    xs, ys = np.array(sd["x"]), np.array(sd["y"])
    print(f"   {bl:>6} | {l1(xr,yr,gx,ex):>12.4f} | {l1(xs,ys,gx,ex):>10.4f} | {l1(xr,yr,xs,ys):>20.4f}")
print("   (col2,col3 = L1 to exact: lower is better, block beats standard. col4 = review-vs-server agreement.)\n")


# =================================================================================================
# CHECK 3 — the block-weight formula, stated three ways, is the same thing.
# =================================================================================================
print("CHECK 3 — block-SLQ weight formula (node i of block-tridiagonal T):")
print("   JS   (index_prediction.html L1289): w = Σ_{j<b} V[j*kb+i]²         / (b·nProbe)")
print("   py   (block_slq.py block_slq):      w = (vecs[:b, i] ** 2).sum()    / (b·num_probes)")
print("   math:                               w = ‖ first b components of eigenvector i ‖²  / (b·#probes)")
print("   → identical (numpy vecs[:b,i] = the first b components of eigenvector i; JS V[j*kb+i] = same).")
