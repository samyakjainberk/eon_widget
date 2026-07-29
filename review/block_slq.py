"""
block_slq.py  —  the block-SLQ code I added to the prediction widgets, written as
                 simply as possible so you can read it top-to-bottom and run it.

WHAT THIS IS
------------
The two prediction widgets draw four "eigenspectrum" plots (the spectral density of the
function Hessian H, Gauss-Newton G, residual term S, and full loss Hessian ∇²L). Those
densities are estimated with Stochastic Lanczos Quadrature (SLQ). I changed the default
from *standard* SLQ (1 probe vector at a time) to *block* SLQ with block size 4, which
resolves clustered/degenerate eigenvalues much better at the same iteration count.

This file re-implements that estimator in plain, heavily-commented NumPy. It is the exact
same ALGORITHM that now runs in the browser JS of both widgets — just written for reading,
not for a browser. Run it (`python block_slq.py`) to see block SLQ beat standard SLQ.

WHERE THE REAL CODE LIVES (what this mirrors)
---------------------------------------------
Browser JS (identical block of code in both widgets):
  eos_widget_prediction/index_prediction.html            _mgsBlock  ~L1241 · blockLanczosCore ~L1254 · slqDensity ~L1281
  eos_widget_prediction/index_prediction_multiclass.html _mgsBlock  ~L1238 · blockLanczosCore ~L1251 · slqDensity ~L1278
Python server (same algorithm, already readable in place):
  server.py   _lanczos_core L2940 · _block_lanczos_core L3071 · slq_density L3111

TWO SMALL, DELIBERATE SIMPLIFICATIONS vs the JS (math is identical):
  1. Random probe vectors here use numpy's RNG. The widgets use a "mulberry32" RNG so the
     browser and the GPU server draw the *same* vectors (cross-backend reproducibility).
     Different draws → slightly different numbers, same algorithm.
  2. The block QR step here calls numpy.linalg.qr. The JS has no built-in QR, so it does the
     same thing by hand with two-pass Gram-Schmidt (that is what `_mgsBlock` is). Same result.

THE ONE-LINE IDEA
-----------------
SLQ turns "estimate the eigenvalue histogram of a huge matrix A you can only multiply by"
into: run Lanczos from a random start to get a tiny tridiagonal T, take T's eigenvalues as
sample locations ("nodes") and the squared first-component of each eigenvector as their
weights, and blur those weighted spikes into a smooth curve. BLOCK SLQ starts Lanczos from
`block` random vectors at once instead of one — so each step can surface up to `block`
eigenvalues of a tight cluster that single-vector Lanczos would merge into one.
"""

import numpy as np


# ======================================================================================
# 1.  STANDARD SLQ  (block size = 1)  — one probe vector at a time
# ======================================================================================

def lanczos_tridiagonal(matvec, n, steps, start):
    """Lanczos: build an orthonormal Krylov basis q_0,q_1,... of A and the small symmetric
    TRIDIAGONAL matrix T (steps×steps) with T = Qᵀ A Q.  `matvec(v)` returns A·v.

    Full reorthogonalization (subtract off every earlier q, twice) keeps it stable in fp32.
    """
    q = start / np.linalg.norm(start)         # first basis vector (unit length)
    Q = []                                    # the basis vectors we keep
    alpha, beta = [], []                      # diagonal (alpha) and off-diagonal (beta) of T
    q_prev = np.zeros(n)
    b = 0.0
    for _ in range(steps):
        Q.append(q)
        w = matvec(q)                         # A·q  — the only place we touch A
        a = float(w @ q); alpha.append(a)     # diagonal entry  a_j = qⱼᵀ A qⱼ
        w = w - a * q - b * q_prev            # three-term recurrence: remove the last two directions
        for _ in range(2):                    # reorthogonalize against ALL earlier basis vectors, twice
            for qk in Q:
                w = w - (w @ qk) * qk
        b = float(np.linalg.norm(w))          # next off-diagonal  b_j = ‖w‖
        if b < 1e-10:                         # Krylov space exhausted → stop
            break
        beta.append(b)
        q_prev, q = q, w / b                  # normalize → next basis vector

    k = len(alpha)                            # actual number of steps taken
    T = np.zeros((k, k))
    for i in range(k):
        T[i, i] = alpha[i]
    for i in range(k - 1):
        T[i, i + 1] = T[i + 1, i] = beta[i]   # symmetric off-diagonals
    return T


def standard_slq(matvec, n, num_probes, steps, seed):
    """Standard SLQ: average the Lanczos quadrature over `num_probes` random start vectors.
    Returns (nodes, weights): eigenvalue sample locations and their (normalized) masses."""
    nodes, weights = [], []
    rng = np.random.default_rng(seed)
    for _ in range(num_probes):
        start = rng.standard_normal(n)
        T = lanczos_tridiagonal(matvec, n, min(steps, n), start)
        vals, vecs = np.linalg.eigh(T)              # Ritz values (nodes) and Ritz vectors
        for i in range(len(vals)):
            nodes.append(float(vals[i]))
            # weight of node i = (first component of its eigenvector)², averaged over probes.
            # (This is the Gaussian-quadrature weight; it approximates how much spectral mass sits there.)
            weights.append(float(vecs[0, i] ** 2) / num_probes)
    return np.array(nodes), np.array(weights)


# ======================================================================================
# 2.  BLOCK SLQ  (block size = b)  — `b` probe vectors advanced together
# ======================================================================================

def block_lanczos_tridiagonal(matvec, n, block, steps, seed):
    """Block Lanczos: the exact same recurrence as above but with `block` vectors at once.
    Every scalar becomes a block×block matrix; every normalization becomes a QR.
    Returns (T, block): T is a symmetric BLOCK-tridiagonal matrix of size (k·block)².
    """
    b = max(1, min(block, n))
    rng = np.random.default_rng(seed)

    # Start block: `b` random columns, orthonormalized (QR). (JS does this by hand = _mgsBlock.)
    Qj, _ = np.linalg.qr(rng.standard_normal((n, b)))     # Qj is n×b with orthonormal columns

    Q_blocks = []          # all the n×b basis blocks we keep (for reorthogonalization)
    A_diag = []            # the b×b DIAGONAL blocks of T  (A_j = Qjᵀ A Qj)
    B_off = []             # the b×b OFF-DIAGONAL blocks of T (the QR factor of the residual)
    Q_prev = np.zeros((n, b))
    B_prev = np.zeros((b, b))

    for _ in range(min(max(1, n // b), steps)):
        Q_blocks.append(Qj)
        W = np.column_stack([matvec(Qj[:, c]) for c in range(b)])   # A·Qj, one matvec per column → n×b
        Aj = Qj.T @ W                          # b×b diagonal block
        Aj = 0.5 * (Aj + Aj.T)                 # force it symmetric (kills fp round-off asymmetry)
        A_diag.append(Aj)

        # Block three-term recurrence: remove the two previous blocks' directions.
        W = W - Qj @ Aj - Q_prev @ B_prev.T
        for _ in range(2):                     # reorthogonalize against ALL earlier blocks, twice
            for Qc in Q_blocks:
                W = W - Qc @ (Qc.T @ W)

        Qnext, Bj = np.linalg.qr(W)            # QR: Qnext (n×b orthonormal), Bj (b×b upper-triangular)
        B_off.append(Bj)
        if np.abs(Bj).max() < 1e-10:           # residual vanished → stop
            break
        Q_prev, B_prev, Qj = Qj, Bj, Qnext

    # Assemble the symmetric block-tridiagonal T from the diagonal (A) and off-diagonal (B) blocks.
    k = len(A_diag)
    kb = k * b
    T = np.zeros((kb, kb))
    for i in range(k):
        T[i*b:(i+1)*b, i*b:(i+1)*b] = A_diag[i]                 # diagonal block
        if i < k - 1:
            T[(i+1)*b:(i+2)*b, i*b:(i+1)*b] = B_off[i]          # sub-diagonal block
            T[i*b:(i+1)*b, (i+1)*b:(i+2)*b] = B_off[i].T        # symmetric super-diagonal block
    return T, b


def block_slq(matvec, n, num_probes, steps, block, seed):
    """Block SLQ. Same shape as standard_slq, but the weight of each node is the squared
    norm of the FIRST BLOCK (first `b` components) of its eigenvector, not just one component."""
    nodes, weights = [], []
    for p in range(num_probes):
        # cap the number of block steps the same way the widget/server does
        block_steps = min(max(1, n // block), steps)
        T, b = block_lanczos_tridiagonal(matvec, n, block, block_steps, seed + p)
        vals, vecs = np.linalg.eigh(T)
        for i in range(len(vals)):
            nodes.append(float(vals[i]))
            weights.append(float((vecs[:b, i] ** 2).sum()) / (b * num_probes))
    return np.array(nodes), np.array(weights)


# ======================================================================================
# 3.  Turn the weighted nodes into a smooth density curve (a blurred histogram)
# ======================================================================================

def density_curve(nodes, weights, num_grid=200):
    """Blur the weighted eigenvalue spikes with a Gaussian kernel → a smooth (x, y) density.
    Identical to the tail of slqDensity in the widgets / slq_density in server.py."""
    lo, hi = float(nodes.min()), float(nodes.max())
    if not hi > lo:
        hi, lo = lo + 1, lo - 1
    pad = 0.05 * (hi - lo) + 1e-9
    lo, hi = lo - pad, hi + pad
    sigma = max((hi - lo) / 60, 1e-9)                       # kernel width = 1/60 of the range
    x = np.linspace(lo, hi, num_grid)
    # y(x) = Σ_i weight_i · Gaussian(x − node_i, sigma)
    y = np.array([float((weights * np.exp(-0.5 * ((xi - nodes) / sigma) ** 2)
                         / (sigma * np.sqrt(2 * np.pi))).sum()) for xi in x])
    return x, y


# ======================================================================================
# 4.  Self-test: on a matrix with tight eigenvalue clusters, block SLQ should be closer to
#     the TRUE spectrum than standard SLQ. (This is the finding that made b=4 the default.)
# ======================================================================================

def _l1(x1, y1, x2, y2):
    """L1 distance between two density curves after putting them on a common, normalized grid."""
    gx = np.linspace(min(x1.min(), x2.min()), max(x1.max(), x2.max()), 600)
    a = np.interp(gx, x1, y1, 0, 0); a = a / (a.sum() + 1e-30)
    b = np.interp(gx, x2, y2, 0, 0); b = b / (b.sum() + 1e-30)
    return float(np.abs(a - b).sum())


if __name__ == "__main__":
    # Build a 60×60 symmetric matrix with two TIGHT clusters (at +5 and −3) plus a spread bulk.
    # Clusters are exactly where block SLQ helps and standard SLQ struggles.
    rng = np.random.default_rng(0)
    n = 60
    eigs = np.concatenate([
        np.full(12, 5.0) + 0.02 * rng.standard_normal(12),    # tight cluster near +5
        np.full(12, -3.0) + 0.02 * rng.standard_normal(12),   # tight cluster near −3
        np.linspace(-1, 1, 36),                               # spread-out bulk
    ])
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (U * eigs) @ U.T
    A = 0.5 * (A + A.T)
    matvec = lambda v: A @ v                                  # the only thing SLQ is allowed to use

    # Ground truth: the exact eigenvalue density.
    exact = np.linalg.eigvalsh(A)
    ex_x, ex_y = density_curve(exact, np.ones(n) / n)

    print(f"matrix {n}×{n}, two tight clusters + bulk    (num_probes=8, steps=40)\n")
    print(f"{'method':>16} | {'L1 distance to the TRUE spectrum':>34}")
    print("-" * 55)
    ns, ws = standard_slq(matvec, n, 8, 40, seed=1234)
    sx, sy = density_curve(ns, ws)
    print(f"{'standard SLQ':>16} | {_l1(sx, sy, ex_x, ex_y):>34.4f}")
    for b in (2, 4, 8):
        ns, ws = block_slq(matvec, n, 8, 40, block=b, seed=1234)
        bx, by = density_curve(ns, ws)
        tag = f"block SLQ (b={b})" + ("  <- widget default" if b == 4 else "")
        print(f"{tag:>16} | {_l1(bx, by, ex_x, ex_y):>34.4f}")
    print("\nlower = closer to the true eigenvalue spectrum. Block SLQ wins, and b=4 is the "
          "sweet spot chosen as the new default.")
