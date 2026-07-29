#!/usr/bin/env python
"""
reference_predictions.py  —  a SIMPLE, RUNNABLE mirror of the prediction widgets' math.
===========================================================================================
This file is a *teaching* re-implementation of what
    eos_widget_prediction/index_prediction.html            (MSE / regression widget)
    eos_widget_prediction/index_prediction_multiclass.html (CE / softmax widget)
compute and plot. The widgets stream these quantities from server.py over SSE and draw them
with Plotly; here the SAME quantities are written as small, self-contained functions you can
read top-to-bottom, run, and modify.

  * It is NOT the production code. It trades the widgets' speed tricks (hand-written
    forward/backward, matrix-free Lanczos, byte-parity RNG) for clarity: everything here uses
    plain autograd on a tiny MLP, so each formula is one obvious block of code.
  * The math matches the widgets. Where a widget plot is `y = <formula>`, the function here
    returns exactly that. Each function's docstring names the widget section it mirrors and
    the server.py function that computes it in production.
  * Read GUIDE (the companion artifact) alongside this for the "why". This file is the "how".

Run a demo:
    python reference_predictions.py            # trains a tiny MLP, prints a few sections/step
    python reference_predictions.py --steps 60 --width 24

Layout:
    PART 0  notation & the mental model (this docstring + comments below)
    PART 1  the model, the data, the losses            (the substrate everything is built on)
    PART 2  the core objects: J, r, g, NTK, M_r, H, sharpness, HVPs, preconditioners
    PART 3  one function per widget section  (filled in section-by-section, each self-contained)
    PART 4  a demo training loop that calls a handful of them

Notation (used everywhere below)
    theta : flat parameter vector, shape (p,)                 p = number of parameters
    X, Y  : inputs (N, in_dim) and targets (N, out_dim)       N = number of samples
    f(theta) : network outputs, shape (N, out_dim); we flatten to (M,) with M = N*out_dim
    J     : Jacobian df/dtheta, shape (M, p)                  "the Jacobian"
    r     : residual, shape (M,).  MSE: r = Y - f.  CE: r = onehot - softmax(f)   (widget sign)
    g     : gradient of the loss wrt theta = -Jᵀ r / tokens   (the thing GD steps along)
    NTK   : J Jᵀ, shape (M, M)                                "neural tangent kernel"  (output space)
    GN    : Jᵀ J, shape (p, p)  (Gauss-Newton / Fisher)      (parameter space)
    Q_k   : ∇²_theta f_k, shape (p, p)  = Hessian of the k-th scalar output
    M_r   : Σ_k r_k Q_k = residual-weighted function Hessian  (a.k.a. Q_r in the Q-spectrum panel)
    H     : Σ_k Q_k     = unweighted function Hessian         (cotangent ≡ 1)
    Hess  : ∇²_theta L  = full loss Hessian = (1/tokens)(Jᵀ J) - M_r/tokens ... (loss-dependent)
    sharpness λmax(Hess) is the "sharpness"; Edge of Stability is λmax ≈ 2/lr.
"""
import argparse, math
import torch
torch.set_default_dtype(torch.float64)          # fp64 keeps every eig/HVP clean for a teaching run


# ======================================================================================
# PART 1 — the model, the data, the losses
# ======================================================================================
# A plain MLP whose parameters live in ONE flat vector `theta`, so J/HVP/Hessian are all
# straightforward autograd on `theta`. This mirrors server.py's MlpModel (which does the same
# thing with a faster hand-written forward). Layers are [in -> width]* -> out with tanh.

class MLP:
    """Flat-vector MLP. `self.forward(theta, X)` is a pure function of theta (autograd-friendly)."""
    def __init__(self, in_dim, out_dim, width=24, depth=2, act=torch.tanh):
        self.act = act
        dims = [in_dim] + [width] * depth + [out_dim]
        self.shapes, self.p = [], 0
        for a, b in zip(dims[:-1], dims[1:]):                 # each layer: weight (a,b) + bias (b,)
            self.shapes.append(('W', (a, b))); self.p += a * b
            self.shapes.append(('b', (b,)));   self.p += b

    def init_theta(self, seed=0, scale=0.3):
        g = torch.Generator().manual_seed(seed)
        parts = []
        for kind, shape in self.shapes:
            n = math.prod(shape)
            fan_in = shape[0] if kind == 'W' else 1
            parts.append(torch.randn(n, generator=g) * (scale / math.sqrt(fan_in)))
        return torch.cat(parts)

    def _unflatten(self, theta):
        out, off = [], 0
        for kind, shape in self.shapes:
            n = math.prod(shape)
            out.append(theta[off:off + n].view(*shape)); off += n
        return out

    def forward(self, theta, X):
        parts = self._unflatten(theta)
        h = X
        n_layers = len(parts) // 2
        for i in range(n_layers):
            W, b = parts[2 * i], parts[2 * i + 1]
            h = h @ W + b
            if i < n_layers - 1:                              # no activation on the readout layer
                h = self.act(h)
        return h                                              # (N, out_dim)


def mse_residual(f, Y):    # r = Y - f  (widget convention: residual points from prediction to target)
    return (Y - f).reshape(-1)

def mse_loss(f, Y):
    return 0.5 * ((f - Y) ** 2).sum() / f.shape[0]

def ce_residual(f, Y_onehot):   # r = onehot - softmax(f)  (the CE "residual" the multiclass widget uses)
    return (Y_onehot - torch.softmax(f, dim=-1)).reshape(-1)

def ce_loss(f, Y_onehot):
    return -(Y_onehot * torch.log_softmax(f, dim=-1)).sum() / f.shape[0]


# ======================================================================================
# PART 2 — the core objects (J, r, g, NTK, M_r, H, sharpness, HVPs, preconditioners)
# ======================================================================================
# Everything the widgets plot is built from these. Each is a few lines of autograd.

def outputs_flat(model, theta, X):
    """f(theta) flattened to (M,), M = N*out_dim."""
    return model.forward(theta, X).reshape(-1)

def jacobian(model, theta, X):
    """J = df/dtheta, shape (M, p).  Mirrors server.py MlpModel.jac_cols."""
    return torch.autograd.functional.jacobian(lambda t: outputs_flat(model, t, X), theta)

def residual(model, theta, X, Y, loss='mse'):
    """r, shape (M,). MSE: Y-f.  CE: onehot - softmax(f)."""
    f = model.forward(theta, X)
    return mse_residual(f, Y) if loss == 'mse' else ce_residual(f, Y)

def loss_grad(model, theta, X, Y, loss='mse'):
    """g = ∇_theta L, shape (p,). GD steps theta <- theta - lr * g."""
    theta = theta.detach().requires_grad_(True)
    f = model.forward(theta, X)
    L = mse_loss(f, Y) if loss == 'mse' else ce_loss(f, Y)
    (grad,) = torch.autograd.grad(L, theta)
    return grad

def ntk(J):                 # J Jᵀ  (M×M, output space)
    return J @ J.T

def gauss_newton(J):        # Jᵀ J  (p×p, parameter space) — the GN / (for CE) Fisher backbone
    return J.T @ J

def weighted_fn_hessian(model, theta, X, cotangent):
    """Σ_a cotangent_a ∇²f_a  (p×p) = Hessian of the scalar (f·cotangent).  cotangent=r ⇒ M_r; =1 ⇒ H.
    This is the SAME operator the Q-spectrum panel and §20 use (server: hvpS / _qspec_full_eigs)."""
    cot = cotangent.detach()
    return torch.autograd.functional.hessian(
        lambda t: (outputs_flat(model, t, X) * cot).sum(), theta)

def M_r(model, theta, X, Y, loss='mse'):
    """M_r = Σ_k r_k Q_k (residual-weighted function Hessian). Q-spectrum 'Q_r'; §20 operator."""
    r = residual(model, theta, X, Y, loss)
    return weighted_fn_hessian(model, theta, X, r)

def loss_hessian(model, theta, X, Y, loss='mse'):
    """∇²_theta L (p×p). Its top eigenvalue is the 'sharpness'. Full Hessian, not GN — includes M_r term."""
    theta = theta.detach().requires_grad_(True)
    f = model.forward(theta, X)
    L = mse_loss(f, Y) if loss == 'mse' else ce_loss(f, Y)
    return torch.autograd.functional.hessian(
        lambda t: (mse_loss(model.forward(t, X), Y) if loss == 'mse'
                   else ce_loss(model.forward(t, X), Y)), theta)

def sharpness(model, theta, X, Y, loss='mse'):
    """λmax(∇²L) — the sharpness. EoS threshold is 2/lr (§1)."""
    return float(torch.linalg.eigvalsh(loss_hessian(model, theta, X, Y, loss))[-1])


# --- preconditioners P^½ for the adaptive-optimizer sections (Pred-6, self-stab) ---------------
# H_P = P^½ ∇²L P^½ is the "preconditioned Hessian" whose sharpness the adaptive optimizer feels.
def precond_half(theta, g, J, opt='gd'):
    """Return P^½ as a function v ↦ P^½ v.  Mirrors server.py _precond_half.
       gd       : P = I
       sign     : P = diag(1/|g|)          ⇒ P^½ = diag(1/sqrt(|g|))
       gaussnewton: P = (JᵀJ)^+            ⇒ P^½ = (JᵀJ)^{-1/2} = V diag(1/s) Vᵀ (SVD of J)
       (spectral/Muon is layerwise whitening — omitted here for brevity; see server _precond_half)"""
    if opt == 'gd':
        return lambda v: v
    if opt == 'sign':
        d = 1.0 / (g.abs() + 1e-12).sqrt()
        return lambda v: d * v
    if opt == 'gaussnewton':
        # P = (JᵀJ)⁺ ⇒ P^½ = (JᵀJ)^{-1/2} = V diag(1/s) Vᵀ  (acts on the row space of J; zero on its null space).
        # J = U diag(s) Vh (reduced SVD), so V = Vhᵀ and (JᵀJ)^{-1/2} v = Vhᵀ diag(1/s) (Vh v).
        # (Do NOT wrap J on both sides — J.T·U·diag(1/s)·Uᵀ·J collapses to V diag(s) Vᵀ = (JᵀJ)^{+1/2}, the inverse of intended.)
        _U, s, Vh = torch.linalg.svd(J, full_matrices=False)
        inv = torch.where(s > 1e-8 * s.max(), s.reciprocal(), torch.zeros_like(s))
        return lambda v: Vh.t() @ (inv * (Vh @ v))
    raise ValueError(opt)


# ======================================================================================
# PART 3 — one function per widget section
# ======================================================================================
# Each function mirrors ONE plot/panel: its docstring names the widget <h2> title and the
# server.py compute function, and its body is the stripped-to-essentials math. Grouped to
# match the guide (GUIDE.html). For clarity these use DENSE operators (the widgets use
# matrix-free Lanczos for speed); on a tiny MLP the results are identical.

# ---- a couple of shared helpers the section functions lean on ----
def topk_bottomk_eig(Hmat, k):
    """Top-k (descending) then bottom-k (ascending) eigenpairs of a symmetric matrix.
    The widgets get these from Lanczos; here just a dense eigh (tiny MLP)."""
    w, V = torch.linalg.eigh(Hmat)                       # ascending
    idx = list(range(len(w) - 1, len(w) - 1 - k, -1)) + list(range(k))
    return [(float(w[i]), V[:, i]) for i in idx]

def mr_pairs(model, theta, X, Y, k, loss='mse'):
    """Top-k⊕bottom-k eigenpairs of M_r = Σ_k r_k Q_k."""
    return topk_bottomk_eig(M_r(model, theta, X, Y, loss), k)

def per_sample_Q_eig(model, theta, X, k):
    """For each sample-output a, top-k⊕bottom-k eigenpairs of Q_a = ∇²f_a. Returns list over a."""
    M = outputs_flat(model, theta, X).numel()
    out = []
    for a in range(M):
        e = torch.zeros(M); e[a] = 1.0                   # cotangent picks out output a ⇒ Q_a
        out.append(topk_bottomk_eig(weighted_fn_hessian(model, theta, X, e), k))
    return out


# ------------------------------- Core dashboard --------------------------------------
def sec1_dashboard(model, theta, X, Y, lr, loss='mse'):
    """§1 — Loss, sharpness, residual, spectral edges of H. (server: inline main loop.)"""
    f = model.forward(theta, X); N = X.shape[0]
    L = (mse_loss(f, Y) if loss == 'mse' else ce_loss(f, Y)).item()
    r = residual(model, theta, X, Y, loss)
    H = weighted_fn_hessian(model, theta, X, torch.ones_like(f).reshape(-1))   # H = Σ_a Q_a
    evH = torch.linalg.eigvalsh(H)
    return dict(loss=L, sharp=sharpness(model, theta, X, Y, loss), edge=2.0 / lr,
                resid=r, hfMax=float(evH[-1]) / N, hfMin=float(evH[0]) / N)


# ------------------------------- The ★ Predictions -----------------------------------
def d_signal(model, theta, X, Y, lr, loss='mse'):
    """d = ‖J·ṙ‖ − ‖J̇·r‖ = ‖JᵀJ ∇L‖ − ‖M_r ∇L‖.  Its upward zero-crossing = PS onset t*.
    (Prediction 1/2 & the anchor for Prediction-3.)"""
    J = jacobian(model, theta, X)
    gL = loss_grad(model, theta, X, Y, loss)
    Mr = M_r(model, theta, X, Y, loss)
    jrd = (J.T @ (J @ gL)).norm()          # ‖JᵀJ ∇L‖
    jdr = (Mr @ gL).norm()                 # ‖M_r ∇L‖
    return float(jrd - jdr)

def pred2c_alignment(model, theta0, X, Y, k=10, l=4, lr=0.05, loss='mse'):
    """Prediction 2c/2d/2e — init curvature-sign-split alignment vs short-horizon Δ‖J‖.
    (server: run_sweep, sweeppt2c.)  Returns (A+ − A−, normalized, gᵀM_r g, Δ‖J‖)."""
    J = jacobian(model, theta0, X); r = residual(model, theta0, X, Y, loss)
    g = J.T @ r; g2 = float(g @ g) + 1e-30
    Ap = sum(s * float(u @ g) ** 2 for s, u in mr_pairs(model, theta0, X, Y, k, loss) if s > 0) / g2
    An = sum(-s * float(u @ g) ** 2 for s, u in mr_pairs(model, theta0, X, Y, k, loss) if s < 0) / g2
    gMrg = float(g @ (M_r(model, theta0, X, Y, loss) @ g))            # full-operator gᵀM_r g (2e)
    J0n, th = float(J.norm()), theta0.clone()
    for _ in range(l): th = th - lr * loss_grad(model, th, X, Y, loss)
    dJ = float(jacobian(model, th, X).norm()) - J0n                  # ‖J_l‖ − ‖J₀‖
    return (Ap - An, (Ap - An) / (Ap + An + 1e-30), gMrg, dJ)

def pred3_frozenJ_residual(J_star, r_star, lr, N, n_steps):
    """Prediction 3 (fixed J · evolving r) — 1st-order frozen-NTK residual decay after PS onset.
    r_{t+1} = (I − (η/N) J* J*ᵀ) r_t.  (server: g_pred3, inline ~4318.)"""
    r, traj = r_star.clone(), [float(r_star.norm())]
    for _ in range(n_steps):
        r = r - (lr / N) * (J_star @ (J_star.T @ r))
        traj.append(float(r.norm()))
    return traj

def pred4_frozenMr_ntk(J0, Mr, lr, N, n_steps, topk=3):
    """Prediction 4 (fixed r · evolving J) — propagate J by frozen M_r, read top-k NTK eigenvalues.
    J_{s+1} = J_s + (η/N) M_r J_s.  (server: g_pred4, inline ~4422.)"""
    J, traj = J0.clone(), []
    for _ in range(n_steps):
        lam = torch.linalg.eigvalsh(J @ J.T)
        traj.append([float(lam[-1 - i]) for i in range(topk)])
        J = J + (lr / N) * (J @ Mr.T)             # each row J_k gets M_r J_k  (Mr symmetric)
    return traj

def phase2a_ray(model, theta_a, w1, r_a, X, lr, N, m=40, s_max=1.0):
    """Prediction 4.2 — walk a frozen straight ray θ(s)=θ_a+s·w1; geometry (σ1, ‖J‖_F) + clock t(s).
    (server: g_ray, inline ~4532.)  w1 = top M_r eigenvector."""
    s = torch.linspace(0, s_max, m + 1); sig1, jfro, V = [], [], []
    for sj in s:
        Jj = jacobian(model, theta_a + sj * w1, X)
        jfro.append(float(Jj.norm()))
        sig1.append(float(torch.linalg.eigvalsh(Jj @ Jj.T)[-1].clamp_min(0).sqrt()))
        V.append((lr / N) * float((Jj.T @ r_a).norm()))            # speed V(s)
    clock = [0.0]
    for j in range(1, m + 1):
        clock.append(clock[-1] + float(s[j] - s[j - 1]) * 0.5 * (1 / V[j - 1] + 1 / V[j]))
    return dict(s=s.tolist(), sig1=sig1, jfro=jfro, clock=clock)

def trace_forecast(model, theta0, X, Y, lr, N, K, loss='mse'):
    """Prediction 5.1 — Tr(NTK) forecast (quadratic, live residual). ΔJ = (η/N) Q_t[J·r].
    (server: g_trace, inline ~4600.)  Returns trajectory of Tr(J Jᵀ)/N."""
    J0 = jacobian(model, theta0, X); J = J0.clone(); traj = []
    for _ in range(K):
        r = residual(model, theta0, X, Y, loss)                    # 'live' residual (frozen θ0 here)
        g = J.T @ r
        Qg = torch.stack([weighted_fn_hessian(model, theta0, X, torch.eye(J.shape[0])[a]) @ g
                          for a in range(J.shape[0])])              # {Q_a · g}_a  (dense; slow but clear)
        J = J + (lr / N) * Qg
        traj.append(float((J * J).sum()) / N)
    return traj

def sharpness_forecast(model, theta0, X, Y, lr, N, K, loss='mse'):
    """Prediction 5.2 — multiplicative sharpness recursion.
    σ_{t+1} = σ_t[1 + 2(η/N)(r·u1)(v1ᵀ Q[u1] v1)].  (server: §9 thP*.)"""
    J0 = jacobian(model, theta0, X)
    lam, U = torch.linalg.eigh(J0 @ J0.T)
    sig0, u1 = float(lam[-1]), U[:, -1]
    v1 = J0.T @ u1; v1 = v1 / v1.norm()
    r = residual(model, theta0, X, Y, loss); prod = 1.0
    Qu1 = weighted_fn_hessian(model, theta0, X, u1)                 # Σ_a u1_a Q_a  = Q[u1]
    for _ in range(K):
        prod *= 1 + 2 * (lr / N) * float(r @ u1) * float(v1 @ (Qu1 @ v1))
    return sig0 * prod

def opt_direction(model, theta, X, Y, opt, loss='mse'):
    """Prediction 6 — the actual update direction d (θ ← θ − lr·d) per optimizer.
    (server: _opt_dir.)  gd/sign/spectral/gaussnewton."""
    gL = loss_grad(model, theta, X, Y, loss)
    if opt == 'gd':   return gL
    if opt == 'sign': return gL.sign()
    if opt == 'gaussnewton':
        J = jacobian(model, theta, X); r = residual(model, theta, X, Y, loss)
        return -J.T @ torch.linalg.solve(J @ J.T + 1e-6 * torch.eye(J.shape[0]), r)
    raise ValueError("spectral/Muon is per-weight-matrix; see server _muon_whiten")


# ------------------------------- New diagnostic sections -----------------------------
def eds_projection_sums(model, theta, X, Y, k=50, loss='mse'):
    """Early-dynamics plot 1 — σ-weighted split of Jᵀr onto ±M_r eigenvectors.
    (server: g_eds via _eds_eigpairs.)"""
    J = jacobian(model, theta, X); Jtr = J.T @ residual(model, theta, X, Y, loss)
    n = Jtr.norm().clamp_min(1e-30); pos = neg = 0.0
    for lam, v in mr_pairs(model, theta, X, Y, k, loss):
        pr = abs(lam) * abs(float(v @ Jtr)) / float(n)
        pos += pr if lam > 0 else 0.0; neg += pr if lam < 0 else 0.0
    return pos, neg

def mean_principal_angle(Vt, Vlag):
    """Early-dynamics plots 2-5 / §12a — mean principal angle (deg) between two eigenspace bases.
    Vt, Vlag are (k, p) row-orthonormal."""
    G = Vt @ Vlag.T
    s2 = torch.linalg.eigvalsh(G @ G.T).clamp(0, 1)
    return float(torch.rad2deg(torch.acos(s2.sqrt().clamp(-1, 1))).mean())

def selfstab_cosines(model, theta, X, Y, opt='gd', ssk=5, loss='mse'):
    """Self-stabilization — cos(∇S, ±ssk eigvecs of H_P) + two projected-gradient cosines.
    (server: _selfstab_payload / _selfstab_gradS.)  ∇L is NEVER preconditioned; only the Hessian is.
    H_P = P^½ ∇²L P^½;  S = λmax(H_P);  ∇S = ∇_θ S.
    ∇S here uses the envelope theorem with P FROZEN:  ∇S = ∇³L[w,w,·],  w = P^½ u₁  — computed by
    a triple-backward. This is EXACT for gd. For adaptive optimizers the TRUE ∇S adds a
    2(∂w/∂θ)ᵀ(∇²L w) term from ∂P/∂θ (which actually dominates) — see server.py _selfstab_gradS."""
    gL = loss_grad(model, theta, X, Y, loss)
    J = jacobian(model, theta, X)
    Ph = precond_half(theta, gL, J, opt)                          # frozen P^½ (v ↦ P^½ v)
    p = theta.numel()
    HL = loss_hessian(model, theta, X, Y, loss)                   # dense ∇²L (one autograd Hessian)
    Pmat = torch.stack([Ph(torch.eye(p)[i]) for i in range(p)])   # P^½ as a (symmetric) matrix
    HP = Pmat @ HL @ Pmat                                         # H_P = P^½ ∇²L P^½
    w_, U = torch.linalg.eigh(0.5 * (HP + HP.T))
    tv = [U[:, -1 - i] for i in range(ssk)]; bv = [U[:, i] for i in range(ssk)]
    u1 = tv[0]; wvec = (Pmat @ u1).detach()                       # w = P^½ u₁ (frozen)
    thg = theta.detach().requires_grad_(True)                     # ∇S = ∇³L[w,w,·] via triple-backward
    L = (mse_loss(model.forward(thg, X), Y) if loss == 'mse' else ce_loss(model.forward(thg, X), Y))
    g1, = torch.autograd.grad(L, thg, create_graph=True)
    hvp, = torch.autograd.grad((g1 * wvec).sum(), thg, create_graph=True)   # ∇²L w
    gS, = torch.autograd.grad((hvp * wvec).sum(), thg)                      # ∇³L[w,w,·]
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-30))
    sh = gS / gS.norm()
    return dict(top=[cos(gS, u) for u in tv], bot=[cos(gS, u) for u in bv],
                p3=cos(gL - (u1 @ gL) * u1, gS), p4=cos(gL - (sh @ gL) * sh, u1))

def qspec_full_eigs(model, theta, X, cotangent):
    """Q eigenspectrum — the FULL spectrum (all p eigenvalues, descending) of Σ_a c_a Q_a.
    cotangent = r ⇒ Q_r; cotangent = 1 ⇒ H.  (server: _qspec_full_eigs.)"""
    ev = torch.linalg.eigvalsh(weighted_fn_hessian(model, theta, X, cotangent))
    return sorted((float(x) for x in ev), reverse=True)


# ------------------------------- Inherited diagnostics -------------------------------
def sec11_grids(Jgt, Qapply, u, r):
    """§11 — 3D tensors T1[i,j,k]=Jᵢᵀ Qⱼ Jₖ, T2=uⱼuₖ·T1, T3=rᵢuⱼuₖ·T1. (server: g3d, inline.)
    Jgt: (N,p) per-sample grads; Qapply(k)->(N,p) = {Q_j · J_k}_j; u,r: (N,)."""
    N = Jgt.shape[0]
    T1 = torch.stack([Jgt @ Qapply(k).T for k in range(N)], dim=2)   # [i,j,k]
    T2 = T1 * u.view(1, N, 1) * u.view(1, 1, N)
    T3 = T2 * r.view(N, 1, 1)
    return T1, T2, T3

def sec12a_angles(TV, BV, ks=(1, 2, 5, 15)):
    """§12a — mean principal angle (deg) between per-sample Qᵢ,Qⱼ eigenspaces + diagonal alignment.
    (server: _sec12_payload.)  TV,BV: (N,K,p) top/bottom per-sample eigenvectors."""
    N = TV.shape[0]; off = ~torch.eye(N, dtype=bool); ang, dal = {}, {}
    for k in ks:
        E = torch.cat([TV[:, :k], BV[:, :k]], 1)                     # (N,2k,p)
        cos = torch.linalg.svdvals(torch.einsum('iap,jbp->ijab', E, E)).clamp(-1, 1)
        deg = (torch.rad2deg(torch.acos(cos))).mean(-1)             # (N,N)
        ang[k] = dict(mn=float(deg[off].min()), mx=float(deg[off].max()), me=float(deg[off].mean()))
        d = torch.einsum('iap,jap->ija', E, E).abs().mean(-1)
        dal[k] = float(d[off].mean())
    return ang, dal

def sec12b_proj(TV, TW, BV, BW, r, Jg, nrank=2):
    """§12b — gradient ∇fᵢ onto top/bottom-nrank eigvecs of Qᵢ: product |⟨J,u⟩|σr and alignment |cos|.
    (server: _sec12_payload.)  TV/BV:(N,K,p) eigvecs, TW/BW:(N,K) eigvals, r:(N,), Jg:(N,p)."""
    jn = Jg.norm(dim=1); N = len(Jg); out = {}
    for name, V, W in (("top", TV, TW), ("bot", BV, BW)):
        ranks = []
        for rk in range(nrank):
            u, s = V[:, rk], W[:, rk]                                # (N,p),(N,)
            jd = (Jg * u).sum(1)
            prod, cos = jd.abs() * s * r, jd.abs() / jn.clamp_min(1e-30)
            ranks.append(dict(prod=(float(prod.mean()), float(prod.std())),
                              cos=(float(cos.mean()), float(cos.std()))))
        out[name] = ranks
    return out

def sec18_persample(proj, sample=0):
    """§18 — the §12b projections for ONE sample over training (3 phases). (reuses _sec12_payload.)"""
    return {(grp, rk): dict(alignment=e.get('vbysamp'), power_iteration=e.get('jbysamp'),
                            residual_dom=e.get('pbysamp'))
            for grp in ('top', 'bot') for rk, e in enumerate(proj[grp])}

def sec6_alignment(J, r, Mr, N, K):
    """§6 — r onto NTK eigvecs; J·r onto M_r eigvecs (with eigenvalue weightings).
    (server: _sec21_payload.)"""
    s2, Vn = torch.linalg.eigh(J @ J.T); s2, Vn = s2.flip(0), Vn.flip(1)   # NTK, desc
    sig = s2[:K] / N; n1 = (Vn[:, :K].T @ r).abs() / r.norm().clamp_min(1e-30)
    Jr = J.T @ r; pairs = topk_bottomk_eig(Mr, K)
    lam = torch.tensor([p[0] for p in pairs]) / N
    p1 = torch.tensor([abs(float(p[1] @ Jr)) for p in pairs]) / Jr.norm().clamp_min(1e-30)
    return dict(sig=sig, n1=n1, n3=sig * n1, lam=lam, p1=p1, p3=lam * p1)

def sec19_gradnorm_vs_D2NTK(model, theta, X, Y, lr, N, loss='mse'):
    """§19 — one-step Δ‖grad‖ (A) and Tr(NTK) (caller forms B = 2nd diff of trNTK).
    (server: _sec19_payload.)  MSE+GD only."""
    J = jacobian(model, theta, X); r = residual(model, theta, X, Y, loss)
    g = J.T @ r; etaN = lr / N; Jgg = J @ g
    r_next = r - etaN * Jgg
    Hg = M_r(model, theta, X, Y, loss) @ g if False else weighted_fn_hessian(
        model, theta, X, r_next.reshape(-1)) @ g                    # (Σ_k r_{t+1,k}Q_k) g
    g_next = g - etaN * (J.T @ Jgg) + etaN * Hg
    return float(g_next.norm() - g.norm()), float((J * J).sum())

def sec20_Mr_histogram(J, Mr, N, K):
    """§20 — M_r spectrum λᵢ and q_k[i] = λᵢ·|⟨vᵢ,uₖ⟩|, uₖ = normalize(Jᵀ gₖ) (NTK dir).
    (server: _sec20_payload.)"""
    pairs = topk_bottomk_eig(Mr, K); lam = torch.tensor([p[0] for p in pairs]) / N
    Vv = torch.stack([p[1] for p in pairs], 1)                      # (p, 2K)
    s2, G = torch.linalg.eigh(J @ J.T); G = G.flip(1)
    q = []
    for k in range(4):
        uk = J.T @ G[:, k]; uk = uk / uk.norm().clamp_min(1e-30)
        q.append([float(lam[i]) * abs(float(Vv[:, i] @ uk)) for i in range(Vv.shape[1])])
    return lam, q

def sec24_first_vs_second_order(model, theta, X, Y, lr, N, loss='mse'):
    """§24 — A = J Jᵀ r (NTK·r) and B_k = (η/2N)(J·r)ᵀ Q_k (J·r); alignments with r and NTK eigvecs.
    (server: g24, inline.)"""
    J = jacobian(model, theta, X); r = residual(model, theta, X, Y, loss)
    M = J.shape[0]; Jr = J.T @ r; A = J @ Jr
    B = (lr / (2 * N)) * torch.tensor(                              # B_a = (η/2N)(J·r)ᵀ Q_a (J·r)
        [float(Jr @ (weighted_fn_hessian(model, theta, X, torch.eye(M)[a]) @ Jr)) for a in range(M)])
    s2, U = torch.linalg.eigh(J @ J.T); U = U.flip(1)
    cos = lambda v, u: abs(float(v @ u)) / (v.norm() * u.norm() + 1e-30)
    return dict(A=A, B=B, cAr=cos(A, r), cBr=cos(B, r),
                cAu=[cos(A, U[:, i]) for i in range(min(4, J.shape[0]))])

def sec27_project(window_vecs, top=3):
    """§7 — project the window-center ∇θ vector into the window's top-3 singular subspace.
    (server: _sec27_top_eig.)  window_vecs: list of W p-vectors for one metric."""
    A = torch.stack(window_vecs, 1)                                 # (p, W)
    lam, V = torch.linalg.eigh(A.T @ A)
    idx = torch.argsort(lam, descending=True)[:top]
    sig, center = lam[idx].clamp_min(0).sqrt(), len(window_vecs) // 2
    return [float(sig[j] * V[center, idx[j]]) for j in range(top)]

# §13 (Jᵢᵀ Qⱼ Jₖ rₖ rank-k approximations) and §14 (per-triplet Tr(ΔNTK) 5-factor decomposition)
# are intricate multi-index sums best read directly in server.py (_sec13_stats / _sec14_payload) or
# the GUIDE cards; their reference sketches live in those cards. The exact reference value they
# approximate is simply:
def sec13_exact(J, Q, r):
    """§13 exact target: E[i,j,k] = Jᵢᵀ Qⱼ Jₖ rₖ.  Q: (N,p,p) per-sample function Hessians.
    (server: jac_hvp reference for _sec13_stats.)"""
    N = J.shape[0]
    E = torch.zeros(N, N, N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                E[i, j, k] = float(J[i] @ (Q[j] @ J[k]) * r[k])
    return E


# ======================================================================================
# PART 4 — demo
# ======================================================================================
def demo(steps=40, width=24, lr=0.05, loss='mse', seed=0):
    torch.manual_seed(seed)
    N, in_dim, out_dim = 16, 8, 4
    model = MLP(in_dim, out_dim, width=width, depth=2)
    theta = model.init_theta(seed=seed)
    X = torch.randn(N, in_dim)
    Y = torch.randn(N, out_dim) if loss == 'mse' else \
        torch.eye(out_dim)[torch.randint(0, out_dim, (N,))]
    print(f"MLP p={model.p}  N={N}  loss={loss}  lr={lr}")
    for t in range(steps):
        g = loss_grad(model, theta, X, Y, loss)
        if t % max(1, steps // 8) == 0:
            f = model.forward(theta, X)
            L = (mse_loss(f, Y) if loss == 'mse' else ce_loss(f, Y)).item()
            sh = sharpness(model, theta, X, Y, loss)
            print(f"  t={t:4d}  loss={L:.4e}  sharpness={sh:8.3f}  2/lr={2/lr:6.1f}  "
                  f"|g|={g.norm():.3e}  {'← above EoS' if sh > 2/lr else ''}")
        theta = theta - lr * g
    return model, theta, X, Y


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--width', type=int, default=24)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--loss', choices=['mse', 'ce'], default='mse')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    demo(a.steps, a.width, a.lr, a.loss, a.seed)
