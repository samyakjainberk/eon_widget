#!/usr/bin/env python
"""Generate the EIGENSPECTRUM-OVERLAY data for the results webpage.

For each run (tanh-teacher on chebyshev, one per optimizer sgd/adam/muon/gn) we train a small MLP,
then at the FINAL iteration compute the EXACT sorted, signed eigenspectrum of four operators:
    hessian = ∇²L            (full loss Hessian,      hvpL)
    sumQ    = Σ_k ∇²f_k       (function Hessian ΣQ_i,  hvpF)
    qr      = Σ_k r_k ∇²f_k   (residual-weighted,      hvpS · r)   [÷N to mean-loss scale]
    jjt     = J Jᵀ            (NTK / Gauss-Newton spectrum; we eig the p×p JᵀJ, whose NONZERO
                               eigenvalues equal JJᵀ's, so it overlays on the same rank axis)
Each spectrum is log-subsampled to ~MAX_SCREE ranks and stored as [rank, signed_lambda].

  python gen_spectra.py --device cuda:0 --out spectra.json
"""
import os, sys, json, argparse, time
import numpy as np, torch

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
sys.path.insert(0, DIR)
import server, capture_run

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="spectra.json")
ap.add_argument("--width", type=int, default=24)
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--nsamp", type=int, default=100)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--maxscree", type=int, default=110)
A = ap.parse_args()

dev = torch.device(A.device)
dtype = torch.float32 if dev.type == "cuda" else torch.float64
server.DTYPE = dtype; server.DEVICE = dev; server._TL.device = dev; server._TL.cifar_dir = None
if dev.type == "cuda": torch.cuda.set_device(dev)

# each run = (label, optimizer, learning-rate). Chebyshev = a fixed scalar "teacher" T_3; tanh MLP student.
RUNS = [
    ("tanh-teacher / sgd",  "sgd",  0.10),
    ("tanh-teacher / adam", "adam", 0.010),
    ("tanh-teacher / muon", "muon", 0.03),
    ("tanh-teacher / gn",   "gn",   0.08),
]

def build(seed=0):
    params = capture_run.default_params()
    params.update({"dataset": "chebyshev", "arch": "mlp", "loss": "mse", "act": "tanh", "bias": "1",
                   "width": A.width, "depth": A.depth, "degree": 3, "nsamp": A.nsamp,
                   "initscheme": "kaiming_normal", "init": "0.5", "seed": seed})
    P = server._parse_params({k: [str(v)] for k, v in params.items()})
    server._TL.model = server.build_model("mlp", 1, 1, P)
    server._TL.model.init_scheme = "kaiming_normal"
    server._TL.loss = server.build_loss("mse")
    th, X, Y, _, _ = server.init_data_theta(P, "chebyshev", A.nsamp, 1, 1)
    return th, X, Y

def exact_spectrum(hvp, p):
    """Dense operator via p HVPs on basis vectors → symmetric → exact eigenvalues (descending)."""
    Amat = torch.zeros(p, p, dtype=dtype, device=dev); e = torch.zeros(p, dtype=dtype, device=dev)
    for i in range(p):
        e[i] = 1.0; Amat[:, i] = hvp(e); e[i] = 0.0
    Amat = 0.5 * (Amat + Amat.t())
    return torch.linalg.eigvalsh(Amat).flip(0).cpu().numpy()   # descending

def log_scree(vals, k):
    """Return [[rank, signed_lambda], ...] log-subsampled so the head (outliers) is dense."""
    n = len(vals)
    if n <= k:
        idx = np.arange(n)
    else:
        idx = np.unique(np.round(np.logspace(0, np.log10(n), k)).astype(int) - 1)
        idx = idx[(idx >= 0) & (idx < n)]
    return [[int(i) + 1, float(vals[i])] for i in idx]     # rank is 1-based (like the screenshot)

def train_and_spectra(label, opt, lr, seed=0):
    th, X, Y = build(seed)
    N = X.shape[0]; p = server._TL.model.p
    m = torch.zeros(p, dtype=dtype, device=dev); v = torch.zeros(p, dtype=dtype, device=dev)  # adam state
    b1, b2, eps = 0.9, 0.999, 1e-8
    t0 = time.time(); loss_curve = []
    for t in range(A.steps):
        g, out = server.gradL(th, X, Y)
        if t % 20 == 0:
            loss_curve.append([t, float(server._TL.loss.value(out, Y, N))])
        if opt == "sgd":
            th = th - lr * g
        elif opt == "adam":
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** (t + 1)); vh = v / (1 - b2 ** (t + 1))
            th = th - lr * mh / (vh.sqrt() + eps)
        elif opt == "muon":
            th = th - lr * server._muon_whiten(server._TL.model, g)
        elif opt == "gn":
            Jc, _ = server.jac_cols(th, X); r = (Y - server._TL.model.forward(th, X)).reshape(-1)
            d = server._opt_dir(server._TL.model, g, "gaussnewton", J=Jc[:N], r=r[:N], th=th, X=X, N=N, outD=1)
            th = th - lr * d
    # --- final-iteration exact spectra of the four operators ---
    out = server._TL.model.forward(th, X); r = (Y - out); rc = r.reshape(N, 1); invN = 1.0 / N
    Jc, _ = server.jac_cols(th, X); Jg = Jc[:N]                       # (N, p)
    JtJ = (Jg.t() @ Jg)                                              # p×p Gauss-Newton (nonzero spectrum ≡ JJᵀ)
    ops = {
        "hessian": exact_spectrum(lambda z: server.hvpL(th, X, Y, z), p),
        "sumQ":    exact_spectrum(lambda z: server.hvpF(th, X, z), p),
        "qr":      exact_spectrum(lambda z: server.hvpS(th, X, z, rc) * invN, p),
        "jjt":     torch.linalg.eigvalsh(0.5 * (JtJ + JtJ.t())).flip(0).cpu().numpy(),
    }
    series = {k: log_scree(vv, A.maxscree) for k, vv in ops.items()}
    npos = {k: int((vv > 0).sum()) for k, vv in ops.items()}
    print(f"  {label:22s} p={p} loss={loss_curve[-1][1]:.2e}  "
          f"npos H={npos['hessian']} ΣQ={npos['sumQ']} Qr={npos['qr']} JJt={npos['jjt']}  ({time.time()-t0:.0f}s)", flush=True)
    return {"label": label, "opt": opt, "P": p, "iter": A.steps, "npos": npos,
            "loss": loss_curve, "series": series}

print(f"device={dev} dtype={dtype}  width={A.width} depth={A.depth}  steps={A.steps}", flush=True)
runs = [train_and_spectra(l, o, lr) for (l, o, lr) in RUNS]
out = {"kind": "eigenspectrum-overlay",
       "operators": [{"key": "qr", "label": "Q_r", "color": "#e05a8f"},
                     {"key": "sumQ", "label": "ΣQ_i", "color": "#8b5cf6"},
                     {"key": "jjt", "label": "JJᵀ", "color": "#16a34a"},
                     {"key": "hessian", "label": "Hessian", "color": "#2563eb"}],
       "runs": runs}
with open(A.out, "w") as f:
    json.dump(out, f)
print(f"wrote {A.out}  ({len(runs)} runs)", flush=True)
