#!/usr/bin/env python
"""Data for the results webpage — reproduces the ACTUAL prediction-widget runs (cheb_q / ksparse_q /
saddle_q), then produces:

SECTION 1 (eigenspectrum overlay, dense slider): exact sorted signed spectra of 4 operators
  hessian ∇²L · ΣQᵢ Σ∇²f_k · Qr Σr_k∇²f_k · JJᵀ NTK  — at EVERY few steps so the slider scrubs finely.

SECTION 2 (three Q modes: evolving θ_t · fixed at init θ₀ · random-fixed θ₀), everything else identical:
  loss & sharpness  — Q-INDEPENDENT (the GD step never uses Q), plotted once.
  Qr=M_r top & bottom eigenvalue        — Q-DEPENDENT, three curves.
  residual↔NTK-eigenvector alignment    — Q-INDEPENDENT (NTK panel, §21 n1), top-4, plotted once.
  Qr→Jr alignment (§6 phase-1)          — Q-DEPENDENT |cos(Jᵀr,u₁)| (M_r panel, §21 p1), three curves.
  power iteration (§6 phase-2·1)         — Q-INDEPENDENT: ‖Jᵀr‖ & λ₁(JJᵀ) & N‖∇L‖, plotted once.
  residual dominance (§6 phase-2·2)      — Q-DEPENDENT: ‖M_r∇L‖−‖JᵀJ∇L‖ = ‖J̇·r‖−‖J·ṙ‖, three curves.

  python gen_all.py --device cuda:0 --specstride 3
"""
import os, sys, json, argparse, time
import numpy as np, torch
DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"; sys.path.insert(0, DIR)
import server, capture_run

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--maxscree", type=int, default=80)
ap.add_argument("--every", type=int, default=8)        # section-2 dynamics cadence
ap.add_argument("--specstride", type=int, default=3)   # section-1 eigenspectrum snapshot every N steps (the slider)
A = ap.parse_args()
dev = torch.device(A.device); dtype = torch.float32 if dev.type == "cuda" else torch.float64
server.DTYPE = dtype; server.DEVICE = dev; server._TL.device = dev; server._TL.cifar_dir = None
if dev.type == "cuda": torch.cuda.set_device(dev)

RUNS = [
    dict(key="chebyshev", label="chebyshev · w16d3 · lr0.1 · kaiming",
         P=dict(dataset="chebyshev", width=16, depth=3, nsamp=100, lr=0.1, initscheme="kaiming_normal",
                indim=1, outdim=1, degree=3, steps=400)),
    dict(key="ksparse", label="ksparse parity · w16d2 · lr0.5 · kaiming",
         P=dict(dataset="ksparse", width=16, depth=2, nsamp=100, lr=0.5, initscheme="kaiming_normal",
                indim=10, outdim=1, steps=400)),
    dict(key="saddle", label="saddle · w16d3 · lr0.3 · kaiming",
         P=dict(dataset="saddle", width=16, depth=3, nsamp=64, lr=0.3, initscheme="kaiming_normal",
                indim=4, outdim=4, steps=400)),
]

def build(cfg):
    params = capture_run.default_params()
    params.update({**{k: str(v) for k, v in cfg["P"].items()}, "arch": "mlp", "loss": "mse",
                   "act": "tanh", "bias": "1", "init": "0.5", "seed": "0"})
    P = server._parse_params({k: [str(v)] for k, v in params.items()})
    ds = cfg["P"]["dataset"]; inD = int(cfg["P"]["indim"]); outD = int(cfg["P"]["outdim"])
    server._TL.model = server.build_model("mlp", inD, outD, P)
    server._TL.model.init_scheme = cfg["P"]["initscheme"]
    server._TL.loss = server.build_loss("mse")
    th, X, Y, _, _ = server.init_data_theta(P, ds, int(cfg["P"]["nsamp"]), inD, outD)
    return th, X, Y, outD

def exact_spectrum(hvp, p):
    Amat = torch.zeros(p, p, dtype=dtype, device=dev); e = torch.zeros(p, dtype=dtype, device=dev)
    for i in range(p):
        e[i] = 1.0; Amat[:, i] = hvp(e); e[i] = 0.0
    Amat = 0.5 * (Amat + Amat.t())
    return torch.linalg.eigvalsh(Amat).flip(0).cpu().numpy()

def log_scree(vals, k):
    n = len(vals)
    idx = np.arange(n) if n <= k else np.unique(np.round(np.logspace(0, np.log10(n), k)).astype(int) - 1)
    idx = idx[(idx >= 0) & (idx < n)]
    return [[int(i) + 1, round(float(vals[i]), 6)] for i in idx]

def spectra_at(th, X, Y, N, outD, p):
    out = server._TL.model.forward(th, X); r = (Y - out); rc = r.reshape(N, outD); M = N * outD; invN = 1.0 / N
    Jc, _ = server.jac_cols(th, X); Jg = Jc[:M]; JtJ = (Jg.t() @ Jg) * invN
    return {"hessian": exact_spectrum(lambda z: server.hvpL(th, X, Y, z), p),
            "sumQ": exact_spectrum(lambda z: server.hvpF(th, X, z) * invN, p),
            "qr": exact_spectrum(lambda z: server.hvpS(th, X, z, rc) * invN, p),
            "jjt": torch.linalg.eigvalsh(0.5 * (JtJ + JtJ.t())).flip(0).cpu().numpy()}

def lanczos_eigs(hvp, dim, m=48, seed=7):
    """top eigenvalue, bottom eigenvalue, and the TOP eigenvector (θ-space) of a symmetric operator."""
    Qb, T, k = server._lanczos_core(hvp, dim, min(dim, m), seed)
    mu, Sv = np.linalg.eigh(T.cpu().numpy())               # ascending
    Qm = torch.stack(Qb)                                   # (k, dim)
    uvec = torch.tensor(Sv[:, -1], dtype=Qm.dtype, device=Qm.device) @ Qm   # top Ritz vector
    return float(mu[-1]), float(mu[0]), uvec

OPS = ["qr", "sumQ", "jjt", "hessian"]; MODES = ["evolve", "fixed", "random"]
spectra_runs, qcmp_runs = [], []
for cfg in RUNS:
    th, X, Y, outD = build(cfg); N = X.shape[0]; p = server._TL.model.p; lr = float(cfg["P"]["lr"])
    steps = int(cfg["P"]["steps"]); th0 = th.clone(); t0 = time.time()
    spec_iters = set(range(0, steps + 1, A.specstride)) | {steps}
    SP = {"frames": [], "series": {k: [] for k in OPS}, "npos": {k: [] for k in OPS}}
    QC = {"frames": [], "loss": [], "sharpness": [], "twoOverLr": 2 / lr, "ntk_align": [[], [], [], []],
          "gnorm": [], "ntk_top": [], "ngradL": [],                          # §6 power-iteration (Q-independent)
          "qr_top": {m: [] for m in MODES}, "qr_bot": {m: [] for m in MODES},
          "mr_align": {m: [] for m in MODES},                               # §6 phase-1 Qr→Jr alignment (three-way)
          "res_dom": {m: [] for m in MODES}}                                # §6 phase-2·2 residual dominance (three-way)
    M0 = N * outD
    QRAND = server._build_randQ(th0, X, M0, "gauss", 12345, 4)     # random Q frozen at θ₀, Frobenius-matched
    modecfg = {"evolve": None, "fixed": {"mode": "fix", "theta_t": th0, "Qrand": None},
               "random": {"mode": "gauss", "theta_t": th0, "M": M0, "seed": 12345, "probes": 4, "Qrand": QRAND}}
    for t in range(steps + 1):
        if t in spec_iters:                                        # SECTION 1: dense eigenspectrum snapshot
            sp = spectra_at(th, X, Y, N, outD, p); SP["frames"].append(t)
            for k in OPS:
                SP["series"][k].append(log_scree(sp[k], A.maxscree)); SP["npos"][k].append(int((sp[k] > 0).sum()))
        if t % A.every == 0:                                       # SECTION 2: three-Q-mode dynamics
            out = server._TL.model.forward(th, X); r = (Y - out); rc = r.reshape(N, outD); M = N * outD
            rv = r.reshape(-1)[:M]; Jc, _ = server.jac_cols(th, X); Jg = Jc[:M]
            Jr = Jg.t() @ rv; Jrn = float(Jr.norm()) + 1e-30
            gL, _ = server.gradL(th, X, Y)                              # ∇L (θ-space, Q-independent)
            JtJ_gL = (Jg.t() @ (Jg @ gL)) / N                          # JᵀJ·∇L = J·ṙ  (Q-independent)
            QC["frames"].append(t); QC["loss"].append(float(server._TL.loss.value(out, Y, N)))
            QC["sharpness"].append(lanczos_eigs(lambda v: server.hvpL(th, X, Y, v), p)[0])
            # residual ↔ NTK-eigenvector alignment (Q-independent; §21 panel-1 n1, top-4)
            ntk = (Jg @ Jg.t()).cpu().numpy(); w, V = np.linalg.eigh(0.5 * (ntk + ntk.T))
            rn = rv.cpu().numpy(); rn = rn / (np.linalg.norm(rn) + 1e-30)
            for kk in range(4):
                QC["ntk_align"][kk].append(float(abs(rn @ V[:, -1 - kk])) if V.shape[1] > kk else 0.0)
            # §6 phase-2·1 "power iteration": GN-gradient norm ‖Jᵀr‖ & top NTK eigenvalue λ₁(JJᵀ) (Q-independent)
            QC["gnorm"].append(Jrn); QC["ntk_top"].append(float(w[-1])); QC["ngradL"].append(float(N * gL.norm()))
            # per Q mode: M_r=Σr_kQ_k top/bottom eigenvalue + §6 phase-1 alignment + §6 phase-2·2 residual dominance
            for m in MODES:
                server._TL.qcfg = modecfg[m]
                tv, bv, u1 = lanczos_eigs(lambda v: server.hvpS(th, X, v, rc) / N, p)
                QC["qr_top"][m].append(tv); QC["qr_bot"][m].append(bv)
                QC["mr_align"][m].append(float(abs(float(u1 @ Jr)) / Jrn))   # Qr→Jr alignment |cos(Jᵀr,u₁)|
                mr_gL = server.hvpS(th, X, gL, rc) / N                        # M_r·∇L = −J̇·r under this Q mode
                QC["res_dom"][m].append(float(mr_gL.norm() - JtJ_gL.norm()))  # ‖J̇·r‖−‖J·ṙ‖  (residual dominance)
            server._TL.qcfg = None
        if t < steps:
            g, _ = server.gradL(th, X, Y); th = th - lr * g
    spectra_runs.append({"key": cfg["key"], "label": cfg["label"], "P": p, **SP})
    qcmp_runs.append({"key": cfg["key"], "label": cfg["label"], "lr": lr, "P": p, **QC})
    print(f"  {cfg['key']:10s} p={p} specframes={len(SP['frames'])} qframes={len(QC['frames'])}  ({time.time()-t0:.0f}s)", flush=True)

json.dump({"kind": "eigenspectrum-overlay",
           "operators": [{"key": "qr", "label": "Qr", "color": "#e0559a"},
                         {"key": "sumQ", "label": "ΣQi", "color": "#8b5cf6"},
                         {"key": "jjt", "label": "JJᵀ", "color": "#16a34a"},
                         {"key": "hessian", "label": "Hessian", "color": "#3b6fd4"}],
           "runs": spectra_runs}, open("spectra.json", "w"))
json.dump({"kind": "q-three-way", "modes": MODES, "runs": qcmp_runs}, open("qcompare.json", "w"))
print(f"wrote spectra.json ({len(spectra_runs)} runs) + qcompare.json ({len(qcmp_runs)} runs)", flush=True)
