#!/usr/bin/env python
"""Q-CHANGING vs Q-FIXED comparison data for the results webpage.

Along ONE training run (tanh MLP on chebyshev, plain SGD at the Edge of Stability) we track over
training the statistics that reveal what "freezing the function Hessian Q" changes:

  loss                : the training loss (the ACTUAL trajectory — identical for both, shown once).
  sharpness           : top eigenvalue of the loss Hessian ∇²L, with the 2/lr EoS line.
  qr_top_evolve       : top eigenvalue of Q_r = Σ_k r_k ∇²f_k(θ_t)   — Q EVOLVING (recomputed at θ_t).
  qr_top_fixed        : top eigenvalue of Q_r = Σ_k r_k ∇²f_k(θ_0)   — Q FROZEN at initialization θ_0.
                        (Same residual r_t both times; only WHERE the function Hessian is evaluated differs.
                         This is exactly the widget's qinit = evolve vs fix, via server._TL.qcfg.)
  ntk_align_k         : residual↔spectrum alignment — |cos(r_t, v_k)| for the top-4 eigenvectors v_k of
                        the NTK J Jᵀ. Shows the residual collapsing onto the top NTK directions (rich regime).

  python gen_qcompare.py --device cuda:0 --out qcompare.json
"""
import os, sys, json, argparse, time
import numpy as np, torch

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
sys.path.insert(0, DIR)
import server, capture_run

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="qcompare.json")
ap.add_argument("--width", type=int, default=16)
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--nsamp", type=int, default=100)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--every", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.1)
A = ap.parse_args()

dev = torch.device(A.device)
dtype = torch.float32 if dev.type == "cuda" else torch.float64
server.DTYPE = dtype; server.DEVICE = dev; server._TL.device = dev; server._TL.cifar_dir = None
if dev.type == "cuda": torch.cuda.set_device(dev)

params = capture_run.default_params()
params.update({"dataset": "chebyshev", "arch": "mlp", "loss": "mse", "act": "tanh", "bias": "1",
               "width": A.width, "depth": A.depth, "degree": 3, "nsamp": A.nsamp,
               "initscheme": "kaiming_normal", "init": "0.5", "seed": 0})
P = server._parse_params({k: [str(v)] for k, v in params.items()})
server._TL.model = server.build_model("mlp", 1, 1, P)
server._TL.model.init_scheme = "kaiming_normal"
server._TL.loss = server.build_loss("mse")
th, X, Y, _, _ = server.init_data_theta(P, "chebyshev", A.nsamp, 1, 1)
N = X.shape[0]; p = server._TL.model.p; lr = A.lr
th0 = th.clone()                                    # θ_0 — where the FIXED Q is frozen
print(f"device={dev} p={p} N={N} lr={lr} steps={A.steps}  2/lr={2/lr}", flush=True)

def top_eig(hvp, dim, m=40, seed=1):
    """Largest (most positive) eigenvalue via Lanczos."""
    _, T, k = server._lanczos_core(hvp, dim, min(dim, m), seed)
    return float(np.linalg.eigvalsh(T.cpu().numpy())[-1])

rows = {"loss": [], "sharpness": [], "qr_top_evolve": [], "qr_top_fixed": [],
        "ntk_align": [[], [], [], []], "twoOverLr": 2 / lr, "frames": []}
t0 = time.time()
for t in range(A.steps + 1):
    if t % A.every == 0:
        out = server._TL.model.forward(th, X)
        r = (Y - out); rc = r.reshape(N, 1); rv = r.reshape(-1)[:N]
        loss = float(server._TL.loss.value(out, Y, N))
        sharp = top_eig(lambda v: server.hvpL(th, X, Y, v), p)
        # --- Q EVOLVING: function Hessian at θ_t ---
        server._TL.qcfg = None
        qr_evo = top_eig(lambda v: server.hvpS(th, X, v, rc) / N, p, seed=7)
        # --- Q FIXED: function Hessian frozen at θ_0 (same residual r_t) ---
        server._TL.qcfg = {"mode": "fix", "theta_t": th0, "Qrand": None}
        qr_fix = top_eig(lambda v: server.hvpS(th, X, v, rc) / N, p, seed=7)
        server._TL.qcfg = None
        # --- residual ↔ NTK eigenvector alignment (top-4) ---
        Jc, _ = server.jac_cols(th, X); Jg = Jc[:N]                # (N, p)
        ntk = (Jg @ Jg.t()).cpu().numpy(); ntk = 0.5 * (ntk + ntk.T)
        w, V = np.linalg.eigh(ntk)                                 # ascending
        rnp = rv.cpu().numpy(); rnp = rnp / (np.linalg.norm(rnp) + 1e-30)
        for k in range(4):
            vk = V[:, -1 - k]                                      # k-th largest NTK eigenvector
            rows["ntk_align"][k].append(float(abs(rnp @ vk)))
        rows["frames"].append(t)
        rows["loss"].append(loss); rows["sharpness"].append(sharp)
        rows["qr_top_evolve"].append(qr_evo); rows["qr_top_fixed"].append(qr_fix)
        if t % 40 == 0:
            print(f"  t={t:4d} loss={loss:.2e} sharp={sharp:.2f} qrEvo={qr_evo:.3f} qrFix={qr_fix:.3f} "
                  f"ntk1={rows['ntk_align'][0][-1]:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    g, _ = server.gradL(th, X, Y)
    th = th - lr * g

out = {"kind": "q-fixed-vs-changing", "lr": lr, "P": p, "steps": A.steps, **rows}
with open(A.out, "w") as f:
    json.dump(out, f)
print(f"wrote {A.out}", flush=True)
