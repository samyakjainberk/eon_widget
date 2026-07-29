#!/usr/bin/env python
"""STANDALONE (not the widget / not a capture): plot the EVOLUTION over training of the eigenspectrum of
three operators —
    Q_r  = Σ_k r_k ∇²f_k   (residual-weighted function Hessian, r = Y−f, reported ÷N ⇒ mean-loss scale)
    JJᵀ  = the empirical NTK Gram (M×M, M = N·outD)
    H    = ∇²L              (the full loss Hessian, mean-loss scale)
estimated FIVE ways at every checkpoint:
    plain SLQ (block=1)  and  BLOCK SLQ with window size b ∈ {2,4,8,16}.
Also computes a DENSE exact-eigenvalue reference at a few checkpoints so the SLQ-vs-block accuracy is visible.

Reuses the widget backend's validated primitives (server.slq_density with block=, server.hvpL/hvpS/jac_cols)
but drives its own GD loop — it does NOT go through run_stream/capture, and writes nothing to runs_captured.

  python spec_evo.py            # runs it (CPU fp64 by default, GPU fp32 if --device cuda)
Outputs -> spectrum_evolution/out/: evo_{Qr,JJt,H}.png, valid_{Qr,JJt,H}.png, sharpness.png, densities.json
"""
import os, sys, json, argparse, time
import numpy as np
import torch

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
sys.path.insert(0, DIR)
import server, capture_run

# ------------------------------------------------------------------ config
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")           # cpu -> fp64 (clean exact ref); cuda:0 -> fp32 (matches widget)
ap.add_argument("--steps", type=int, default=320)
ap.add_argument("--every", type=int, default=16)     # checkpoint cadence (steps/every ≈ #frames in the evolution heatmap)
ap.add_argument("--nprobe", type=int, default=10)    # SLQ Hutchinson probes
ap.add_argument("--m", type=int, default=48)         # Lanczos iterations per probe
ap.add_argument("--ngrid", type=int, default=220)    # density grid resolution
ap.add_argument("--dataset", default="chebyshev")
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--nsamp", type=int, default=100)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--degree", type=int, default=3)
ap.add_argument("--initscheme", default="default")
ap.add_argument("--init", default="0.5")
A = ap.parse_args()

BLOCKS = [1, 2, 4, 8, 16]      # 1 = plain SLQ; the rest = block SLQ window sizes the user asked for
OUT = os.path.join(DIR, "spectrum_evolution", "out"); os.makedirs(OUT, exist_ok=True)

dev = torch.device(A.device)
dtype = torch.float32 if dev.type == "cuda" else torch.float64
server.DTYPE = dtype
server.DEVICE = dev
server._TL.device = dev
server._TL.cifar_dir = None
if dev.type == "cuda":
    torch.cuda.set_device(dev)
print(f"device={dev} dtype={dtype}  dataset={A.dataset} w{A.width}d{A.depth} n{A.nsamp} lr{A.lr} "
      f"steps={A.steps} every={A.every} nprobe={A.nprobe} m={A.m}", flush=True)

# ------------------------------------------------------------------ model + data (widget builders, own loop)
params = capture_run.default_params()
params.update({"dataset": A.dataset, "arch": "mlp", "loss": "mse", "act": "tanh", "bias": "1",
               "width": A.width, "depth": A.depth, "degree": A.degree, "nsamp": A.nsamp,
               "lr": A.lr, "seed": 0, "initscheme": A.initscheme, "init": A.init, "steps": A.steps})
P = server._parse_params({k: [str(v)] for k, v in params.items()})
inD = outD = 1                                    # chebyshev: scalar in -> scalar out
N = P["nsamp"]; M = N * outD
server._TL.model = server.build_model("mlp", inD, outD, P)
server._TL.model.init_scheme = P.get("initscheme", "default")
server._TL.loss = server.build_loss("mse")
th, X, Y, _, _ = server.init_data_theta(P, A.dataset, N, inD, outD)
p = server._TL.model.p
lr = float(P["lr"])
print(f"p={p} params, M={M} (NTK dim)", flush=True)

# ------------------------------------------------------------------ operators at a fixed theta
def make_ops(theta):
    """Return {name: (hvp, dim)} for Q_r, JJᵀ, H at parameters `theta` (residual r = Y − f)."""
    out = server._TL.model.forward(theta, X)
    r = (Y - out)                                  # residual, widget §20/qspec convention (Y − f)
    rc = r.reshape(N, outD)
    Jc, _ = server.jac_cols(theta, X)              # (M, p) per-output Jacobian
    Jg = Jc[:M]
    invN = 1.0 / max(N, 1)
    ops = {
        "Qr":  (lambda v: server.hvpS(theta, X, v, rc) * invN, p),          # (1/N) Σ_k r_k ∇²f_k
        "JJt": (lambda u: Jg @ (Jg.t() @ u), M),                            # empirical NTK Gram, M×M
        "H":   (lambda v: server.hvpL(theta, X, Y, v), p),                  # ∇²L (mean loss)
    }
    return ops, Jg

def exact_eigs(hvp, dim):
    """Dense reference: materialize the operator by `dim` HVPs on basis vectors, symmetrize, eigvalsh."""
    Amat = torch.zeros(dim, dim, dtype=dtype, device=dev)
    e = torch.zeros(dim, dtype=dtype, device=dev)
    for i in range(dim):
        e[i] = 1.0
        Amat[:, i] = hvp(e)
        e[i] = 0.0
    Amat = 0.5 * (Amat + Amat.t())
    return torch.linalg.eigvalsh(Amat).cpu().numpy()

# ------------------------------------------------------------------ train + capture spectra
ckpts = list(range(0, A.steps + 1, A.every))
ck_set = set(ckpts)
# exact reference at ~3 representative frames (early / mid / late) — dense eig is O(dim) HVPs, do it sparingly
exact_at = sorted({ckpts[0], ckpts[len(ckpts) // 2], ckpts[-1]})
OPS = ["Qr", "JJt", "H"]
OPLABEL = {"Qr": "Q_r = Σ r_k ∇²f_k  (÷N)", "JJt": "NTK  J Jᵀ", "H": "loss Hessian  ∇²L"}
dens = {op: {b: [] for b in BLOCKS} for op in OPS}   # dens[op][block] = list over checkpoints of {x,y}
exact = {op: {} for op in OPS}                        # exact[op][iter] = sorted eigenvalue array
topeig = {op: [] for op in OPS}                       # top |eigenvalue| over training (from exact where avail, else SLQ max)
matvecs = {op: {b: 0 for b in BLOCKS} for op in OPS}  # cost accounting (one representative checkpoint)
frames = []                                           # iteration index per stored density row

t_start = time.time()
for t in range(A.steps + 1):
    if t in ck_set:
        ops, Jg = make_ops(th)
        frames.append(t)
        for op in OPS:
            hvp, dim = ops[op]
            for b in BLOCKS:
                d = server.slq_density(hvp, dim, A.nprobe, A.m, A.ngrid, seed=1234, block=b)
                dens[op][b].append(d)
                if t == exact_at[0]:                 # count matvecs once (first frame) for the cost table
                    if b > 1:
                        it = min(max(1, dim // b), A.m)
                        matvecs[op][b] = A.nprobe * it * b
                    else:
                        matvecs[op][b] = A.nprobe * min(dim, A.m)
            if t in exact_at:
                ev = exact_eigs(hvp, dim)
                exact[op][t] = ev
                topeig[op].append((t, float(np.max(np.abs(ev)))))
        sh = exact["H"].get(t)
        msg = f"  step {t:4d}/{A.steps}  ({time.time()-t_start:5.0f}s)"
        if sh is not None:
            msg += f"   λmax(H)={np.max(sh):.3f}  (2/lr={2/lr:.1f})"
        print(msg, flush=True)
    # --- GD step (mean-loss, step = lr) : reproduces run_stream's θ ← θ − lr·∇L ---
    g, _ = server.gradL(th, X, Y)
    th = th - lr * g

print(f"training + spectra done in {time.time()-t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ save raw densities
with open(os.path.join(OUT, "densities.json"), "w") as f:
    json.dump({"frames": frames, "blocks": BLOCKS, "ops": OPS, "exact_at": exact_at,
               "lr": lr, "p": p, "M": M, "matvecs": matvecs,
               "dens": {op: {str(b): dens[op][b] for b in BLOCKS} for op in OPS},
               "exact": {op: {str(t): exact[op][t].tolist() for t in exact[op]} for op in OPS}},
              f)
print("wrote densities.json", flush=True)

# ------------------------------------------------------------------ plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def evolution_fig(op):
    """One figure per operator: 5 side-by-side heatmaps (SLQ, block2/4/8/16). x=eigenvalue, y=iteration,
    color=spectral density. Shows both the EVOLUTION (down each column) and the method contrast (across)."""
    rows = dens[op]
    # common eigenvalue axis across all methods/frames for this operator — IGNORE non-finite cells
    # (a block-Lanczos breakdown on a near-degenerate operator can emit a NaN density; those rows are
    #  drawn as blank/zero and reported below, rather than corrupting the shared axis).
    def _fin(a): a = np.asarray(a, float); return a[np.isfinite(a)]
    xs = np.concatenate([_fin(d["x"]) for b in BLOCKS for d in rows[b] if _fin(d["x"]).size])
    xmin, xmax = float(xs.min()), float(xs.max())
    grid = np.linspace(xmin, xmax, A.ngrid)
    nan_cells = [(b, frames[i]) for b in BLOCKS for i, d in enumerate(rows[b])
                 if not (np.all(np.isfinite(d["x"])) and np.all(np.isfinite(d["y"])))]
    if nan_cells:
        print(f"  [{op}] blanked NaN density cells (block,step): {nan_cells}", flush=True)
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(3.1 * len(BLOCKS), 5.2), sharey=True)
    vmax = 0.0
    IMG = {}
    for b in BLOCKS:
        img = np.array([np.interp(grid, np.asarray(d["x"], float), np.nan_to_num(np.asarray(d["y"], float)),
                                  left=0, right=0) if np.all(np.isfinite(d["x"])) else np.zeros_like(grid)
                        for d in rows[b]])
        img = np.nan_to_num(img)
        IMG[b] = img
        vmax = max(vmax, img.max())
    for ax, b in zip(axes, BLOCKS):
        img = IMG[b]
        pc = ax.pcolormesh(grid, frames, np.clip(img, 1e-6 * vmax, None),
                           norm=LogNorm(vmin=max(1e-6 * vmax, vmax * 1e-4), vmax=vmax),
                           cmap="magma", shading="auto")
        ax.set_title("SLQ (b=1)" if b == 1 else f"block SLQ  b={b}", fontsize=10)
        ax.set_xlabel("eigenvalue")
        for tt in exact_at:                          # mark frames where an exact reference exists
            ax.axhline(tt, color="cyan", lw=0.5, alpha=0.4, ls=":")
    axes[0].set_ylabel("training step")
    fig.suptitle(f"Eigenspectrum evolution — {OPLABEL[op]}   (dataset={A.dataset}, lr={lr}, p={p})", fontsize=12)
    fig.colorbar(pc, ax=axes, label="spectral density (log)", fraction=0.03, pad=0.01)
    path = os.path.join(OUT, f"evo_{op}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote", path, flush=True)

def validation_fig(op):
    """One figure per operator: rows = the exact-reference frames; each overlays the exact eigenvalue
    histogram (grey) against all 5 SLQ/block density curves — how well each method matches ground truth."""
    ts = exact_at
    fig, axes = plt.subplots(len(ts), 1, figsize=(8.5, 3.0 * len(ts)), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(BLOCKS)))
    for ax, t in zip(axes[:, 0], ts):
        ev = exact[op][t]
        ax.hist(ev, bins=60, density=True, color="0.8", edgecolor="0.6", label="exact eigenvalues")
        fi = frames.index(t)
        for b, c in zip(BLOCKS, colors):
            d = dens[op][b][fi]
            lab = "SLQ b=1" if b == 1 else f"block b={b}"
            if np.all(np.isfinite(d["x"])) and np.all(np.isfinite(d["y"])):
                ax.plot(d["x"], d["y"], color=c, lw=1.6, label=lab)
        ax.set_title(f"step {t}   (λmin={ev.min():.3g}, λmax={ev.max():.3g})", fontsize=9)
        ax.set_ylabel("density")
    axes[-1, 0].set_xlabel("eigenvalue")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"SLQ vs block-SLQ vs EXACT — {OPLABEL[op]}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, f"valid_{op}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote", path, flush=True)

for op in OPS:
    evolution_fig(op)
    validation_fig(op)

# sharpness / top-eig context
fig, ax = plt.subplots(figsize=(7, 4))
for op in OPS:
    if topeig[op]:
        xs, ys = zip(*topeig[op])
        ax.plot(xs, ys, "o-", label=f"top |λ|  {op}")
ax.axhline(2 / lr, color="r", ls="--", lw=1, label=f"2/lr = {2/lr:.1f}")
ax.set_xlabel("training step"); ax.set_ylabel("top |eigenvalue| (exact)"); ax.legend(); ax.set_yscale("log")
ax.set_title("Progressive sharpening context (exact top eigenvalue)")
fig.savefig(os.path.join(OUT, "sharpness.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
print("wrote sharpness.png", flush=True)

# ------------------------------------------------------------------ accuracy summary (L1 vs exact, equal grid)
print("\n=== L1 density deviation from EXACT (mean over exact frames; lower = better) ===", flush=True)
summary = {}
for op in OPS:
    line = [f"{op:4s}"]
    summary[op] = {}
    for b in BLOCKS:
        errs = []
        for t in exact_at:
            fi = frames.index(t); d = dens[op][b][fi]
            ev = exact[op][t]
            lo, hi = float(min(d["x"])), float(max(d["x"]))
            gx = np.linspace(lo, hi, 400)
            # exact density via same Gaussian KDE width the SLQ grid uses (60 bins over range)
            sig = max((hi - lo) / 60, 1e-9)
            ex = np.zeros_like(gx)
            for lam in ev:
                ex += np.exp(-0.5 * ((gx - lam) / sig) ** 2)
            ex /= (ex.sum() * (gx[1] - gx[0]) + 1e-30)
            sy = np.interp(gx, d["x"], d["y"], left=0, right=0)
            sy /= (sy.sum() * (gx[1] - gx[0]) + 1e-30)
            errs.append(float(np.abs(sy - ex).sum() * (gx[1] - gx[0])))
        m = float(np.mean(errs)); summary[op][b] = m
        tag = "SLQ " if b == 1 else f"b={b:<2d}"
        line.append(f"{tag}:{m:.4f}(mv{matvecs[op][b]:>5d})")
    print("  " + "  ".join(line), flush=True)

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump({"L1_vs_exact": summary, "matvecs": matvecs}, f, indent=2)
print("\nALL DONE ->", OUT, flush=True)
