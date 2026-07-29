# Prediction & Prediction-Multiclass widgets — a code guide

A map for reading the actual source. Everything lives under `/nas/ucb/samsj/TestingPSTheory/eos_widget/`.
Line numbers are for the current tree (`server.py` is ~6,800 lines; the two HTML files ~6,400 each).

---

## 1. The files that ARE the widgets

| File | What it is |
|---|---|
| `eos_widget_prediction/index_prediction.html` | The **prediction widget** — the whole UI, the plots, and a browser-local JS compute engine. Scalar-output tasks (chebyshev, ksparse-parity, saddle, …). |
| `eos_widget_prediction/index_prediction_multiclass.html` | The **multiclass** variant — same panels, but datasets whose output is n·d-flattened multiclass (CIFAR-10, MNIST-10, max-finder, modulo-add) and CE + Fisher curvature. Its JS is byte-identical to `index_prediction.html` except the multiclass data/CE paths. |
| `server.py` | The **GPU compute backend**. THE file for the math. One `/run` request → `run_stream()` streams per-step diagnostics over SSE. Every operator, every panel, SLQ, and the prediction trackers are here. |
| `capture_run.py` | Headless **capture harness** — runs `run_stream` once with a param dict and writes a `.json` you load in the widget via "⬆ load run". `default_params()` (L44) is the full list of every widget knob + its default. |
| `eos_prediction.py` / `eos_prediction_multiclass.py` | Thin CLI wrappers around `capture_run` (what the SLURM sweeps invoke). |

There are **two compute paths** for the same panels (kept at numerical parity):
- **GPU backend** (`compute = GPU`): browser → SSE → `server.run_stream`. Used for real runs and all captures.
- **Browser-local** (`compute = browser`): a pure-JS reimplementation inside the HTML (`runLocal`, ~L4286 in `index_prediction.html`) for tiny MLPs. Same formulas, JS instead of torch.

---

## 2. Compute flow

```
capture:   eos_prediction.py ─► capture_run.capture() ─► server.run_stream(P)  ─► records[] ─► <name>.json
live GPU:  browser  ── POST /run ──►  server.run_stream(P)  ── SSE messages ──►  browser plots
live local: browser ─────────────────────────────────────►  runLocal() in the HTML (JS)
```

`run_stream(P)` (server.py **L3838**) is the spine: build model+data+θ, then a GD loop that, at each
eig-tick, yields message dicts. Read this top-to-bottom to understand the widget. The param dict `P` is
produced by `_parse_params` (**L6100**) — coerces the browser's string query into typed values (this is
also where per-panel toggles `s1…s42`, `slqblock`, `qspec`, etc. get their defaults).

---

## 3. The math core — operators

The model is one flat parameter vector `θ` (so all the matrix-free Lanczos/HVP code is generic).
`f(θ,X)` is the network output; `r = Y − f` is the residual; `J` the per-output Jacobian.

| Operator | Symbol | Function | server.py |
|---|---|---|---|
| Loss gradient | ∇L | `gradL(th,X,Y)` | **L1330** |
| Jacobian (per-output rows) | J (M×p) | `jac_cols(th,X)` | **L2575** |
| **Function Hessian** | ΣQᵢ = Σₖ∇²fₖ | `hvpF(th,X,v)` | **L1813** |
| **Loss Hessian** | ∇²L | `hvpL(th,X,Y,v)` | **L1820** |
| **Residual-weighted (Qr)** | Σₖrₖ∇²fₖ | `hvpS(th,X,v,c)` with c=r | **L1827** |
| Per-output Q·v (for predictions) | ∇²fₖ·v | `jac_hvp(th,X,z)` | **L2677** |

Derived (formed from J in the panels, not standalone fns):
- **NTK / JJᵀ** = J Jᵀ (M×M) — its nonzero spectrum equals the p×p **Gauss–Newton** G = JᵀJ.
- Exact identity the widget leans on: **∇²L = G + Qr** (loss Hessian = Gauss–Newton + residual curvature).

HVPs are matrix-free: exact via autograd for conv/GPT (`exact_hvp=True`), finite-difference for the
functional MLP (`MlpModel`, L1410, `exact_hvp=False` — FD is exact for its hand-written vjp). The loss
layer decides the cotangent: `MSELoss`/`CELoss` at **L1436 / L1465** (`resid_cotangent` = ∂L/∂f).

---

## 4. Training step & optimizers

The GD update is `θ ← θ − lr·_opt_dir(∇L, opt)`. `_opt_dir` (**L1057**) selects the direction:
`gd` (plain), `sign`, `spectral` (Muon two-sided whitening, `_muon_whiten` L1025), `gaussnewton`
(min-norm GN, `_gn_precond` L1042 / Fisher natural-grad for CE). Note `gd ⇒ θ←θ−(lr/N)·Jᵀr` reproduces
mean-loss GD byte-for-byte.

---

## 5. What `run_stream` emits — the record/panel types

Each capture is `{…, "records": [...]}`. Record `type` and what it carries (see §1 of your capture JSONs):

| type | count | key fields | panel |
|---|---|---|---|
| `meta` | 1 | p, n, dataset, arch, loss | — |
| `step` | per eig-tick | `loss`, `sharp` (λ₁∇²L), `r` (residual), and **~80 nested panels** `g_qspec`, `g_pred3/4/6`, `g24…g28`, `g_ray`, `g_trace`, … | §1–§28 + predictions |
| `slq` | strided | `sH,sG,sS,sHL` (SLQ **densities** of ΣQ, G, Qr, ∇²L) + `trH…` traces | §5 spectra |
| `g21` | per-step | residual↔spectrum alignment: `n1..n4` (NTK-eigvec), `p1..p4` (M_r), `lam` | §21 |
| `sweeppt`, `sweeppt2c` | sweep | Prediction-1&2 sign-change sweep points | — |
| `done` | 1 | final marker | — |

The one you'll care about most for spectra: **`step.g_qspec`** = `{mr:[…], h:[…], p, full}` — the FULL
sorted eigenspectra of **Qr** (`mr`) and **ΣQᵢ** (`h`) at that step (the "Q-spectrum" scree + scrubber).
This is exactly the data behind the eigenspectrum webpage's section 1.

---

## 6. Key subsystems (read these after `run_stream`)

- **SLQ / block-SLQ** (spectral density): `slq_density` **L3111**, `_lanczos_core` **L2940**,
  `_block_lanczos_core` **L3071**. Block size `slqblock` (default **4**). Clean standalone walkthrough:
  [`review/block_slq.py`](block_slq.py); the exact JS↔Python correspondence: [`review/changes_catalog.py`](changes_catalog.py).
- **qinit toggle** (freeze/randomize the function Hessian Q): the chokepoints are `hvpS` (L1827) and
  `jac_hvp` (L2677), both honoring `_TL.qcfg`. Modes evolve/fix/gauss/bern/unif. This is what the webpage's
  "Q evolving vs Q fixed" section drives.
- **Residual↔spectrum alignment** (§19/§20/§21): `_sec19_payload` **L1848**, `_sec20_payload` **L1873**
  (M_r histograms), `_sec21_payload` **L2144** (NTK panel + M_r panel — the "residual↔NTK-eigenvector" plot).
- **Prediction trackers** (Pred-3/4/5/6, the frozen-J / frozen-Q quadratic forecasts): search `g_pred3`,
  `g_pred4`, `g_pred6`, `g_ray`, `g_trace` inside `run_stream`.

---

## 7. The two HTML files (UI + browser-local numerics)

- Controls: the `<div class="ctl">` blocks near the top set every knob (dataset, arch, lr, init, `slqblock`,
  section toggles, `qinit`, …). Their `id`s match the `P` keys `_parse_params` reads.
- Browser-local engine: `runLocal` (~L4286) mirrors `run_stream`; the SLQ density plots call `slqDensity`
  (~L1281, now with the block branch I added — see `review/block_slq.py`).
- Parity: the browser JS and `server.py` are kept numerically equivalent panel-by-panel; the JS uses a
  `mulberry32` RNG so probe vectors match cross-backend where it matters.

---

## 8. Suggested reading order

1. `capture_run.default_params()` — the full knob list (what the widget can do), 40 lines.
2. `server.run_stream` **L3838** — the spine; skim the GD loop and see where each `g_*` panel is filled.
3. The four operators (§3 above) — 15 lines each.
4. `slq_density` + `review/block_slq.py` — the spectra.
5. `_sec21_payload` — the residual↔NTK alignment.
6. One `index_prediction.html` control block + `runLocal` — how the UI drives it.

To watch it live instead of reading: a GPU server is (or was) running at **http://localhost:8756/**
(`server.py --device cuda:6`); press Run and open the browser devtools to see the `/run` SSE payloads,
which are exactly the records described in §5.
```
