#!/usr/bin/env python
"""Generate + submit a grid of STACKED prediction_multiclass captures to SLURM.

Grid = (dataset × arch) × learning-rate × init-config, ALL of the prediction_multiclass widget's plots
on (base + SLQ scree + Pred-3/4/5/6 + ray + early-dynamics + 2c/2d — the widget's own sections, incl. the
non-default toggles; NOT the heavy main-widget diagnostics). Each capture → one loadable .json (+ .min.json).
Captures are packed into SLURM jobs so each job is ~12–15 h, then submitted via run_capture_pred_stack.sh.

  python gen_pred_sweep.py --dry-run      # print grid + per-job packing + scale, submit NOTHING
  python gen_pred_sweep.py --submit        # actually sbatch the jobs
  python gen_pred_sweep.py --dry-run --archs mlp   # restrict to a subset (mlp only)
"""
import argparse, math, os, subprocess, sys

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"

# dataset × arch presets (mirror index_prediction_multiclass.html PRESETS). N·outD ≤ grid3dcap(500) so predictions
# produce. est_h = rough GPU-hours per capture at the given steps (for job packing only).
COMBOS = [
    # dataset, arch,   loss, nsamp, dims,                         arch_flags,                                               steps, est_h
    ("cifar10", "mlp",   "mse", 20, "--indim 10 --outdim 10", "--width 32 --depth 2 --act tanh",                            1000, 2.5),
    ("cifar10", "cnn",   "mse", 20, "--indim 10 --outdim 10", "--chmul 0.25 --act relu",                                     800, 4.0),
    ("cifar10", "vgg11", "mse", 10, "--indim 10 --outdim 10", "--chmul 0.125 --act relu",                                    500, 3.0),   # shrunk (chmul 0.25→0.125, nsamp 15→10) — chmul 0.25 OOM'd (p=578K + big conv activations filled the 48GB GPU)
    ("mnist",   "mlp",   "mse", 20, "--indim 10 --outdim 10", "--width 32 --depth 2 --act tanh",                            1000, 2.5),
    ("mnist",   "cnn",   "mse", 20, "--indim 10 --outdim 10", "--chmul 0.25 --act relu",                                     800, 4.0),
    ("mnist",   "vgg11", "mse", 10, "--indim 10 --outdim 10", "--chmul 0.125 --act relu",                                    500, 3.0),   # shrunk (chmul 0.25→0.125, nsamp 15→10) — see cifar10 vgg11 note
    ("maxfind", "mlp",   "ce",  20, "--indim 10 --outdim 10", "--width 48 --depth 2 --act relu",                            1000, 2.5),
    ("maxfind", "gpt",   "ce",  20, "--indim 10 --outdim 10", "--dmodel 32 --nhead 2 --nlayer 2 --seqlen 10 --act relu",     800, 4.0),
    ("modadd",  "mlp",   "ce",  22, "--indim 22 --outdim 11", "--width 48 --depth 2 --act relu",                            1000, 2.5),
    ("modadd",  "gpt",   "ce",  22, "--indim 22 --outdim 11", "--dmodel 32 --nhead 2 --nlayer 2 --seqlen 22 --act relu",     800, 4.0),
]
LRS = [0.02, 0.05, 0.12]
# init-config: (initscheme, init_scale_or_None). default varies the scale; mup/kaiming ignore scale (canonical variance).
INITS = [("mup", None), ("kaiming_normal", None), ("kaiming_uniform", None),
         ("default", 0.1), ("default", 0.02), ("default", 0.5)]
SWPAIRS, SWSTEPS, SEED = 120, 100, 0
JOB_HOURS = 14.5   # target GPU-hours per stacked job (SLURM limit 16h)

# ── OPTIMIZER-VARIANT MSE sweep (--opt-sweep) ─────────────────────────────────────────────────────────────
# The main grid is ALL gd, so the optimizer-aware machinery (preconditioned-Hessian self-stabilization + the
# Prediction-6 adaptive-optimizer panel) never gets exercised. This sweep runs MSE + TANH MLP (where ∇S is
# non-trivial, so self-stab is meaningful) across the ADAPTIVE optimizers, each at optimizer-appropriate LRs.
OPT_COMBOS = [   # dataset, arch, loss, nsamp, dims, aflags, steps, est_h
    ("cifar10", "mlp", "mse", 20, "--indim 10 --outdim 10", "--width 32 --depth 2 --act tanh", 800, 2.0),
    ("mnist",   "mlp", "mse", 20, "--indim 10 --outdim 10", "--width 32 --depth 2 --act tanh", 800, 2.0),
]
OPT_LRS = {"sign": [0.005, 0.02], "spectral": [0.01, 0.05], "gaussnewton": [0.05, 0.2]}   # per-optimizer LR ranges
OPT_INITS = [("mup", None), ("default", 0.1)]


def opt_captures():
    """Optimizer-variant MSE captures — tanh MLP × {sign, spectral, gaussnewton} × per-opt LRs × inits. ss ON (tanh)."""
    out = []
    for ds, arch, loss, nsamp, dims, aflags, steps, est_h in OPT_COMBOS:
        for opt, lrs in OPT_LRS.items():
            for lr in lrs:
                for scheme, scale in OPT_INITS:
                    itag = scheme if scale is None else f"default{scale}"
                    iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
                    name = f"{ds}_{arch}_{loss}_{opt}_lr{lr}_{itag}_n{nsamp}_s{SEED}"
                    cmd = (f"eos_prediction_multiclass.py --dataset {ds} --loss {loss} --arch {arch} "
                           f"--nsamp {nsamp} {dims} {aflags} --lr {lr} {iflag} --steps {steps} "
                           f"--swpairs {SWPAIRS} --swsteps {SWSTEPS} --seed {SEED} --label {name} "
                           f"--set optimizer={opt} "   # ss stays ON (tanh) ⇒ exercises the preconditioned-Hessian self-stab
                           f"--out runs_captured/{name}.json")
                    out.append((name, cmd, est_h))
    return out


def captures(arch_filter=None, ds_filter=None):
    out = []
    for ds, arch, loss, nsamp, dims, aflags, steps, est_h in COMBOS:
        if arch_filter and arch not in arch_filter:
            continue
        if ds_filter and ds not in ds_filter:
            continue
        for lr in LRS:
            for scheme, scale in INITS:
                itag = scheme if scale is None else f"default{scale}"
                iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
                name = f"{ds}_{arch}_{loss}_lr{lr}_{itag}_n{nsamp}_s{SEED}"
                # self-stabilization (ss) is only meaningful for SMOOTH activations — for relu nets ∇S≈0
                # (sharpness ~piecewise-constant), so the cosines are ~blank while still costing Lanczos+2 HVPs
                # per eig-tick. Disable ss for relu configs; keep it for tanh (cifar10_mlp / mnist_mlp).
                ss_off = " --set ss=0" if "--act relu" in aflags else ""
                cmd = (f"eos_prediction_multiclass.py --dataset {ds} --loss {loss} --arch {arch} "
                       f"--nsamp {nsamp} {dims} {aflags} --lr {lr} {iflag} --steps {steps} "
                       f"--swpairs {SWPAIRS} --swsteps {SWSTEPS} --seed {SEED} --label {name}{ss_off} "
                       f"--out runs_captured/{name}.json")
                out.append((name, cmd, est_h))
    return out


def pack(caps):
    """Greedy bin-pack captures into jobs targeting ~JOB_HOURS each (never split a capture)."""
    jobs, cur, cur_h = [], [], 0.0
    for name, cmd, est_h in caps:
        if cur and cur_h + est_h > JOB_HOURS:
            jobs.append(cur); cur, cur_h = [], 0.0
        cur.append((name, cmd, est_h)); cur_h += est_h
    if cur:
        jobs.append(cur)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--archs", nargs="*", default=None, help="restrict arches, e.g. --archs mlp cnn")
    ap.add_argument("--datasets", nargs="*", default=None, help="restrict datasets")
    ap.add_argument("--rerun-missing", action="store_true", help="only (re)submit captures whose .min.json is absent (e.g. failed runs)")
    ap.add_argument("--opt-sweep", action="store_true", help="optimizer-variant MSE grid (tanh MLP × sign/spectral/gaussnewton × per-opt LRs) instead of the main grid")
    ap.add_argument("--tag", default="psweep", help="manifest + job-name prefix (use a UNIQUE tag per submission so re-runs don't overwrite each other's manifests, e.g. --tag vgg / --tag opt)")
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True

    caps = opt_captures() if a.opt_sweep else captures(a.archs, a.datasets)
    if a.rerun_missing:
        caps = [(n, c, e) for (n, c, e) in caps if not os.path.exists(f"{DIR}/runs_captured/{n}.min.json")]
    jobs = pack(caps)
    total_h = sum(c[2] for c in caps)
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    os.makedirs(f"{DIR}/runs_captured/logs", exist_ok=True)

    print(f"GRID: {len(caps)} captures  ·  {len(COMBOS) if not (a.archs or a.datasets) else '(filtered)'} dataset×arch "
          f"× {len(LRS)} lrs × {len(INITS)} init-configs")
    print(f"PACKING: {len(jobs)} SLURM jobs (~{JOB_HOURS:.0f}h target each)  ·  est total ≈ {total_h:.0f} GPU-hours "
          f"({total_h/24:.1f} GPU-days)")
    print()
    for ji, job in enumerate(jobs):
        mf = f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        with open(mf, "w") as f:
            f.write(f"# job {ji:02d}: {len(job)} captures, est {sum(c[2] for c in job):.1f}h\n")
            for name, cmd, est_h in job:
                f.write(cmd + "\n")
        jh = sum(c[2] for c in job)
        print(f"  job {ji:02d}  {len(job)} caps  est {jh:4.1f}h  [{', '.join(c[0] for c in job)}]")
        if a.submit:
            r = subprocess.run(["sbatch", f"--job-name={a.tag}_{ji:02d}",
                                f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"],
                               cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip())
    print()
    if a.submit:
        print(f"SUBMITTED {len(jobs)} jobs. Watch:  squeue -u samsj | grep psweep    Cancel:  scancel --name=psweep_NN  (or -u samsj)")
    else:
        print(f"DRY-RUN — manifests written to runs_captured/manifests/, submitted nothing. Re-run with --submit to launch.")


if __name__ == "__main__":
    main()
