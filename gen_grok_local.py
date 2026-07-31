#!/usr/bin/env python
"""Generate STACKED local-GPU capture sweeps for the grokking study.

Emits one bash file per GPU (runs_captured captures stacked sequentially) + a master launcher.
Every capture is a LOADABLE prediction-widget .json (→ runs_captured/, pick via '⬆ load from server').

Design constraints (user):
  * tanh activation, depth>=3 (=3), width>=100 (=100), full-batch (batch=0) except a few minibatch,
    nsamp>=500, batch>=500 (full-batch @ nsamp>=500 => effective batch>=500), steps>=1000 (=1200).
  * §6 residual<->curvature alignment (s29) ALWAYS on.
  * grok sub-run config == main config (capture_run.py now inherits gw_nsamp/gw_batch/gw_steps).
  * Heavy O(N^2 p)/O(N^3) sections (s2/s3 eig-Lanczos, s6-s28 cubes/optimizers/surrogates) OFF —
    they OOM / are too slow at nsamp=500/width=100. Kept: §1 (loss+TESTLOSS+sharp+spectral edges),
    §5 SLQ scree, §6 (s29). Grokking is visible directly in §1 (train vs held-out test loss).
  * Priority: SPARSE PARITY, many (lr x init). Grok configs (init 2-4, lr 0.1-0.3) FIRST.

Usage:
  python gen_grok_local.py                 # write bash files + launcher (no run)
  python gen_grok_local.py --launch        # write AND launch all bash files in the background
  python gen_grok_local.py --gpus 0,2,4,7  # choose GPUs
"""
import argparse, os, stat

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
PY = "/nas/ucb/samsj/conda_env/envs/samsenv/bin/python"
CIFAR = f"{DIR}/data/cifar-10-batches-py"
OUT = f"{DIR}/runs_captured"
BASHDIR = f"{DIR}/grok_local_jobs"

# disable heavy sections: s2,s3 (eig-Lanczos) + s4 + s6..s35 EXCEPT s29 (§6 residual↔curvature — ALWAYS ON).
# CRITICAL: s29 must NOT be in DIS — DIS is appended AFTER "--set s29=1" and the last --set wins, so including
# 29 here would silently clobber §6 back to 0 (it did, on the first launch — §6 records went missing).
DIS = " ".join(f"--set s{i}=0" for i in [2, 3, 4] + [i for i in range(6, 36) if i != 29])
# shared template: tanh depth3 width100, steps2000 (captures the full grok ~t900-1200), §6 on (s29k=30),
# WEIGHT DECAY wd=5e-3 (ESSENTIAL for grokking — coupled SGD weight_decay), lean+fast cadence (~60-70s/capture)
TMPL = ("--arch mlp --act tanh --set depth=3 --set width=100 --set bias=1 --set initscheme=default "
        "--set wd=5e-3 --steps 2000 --set eigevery=200 --set slqevery=400 --set grid3dcap=11000 "
        "--set s1=1 --set s5=1 --set s29=1 --set s29k=30 " + DIS)

def cap(name, ds_flags, init, lr, nsamp=500, batch=0, seed=0):
    """One capture_run.py invocation string (batch=0 => full batch)."""
    out = f"{OUT}/{name}.json"
    return (f"OMP_NUM_THREADS=6 {PY} -u capture_run.py {ds_flags} {TMPL} "
            f"--nsamp {nsamp} --set batch={batch} --init {init} --lr {lr} --seed {seed} "
            f"--cifar-dir {CIFAR} --device __DEV__ --out {out} "
            f"--label '{name}'")

def build_runs():
    runs = []  # (name, cmd)  — ORDER = priority (parity-grok first)
    # ---------- SPARSE PARITY (priority) ----------
    P = "--dataset ksparse --loss mse --set indim=30 --set ksparse=3"
    # A) GROK regime: init 2-4 x lr 0.1-0.3 @ nsamp500 (init>=2 + lr>=0.1 => grokking; lr0.3 groks <1000)
    for it in (2.0, 3.0, 4.0):
        for lr in (0.1, 0.2, 0.3):
            n = f"grok_parity_k3_n500_i{it}_lr{lr}_s0"
            runs.append((n, cap(n, P, it, lr, nsamp=500)))
    # B) the <1000-iter headline grok, extra seeds (reproducibility)
    for sd in (1, 2, 3):
        n = f"grok_parity_k3_n500_i3.0_lr0.3_s{sd}"
        runs.append((n, cap(n, P, 3.0, 0.3, nsamp=500, seed=sd)))
    # C) nsamp700 grok
    for lr in (0.2, 0.3):
        n = f"grok_parity_k3_n700_i3.0_lr{lr}_s0"
        runs.append((n, cap(n, P, 3.0, lr, nsamp=700)))
    # D) lr x init CONTRAST sweep (low init => no grok / immediate) @ nsamp500
    for it in (0.5, 1.0):
        for lr in (0.05, 0.1, 0.2):
            n = f"parity_k3_n500_i{it}_lr{lr}_s0"
            runs.append((n, cap(n, P, it, lr, nsamp=500)))
    # E) a k=5 grok (init4, more samples)
    n = "grok_parity_k5_n800_i4.0_lr0.3_s0"
    runs.append((n, cap(n, "--dataset ksparse --loss mse --set indim=30 --set ksparse=5", 4.0, 0.3, nsamp=800)))
    # F) the "few" MINIBATCH runs (batch=500 < nsamp=1000)
    for lr in (0.2, 0.3):
        n = f"grok_parity_k3_n1000_b500_i3.0_lr{lr}_s0"
        runs.append((n, cap(n, P, 3.0, lr, nsamp=1000, batch=500)))
    # ---------- CHEBYSHEV ----------
    C = "--dataset chebyshev --loss mse --set degree=3"
    for it in (0.5, 1.0, 2.0):
        for lr in (0.05, 0.1):
            n = f"chebyshev_deg3_n500_i{it}_lr{lr}_s0"
            runs.append((n, cap(n, C, it, lr, nsamp=500)))
    # ---------- MAXFIND (CE, 10-class) ----------
    M = "--dataset maxfind --loss ce --set indim=10 --set outdim=10"
    for it in (0.5, 1.0):
        for lr in (0.05, 0.1):
            n = f"maxfind10_n500_i{it}_lr{lr}_s0"
            runs.append((n, cap(n, M, it, lr, nsamp=500)))
    # ---------- CIFAR-10 (MSE, real data) ----------
    CF = "--dataset cifar10 --loss mse"
    for it in (0.5, 1.0):
        for lr in (0.02, 0.05):
            n = f"cifar10_mlp_n500_i{it}_lr{lr}_s0"
            runs.append((n, cap(n, CF, it, lr, nsamp=500)))
    return runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,2,4,5,7")
    ap.add_argument("--launch", action="store_true")
    a = ap.parse_args()
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    os.makedirs(OUT, exist_ok=True); os.makedirs(BASHDIR, exist_ok=True)
    runs = build_runs()
    # round-robin assign runs to GPUs, one stacked bash file per GPU
    per = {g: [] for g in gpus}
    for i, (name, cmd) in enumerate(runs):
        g = gpus[i % len(gpus)]
        per[g].append((name, cmd.replace("__DEV__", f"cuda:{g}")))
    launcher = [f"#!/bin/bash", f"# master launcher — {len(runs)} captures across GPUs {gpus}", f"cd {DIR}"]
    for g in gpus:
        bf = f"{BASHDIR}/job_gpu{g}.sh"
        lines = [f"#!/bin/bash", f"# GPU {g}: {len(per[g])} captures (stacked, sequential)", f"cd {DIR}", "set +e"]
        for j, (name, cmd) in enumerate(per[g]):
            lines.append(f'echo "[gpu{g}] ({j+1}/{len(per[g])}) {name} ..."')
            lines.append(f'{cmd} && echo "[gpu{g}] DONE {name}" || echo "[gpu{g}] FAIL {name}"')
        lines.append(f'echo "[gpu{g}] ALL DONE"')
        open(bf, "w").write("\n".join(lines) + "\n")
        os.chmod(bf, os.stat(bf).st_mode | stat.S_IEXEC)
        launcher.append(f"setsid nohup bash {bf} > {BASHDIR}/log_gpu{g}.txt 2>&1 &")
    launcher.append('echo "launched all GPU jobs"')
    lf = f"{BASHDIR}/launch_all.sh"
    open(lf, "w").write("\n".join(launcher) + "\n"); os.chmod(lf, os.stat(lf).st_mode | stat.S_IEXEC)
    print(f"{len(runs)} captures → {len(gpus)} GPUs ({', '.join(str(len(per[g])) for g in gpus)} each)")
    print(f"bash files: {BASHDIR}/job_gpu*.sh ; launcher: {lf}")
    print("first 6 run names:", ", ".join(n for n, _ in runs[:6]))
    if a.launch:
        import subprocess
        subprocess.run(["bash", lf], check=True)
        print("LAUNCHED in background.")

if __name__ == "__main__":
    main()
