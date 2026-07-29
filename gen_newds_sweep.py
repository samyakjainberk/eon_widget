#!/usr/bin/env python
"""Generate + submit MSE captures for the NEW datasets the user asked for — all populate EVERY
prediction-widget plot, MSE loss only (no CE), with mup-init emphasis at medium/high LR:

  vgg      cifar2 + VGG11 conv net (chmul=0.125 ⇒ ~145k params).  Uses cifar2 (scalar ±1 → outD=1) so
           M=N·outD stays =N=100 and the dense (M,p) Jacobian predictions fit in <10 GB (cifar10's
           outD=10 makes M=1000 ⇒ 46 GB OOM; the backend is single-device so we shrink M, not shard GPUs).
  ksparse  k-sparse parity — BOTH mlp AND gpt/transformer (the "sparse parity, both mlp and transformer").
  chebyshev  Chebyshev T_k regression (Cohen et al. EoS toy), deep-ish tanh MLP.
  saddle   saddle-to-saddle linear-regression staircase dynamics (small init).

Every capture turns on all sections + lifts the sweep guard (same allsec flags as gen_bign_sweep.py) and
sets grid3dcap=N·outD so the dense-Jacobian predictions run. MLP-only panels (full qspec, §35) auto-skip
for vgg/gpt; the rest (Pred-1..6, ray, EDS, self-stab, truncated qspec) populate — verified by smoke test.

mup emphasis: mup is the primary init (run at every base LR = medium+high) AND each dataset gets an extra
higher-LR mup run (mup_hi) — satisfying "more runs with mup init with high and medium learning rate".

  python gen_newds_sweep.py --dry-run      # print grid + packing
  python gen_newds_sweep.py --submit       # sbatch the jobs
"""
import argparse, os, subprocess

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
SEED = 0
JOB_HOURS = 24.0
CIFAR = "--cifar-dir data/cifar-10-batches-py"   # cifar2 is NOT auto-matched by the stack script (it greps cifar10) ⇒ pass explicitly

# default init list — mup PRIMARY, then two references
INITS_STD = [("mup", None), ("kaiming_normal", None), ("default", "0.1")]

# key, dataset, arch, per-dataset flags, arch/mlp flags, outD, base LRs, extra-high mup LR, nsamps, steps, base GPU-h
DATASETS = [
    dict(key="chebyshev", ds="chebyshev", arch="mlp",
         extra="--indim 1 --outdim 1 --degree 3", aflags="--width 50 --depth 4 --act tanh",
         outD=1, lrs=[0.02, 0.05, 0.1], mup_hi=0.2, nsamps=[100, 250], steps=600, base_h=2.0),
    dict(key="ksparse", ds="ksparse", arch="mlp",
         extra="--indim 10 --outdim 1 --set ksparse=3", aflags="--width 64 --depth 2 --act tanh",
         outD=1, lrs=[0.02, 0.05, 0.12], mup_hi=0.2, nsamps=[100, 250], steps=600, base_h=2.0),
    dict(key="saddle", ds="saddle", arch="mlp",
         extra="--indim 4 --outdim 4 --set saddlesep=0.4", aflags="--width 64 --depth 2 --act tanh",
         outD=4, lrs=[0.1, 0.2], mup_hi=0.3, nsamps=[100], steps=800, base_h=3.0,
         inits=[("mup", None), ("kaiming_normal", None), ("default", "0.05")]),   # saddle-to-saddle wants SMALL init
    dict(key="ksparse_gpt", ds="ksparse", arch="gpt",
         extra="--indim 10 --outdim 1 --set ksparse=3 --dmodel 32 --nlayer 2 --nhead 4", aflags="",
         outD=1, lrs=[0.003, 0.01], mup_hi=0.02, nsamps=[100], steps=500, base_h=3.2),
    dict(key="cifar2_vgg", ds="cifar2", arch="vgg11",
         extra=f"--set chmul=0.125 {CIFAR}", aflags="--act tanh",
         outD=1, lrs=[0.02, 0.05], mup_hi=0.08, nsamps=[100], steps=400, base_h=5.5),
]

def swpairs_steps(nsamp):
    return (40, 80) if nsamp <= 100 else (30, 70)

def captures(keys=None, nsamps=None):
    out = []
    for d in DATASETS:
        if keys and d["key"] not in keys:
            continue
        inits = d.get("inits", INITS_STD)
        # (init, lr) grid = base inits × base lrs, PLUS an extra high-LR mup run (mup emphasis)
        combos = [(s, sc, lr) for (s, sc) in inits for lr in d["lrs"]]
        combos.append(("mup", None, d["mup_hi"]))
        for nsamp in d["nsamps"]:
            if nsamps and nsamp not in nsamps:
                continue
            g3d = nsamp * d["outD"]
            swp, sws = swpairs_steps(nsamp)
            for scheme, scale, lr in combos:
                itag = scheme if scale is None else f"default{scale}"
                iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
                name = f"{d['key']}_mse_{d['arch']}_n{nsamp}_lr{lr}_{itag}_s{SEED}"
                allsec = (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={g3d} --set sw2cd=1 --set sw2e=1 "
                          f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")
                cmd = (f"eos_prediction_multiclass.py --dataset {d['ds']} --loss mse --arch {d['arch']} "
                       f"--nsamp {nsamp} {d['extra']} {d['aflags']} --lr {lr} {iflag} --steps {d['steps']} "
                       f"--swpairs {swp} --swsteps {sws} --seed {SEED} --label {name} {allsec} "
                       f"--out runs_captured/{name}.json")
                out.append((name, cmd, d["base_h"] * (1.0 if nsamp <= 100 else 1.4)))
    return out

def pack(caps, jobhours):
    jobs, cur, cur_h = [], [], 0.0
    for name, cmd, e in caps:
        if cur and cur_h + e > jobhours:
            jobs.append(cur); cur, cur_h = [], 0.0
        cur.append((name, cmd, e)); cur_h += e
    if cur:
        jobs.append(cur)
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--keys", nargs="*", default=None, help="restrict to dataset keys")
    ap.add_argument("--nsamps", nargs="*", type=int, default=None)
    ap.add_argument("--rerun-missing", action="store_true")
    ap.add_argument("--tag", default="newds")
    ap.add_argument("--maxjobs", type=int, default=12)
    ap.add_argument("--jobhours", type=float, default=JOB_HOURS)
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True
    caps = captures(a.keys, a.nsamps)
    if a.rerun_missing:
        caps = [c for c in caps if not os.path.exists(f"{DIR}/runs_captured/{c[0]}.min.json")]
    jobs = pack(caps, a.jobhours)
    total_h = sum(c[2] for c in caps)
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    os.makedirs(f"{DIR}/runs_captured/logs", exist_ok=True)
    print(f"GRID: {len(caps)} captures across {len({c[0].split('_mse_')[0] for c in caps})} dataset-keys")
    print(f"PACK: {len(jobs)} jobs (~{a.jobhours:.0f}h target)  ·  est {total_h:.0f} GPU-h ({total_h/24:.1f} GPU-days)")
    if len(jobs) > a.maxjobs:
        print(f"  !! {len(jobs)} jobs > --maxjobs {a.maxjobs}; submitting only the first {a.maxjobs}")
    print()
    submitted = 0
    for ji, job in enumerate(jobs):
        mf = f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        with open(mf, "w") as f:
            f.write(f"# job {ji:02d}: {len(job)} captures, est {sum(c[2] for c in job):.1f}h\n")
            for name, cmd, e in job:
                f.write(cmd + "\n")
        print(f"  job {ji:02d}  {len(job)} caps  est {sum(c[2] for c in job):4.1f}h  [{', '.join(c[0] for c in job)}]")
        if a.submit and ji < a.maxjobs:
            r = subprocess.run(["sbatch", f"--job-name={a.tag}_{ji:02d}", "--time=30:00:00",
                                f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"],
                               cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip()); submitted += 1
    print()
    print(f"{'SUBMITTED '+str(submitted)+' jobs.' if a.submit else 'DRY-RUN — manifests written, submitted nothing.'}"
          f"  Watch: squeue -u samsj | grep {a.tag}")

if __name__ == "__main__":
    main()
