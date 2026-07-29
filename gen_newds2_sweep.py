#!/usr/bin/env python
"""Wave-2 MSE captures for the new datasets — EXTENDS gen_newds_sweep.py with:
  (1) a DENSE mup-init LR sweep (medium+high) across every dataset — the user asked twice for
      "more runs with mup init at high and medium learning rate", so mup gets a thick LR grid here;
  (2) cifar10 + VGG11 (10-class MSE) at a REDUCED batch nsamp=25 (M=N·outD=250) — the 10-class vgg
      needs a smaller batch to fit 16 GB (n50→M500→17.7 GB OOMs; n25→M250→10.3 GB fits). The user
      OK'd decreasing the vgg batch further. cifar2-vgg (scalar, M=100) already runs at n100.
  (3) seed=1 replication of the strongest mup configs (variance / robustness).

All MSE. Memory-verified vgg ceiling on A4000-16 GB: chmul=0.125 (~145k) only — chmul=0.25 (~600k)
hits 40 GB even at M=100. Labels carry distinct LRs/seeds so nothing collides with wave-1 (newds_*);
a dedup skips any label already captured or manifested.

  python gen_newds2_sweep.py --dry-run
  python gen_newds2_sweep.py --submit
"""
import argparse, glob, os, re, subprocess

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
JOB_HOURS = 24.0
CIFAR = "--cifar-dir data/cifar-10-batches-py"

def allsec(g3d):
    return (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={g3d} --set sw2cd=1 --set sw2e=1 "
            f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")

def swpairs_steps(nsamp):
    return (40, 80) if nsamp <= 100 else (30, 70)

# ---- (1) dense mup LR sweep (mup init only) ---------------------------------------------------------
#   LRs chosen DISJOINT from wave-1's mup LRs so no capture repeats.
MUP_DENSE = [
    dict(key="chebyshev",   ds="chebyshev", arch="mlp", extra="--indim 1 --outdim 1 --degree 3",
         aflags="--width 50 --depth 4 --act tanh", outD=1, nsamps=[100, 250], steps=600, base_h=2.0,
         lrs=[0.03, 0.08, 0.15, 0.3]),
    dict(key="ksparse",     ds="ksparse", arch="mlp", extra="--indim 10 --outdim 1 --set ksparse=3",
         aflags="--width 64 --depth 2 --act tanh", outD=1, nsamps=[100, 250], steps=600, base_h=2.0,
         lrs=[0.03, 0.08, 0.15, 0.3]),
    dict(key="saddle",      ds="saddle", arch="mlp", extra="--indim 4 --outdim 4 --set saddlesep=0.4",
         aflags="--width 64 --depth 2 --act tanh", outD=4, nsamps=[100], steps=800, base_h=3.0,
         lrs=[0.15, 0.25, 0.4]),
    dict(key="ksparse_gpt", ds="ksparse", arch="gpt",
         extra="--indim 10 --outdim 1 --set ksparse=3 --dmodel 32 --nlayer 2 --nhead 4",
         aflags="", outD=1, nsamps=[100], steps=500, base_h=3.2, lrs=[0.005, 0.015, 0.03]),
    dict(key="cifar2_vgg",  ds="cifar2", arch="vgg11", extra=f"--set chmul=0.125 {CIFAR}",
         aflags="--act tanh", outD=1, nsamps=[100], steps=400, base_h=5.5, lrs=[0.03, 0.065, 0.1]),
]

# ---- (2) cifar10 (10-class) vgg at reduced batch -----------------------------------------------------
CIFAR10_VGG = dict(key="cifar10_vgg", ds="cifar10", arch="vgg11", extra=f"--set chmul=0.125 {CIFAR}",
                   aflags="--act tanh", outD=10, nsamps=[25], steps=400, base_h=4.0,
                   inits=[("mup", None), ("kaiming_normal", None), ("default", "0.1")],
                   lrs=[0.02, 0.05], mup_hi=0.08)

# ---- (3) seed=1 replication of the strongest mup configs (these LRs exist in wave-1 ⇒ true replicas) --
SEED_REPS = [
    ("chebyshev",  "chebyshev", "mlp", "--indim 1 --outdim 1 --degree 3",           "--width 50 --depth 4 --act tanh", 1,  [0.05, 0.1],  600, 2.0),
    ("ksparse",    "ksparse",   "mlp", "--indim 10 --outdim 1 --set ksparse=3",      "--width 64 --depth 2 --act tanh", 1,  [0.05, 0.12], 600, 2.0),
    ("saddle",     "saddle",    "mlp", "--indim 4 --outdim 4 --set saddlesep=0.4",   "--width 64 --depth 2 --act tanh", 4,  [0.1, 0.2],   800, 3.0),
    ("cifar2_vgg", "cifar2",    "vgg11", f"--set chmul=0.125 {CIFAR}",               "--act tanh",                      1,  [0.02, 0.05], 400, 5.5),
]

def _cap(key, ds, arch, extra, aflags, outD, nsamp, lr, scheme, scale, steps, base_h, seed):
    itag = scheme if scale is None else f"default{scale}"
    iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
    name = f"{key}_mse_{arch}_n{nsamp}_lr{lr}_{itag}_s{seed}"
    swp, sws = swpairs_steps(nsamp)
    cmd = (f"eos_prediction_multiclass.py --dataset {ds} --loss mse --arch {arch} "
           f"--nsamp {nsamp} {extra} {aflags} --lr {lr} {iflag} --steps {steps} "
           f"--swpairs {swp} --swsteps {sws} --seed {seed} --label {name} {allsec(nsamp*outD)} "
           f"--out runs_captured/{name}.json")
    return (name, cmd, base_h * (1.0 if nsamp <= 100 else 1.4))

def captures():
    out = []
    for d in MUP_DENSE:                                           # (1) dense mup
        for nsamp in d["nsamps"]:
            for lr in d["lrs"]:
                out.append(_cap(d["key"], d["ds"], d["arch"], d["extra"], d["aflags"], d["outD"],
                                nsamp, lr, "mup", None, d["steps"], d["base_h"], 0))
    d = CIFAR10_VGG                                               # (2) cifar10 vgg (reduced batch)
    combos = [(s, sc, lr) for (s, sc) in d["inits"] for lr in d["lrs"]] + [("mup", None, d["mup_hi"])]
    for nsamp in d["nsamps"]:
        for scheme, scale, lr in combos:
            out.append(_cap(d["key"], d["ds"], d["arch"], d["extra"], d["aflags"], d["outD"],
                            nsamp, lr, scheme, scale, d["steps"], d["base_h"], 0))
    for key, ds, arch, extra, aflags, outD, lrs, steps, base_h in SEED_REPS:   # (3) seed=1
        for lr in lrs:
            out.append(_cap(key, ds, arch, extra, aflags, outD, 100, lr, "mup", None, steps, base_h, 1))
    return out

def existing_labels(skip_tag=None):
    labs = {os.path.basename(p)[:-len(".min.json")] for p in glob.glob(f"{DIR}/runs_captured/*.min.json")}
    for mf in glob.glob(f"{DIR}/runs_captured/manifests/*.txt"):
        if skip_tag and os.path.basename(mf).startswith(skip_tag + "_"):
            continue                                                 # don't dedup against OUR OWN prior manifests
        for line in open(mf):
            m = re.search(r"--label (\S+)", line)
            if m:
                labs.add(m.group(1))
    return labs

def pack(caps, jobhours):
    jobs, cur, cur_h = [], [], 0.0
    for c in caps:
        if cur and cur_h + c[2] > jobhours:
            jobs.append(cur); cur, cur_h = [], 0.0
        cur.append(c); cur_h += c[2]
    if cur:
        jobs.append(cur)
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--tag", default="newds2")
    ap.add_argument("--maxjobs", type=int, default=10)
    ap.add_argument("--jobhours", type=float, default=JOB_HOURS)
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True
    have = existing_labels(skip_tag=a.tag)
    caps = [c for c in captures() if c[0] not in have]
    dropped = len(captures()) - len(caps)
    jobs = pack(caps, a.jobhours)
    total_h = sum(c[2] for c in caps)
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    print(f"GRID: {len(caps)} captures ({dropped} dropped as already captured/manifested)")
    print(f"PACK: {len(jobs)} jobs (~{a.jobhours:.0f}h)  ·  est {total_h:.0f} GPU-h ({total_h/24:.1f} GPU-days)")
    print()
    submitted = 0
    for ji, job in enumerate(jobs):
        mf = f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        with open(mf, "w") as f:
            f.write(f"# {a.tag} job {ji:02d}: {len(job)} captures, est {sum(c[2] for c in job):.1f}h\n")
            for name, cmd, e in job:
                f.write(cmd + "\n")
        print(f"  job {ji:02d}  {len(job)} caps  est {sum(c[2] for c in job):4.1f}h  [{', '.join(c[0] for c in job)}]")
        if a.submit and ji < a.maxjobs:
            r = subprocess.run(["sbatch", f"--job-name={a.tag}_{ji:02d}", "--time=30:00:00",
                                f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"],
                               cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip()); submitted += 1
    print()
    print(f"{'SUBMITTED '+str(submitted)+' jobs.' if a.submit else 'DRY-RUN — manifests written, submitted nothing.'}")

if __name__ == "__main__":
    main()
