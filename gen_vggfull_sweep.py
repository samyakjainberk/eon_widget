#!/usr/bin/env python
"""COMPREHENSIVE VGG MSE sweep across the full initialization-scale x learning-rate grid.

The prior VGG captures only ever used 3 of the 6 canonical init schemes (mup, kaiming_normal,
default@0.1) and clustered at lr 0.02/0.05. This fills the grid the user asked for: EVERY init scheme
(incl. the small-init default@0.02 and large-init default@0.5 that were entirely absent, plus
kaiming_uniform) x a dense LR span, across 2 datasets (cifar2 scalar, cifar10 10-class) and 2 net
sizes each. MSE only.

Canonical 6 inits (matches gen_bign_sweep.INITS):
  mup, kaiming_normal, kaiming_uniform, default@0.02 (small), default@0.1 (mid), default@0.5 (large)

MEMORY TIERS — probe-MEASURED (M = N*outD), routed so each job requests only the card it needs:
  cifar2-vgg  chmul0.125 n100  M=100  ->  9.7 GB  [A4000-ok]  <- the DENSE grid lives here (cheap)
  cifar10-vgg chmul0.125 n25   M=250  -> 10.3 GB  [A4000-ok]
  cifar2-vgg  chmul0.25  n100  M=100  -> 28.1 GB  [A6000-48]  (4x params)
  cifar10-vgg chmul0.125 n50   M=500  -> 29.7 GB  [A6000-48]
cifar10 n100 (54GB, A100) and cifar2 n1000 (62GB, A100) are covered by gen_a100_vgg / gen_vgg1000 and
are NOT re-swept here (too expensive for a dense init grid). A dedup drops anything already captured or
manifested, so re-running is safe.

  python gen_vggfull_sweep.py --dry-run | --submit
"""
import argparse, glob, os, re, subprocess

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
CIFAR = "--cifar-dir data/cifar-10-batches-py"
JOB_HOURS = 24.0
A6000 = "gpu:A6000:1"

# 6 canonical inits: (scheme, scale)  scale=None => named scheme; else default@scale
INITS_FULL = [("mup", None), ("kaiming_normal", None), ("kaiming_uniform", None),
              ("default", "0.02"), ("default", "0.1"), ("default", "0.5")]
# the 3 inits that were ENTIRELY ABSENT from every prior VGG capture — the big/expensive A6000 tiers
# only fill these holes (x a 3-LR subset) rather than re-running the full grid at a 2nd net size.
INITS_FILL = [("kaiming_uniform", None), ("default", "0.02"), ("default", "0.5")]

# LR span: small -> large, denser than the old 0.02/0.05 cluster
LRS_SMALL = [0.01, 0.02, 0.05, 0.08, 0.12, 0.2]     # scalar-output cifar2 (stable to higher lr)
LRS_10CLS = [0.01, 0.02, 0.05, 0.08, 0.12]          # 10-class cifar10
LRS_FILL  = [0.02, 0.05, 0.12]                       # 3-LR confirmation subset for the big tiers

# key, dataset, chmul, nsamp, outD, inits, lrs, base_h, needs_A6000
TIERS = [
    # cheap A4000 tiers — the PRIMARY init-scale x LR map (full 6-init grid), one per dataset:
    dict(key="cifar2_vgg",   ds="cifar2",  chmul="0.125", nsamp=100, outD=1,  inits=INITS_FULL, lrs=LRS_SMALL, base_h=6.0, big=False),
    dict(key="cifar10_vgg",  ds="cifar10", chmul="0.125", nsamp=25,  outD=10, inits=INITS_FULL, lrs=LRS_10CLS, base_h=6.0, big=False),
    # expensive A6000 tiers (bigger net / bigger batch) — fill only the 3 absent inits x 3 LRs:
    dict(key="cifar2_vgg25", ds="cifar2",  chmul="0.25",  nsamp=100, outD=1,  inits=INITS_FILL, lrs=LRS_FILL, base_h=9.0, big=True),
    dict(key="cifar10_vgg",  ds="cifar10", chmul="0.125", nsamp=50,  outD=10, inits=INITS_FILL, lrs=LRS_FILL, base_h=9.0, big=True),
]
STEPS = 400


def allsec(M):
    return (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={M} --set sw2cd=1 --set sw2e=1 "
            f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")


def swp(nsamp):
    return (40, 80) if nsamp <= 100 else (30, 70)


def captures():
    out = []
    for t in TIERS:
        M = t["nsamp"] * t["outD"]
        a, b = swp(t["nsamp"])
        for scheme, scale in t["inits"]:
            for lr in t["lrs"]:
                itag = scheme if scale is None else f"default{scale}"
                iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
                name = f"{t['key']}_mse_vgg11_n{t['nsamp']}_lr{lr}_{itag}_s0"
                cmd = (f"eos_prediction_multiclass.py --dataset {t['ds']} --loss mse --arch vgg11 "
                       f"--nsamp {t['nsamp']} --batch 0 --outdim {t['outD']} --act tanh "
                       f"--set chmul={t['chmul']} {CIFAR} --lr {lr} {iflag} --steps {STEPS} "
                       f"--swpairs {a} --swsteps {b} --seed 0 --label {name} {allsec(M)} "
                       f"--out runs_captured/{name}.json")
                out.append(dict(name=name, cmd=cmd, hours=t["base_h"], big=t["big"]))
    return out


def existing_labels(skip_tag=None):
    labs = {os.path.basename(p)[:-len(".min.json")] for p in glob.glob(f"{DIR}/runs_captured/*.min.json")}
    for mf in glob.glob(f"{DIR}/runs_captured/manifests/*.txt"):
        if skip_tag and os.path.basename(mf).startswith(skip_tag + "_"):
            continue
        for line in open(mf):
            m = re.search(r"--label (\S+)", line)
            if m:
                labs.add(m.group(1))
    return labs


def pack(caps, jobhours):
    jobs = []
    for isbig in (True, False):
        cur, cur_h = [], 0.0
        for c in [c for c in caps if c["big"] == isbig]:
            if cur and cur_h + c["hours"] > jobhours:
                jobs.append((isbig, cur)); cur, cur_h = [], 0.0
            cur.append(c); cur_h += c["hours"]
        if cur:
            jobs.append((isbig, cur))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--tag", default="vggfull")
    ap.add_argument("--maxjobs", type=int, default=20)
    ap.add_argument("--jobhours", type=float, default=JOB_HOURS)
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True
    have = existing_labels(skip_tag=a.tag)
    caps = [c for c in captures() if c["name"] not in have]
    dropped = len(captures()) - len(caps)
    jobs = pack(caps, a.jobhours)
    total_h = sum(c["hours"] for c in caps)
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    print(f"VGG MSE full init-scale x LR sweep — 6 inits x dense LR x 4 (dataset,size) tiers")
    print(f"GRID: {len(caps)} new captures ({dropped} dropped as already captured/manifested)")
    print(f"PACK: {len(jobs)} jobs  ·  est {total_h:.0f} GPU-h ({total_h/24:.1f} GPU-days)")
    print()
    submitted = 0
    for ji, (isbig, job) in enumerate(jobs):
        mf = f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        with open(mf, "w") as f:
            f.write(f"# {a.tag} job {ji:02d}: {len(job)} captures, est {sum(c['hours'] for c in job):.1f}h"
                    f"{' [A6000-48GB]' if isbig else ' [A4000-ok]'}\n")
            for c in job:
                f.write(c["cmd"] + "\n")
        tag = 'A6000' if isbig else 'any  '
        print(f"  job {ji:02d}  {len(job):2d} caps  est {sum(c['hours'] for c in job):4.1f}h  {tag}")
        if a.submit and ji < a.maxjobs:
            cmd = ["sbatch", f"--job-name={a.tag}_{ji:02d}", "--time=30:00:00"]
            if isbig:
                cmd += [f"--gres={A6000}", "--mem=64G"]
            cmd += [f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"]
            r = subprocess.run(cmd, cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip()); submitted += 1
    print()
    print(f"{'SUBMITTED '+str(submitted)+' jobs.' if a.submit else 'DRY-RUN — manifests written, nothing submitted.'}")


if __name__ == "__main__":
    main()
