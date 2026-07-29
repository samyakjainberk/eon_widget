#!/usr/bin/env python
"""cifar2 + VGG11 at FULL BATCH 1000 — the literal ask: full-batch GD, batch size 1000, on vgg,
without OOM. Runs on an A100-80GB.

MEASURED, not predicted: this config peaks at **62.35 GB and COMPLETES** on an A100-SXM4-80GB
(.cltmp/probe/probe_vgg1000_a100.sh). It does NOT fit a 48 GB A6000. I had predicted it could not fit
at all — that prediction was wrong; a bigger SINGLE card is what makes it work. (Multi-GPU still
cannot help: one run = one device, and the peak is a single dense (M,p) tensor that DDP would
replicate, not shard.)

M = N*outD = 1000*1 = 1000 (cifar2 is scalar-output, which is what keeps M sane at batch 1000;
cifar10's outD=10 would give M=10000, far past anything measured to fit).

COST (measured ~227 s per step-equivalent): steps=200 + a 5x16 sweep ~= 280 step-equivalents ~= 18 h,
so 1 capture per job. TRADE-OFFS vs the n100 vgg captures: shorter trajectory (200 vs 400 steps) and
a 5-point sweep scatter instead of 40. Stock settings would be ~100 h/capture.

  python gen_vgg1000.py --dry-run | --submit
"""
import argparse, glob, os, re, subprocess

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
A100 = "gpu:A100-SXM4-80GB:1"
NSAMP, OUTD = 1000, 1
M = NSAMP * OUTD
STEPS, SWPAIRS, SWSTEPS = 200, 5, 16
COMBOS = [("mup", None, 0.02), ("mup", None, 0.05), ("mup", None, 0.08)]


def captures():
    out = []
    for scheme, scale, lr in COMBOS:
        itag = scheme if scale is None else f"default{scale}"
        iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
        name = f"cifar2_vgg_mse_vgg11_n{NSAMP}_lr{lr}_{itag}_s0"
        cmd = (f"eos_prediction_multiclass.py --dataset cifar2 --loss mse --arch vgg11 "
               f"--nsamp {NSAMP} --batch 0 --outdim {OUTD} --act tanh --set chmul=0.125 "
               f"--cifar-dir data/cifar-10-batches-py --lr {lr} {iflag} --steps {STEPS} "
               f"--swpairs {SWPAIRS} --swsteps {SWSTEPS} --seed 0 --label {name} "
               f"--set qspec=1 --set qspecevery=5 --set grid3dcap={M} --set sw2cd=1 --set sw2e=1 "
               f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03 "
               f"--out runs_captured/{name}.json")
        out.append(dict(name=name, cmd=cmd))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--tag", default="vgg1k")
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True
    have = existing_labels(skip_tag=a.tag)
    caps = [c for c in captures() if c["name"] not in have]
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    print(f"cifar2-vgg FULL BATCH {NSAMP} (M={M}) on A100-80GB — measured peak 62.35GB, ~18h/capture")
    print(f"GRID: {len(caps)} captures, 1 per job\n")
    submitted = 0
    for ji, c in enumerate(caps):
        mf = f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        with open(mf, "w") as f:
            f.write(f"# {a.tag} job {ji:02d}: 1 capture [A100-80GB] est ~18h\n")
            f.write(c["cmd"] + "\n")
        print(f"  job {ji:02d}  {c['name']}")
        if a.submit:
            cmd = ["sbatch", f"--job-name={a.tag}_{ji:02d}", "--time=30:00:00",
                   f"--gres={A100}", "--mem=96G",
                   f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"]
            r = subprocess.run(cmd, cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip()); submitted += 1
    print()
    print(f"{'SUBMITTED '+str(submitted)+' jobs.' if a.submit else 'DRY-RUN — nothing submitted.'}")


if __name__ == "__main__":
    main()
