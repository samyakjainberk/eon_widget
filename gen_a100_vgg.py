#!/usr/bin/env python
"""cifar10 + VGG11, 10-class MSE at nsamp=100 (M = N*outD = 1000) — the FULL-batch 10-class vgg,
run on an A100-80GB.

WHY THIS EXISTS: this exact config was previously written off as impossible. It OOMs at 50.34 GB on
a 48 GB A6000 (measured), which is where "cifar10 n100 is ruled out" came from. That conclusion was
card-specific: probe-MEASURED on an A100-SXM4-80GB it peaks at **54.42 GB and completes** (~25 GB
headroom). A bigger single card — NOT multi-GPU — is what clears this: a run is pinned to one device
(server.py build_device_pool = one GPU per run; no DDP/FSDP/sharding anywhere), and the peak is a
single dense (M,p) tensor that data-parallelism replicates rather than shards.

COST (measured): ~109 s per step-equivalent at M=1000 => the stock steps=400 + 40x80 sweep would be
~100 h. Reduced to steps=250 + an 8x25 sweep (~450 step-equivalents, ~14 h/capture, 1 per job).
TRADE-OFFS vs the n25/n50 cifar10-vgg captures: shorter trajectory (250 vs 400 steps) and a sweep
scatter with 8 points instead of 40. Everything else (all sections, grid3dcap=M) is unchanged.

  python gen_a100_vgg.py --dry-run
  python gen_a100_vgg.py --submit
"""
import argparse, glob, os, re, subprocess

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
A100 = "gpu:A100-SXM4-80GB:1"
CIFAR = "--cifar-dir data/cifar-10-batches-py"
NSAMP, OUTD = 100, 10
M = NSAMP * OUTD                 # 1000
STEPS, SWPAIRS, SWSTEPS = 250, 8, 25

COMBOS = [("mup", None, 0.02), ("mup", None, 0.05), ("mup", None, 0.08),
          ("kaiming_normal", None, 0.05)]


def allsec():
    return (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={M} --set sw2cd=1 --set sw2e=1 "
            f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")


def captures():
    out = []
    for scheme, scale, lr in COMBOS:
        itag = scheme if scale is None else f"default{scale}"
        iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
        name = f"cifar10_vgg_mse_vgg11_n{NSAMP}_lr{lr}_{itag}_s0"
        cmd = (f"eos_prediction_multiclass.py --dataset cifar10 --loss mse --arch vgg11 "
               f"--nsamp {NSAMP} --batch 0 --outdim {OUTD} --act tanh --set chmul=0.125 {CIFAR} "
               f"--lr {lr} {iflag} --steps {STEPS} --swpairs {SWPAIRS} --swsteps {SWSTEPS} "
               f"--seed 0 --label {name} {allsec()} --out runs_captured/{name}.json")
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
    ap.add_argument("--tag", default="a100vgg")
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True
    have = existing_labels(skip_tag=a.tag)
    caps = [c for c in captures() if c["name"] not in have]
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    print(f"cifar10-vgg 10-class, FULL batch n={NSAMP} (M={M}) on A100-80GB — measured peak 54.42GB")
    print(f"GRID: {len(caps)} captures, 1 per job (~14 h each)")
    print()
    submitted = 0
    for ji, c in enumerate(caps):
        mf = f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        with open(mf, "w") as f:
            f.write(f"# {a.tag} job {ji:02d}: 1 capture [A100-80GB] est ~14h\n")
            f.write(c["cmd"] + "\n")
        print(f"  job {ji:02d}  {c['name']}")
        if a.submit:
            cmd = ["sbatch", f"--job-name={a.tag}_{ji:02d}", "--time=30:00:00",
                   f"--gres={A100}", "--mem=96G",
                   f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"]
            r = subprocess.run(cmd, cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip()); submitted += 1
    print()
    print(f"{'SUBMITTED '+str(submitted)+' jobs.' if a.submit else 'DRY-RUN — manifests written, submitted nothing.'}")


if __name__ == "__main__":
    main()
