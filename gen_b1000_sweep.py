#!/usr/bin/env python
"""FULL-BATCH GD at batch size 1000 (nsamp=1000) — 4x bigger batch than anything captured so far
(gen_bign_sweep tops out at nsamp=250).

"Full batch" needs no flag: server.py:3751 does `bs = Nfull if int(P.get("batch",0)) <= 0` with
batch defaulting to 0, i.e. the whole pool every step, deterministic (server.py:3749). So batch
size 1000 == `--nsamp 1000`; we pass `--batch 0` explicitly to make that self-documenting.

WHAT FITS (probe-MEASURED on A6000-48GB via .cltmp/probe/probe_b1000.sh — NOT extrapolated):
  chebyshev n1000  M=1000  -> 17.38 GB, ~24 s/step  [A6000: >16GB, so NOT an A4000]
  ksparse   n1000  M=1000  ->  9.80 GB, ~15 s/step  [fits a 16GB A4000 too]
  saddle    n1000  M=4000  -> 17.05 GB, ~32 s/step  [A6000; M=4000 sits exactly AT swMmax]
NOT INCLUDED, and why:
  ksparse_gpt n1000 -> measured ~105 s/step (2812s for ~20 steps): a 500-step capture is ~15 h of
      main-run alone. Impractical; excluded deliberately, not forgotten.
  cifar2/cifar10 + vgg n1000 -> M*p is identical to the cifar10-vgg n100 config that MEASURED an
      OOM at 50.34 GB on this same 48 GB card, plus 10x the conv activations => cannot fit. There is
      no multi-GPU escape: a run is pinned to ONE device (server.py build_device_pool assigns one GPU
      PER RUN for throughput; there is no DDP/FSDP/tensor-sharding anywhere), and the OOM is a single
      dense (M,p) tensor, which data-parallelism does not shrink.

M = N*outD drives both the dense-Jacobian block (grid3dcap) and server.py's sweep guard swMmax
(default 400 => raised to 4000). saddle has outD=4 => M=4000 at n=1000, exactly at the cap.

COST: the sweep trains `swpairs` mini-nets x `swsteps` full-batch steps at n=1000, so the stock
40x80=3200 sweep-steps would dominate (~18 h). Reduced to 16x40=640 here => the Prediction-1&2
sweep scatter gets 16 points instead of 40. That is a deliberate cost trade, and the only knob
lowered relative to the n<=250 captures.

  python gen_b1000_sweep.py --dry-run
  python gen_b1000_sweep.py --submit
"""
import argparse, glob, os, re, subprocess

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
JOB_HOURS = 24.0
A6000 = "gpu:A6000:1"
NSAMP = 1000
STEPS = 400
SWPAIRS, SWSTEPS = 16, 40          # 640 sweep-steps (vs 3200 stock) — see COST above


def allsec(M):
    return (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={M} --set sw2cd=1 --set sw2e=1 "
            f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")


GRID = [
    dict(key="chebyshev", ds="chebyshev", arch="mlp", extra="--indim 1 --outdim 1 --degree 3",
         aflags="--width 50 --depth 4 --act tanh", outD=1, base_h=7.0, big=True,
         combos=[("mup", None, 0.02), ("mup", None, 0.05), ("mup", None, 0.1),
                 ("kaiming_normal", None, 0.05), ("default", "0.1", 0.05)]),
    dict(key="ksparse", ds="ksparse", arch="mlp", extra="--indim 10 --outdim 1 --set ksparse=3",
         aflags="--width 64 --depth 2 --act tanh", outD=1, base_h=5.0, big=False,
         combos=[("mup", None, 0.02), ("mup", None, 0.05), ("mup", None, 0.12),
                 ("kaiming_normal", None, 0.05), ("default", "0.1", 0.05)]),
    dict(key="saddle", ds="saddle", arch="mlp", extra="--indim 4 --outdim 4 --set saddlesep=0.4",
         aflags="--width 64 --depth 2 --act tanh", outD=4, base_h=9.0, big=True,
         combos=[("mup", None, 0.1), ("mup", None, 0.2), ("mup", None, 0.3),
                 ("kaiming_normal", None, 0.2), ("default", "0.05", 0.2)]),
]


def captures():
    out = []
    for d in GRID:
        M = NSAMP * d["outD"]
        for scheme, scale, lr in d["combos"]:
            itag = scheme if scale is None else f"default{scale}"
            iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
            name = f"{d['key']}_mse_{d['arch']}_n{NSAMP}_lr{lr}_{itag}_s0"
            cmd = (f"eos_prediction_multiclass.py --dataset {d['ds']} --loss mse --arch {d['arch']} "
                   f"--nsamp {NSAMP} --batch 0 {d['extra']} {d['aflags']} --lr {lr} {iflag} "
                   f"--steps {STEPS} --swpairs {SWPAIRS} --swsteps {SWSTEPS} --seed 0 --label {name} "
                   f"{allsec(M)} --out runs_captured/{name}.json")
            out.append(dict(name=name, cmd=cmd, hours=d["base_h"], big=d["big"]))
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
    ap.add_argument("--tag", default="b1000")
    ap.add_argument("--maxjobs", type=int, default=12)
    ap.add_argument("--jobhours", type=float, default=JOB_HOURS)
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True
    have = existing_labels(skip_tag=a.tag)
    caps = [c for c in captures() if c["name"] not in have]
    jobs = pack(caps, a.jobhours)
    total_h = sum(c["hours"] for c in caps)
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    print(f"FULL-BATCH GD, batch=nsamp={NSAMP} (server.py: batch=0 => bs=Nfull, deterministic)")
    print(f"GRID: {len(caps)} captures  ·  PACK: {len(jobs)} jobs  ·  est {total_h:.0f} GPU-h ({total_h/24:.1f} GPU-days)")
    print()
    submitted = 0
    for ji, (isbig, job) in enumerate(jobs):
        mf = f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        with open(mf, "w") as f:
            f.write(f"# {a.tag} job {ji:02d}: {len(job)} captures, est {sum(c['hours'] for c in job):.1f}h"
                    f"{' [A6000-48GB]' if isbig else ''}\n")
            for c in job:
                f.write(c["cmd"] + "\n")
        print(f"  job {ji:02d}  {len(job):2d} caps  est {sum(c['hours'] for c in job):4.1f}h"
              f"  {'A6000' if isbig else 'any  '}  [{', '.join(c['name'] for c in job)}]")
        if a.submit and ji < a.maxjobs:
            cmd = ["sbatch", f"--job-name={a.tag}_{ji:02d}", "--time=30:00:00"]
            if isbig:
                cmd += [f"--gres={A6000}", "--mem=64G"]
            cmd += [f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"]
            r = subprocess.run(cmd, cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip()); submitted += 1
    print()
    print(f"{'SUBMITTED '+str(submitted)+' jobs.' if a.submit else 'DRY-RUN — manifests written, submitted nothing.'}")


if __name__ == "__main__":
    main()
