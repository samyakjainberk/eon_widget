#!/usr/bin/env python
"""Wave-3 MSE captures — the runs the 16 GB A4000 made impossible, plus depth on the thin datasets.

Waves 1-2 (gen_newds_sweep / gen_newds2_sweep) had to DEGRADE vgg to fit an A4000-16GB:
cifar10-vgg was capped at nsamp=25 (M=N*outD=250) and cifar2-vgg at chmul=0.125 (~145k params).
The A6000 nodes (dqn/gan) have 48 GB, so this wave lifts both ceilings:

  (A) cifar10 + vgg11, 10-class MSE at nsamp 50/100  (M=500/1000) -- the full-batch 10-class vgg.
  (B) cifar2 + vgg11 at chmul=0.25 (~600k params, 4x wave-1's net) -- the BIGGER network.
  (C) ksparse_gpt depth: only 10 caps existed (thinnest dataset) -- adds n250, seed=1, more inits.
  (D) saddle depth: adds n250 + seed=1 + inits.

(A)/(B) are memory-heavy => submitted with --gres=gpu:A6000:1 (48 GB). (C)/(D) are small => any GPU.
Peak memory for (A)/(B) is MEASURED by .cltmp/probe/probe_job.sh, not extrapolated -- pass the
surviving configs via --big-ok. Everything MSE, mup-primary, labels disjoint from waves 1-2.

LABEL COLLISION NOTE: chmul is NOT in the label, so a chmul=0.25 cifar2 run would collide with (and
overwrite) the existing chmul=0.125 captures. Hence (B) uses its own key `cifar2_vgg25`.

  python gen_newds3_sweep.py --dry-run
  python gen_newds3_sweep.py --submit --big-ok c10n50,c10n100,c2ch25
"""
import argparse, glob, os, re, subprocess

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
JOB_HOURS = 24.0
CIFAR = "--cifar-dir data/cifar-10-batches-py"
A6000 = "gpu:A6000:1"


def allsec(g3d):
    return (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={g3d} --set sw2cd=1 --set sw2e=1 "
            f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")


def swpairs_steps(nsamp):
    return (40, 80) if nsamp <= 100 else (30, 70)


def _cap(key, ds, arch, extra, aflags, outD, nsamp, lr, scheme, scale, steps, base_h, seed, big=False):
    itag = scheme if scale is None else f"default{scale}"
    iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
    name = f"{key}_mse_{arch}_n{nsamp}_lr{lr}_{itag}_s{seed}"
    swp, sws = swpairs_steps(nsamp)
    cmd = (f"eos_prediction_multiclass.py --dataset {ds} --loss mse --arch {arch} "
           f"--nsamp {nsamp} {extra} {aflags} --lr {lr} {iflag} --steps {steps} "
           f"--swpairs {swp} --swsteps {sws} --seed {seed} --label {name} {allsec(nsamp*outD)} "
           f"--out runs_captured/{name}.json")
    return dict(name=name, cmd=cmd, hours=base_h * (1.0 if nsamp <= 100 else 1.4), big=big)


# ---- (A)/(B) the memory-gated vgg runs — each keyed so --big-ok can enable only what the probe cleared
def big_caps(ok):
    out = []
    if "c10n50" in ok:      # cifar10 10-class vgg, M=500 (A4000 OOMed at 17.7GB; fits 48GB)
        for scheme, scale, lr in [("mup", None, 0.02), ("mup", None, 0.05), ("mup", None, 0.08),
                                  ("kaiming_normal", None, 0.02), ("kaiming_normal", None, 0.05),
                                  ("default", "0.1", 0.02), ("default", "0.1", 0.05)]:
            out.append(_cap("cifar10_vgg", "cifar10", "vgg11", f"--set chmul=0.125 {CIFAR}", "--act tanh",
                            10, 50, lr, scheme, scale, 400, 7.0, 0, big=True))
    if "c10n100" in ok:     # cifar10 10-class vgg, M=1000 — the full batch
        for scheme, scale, lr in [("mup", None, 0.02), ("mup", None, 0.05), ("mup", None, 0.08),
                                  ("kaiming_normal", None, 0.05), ("default", "0.1", 0.05)]:
            out.append(_cap("cifar10_vgg", "cifar10", "vgg11", f"--set chmul=0.125 {CIFAR}", "--act tanh",
                            10, 100, lr, scheme, scale, 400, 12.0, 0, big=True))
    if "c2ch25" in ok:      # cifar2 vgg at 4x params (~600k) — distinct key, see collision note
        for scheme, scale, lr in [("mup", None, 0.02), ("mup", None, 0.05), ("mup", None, 0.08),
                                  ("kaiming_normal", None, 0.05), ("default", "0.1", 0.05)]:
            out.append(_cap("cifar2_vgg25", "cifar2", "vgg11", f"--set chmul=0.25 {CIFAR}", "--act tanh",
                            1, 100, lr, scheme, scale, 400, 9.0, 0, big=True))
    return out


# ---- (C)/(D) depth on the two thinnest datasets (small nets, any GPU) -------------------------------
GPT_X = "--indim 10 --outdim 1 --set ksparse=3 --dmodel 32 --nlayer 2 --nhead 4"
SAD_X = "--indim 4 --outdim 4 --set saddlesep=0.4"
SAD_A = "--width 64 --depth 2 --act tanh"


def small_caps():
    out = []
    # (C) ksparse_gpt: n250 scale-out, seed=1 replicas, higher LR, two unseen inits
    for lr in [0.005, 0.01, 0.02]:
        out.append(_cap("ksparse_gpt", "ksparse", "gpt", GPT_X, "", 1, 250, lr, "mup", None, 500, 3.2, 0))
    for lr in [0.01, 0.02]:
        out.append(_cap("ksparse_gpt", "ksparse", "gpt", GPT_X, "", 1, 100, lr, "mup", None, 500, 3.2, 1))
    out.append(_cap("ksparse_gpt", "ksparse", "gpt", GPT_X, "", 1, 100, 0.05, "mup", None, 500, 3.2, 0))
    out.append(_cap("ksparse_gpt", "ksparse", "gpt", GPT_X, "", 1, 100, 0.01, "kaiming_uniform", None, 500, 3.2, 0))
    out.append(_cap("ksparse_gpt", "ksparse", "gpt", GPT_X, "", 1, 100, 0.01, "default", "0.02", 500, 3.2, 0))
    # (D) saddle: n250 scale-out, seed=1 at the LRs lacking a replica, two unseen inits
    for lr in [0.1, 0.2, 0.3]:
        out.append(_cap("saddle", "saddle", "mlp", SAD_X, SAD_A, 4, 250, lr, "mup", None, 800, 3.0, 0))
    for lr in [0.15, 0.3]:
        out.append(_cap("saddle", "saddle", "mlp", SAD_X, SAD_A, 4, 100, lr, "mup", None, 800, 3.0, 1))
    out.append(_cap("saddle", "saddle", "mlp", SAD_X, SAD_A, 4, 100, 0.2, "kaiming_uniform", None, 800, 3.0, 0))
    out.append(_cap("saddle", "saddle", "mlp", SAD_X, SAD_A, 4, 100, 0.15, "default", "0.05", 800, 3.0, 0))
    return out


def existing_labels(skip_tag=None):
    labs = {os.path.basename(p)[:-len(".min.json")] for p in glob.glob(f"{DIR}/runs_captured/*.min.json")}
    for mf in glob.glob(f"{DIR}/runs_captured/manifests/*.txt"):
        if skip_tag and os.path.basename(mf).startswith(skip_tag + "_"):
            continue                       # never dedup against OUR OWN prior manifests (self-poison)
        for line in open(mf):
            m = re.search(r"--label (\S+)", line)
            if m:
                labs.add(m.group(1))
    return labs


def pack(caps, jobhours):
    """Pack into jobs, keeping big(A6000) and small(any-GPU) captures in SEPARATE jobs so each job
    requests only the GPU it needs — mixing would force small caps to wait on a scarce A6000."""
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
    ap.add_argument("--tag", default="newds3")
    ap.add_argument("--maxjobs", type=int, default=12)
    ap.add_argument("--jobhours", type=float, default=JOB_HOURS)
    ap.add_argument("--big-ok", default="", help="comma list of probe-CLEARED big configs: c10n50,c10n100,c2ch25")
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True
    ok = {s.strip() for s in a.big_ok.split(",") if s.strip()}
    allcaps = big_caps(ok) + small_caps()
    have = existing_labels(skip_tag=a.tag)
    caps = [c for c in allcaps if c["name"] not in have]
    jobs = pack(caps, a.jobhours)
    total_h = sum(c["hours"] for c in caps)
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    print(f"BIG-OK: {sorted(ok) or '(none — probe cleared nothing / not passed)'}")
    print(f"GRID: {len(caps)} captures ({len(allcaps)-len(caps)} dropped as already captured/manifested)")
    print(f"PACK: {len(jobs)} jobs  ·  est {total_h:.0f} GPU-h ({total_h/24:.1f} GPU-days)")
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
                cmd += [f"--gres={A6000}", "--mem=64G"]      # 48GB card + headroom for the (M,p) block
            cmd += [f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"]
            r = subprocess.run(cmd, cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip()); submitted += 1
    print()
    print(f"{'SUBMITTED '+str(submitted)+' jobs.' if a.submit else 'DRY-RUN — manifests written, submitted nothing.'}")


if __name__ == "__main__":
    main()
