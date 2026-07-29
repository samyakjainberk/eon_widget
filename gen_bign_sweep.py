#!/usr/bin/env python
"""Generate + submit BIG-BATCH comprehensive prediction captures (nsamp≥100) that populate EVERY
prediction-widget plot — for BOTH the prediction (MSE) and prediction_multiclass (CE) widgets.

Every capture turns ON all of the widget's sections and lifts the size guards so nothing is skipped
at large batch:
  --set qspec=1                       Q-eigenspectrum (full spectrum, MLP)
  --set grid3dcap=<N·outD>            predictions need N·outD ≤ grid3dcap (default 500 blocks nsamp≥100)
  --set sw2cd=1 --set sw2e=1          Prediction-2c/2d/2e scatter
  --set swMmax/ swMPmax (big)         raise the sweep size guard so Prediction-1&2 runs at big M·p
  --set p4thr=0.04 --set prthr=0.03   Prediction-4 & 4.2-ray anchors fire (at N≥100 the Jᵀr↔u₁ alignment
                                      maxes ~0.08, so the default 0.8 never anchors ⇒ empty plots)
(PRED_ON already enables Pred-3/4/5/6 + ray + early-dynamics + self-stab; BASE_ON enables §1/§5/§6/§7 + §9/§10.)

Grid = 4 dataset×loss (cifar10/mnist MSE → prediction widget · maxfind/modadd CE → multiclass widget),
all TANH MLP (smooth ⇒ self-stab + qspec meaningful), × 6 init-schemes × 3 lrs × {nsamp 100, 250}.
Captures are cost-packed into stacked SLURM jobs (~26h each, ≤30h limit) and submitted via
run_capture_pred_stack.sh.

  python gen_bign_sweep.py --dry-run     # print grid + packing, submit nothing
  python gen_bign_sweep.py --submit      # sbatch the jobs
  python gen_bign_sweep.py --dry-run --nsamps 100     # restrict nsamp
"""
import argparse, os, subprocess

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
PY = "/nas/ucb/samsj/conda_env/envs/samsenv/bin/python"   # (run_capture_pred_stack.sh sets its own PY; here only for reference)

# dataset, loss, dims,                       arch_flags,                         outD, has_sweep(MSE only)
DATASETS = [
    ("cifar10", "mse", "--indim 10 --outdim 10", "--width 32 --depth 2 --act tanh", 10, True),   # → prediction widget
    ("mnist",   "mse", "--indim 10 --outdim 10", "--width 32 --depth 2 --act tanh", 10, True),    # → prediction widget
    ("maxfind", "ce",  "--indim 10 --outdim 10", "--width 48 --depth 2 --act tanh", 10, False),   # → multiclass widget (no sweep: MSE-only theory)
    ("modadd",  "ce",  "--indim 22 --outdim 11", "--width 48 --depth 2 --act tanh", 11, False),   # → multiclass widget
]
INITS = [("mup", None), ("kaiming_normal", None), ("kaiming_uniform", None),
         ("default", 0.1), ("default", 0.5), ("default", 0.02)]
LRS = [0.02, 0.05, 0.12]
NSAMPS = [100, 250]
# nsamp=250 is heavier ⇒ run it only on a representative SUBSET (else the grid overflows 20 jobs); nsamp=100 covers the full grid.
N250_INITS = {"mup", "kaiming_normal", "default"}     # (default here = default@0.1 via the scale check below)
N250_LRS = {0.02, 0.05}
STEPS = 600
SEED = 0
JOB_HOURS = 24.0          # target GPU-hours per stacked job (SLURM limit 30h) — 6h headroom for est error

def swpairs_steps(nsamp):
    return (40, 80) if nsamp <= 100 else (30, 70)   # fewer sweep pairs at nsamp=250 (each mini-net is heavier)

def est_h(loss, nsamp):
    base = 3.2 if loss == "mse" else 2.7            # measured ~17s/step at N=100 (all sections); MSE also has the sweep
    return base * (1.0 if nsamp <= 100 else 1.4)    # nsamp=250 main-run + sweep ~1.4× heavier

def _do_250(scheme, scale):     # nsamp=250 init subset: mup, kaiming_normal, default@0.1
    return (scheme in N250_INITS) and (scale is None or scale == 0.1)

def captures(datasets=None, nsamps=None, inits=None, extra=False, full=False):
    # extra=False (default): full nsamp=100 grid + the nsamp=250 SUBSET (_do_250 × N250_LRS).
    # extra=True: ONLY the nsamp=250 COMPLEMENT (configs NOT in that subset).
    # full=True: ignore all subset gating (the ENTIRE grid for the given nsamps) — for --complement-of.
    out = []
    for ds, loss, dims, aflags, outD, has_sweep in DATASETS:
        if datasets and ds not in datasets:
            continue
        for nsamp in NSAMPS:
            if extra and nsamp != 250:               # extra mode = nsamp=250 complement only
                continue
            if nsamps and nsamp not in nsamps:
                continue
            g3d = nsamp * outD                       # predictions need N·outD ≤ grid3dcap
            swp, sws = swpairs_steps(nsamp)
            for scheme, scale in INITS:
                if inits and scheme not in inits:
                    continue
                itag = scheme if scale is None else f"default{scale}"
                iflag = f"--initscheme {scheme}" + ("" if scale is None else f" --init {scale}")
                for lr in LRS:
                    if nsamp == 250 and not full:
                        in_sub = _do_250(scheme, scale) and (lr in N250_LRS)   # the already-submitted subset
                        if (extra and in_sub) or (not extra and not in_sub):
                            continue
                    name = f"{ds}_{loss}_mlp_n{nsamp}_lr{lr}_{itag}_s{SEED}"
                    allsec = (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={g3d} --set sw2cd=1 --set sw2e=1 "
                              f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")
                    cmd = (f"eos_prediction_multiclass.py --dataset {ds} --loss {loss} --arch mlp "
                           f"--nsamp {nsamp} {dims} {aflags} --lr {lr} {iflag} --steps {STEPS} "
                           f"--swpairs {swp} --swsteps {sws} --seed {SEED} --label {name} {allsec} "
                           f"--out runs_captured/{name}.json")
                    out.append((name, cmd, est_h(loss, nsamp)))
    return out

def pack(caps, jobhours=JOB_HOURS):
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
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--nsamps", nargs="*", type=int, default=None)
    ap.add_argument("--inits", nargs="*", default=None)
    ap.add_argument("--rerun-missing", action="store_true", help="only (re)submit captures whose .min.json is absent")
    ap.add_argument("--tag", default="bign", help="manifest + job-name prefix")
    ap.add_argument("--maxjobs", type=int, default=20, help="cap submitted jobs (SLURM courtesy)")
    ap.add_argument("--jobhours", type=float, default=JOB_HOURS, help="target GPU-hours per stacked job (packing)")
    ap.add_argument("--extra", action="store_true", help="generate ONLY the nsamp=250 complement (configs not in the default subset) — for extra jobs; use with a distinct --tag")
    ap.add_argument("--complement-of", default=None, help="generate the FULL grid (for --nsamps) MINUS labels already present in <this-tag>_*.txt manifests — robust way to add non-overlapping extra jobs")
    a = ap.parse_args()
    if not (a.dry_run or a.submit):
        a.dry_run = True
    if a.complement_of:
        import glob, re
        existing = set()
        for mf in glob.glob(f"{DIR}/runs_captured/manifests/{a.complement_of}_*.txt"):
            for line in open(mf):
                m = re.search(r"--label (\S+)", line)
                if m:
                    existing.add(m.group(1))
        caps = [c for c in captures(a.datasets, a.nsamps, a.inits, full=True) if c[0] not in existing]
        print(f"[complement-of {a.complement_of}: excluded {len(existing)} already-manifested labels]")
    else:
        caps = captures(a.datasets, a.nsamps, a.inits, extra=a.extra)
    if a.rerun_missing:
        caps = [c for c in caps if not os.path.exists(f"{DIR}/runs_captured/{c[0]}.min.json")]
    jobs = pack(caps, a.jobhours)
    total_h = sum(c[2] for c in caps)
    os.makedirs(f"{DIR}/runs_captured/manifests", exist_ok=True)
    os.makedirs(f"{DIR}/runs_captured/logs", exist_ok=True)
    print(f"GRID: {len(caps)} captures  ·  {len(DATASETS)} dataset×loss × {len(INITS)} inits × {len(LRS)} lrs × {len(NSAMPS)} nsamp")
    print(f"PACK: {len(jobs)} jobs (~{JOB_HOURS:.0f}h target)  ·  est {total_h:.0f} GPU-h ({total_h/24:.1f} GPU-days)")
    if len(jobs) > a.maxjobs:
        print(f"  !! {len(jobs)} jobs > --maxjobs {a.maxjobs}; submitting only the first {a.maxjobs} (rest: re-run with --rerun-missing later)")
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
            r = subprocess.run(["sbatch", f"--job-name={a.tag}_{ji:02d}", f"--time=30:00:00",
                                f"--export=ALL,MANIFEST={mf}", "run_capture_pred_stack.sh"],
                               cwd=DIR, capture_output=True, text=True)
            print("     ->", (r.stdout or r.stderr).strip()); submitted += 1
    print()
    print(f"{'SUBMITTED '+str(submitted)+' jobs.' if a.submit else 'DRY-RUN — manifests written, submitted nothing.'}"
          f"  Watch: squeue -u samsj | grep {a.tag}")

if __name__ == "__main__":
    main()
