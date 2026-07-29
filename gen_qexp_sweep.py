#!/usr/bin/env python
"""Q-INIT EXPERIMENT: for PROBE-VERIFIED rich small-p configs, run all 5 Q-modes (evolve / fix@0 /
gauss / bern / unif) at the SAME (init,lr,nsamp) so the prediction trackers (Pred-3/4/5.1/6) can be
compared across the nature of the function Hessian Q. Answers "how important is Q's true structure to
the predictive theory of learning?" (2026-07-18, uses the qinit toggle in server.py).

Configs are chosen from regime_small.py to be BOTH (a) dynamically RICH — loss→~0, sharpness reaches
EoS, so there ARE dynamics to predict — AND (b) small-p so the DENSE random Q (M,p,p) fits qrandcap=2e8:
  chebyshev w16d3 p544 kaiming_normal lr0.1 n100  (rel0.02, srr1.32; M=100, M·p²=3.0e7 ✓)
  ksparse   w16d2 p432 kaiming_normal lr0.5 n100  (rel0.00, srr1.14; M=100, M·p²=1.9e7 ✓)
  saddle    w16d3 p640 kaiming_normal lr0.3 n64   (rel0.01, srr1.35; M=256, M·p²=1.0e8 ✓)
The ACTUAL GD trajectory is identical across modes (qinit only changes Q INSIDE the predictions, not the
gradient step) ⇒ the comparison isolates how well the theory predicts when Q is true vs frozen vs random.
qprobes=16 for a tight per-sample Frobenius match. Full sections + sweep, 400 steps.

  python gen_qexp_sweep.py --dry-run | --submit
"""
import argparse, glob, os, re, subprocess
DIR="/nas/ucb/samsj/TestingPSTheory/eos_widget"; STEPS=400; QPROBES=16
# key, dataset, extra, arch_flags, indim, outD, nsamp, lr, init, base_h
CONFIGS=[
  dict(key="cheb_q",   ds="chebyshev", extra="--degree 3",          af="--width 16 --depth 3 --act tanh", indim=1, outD=1, nsamp=100, lr=0.1, init="kaiming_normal", h=1.5),
  dict(key="ksparse_q",ds="ksparse",   extra="--set ksparse=3",     af="--width 16 --depth 2 --act tanh", indim=10,outD=1, nsamp=100, lr=0.5, init="kaiming_normal", h=1.5),
  dict(key="saddle_q", ds="saddle",    extra="--set saddlesep=0.4", af="--width 16 --depth 3 --act tanh", indim=4, outD=4, nsamp=64,  lr=0.3, init="kaiming_normal", h=1.5),
]
QMODES=["evolve","fix","gauss","bern","unif"]
def allsec(M): return (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={M} --set sw2cd=1 --set sw2e=1 "
    f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")
def caps():
    out=[]
    for c in CONFIGS:
        M=c["nsamp"]*c["outD"]; a,b=(40,80)
        for qm in QMODES:
            qflags=f"--set qinit={qm}"+("" if qm=="evolve" else f" --set qfixt=0 --set qseed=0 --set qprobes={QPROBES}")
            name=f"{c['key']}_mse_mlp_n{c['nsamp']}_lr{c['lr']}_{c['init']}_Q{qm}_s0"
            cmd=(f"eos_prediction_multiclass.py --dataset {c['ds']} --loss mse --arch mlp "
                 f"--nsamp {c['nsamp']} --batch 0 --indim {c['indim']} --outdim {c['outD']} {c['extra']} {c['af']} "
                 f"--lr {c['lr']} --initscheme {c['init']} --steps {STEPS} --swpairs {a} --swsteps {b} "
                 f"--seed 0 --label {name} {allsec(M)} {qflags} --out runs_captured/{name}.json")
            out.append(dict(name=name,cmd=cmd,hours=c["h"]))
    return out
def existing(skip):
    labs={os.path.basename(p)[:-len(".min.json")] for p in glob.glob(f"{DIR}/runs_captured/*.min.json")}
    for mf in glob.glob(f"{DIR}/runs_captured/manifests/*.txt"):
        if skip and os.path.basename(mf).startswith(skip+"_"): continue
        for line in open(mf):
            m=re.search(r"--label (\S+)",line)
            if m: labs.add(m.group(1))
    return labs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--submit",action="store_true")
    ap.add_argument("--tag",default="qexp"); a=ap.parse_args()
    if not(a.dry_run or a.submit): a.dry_run=True
    have=existing(a.tag); cs=[c for c in caps() if c["name"] not in have]
    os.makedirs(f"{DIR}/runs_captured/manifests",exist_ok=True)
    # pack all into few small jobs (tiny nets ⇒ cheap; group by config so modes land together)
    print(f"Q-EXPERIMENT: {len(cs)} caps ({len(CONFIGS)} rich configs × {len(QMODES)} Q-modes). Tiny nets ⇒ A4000.")
    # one job per config (5 modes each, ~7.5h)
    byc={}
    for c in cs: byc.setdefault(c["name"].split("_mse_")[0],[]).append(c)
    sub=0
    for ji,(k,job) in enumerate(sorted(byc.items())):
        mf=f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        open(mf,"w").write(f"# {a.tag} job {ji:02d}: {k} × {len(job)} Q-modes\n"+"\n".join(c["cmd"] for c in job)+"\n")
        print(f"  job {ji:02d} {k}: {[c['name'].split('_Q')[1].split('_')[0] for c in job]}")
        if a.submit:
            cmd=["sbatch",f"--job-name={a.tag}_{ji:02d}","--time=20:00:00","--gres=gpu:A4000:1","--mem=24G",f"--export=ALL,MANIFEST={mf}","run_capture_pred_stack.sh"]
            r=subprocess.run(cmd,cwd=DIR,capture_output=True,text=True); print("     ->",(r.stdout or r.stderr).strip()); sub+=1
    print(f"\n{'SUBMITTED '+str(sub) if a.submit else 'DRY-RUN'}")
if __name__=="__main__": main()
