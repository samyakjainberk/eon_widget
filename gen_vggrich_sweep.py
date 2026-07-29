#!/usr/bin/env python
"""VGG MSE captures in the PROBE-VERIFIED RICH regime (2026-07-18).

Previous VGG sweeps were dominated by mup + low-LR = DEAD dynamics (loss 0.5->0.5, sharpness pinned
at ~1 while 2/lr=40 ⇒ every prediction panel flat = "didn't yield"). A regime probe
(.cltmp/probe/regime_vgg.py: short runs, loss+sharpness only) mapped exactly where VGG learns AND
reaches Edge of Stability. Rich cells (loss drops + sharpness→/overshoots 2/lr):
  kaiming_normal / kaiming_uniform at lr 0.1-0.5 on cifar2 (chmul 0.125 & 0.25) and cifar10 (n25).
  mup is DEAD except weakly at lr>=0.3; default only marginal. So this grid = kaiming-first, lr>=0.1,
  with a few default/mup high-lr CONTRAST cells (init-scale story) so the user sees dead-vs-rich too.
All MSE, full sections + sweep, 400 steps. Every cell here is verified to produce real dynamics.

  python gen_vggrich_sweep.py --dry-run | --submit
"""
import argparse, glob, os, re, subprocess
DIR="/nas/ucb/samsj/TestingPSTheory/eos_widget"; CIFAR="--cifar-dir data/cifar-10-batches-py"
A6000="gpu:A6000:1"; JOB_HOURS=27.0; STEPS=400

# (scheme, scale): the RICH inits first, then contrast cells
RICH_INITS=[("kaiming_normal",None),("kaiming_uniform",None)]
CONTRAST_INITS=[("default","0.5"),("default","0.1"),("mup",None)]   # tested only at high lr (contrast)
RICH_LRS=[0.1,0.15,0.2,0.3,0.5]
CONTRAST_LRS=[0.3,0.5]        # where even mup/default show *some* life (probe)

# key, dataset, chmul, nsamp, outD, base_h, needs_A6000
TIERS=[
  dict(key="cifar2_vgg",   ds="cifar2",  chmul="0.125", nsamp=100, outD=1,  base_h=6.0, big=False),   # scalar M=100 → fits A4000-16GB
  # cifar10 is 10-CLASS ⇒ M=N·outD=250 (2.5× cifar2). With the FULL prediction suite (Pred-3/4/6 + qspec + §12 +
  # sweep) the dense (M,p) block at p=145k OOMs a 16GB A4000 — the dense-Jacobian probe only measured ONE tensor,
  # not the whole suite. Route 10-class VGG to A6000-48GB. (n50→A6000, n100→A100; see gen_a100_vgg.)
  dict(key="cifar10_vgg",  ds="cifar10", chmul="0.125", nsamp=25,  outD=10, base_h=6.0, big=True),    # 10-class M=250 → A6000 (was A4000 → OOM)
  dict(key="cifar2_vgg25", ds="cifar2",  chmul="0.25",  nsamp=100, outD=1,  base_h=9.0, big=True),
]
def allsec(M): return (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={M} --set sw2cd=1 --set sw2e=1 "
    f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")
def swp(n): return (40,80) if n<=100 else (30,70)
def caps():
    out=[]
    for t in TIERS:
        M=t["nsamp"]*t["outD"]; a,b=swp(t["nsamp"])
        combos=[(s,sc,lr) for (s,sc) in RICH_INITS for lr in RICH_LRS] \
             + [(s,sc,lr) for (s,sc) in CONTRAST_INITS for lr in CONTRAST_LRS]
        for scheme,scale,lr in combos:
            itag=scheme if scale is None else f"default{scale}"
            iflag=f"--initscheme {scheme}"+("" if scale is None else f" --init {scale}")
            name=f"{t['key']}_mse_vgg11_n{t['nsamp']}_lr{lr}_{itag}_rich_s0"   # _rich tag ⇒ never collides with old dead captures
            cmd=(f"eos_prediction_multiclass.py --dataset {t['ds']} --loss mse --arch vgg11 "
                 f"--nsamp {t['nsamp']} --batch 0 --outdim {t['outD']} --act tanh --set chmul={t['chmul']} {CIFAR} "
                 f"--lr {lr} {iflag} --steps {STEPS} --swpairs {a} --swsteps {b} --seed 0 --label {name} "
                 f"{allsec(M)} --out runs_captured/{name}.json")
            out.append(dict(name=name,cmd=cmd,hours=t["base_h"],big=t["big"]))
    return out
def existing(skip):
    labs={os.path.basename(p)[:-len(".min.json")] for p in glob.glob(f"{DIR}/runs_captured/*.min.json")}
    for mf in glob.glob(f"{DIR}/runs_captured/manifests/*.txt"):
        if skip and os.path.basename(mf).startswith(skip+"_"): continue
        for line in open(mf):
            m=re.search(r"--label (\S+)",line)
            if m: labs.add(m.group(1))
    return labs
def pack(cs,jh):
    jobs=[]
    for big in (True,False):
        cur,h=[],0.0
        for c in [c for c in cs if c["big"]==big]:
            if cur and h+c["hours"]>jh: jobs.append((big,cur)); cur,h=[],0.0
            cur.append(c); h+=c["hours"]
        if cur: jobs.append((big,cur))
    return jobs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--submit",action="store_true")
    ap.add_argument("--tag",default="vggrich"); ap.add_argument("--jobhours",type=float,default=JOB_HOURS); a=ap.parse_args()
    if not(a.dry_run or a.submit): a.dry_run=True
    have=existing(a.tag); cs=[c for c in caps() if c["name"] not in have]; jobs=pack(cs,a.jobhours)
    th=sum(c["hours"] for c in cs)
    os.makedirs(f"{DIR}/runs_captured/manifests",exist_ok=True)
    print(f"VGG RICH regime: kaiming-first lr>=0.1 (+ contrast). {len(cs)} caps, {len(jobs)} jobs, ~{th:.0f} GPU-h")
    sub=0
    for ji,(big,job) in enumerate(jobs):
        mf=f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        open(mf,"w").write(f"# {a.tag} job {ji:02d}: {len(job)} caps{' [A6000]' if big else ' [A4000]'}\n"+"\n".join(c["cmd"] for c in job)+"\n")
        print(f"  job {ji:02d} {len(job):2d} caps {'A6000' if big else 'any  '}: {[c['name'].split('_mse_')[0]+'/'+c['name'].split('_lr')[1].split('_rich')[0] for c in job]}")
        if a.submit:
            cmd=["sbatch",f"--job-name={a.tag}_{ji:02d}","--time=30:00:00"]+(["--gres="+A6000,"--mem=64G"] if big else [])+[f"--export=ALL,MANIFEST={mf}","run_capture_pred_stack.sh"]
            r=subprocess.run(cmd,cwd=DIR,capture_output=True,text=True); print("     ->",(r.stdout or r.stderr).strip()); sub+=1
    print(f"\n{'SUBMITTED '+str(sub) if a.submit else 'DRY-RUN'}")
if __name__=="__main__": main()
