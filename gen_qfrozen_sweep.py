#!/usr/bin/env python
"""FROZEN-Q sweep: chebyshev/parity/CIFAR-10 x {MLP, transformer, VGG} x Q-initialization distribution.

Every run here has Q FROZEN for the whole of training (verified: the random Q is drawn once at theta_0
and is byte-identical + theta-independent for the entire run). Labels lead with the frozen distribution
so they group together in the widget's load list:

    Qfrozen-random-gauss_...  Q FROZEN, RANDOM, iid Gaussian   (symmetrised, Frobenius-matched to true Q)
    Qfrozen-random-bern_...   Q FROZEN, RANDOM, iid Bernoulli +/-1
    Qfrozen-random-unif_...   Q FROZEN, RANDOM, iid Uniform(-1,1)
    Qfrozen-true_...          Q FROZEN at the TRUE function Hessian Q(theta_0)   (qinit=fix)
    Qevolving-true_...        BASELINE: TRUE Q, EVOLVING over training           (qinit=evolve)

FEASIBILITY (why not every config gets every distribution): the random modes materialise a dense
(M,p,p) tensor, gated by M*p^2 <= qrandcap. Measured:
    chebyshev MLP  p=544    M=100  -> 3.0e7  OK
    ksparse   MLP  p=432    M=100  -> 1.9e7  OK
    ksparse   GPT  p=25889  M=100  -> 6.7e10 (268 GB tensor)  IMPOSSIBLE
    cifar10   MLP  p=17568  M=1000 -> 3.1e11                  IMPOSSIBLE
    cifar10   VGG  p=145210 M=250  -> 5.3e12                  IMPOSSIBLE
So GPT/CIFAR-10 get Qfroz-true + Qevolve-true only (those need no dense tensor and work at any p).
Reaching random-Q on those needs a MATRIX-FREE random operator, not the dense one.

Configs are the REGIME-VERIFIED rich ones (loss drops + reaches EoS) so the predictions actually carry signal.
  python gen_qfrozen_sweep.py --dry-run | --submit
"""
import argparse, glob, os, re, subprocess
DIR="/nas/ucb/samsj/TestingPSTheory/eos_widget"; CIFAR="--cifar-dir data/cifar-10-batches-py"
A6000="gpu:A6000:1"
RAND=["gauss","bern","unif"]                    # dense random -> small-p only
ALWAYS=[("fix","Qfrozen-true"),("evolve","Qevolving-true")]   # work at ANY p

# key, dataset, arch, extra flags, arch flags, outD, nsamp, lr, init, steps, hours, big(A6000), random-ok
CONFIGS=[
 dict(key="cheb",     ds="chebyshev", arch="mlp", extra="--degree 3",          af="--width 16 --depth 3 --act tanh",
      indim=1,  outD=1,  nsamp=100, lr=0.1,  init="kaiming_normal", steps=400, h=2.0, big=False, rand=True),
 dict(key="parity",   ds="ksparse",   arch="mlp", extra="--set ksparse=3",     af="--width 16 --depth 2 --act tanh",
      indim=10, outD=1,  nsamp=100, lr=0.5,  init="kaiming_normal", steps=400, h=2.0, big=False, rand=True),
 dict(key="parityGPT",ds="ksparse",   arch="gpt", extra="--set ksparse=3 --dmodel 32 --nlayer 2 --nhead 4", af="",
      indim=10, outD=1,  nsamp=100, lr=0.01, init="kaiming_normal", steps=400, h=5.0, big=True,  rand=False),
 dict(key="cifar10",  ds="cifar10",   arch="mlp", extra=f"--outdim 10 {CIFAR}", af="--width 32 --depth 2 --act tanh",
      indim=None,outD=10, nsamp=100, lr=0.02, init="kaiming_normal", steps=400, h=6.0, big=True,  rand=False),
 dict(key="cifar10VGG",ds="cifar10",  arch="vgg11",extra=f"--outdim 10 --set chmul=0.125 {CIFAR}", af="--act tanh",
      indim=None,outD=10, nsamp=25,  lr=0.3,  init="kaiming_normal", steps=400, h=7.0, big=True,  rand=False),
]
def allsec(M): return (f"--set qspec=1 --set qspecevery=5 --set grid3dcap={M} --set sw2cd=1 --set sw2e=1 "
    f"--set swMmax=4000 --set swMPmax=1e11 --set p4thr=0.04 --set prthr=0.03")
def caps():
    out=[]
    for c in CONFIGS:
        M=c["nsamp"]*c["outD"]
        modes=[(d,f"Qfrozen-random-{d}") for d in RAND] if c["rand"] else []
        modes+=ALWAYS
        for qmode,tag in modes:
            name=f"{tag}_{c['key']}_{c['arch']}_n{c['nsamp']}_lr{c['lr']}_s0"
            qf=f"--set qinit={qmode}" + ("" if qmode=="evolve" else " --set qfixt=0 --set qseed=0 --set qprobes=16")
            ind=(f"--indim {c['indim']} --outdim {c['outD']} " if c["indim"] else "")
            cmd=(f"eos_prediction_multiclass.py --dataset {c['ds']} --loss mse --arch {c['arch']} "
                 f"--nsamp {c['nsamp']} --batch 0 {ind}{c['extra']} {c['af']} --lr {c['lr']} "
                 f"--initscheme {c['init']} --steps {c['steps']} --swpairs 40 --swsteps 80 --seed 0 "
                 f"--label {name} {allsec(M)} {qf} --out runs_captured/{name}.json")
            out.append(dict(name=name,cmd=cmd,hours=c["h"],big=c["big"]))
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
    ap.add_argument("--tag",default="qfroz"); a=ap.parse_args()
    if not(a.dry_run or a.submit): a.dry_run=True
    have=existing(a.tag); cs=[c for c in caps() if c["name"] not in have]
    os.makedirs(f"{DIR}/runs_captured/manifests",exist_ok=True)
    print(f"FROZEN-Q sweep: {len(cs)} captures")
    # one job per (config-size class) grouping ~2-3 caps
    big=[c for c in cs if c["big"]]; small=[c for c in cs if not c["big"]]
    jobs=[]; 
    for grp,isbig,per in ((small,False,3),(big,True,2)):
        for i in range(0,len(grp),per): jobs.append((isbig,grp[i:i+per]))
    sub=0
    for ji,(isbig,job) in enumerate(jobs):
        mf=f"{DIR}/runs_captured/manifests/{a.tag}_{ji:02d}.txt"
        open(mf,"w").write(f"# {a.tag} job {ji:02d}: {len(job)} frozen-Q captures{' [A6000]' if isbig else ' [A4000]'}\n"+"\n".join(c["cmd"] for c in job)+"\n")
        print(f"  job {ji:02d} {'A6000' if isbig else 'any  '}: {[c['name'] for c in job]}")
        if a.submit:
            cmd=["sbatch",f"--job-name={a.tag}_{ji:02d}","--time=24:00:00"]+(["--gres="+A6000,"--mem=64G"] if isbig else ["--gres=gpu:A4000:1","--mem=24G"])+[f"--export=ALL,MANIFEST={mf}","run_capture_pred_stack.sh"]
            r=subprocess.run(cmd,cwd=DIR,capture_output=True,text=True); print("     ->",(r.stdout or r.stderr).strip()); sub+=1
    print(f"\n{'SUBMITTED '+str(sub)+' jobs' if a.submit else 'DRY-RUN'}")
if __name__=="__main__": main()
