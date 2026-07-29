#!/usr/bin/env python
"""Audit every landed capture for COMPLETENESS — catches the failure modes that are silent in the widget.

A capture can exist on disk and still be hollow. Checked here:
  PARTIAL   top-level `partial` is True   => the run was cut off (walltime/crash mid-capture)
  NODONE    no `done` record              => same, capture never finalized
  SWEEP0    sweeppt == 0 (MSE runs)       => server.py's swMmax/swMPmax guard tripped and run_sweep
                                             yielded a meta `error` instead of points. The stack script
                                             CONTINUES past it, so this fails QUIETLY.
  SWEEPLOW  sweeppt < the manifest's --swpairs
  STEPS     step records != --steps + 1
  NOPRED    g_pred3/g_pred4/g_trace/g_pred6 all absent => the dense-Jacobian block was skipped
                                             (N*outD > grid3dcap)
  METAERR   meta record carries an `error`
  BADJSON   unreadable / truncated file

Expected values come from each capture's OWN manifest line (--steps / --swpairs), not a guess.

  python audit_captures.py            # audit everything
  python audit_captures.py --tag b1000  # only labels from manifests matching a tag
"""
import argparse, glob, json, os, re, sys

DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
PRED_KEYS = ("g_pred3", "g_pred3m", "g_pred4", "g_pred4m", "g_trace", "g_pred6")


def manifest_expectations(tag=None):
    """label -> {steps, swpairs, loss, manifest} parsed from the manifest command lines."""
    exp = {}
    for mf in sorted(glob.glob(f"{DIR}/runs_captured/manifests/*.txt")):
        base = os.path.basename(mf)[:-4]
        if tag and not base.startswith(tag):
            continue
        for line in open(mf):
            if line.startswith("#") or "--label" not in line:
                continue
            g = lambda pat, d=None: (re.search(pat, line).group(1) if re.search(pat, line) else d)
            lab = g(r"--label (\S+)")
            exp[lab] = dict(steps=int(g(r"--steps (\d+)", "0")),
                            swpairs=int(g(r"--swpairs (\d+)", "0")),
                            loss=g(r"--loss (\S+)", "mse"),
                            manifest=base)
    return exp


def audit_one(path, e):
    try:
        d = json.load(open(path))
    except Exception as ex:
        return ["BADJSON:" + type(ex).__name__], {}
    rc = d.get("record_counts") or {}
    if not rc:
        rc = {}
        for r in d.get("records", []):
            rc[r.get("type")] = rc.get(r.get("type"), 0) + 1
    problems = []
    if d.get("partial"):
        problems.append("PARTIAL")
    if rc.get("done", 0) < 1:
        problems.append("NODONE")
    steps = rc.get("step", 0)
    if e and e["steps"] and steps != e["steps"] + 1:
        problems.append(f"STEPS({steps}vs{e['steps']+1})")
    sw = rc.get("sweeppt", 0)
    if e and e["loss"] == "mse":
        if sw == 0:
            problems.append("SWEEP0")
        elif e["swpairs"] and sw < e["swpairs"]:
            problems.append(f"SWEEPLOW({sw}vs{e['swpairs']})")
    # prediction keys: scan records once
    npred = 0
    for r in d.get("records", []):
        if any(r.get(k) is not None for k in PRED_KEYS):
            npred += 1
            break
    if npred == 0:
        problems.append("NOPRED")
    for r in d.get("records", []):
        if r.get("type") == "meta" and r.get("error"):
            problems.append("METAERR:" + str(r["error"])[:60])
            break
    return problems, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    exp = manifest_expectations(a.tag)
    landed = {os.path.basename(p)[:-5]: p for p in glob.glob(f"{DIR}/runs_captured/*.json")
              if not p.endswith(".min.json")}
    labels = sorted(exp) if a.tag else sorted(set(exp) | set(landed))
    ok, bad, missing = [], [], []
    for lab in labels:
        if lab not in landed:
            if lab in exp:
                missing.append(lab)
            continue
        problems, rc = audit_one(landed[lab], exp.get(lab))
        (bad if problems else ok).append((lab, problems, rc))
    print(f"AUDIT: {len(ok)} clean · {len(bad)} degraded · {len(missing)} planned-but-absent")
    if bad:
        print("\n--- DEGRADED ---")
        for lab, pr, rc in bad:
            print(f"  {lab}\n      {', '.join(pr)}  | counts={ {k:v for k,v in rc.items() if k in ('step','sweeppt','done','meta')} }")
    if missing:
        print(f"\n--- PLANNED BUT ABSENT ({len(missing)}) — still running, or died ---")
        for lab in missing[:40]:
            print("  ", lab, " [", exp[lab]["manifest"], "]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
