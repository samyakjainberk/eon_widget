#!/usr/bin/env python
"""Create duplicateQchanging_<frozen label> = a COPY of the corresponding Qevolving-true (qinit=evolve)
capture. These are byte-identical to actually re-running (evolve is deterministic: same config, seed, mode),
so copying avoids burning capped GPU on redundant runs. Idempotent: copies only sources that exist and
targets that don't. Patches the internal 'label' field so the capture self-reports its duplicate name."""
import glob, os, re, json, shutil
DIR="/nas/ucb/samsj/TestingPSTheory/eos_widget/runs_captured"
frozen=set()
for mf in glob.glob(f"{DIR}/manifests/qfroz_*.txt"):
    for line in open(mf):
        m=re.search(r"--label (\S+)", line)
        if m and m.group(1).startswith("Qfrozen"): frozen.add(m.group(1))
made=skipped=waiting=0
for fl in sorted(frozen):
    src=re.sub(r"^Qfrozen-(random-[a-z]+|true)_", "Qevolving-true_", fl)
    dst="duplicateQchanging_"+fl
    if os.path.exists(f"{DIR}/{dst}.json"): skipped+=1; continue
    if not os.path.exists(f"{DIR}/{src}.json"): waiting+=1; continue
    try:
        _sd=json.load(open(f"{DIR}/{src}.json"))
        if _sd.get("partial") or _sd.get("record_counts",{}).get("done",0)<1: waiting+=1; continue  # source not COMPLETE yet
    except Exception: waiting+=1; continue
    for ext in (".json",".min.json"):
        s=f"{DIR}/{src}{ext}"; d=f"{DIR}/{dst}{ext}"
        if not os.path.exists(s): continue
        shutil.copy(s,d)
        try:
            j=json.load(open(d)); j["label"]=dst
            for r in j.get("records",[]):
                if r.get("type")=="meta" and "label" in r: r["label"]=dst
            json.dump(j, open(d,"w"))
        except Exception as e: print(f"  warn: label-patch {dst}{ext}: {e}")
    made+=1; print(f"  made {dst}")
print(f"DONE: created {made}, already-present {skipped}, waiting-on-evolve-source {waiting}")
