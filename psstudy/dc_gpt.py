import sys,json,time,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
B=dict(dataset="maxfind",arch="gpt",loss="ce",dmodel="32",nhead="2",nlayer="2",seqlen="10",nsamp="20",indim="10",outdim="10",
       lr="0.02",initscheme="kaiming_normal",init="0.3",seed="0",steps="2",eigevery="1",bias="1",
       gw5="1",gw6="1",gw7="1",gw8="1",gw9="1",gw_n="3",gw_steps="12",gw_ev="6",gw_nsamp="24",gw_nte="24",gw9K="4",gw9refresh="6",gw9try="4")
B.update(OFF); B["s19"]="1"
pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
got={f"g_gw{i}":0 for i in range(5,10)}; err=None; t0=time.time()
try:
    for msg in server.run_stream(P):
        if msg.get("type")=="error": err=msg;break
        if msg.get("type")=="step":
            for k in got:
                if msg.get(k) is not None: got[k]+=1
except Exception as e:
    import traceback; err=traceback.format_exc()[:300]
print(json.dumps({"arch":"gpt-transformer","err":str(err)[:200] if err else None,"emit":got,
                  "gw5679_ok":all(got[k]>0 for k in ["g_gw5","g_gw6","g_gw7","g_gw9"]),
                  "gw8_skipped(MLP-only)":got["g_gw8"]==0,"secs":round(time.time()-t0,1)}))
