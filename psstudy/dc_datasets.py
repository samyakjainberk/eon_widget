import sys,json,torch
sys.path.insert(0,"/nas/ucb/samsj/TestingPSTheory/eos_widget"); import server,capture_run
cfg=json.loads(sys.argv[1])
dev=torch.device("cuda:0"); server.DTYPE=torch.float64; server.DEVICE=dev; server._TL.device=dev; server._TL.cifar_dir=None; torch.cuda.set_device(dev)
OFF={f"s{i}":"0" for i in range(1,43)}
B=dict(arch="mlp",initscheme="kaiming_normal",init="0.5",seed="0",steps="4",eigevery="1",bias="1",
       gw1="1",gw1n="15",gw1steps="2",gw2="1",gw3="1",gw3k="5",gw4="1",gw4K="4",gw4t0s="0,2",
       gw5="1",gw6="1",gw7="1",gw8="1",gw_n="3",gw_steps="40",gw_ev="20",gw_nte="32")
B.update(OFF); B["s19"]="1"; B.update(cfg)
pp=capture_run.default_params(); pp.update(B); P=server._parse_params({k:[str(v)] for k,v in pp.items()})
got={f"g_gw{i}":0 for i in [1,2,3,4,5,6,7,8]}; err=None
try:
    for msg in server.run_stream(P):
        if msg.get("type")=="error": err=msg;break
        if msg.get("type")=="step":
            for k in got:
                if msg.get(k) is not None: got[k]+=1
except Exception as e:
    import traceback; err=traceback.format_exc()[:300]
print(json.dumps({"tag":cfg.get("_tag"),"err":str(err)[:200] if err else None,"emit":got,"all8":all(v>0 for v in got.values())}))
