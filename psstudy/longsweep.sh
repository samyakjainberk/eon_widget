#!/bin/bash
# long standard-init runs to test cycle persistence vs lr over a long horizon
GPU=$1; shift; LRS="$@"
S=/nas/ucb/samsj/tmp/spec_login
export CUDA_VISIBLE_DEVICES=$GPU TMPDIR="$S" CUDA_CACHE_PATH="$S/nv" MPLCONFIGDIR="$S/mpl" PYTHONDONTWRITEBYTECODE=1
PY=/nas/ucb/samsj/conda_env/envs/samsenv/bin/python
cd /nas/ucb/samsj/TestingPSTheory/eos_widget
DROP="--set s4=0 --set s5=0 --set s12=0 --set s14=0 --set s16=0 --set s17=0 --set s35=0 --set s37=0 --set s38=0 --set s39=0 --set s42=0 --set ss=0"
for LR in $LRS; do
  echo "[$(date +%H:%M)] GPU$GPU cheb lr=$LR start"
  $PY eos_prediction.py --dataset chebyshev --nsamp 100 --indim 1 --outdim 1 \
    --width 16 --depth 3 --act tanh --lr $LR --initscheme kaiming_normal --steps 1500 --seed 0 \
    --no-sweep $DROP --label longcheb_lr${LR} \
    --out psstudy/longcheb_lr${LR}.json --device cuda:0 >> psstudy/long_gpu${GPU}.log 2>&1
  echo "[$(date +%H:%M)] GPU$GPU cheb lr=$LR done -> $(ls -la psstudy/longcheb_lr${LR}.json 2>/dev/null | awk '{print $5}')"
done
echo "[$(date +%H:%M)] GPU$GPU ALL DONE"
