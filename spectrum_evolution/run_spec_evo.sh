#!/bin/bash
#SBATCH --job-name=spec_evo
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/nas/ucb/samsj/TestingPSTheory/eos_widget/spectrum_evolution/spec_evo-%j.out

set -u
PY=${PY:-/nas/ucb/samsj/conda_env/envs/samsenv/bin/python}
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget/spectrum_evolution
cd "$DIR"

# per-job scratch/caches (same isolation the capture jobs use)
SCRATCH="/nas/ucb/samsj/tmp/spec_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH"/{nv,triton,mpl}
export TMPDIR="$SCRATCH" CUDA_CACHE_PATH="$SCRATCH/nv" TRITON_CACHE_DIR="$SCRATCH/triton" MPLCONFIGDIR="$SCRATCH/mpl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host $(hostname)  gpu $CUDA_VISIBLE_DEVICES  $(date)"
# GPU fp32 (matches the widget's fidelity). The SLQ estimator is tested against the DENSE eigenvalues of the
# SAME fp32 operator, so the SLQ-vs-block comparison is exact regardless of the fp32 operator precision.
"$PY" spec_evo.py --device cuda:0 --steps 320 --every 16 --nprobe 12 --m 48 \
      --width 32 --depth 3 --nsamp 100 --lr 0.1 --degree 3
echo "exit $?  $(date)"
