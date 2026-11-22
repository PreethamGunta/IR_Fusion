#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=train_all_translators
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --output=logs/train_all_translators_%j.out
#SBATCH --error=logs/train_all_translators_%j.err
#SBATCH --open-mode=append

cd $ROOT
source $HOME/.conda/etc/profile.d/conda.sh
conda activate gateir

echo "======================================================"
echo " train_all_translators.sbatch"
echo " started  $(date)"
echo " jobid=$SLURM_JOB_ID"
echo "======================================================"

export PY="python"
export REPO="$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ———————————————————————————————————————
# Step 0: Verify Heavy_Snow is fully downloaded
# ———————————————————————————————————————
if [ ! -d "$REPO/data/DSERT-RoLL/Heavy_Snow/rectified_THERMAL_L" ]; then
    n_files=$(find "$REPO/data/DSERT-RoLL/Heavy_Snow" -type f 2>/dev/null | wc -l)
    echo "WARNING: Heavy_Snow has $n_files files (expected 8827). Training snow models may fail."
fi

# ———————————————————————————————————————
# Step 1: CycleGAN (Methods 3) — fog, rain, snow
# ———————————————————————————————————————
echo ""
echo ">>> Step 1: CycleGAN training (Method 3)"
bash "$REPO/experiments/01_flir_weather_translation/method3_cyclegan/scripts/train_cyclegan.sh"
rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: CycleGAN training failed with exit code $rc"
    exit $rc
fi

# ———————————————————————————————————————
# Step 2: CycleDiff (Method 4) — fog, rain, snow
# ———————————————————————————————————————
echo ""
echo ">>> Step 2: CycleDiff training (Method 4)"
bash "$REPO/experiments/01_flir_weather_translation/method4_cyclediff/scripts/train_cyclediff.sh"
rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: CycleDiff training failed with exit code $rc"
    exit $rc
fi

echo ""
echo "======================================================"
echo " All translators trained."
echo " finished  $(date)"
echo "======================================================"
echo ""
echo "Check results:"
for cond in fog rain snow; do
    cg="$REPO/experiments/01_flir_weather_translation/method3_cyclegan/runs/cyclegan_$cond"
    cd="$REPO/experiments/01_flir_weather_translation/method4_cyclediff/results/train_cycle_C_disc_G_blocks_12_clear2${cond}"
    echo "  CycleGAN $cond: $([ -d "$cg" ] && echo "READY" || echo 'missing')"
    echo "  CycleDiff $cond: $([ -d "$cd" ] && echo "READY" || echo 'missing')"
done
