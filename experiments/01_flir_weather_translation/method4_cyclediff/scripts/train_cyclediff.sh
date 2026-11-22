#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# train_cyclediff.sh — Orchestrate all CycleDiff training for FLIR ADAS v2.
#
# CycleDiff has 3 stages per condition:
#   1. VAE training (clear + target domain)
#   2. LDM training (clear + target domain)
#   3. Cycle translator training
#
# Runs fog, rain, snow in sequence. Each stage checks for existing checkpoints
# and skips if complete. Resumable via checkpoint resume_milestone.

set -uo pipefail

REPO="$ROOT"
export PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HERE="$REPO/experiments/01_flir_weather_translation"
CDIFF="$HERE/method4_cyclediff"
CONFIGS="$CDIFF/configs"
RESULTS="$CDIFF/results"
STATE="$CDIFF/logs/state"
mkdir -p "$STATE" "$RESULTS"

# Each condition maps to its config directory
CONDITIONS=("fog" "rain" "snow")
if [ $# -gt 0 ]; then CONDITIONS=("$@"); fi

echo "================================================================"
echo " Method 4 CycleDiff — FLIR ADAS v2"
echo " started $(date)"
echo " python=$PY"
echo " conditions=${CONDITIONS[*]}"
echo "=============================================================="

for cond in "${CONDITIONS[@]}"; do
    echo ""
    echo "=============================================================="
    echo ">>> $cond  $(date)"
    echo "=============================================================="

    cfgdir="$CONFIGS/flir_clear2${cond}"
    cond_results="$RESULTS/train_cycle_C_disc_G_blocks_12_clear2${cond}"

    # Check if already done
    final_ckpt="$cond_results/final_model.pt"
    if [ -f "$final_ckpt" ]; then
        echo "  [$cond] CycleDiff training already complete (found $final_ckpt). Skipping."
        continue
    fi

    # Stage 1: VAE training (clear + target)
    for domain in clear "${cond}"; do
        vae_ckpt="$RESULTS/ae_kl_256x256_d4_${domain}/final_model.pt"
        if [ -f "$vae_ckpt" ]; then
            echo "  [$cond] VAE ${domain} already trained. Skipping."
        else
            echo "  [$cond] Training VAE ${domain}..."
            $PY "$CDIFF/train_vae.py" \
                --cfg "$cfgdir/${domain}_ae_kl_256x256_d4.yaml"
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "  [$cond] VAE ${domain} failed with exit code $rc"
                exit $rc
            fi
        fi
    done

    # Stage 2: LDM training (clear + target)
    for domain in clear "${cond}"; do
        ldm_ckpt="$RESULTS/ddm_const_uncond_unet_ldm_${domain}/final_model.pt"
        if [ -f "$ldm_ckpt" ]; then
            echo "  [$cond] LDM ${domain} already trained. Skipping."
        else
            echo "  [$cond] Training LDM ${domain}..."
            $PY "$CDIFF/train_uncond_ldm.py" \
                --cfg "$cfgdir/${domain}_ddm_const4_ldm_unet6_114_ode_2.yaml"
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "  [$cond] LDM ${domain} failed with exit code $rc"
                exit $rc
            fi
        fi
    done

    # Stage 3: Cycle translator
    if [ -f "$final_ckpt" ]; then
        echo "  [$cond] Cycle translator already trained. Skipping."
    else
        echo "  [$cond] Training Cycle translator..."
        # Train Cycle translator
        echo "  [$cond] Training Cycle translator..."
        $PY "$CDIFF/train_uncond_ldm_cycle.py" \
            --cfg "$cfgdir/translation_C_disc_timestep_ode_2.yaml"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  [$cond] Cycle translator failed with exit code $rc"
            exit $rc
        fi
    fi

    echo "  [$cond] done."

    # Create DONE marker for this condition
    mkdir -p "$cond_results"
    cat > "$cond_results/DONE" <<DONE_EOF
condition=$cond
completed=$(date -Iseconds)
checkpoint=$final_ckpt
DONE_EOF
    echo "  [$cond] DONE marker created at $cond_results/DONE"
done

echo ""
echo "=============================================================="
echo " finished $(date)"
for cond in "${CONDITIONS[@]}"; do
    cond_results="$RESULTS/train_cycle_C_disc_G_blocks_12_clear2${cond}"
    final_ckpt="$cond_results/final_model.pt"
    if [ -f "$final_ckpt" ]; then
        echo "  [DONE]    $cond  -> $final_ckpt"
    elif [ -d "$cond_results" ]; then
        echo "  [PARTIAL] $cond  -> $cond_results (resubmit to resume)"
    else
        echo "  [TODO]    $cond"
    fi
done
echo "=============================================================="
