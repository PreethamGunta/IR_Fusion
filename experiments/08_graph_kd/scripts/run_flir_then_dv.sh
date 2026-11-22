#!/usr/bin/env bash
# Reduced FLIR core matrix, then DroneVehicle.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXP" || exit 1
export ARMS="${FLIR_ARMS:-a0_gt a2_ir_control a1_pred_ir a3_node_edge_ir a5_fgw_ir}"
echo "[$(date)] FLIR core arms: $ARMS"
bash scripts/run_all.sh
echo "[$(date)] FLIR phase finished; starting DroneVehicle"
bash scripts/dronevehicle_all.sh
echo "[$(date)] FLIR + DroneVehicle chain finished"
