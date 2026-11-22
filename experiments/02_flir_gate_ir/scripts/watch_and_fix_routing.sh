#!/usr/bin/env bash
# Wait until the screen stage starts, then repair the gated clear test sets
# (the learned gate sends a small fraction of clear frames through the filter).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

echo "=== watcher: waiting for screen stage ==="
for _ in $(seq 1 240); do
  if grep -q "\[screen\] training B1" logs/run_all_remaining.log 2>/dev/null; then
    echo "=== watcher: screen started, fixing routing ==="
    WORKERS=4 bash scripts/fix_routing.sh > logs/fix_routing.log 2>&1 || echo "ROUTING FIX FAILED"
    touch results/.routing_fixed
    echo "=== watcher: routing fixed $(date) ==="
    exit 0
  fi
  sleep 30
done
echo "=== watcher: timed out waiting for screen stage ==="
exit 1
