#!/bin/bash
# Continuous checkpoint sync from Vast.ai back to local
#
# Usage:
#   ./scripts/sync_checkpoints.sh <ssh_host> <ssh_port> [interval_seconds]
#
# Example:
#   ./scripts/sync_checkpoints.sh root@142.170.89.112 31494 300

set -e

MACHINE=$1
PORT=$2
INTERVAL=${3:-300}  # default 5 minutes
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing checkpoints from $MACHINE:$PORT every ${INTERVAL}s"
echo "Local destination: $PROJECT_DIR/models/"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    echo "[$(date '+%H:%M:%S')] Syncing..."
    rsync -avz --progress -e "ssh -p $PORT" \
        "$MACHINE:/workspace/babylm/models/" \
        "$PROJECT_DIR/models/" 2>/dev/null || echo "  sync failed, retrying next interval"

    # Also grab the log
    rsync -avz -e "ssh -p $PORT" \
        "$MACHINE:/workspace/babylm/logs/training.log" \
        "$PROJECT_DIR/logs/" 2>/dev/null || true

    # Show latest training status
    if [ -f "$PROJECT_DIR/logs/training.log" ]; then
        echo "  Latest: $(tail -1 "$PROJECT_DIR/logs/training.log")"
    fi

    echo "  Next sync in ${INTERVAL}s"
    sleep "$INTERVAL"
done
