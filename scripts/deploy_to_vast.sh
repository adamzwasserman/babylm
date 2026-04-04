#!/bin/bash
# Deploy BabyLM training to Vast.ai
#
# Usage:
#   ./scripts/deploy_to_vast.sh <ssh_host> <ssh_port>
#   ./scripts/deploy_to_vast.sh root@<ip> <port>
#
# Example:
#   ./scripts/deploy_to_vast.sh root@142.170.89.112 31494
#
# Steps:
#   1. Run cloud_setup.sh on the remote machine first
#   2. Then run this script to upload data and start training

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ssh_host> <ssh_port>"
    echo "Example: $0 root@142.170.89.112 31494"
    exit 1
fi

MACHINE=$1
PORT=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SSH_CMD="ssh -o StrictHostKeyChecking=no -p $PORT $MACHINE"

echo "================================================"
echo "Deploying BabyLM to $MACHINE:$PORT"
echo "================================================"

# 1. Create directories
echo ""
echo "--- Creating remote directories ---"
$SSH_CMD "mkdir -p /workspace/babylm/corpus/final /workspace/babylm/models/tokenizer /workspace/babylm/logs"

# 2. Upload tokenizer (small, fast)
echo ""
echo "--- Uploading tokenizer ---"
rsync -avz --progress -e "ssh -p $PORT" \
    "$PROJECT_DIR/models/tokenizer/" \
    "$MACHINE:/workspace/babylm/models/tokenizer/"

# 3. Upload training corpus (large, ~600MB)
echo ""
echo "--- Uploading training corpus ---"
rsync -avz --progress -e "ssh -p $PORT" \
    "$PROJECT_DIR/corpus/final/train_french.txt" \
    "$MACHINE:/workspace/babylm/corpus/final/"

# 4. Upload training script
echo ""
echo "--- Uploading training script ---"
rsync -avz --progress -e "ssh -p $PORT" \
    "$SCRIPT_DIR/train.py" \
    "$MACHINE:/workspace/babylm/"

# 5. Upload cloud setup script
rsync -avz --progress -e "ssh -p $PORT" \
    "$SCRIPT_DIR/cloud_setup.sh" \
    "$MACHINE:/workspace/babylm/"

# 6. Run setup if needed
echo ""
echo "--- Running cloud setup ---"
$SSH_CMD "cd /workspace/babylm && bash cloud_setup.sh"

# 7. Start training
echo ""
echo "--- Starting training ---"
$SSH_CMD "cd /workspace/babylm && source .venv/bin/activate && nohup python train.py \
    --corpus /workspace/babylm/corpus/final/train_french.txt \
    --epochs 1 \
    --batch_size 32 \
    > /workspace/babylm/logs/training.log 2>&1 &"

echo ""
echo "================================================"
echo "Deployment complete!"
echo "================================================"
echo ""
echo "Monitor training:"
echo "  ssh -p $PORT $MACHINE 'tail -f /workspace/babylm/logs/training.log'"
echo ""
echo "Sync checkpoints back:"
echo "  rsync -avz --progress -e \"ssh -p $PORT\" $MACHINE:/workspace/babylm/models/ $PROJECT_DIR/models/"
echo ""
echo "Quick status:"
echo "  ssh -p $PORT $MACHINE 'tail -1 /workspace/babylm/logs/training.log'"
