#!/bin/bash
# Cloud setup script for Vast.ai - BabyLM 2026
# Run this after SSH'ing into your instance
set -e

echo "=== Setting up BabyLM 2026 Training ==="

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd /workspace
mkdir -p /workspace/babylm
cd /workspace/babylm

# Create venv and install deps
uv venv
source .venv/bin/activate

# Install PyTorch with CUDA support + HuggingFace
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers tokenizers safetensors numpy pandas tqdm wandb

# Create directory structure
mkdir -p /workspace/babylm/corpus/final
mkdir -p /workspace/babylm/models/tokenizer
mkdir -p /workspace/babylm/logs

# Check GPU
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'CUDA: {torch.version.cuda}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Directory layout:"
echo "  /workspace/babylm/corpus/final/train_french.txt"
echo "  /workspace/babylm/models/tokenizer/             (shared across seeds)"
echo "  /workspace/babylm/models/seed{S}/chck_*M/       (per-seed checkpoints)"
echo "  /workspace/babylm/logs/seed{S}.log              (per-seed log)"
echo ""
echo "For wandb logging, run 'wandb login' once on this host:"
echo "  wandb login    # stores the key in ~/.netrc; train.py picks it up"
echo "To run without wandb, pass --wandb_mode disabled to train.py."
echo ""
echo "Next:"
echo "  single seed (vast.ai):  run deploy_to_vast.sh from local"
echo "  multi-seed (local GPU): bash scripts/run_multi_seed.sh 1 2 3 4 5"
