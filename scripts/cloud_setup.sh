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
uv pip install transformers tokenizers safetensors numpy tqdm

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
echo "  /workspace/babylm/models/tokenizer/"
echo "  /workspace/babylm/models/chck_*M/"
echo "  /workspace/babylm/logs/training.log"
echo ""
echo "Next: run deploy_to_vast.sh from your local machine"
