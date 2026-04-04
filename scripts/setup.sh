#!/bin/bash
# BabyLM 2026 Setup Script
# Run from /Users/adam/dev/babylm/

set -e

echo "=== BabyLM 2026 Setup ==="
echo ""

# Check dependencies
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q datasets huggingface_hub transformers torch
pip install -q sentencepiece spacy
python -m spacy download fr_core_news_sm 2>/dev/null || true

echo ""
echo "=== Downloading BabyLM 2026 Official Corpus ==="
python3 scripts/download_babylm_corpus.py

echo ""
echo "=== Cloning BabyLM Evaluation Pipeline ==="
if [ ! -d "eval/evaluation-pipeline" ]; then
    git clone https://github.com/babylm/evaluation-pipeline-2024 eval/evaluation-pipeline
    echo "NOTE: Update to 2025 pipeline when released (early April 2026)"
else
    echo "Eval pipeline already cloned."
fi

echo ""
echo "=== Setup Complete ==="
echo "Next: run scripts/inspect_corpus.py to see what French data exists"
echo "Next: run scripts/download_childes_french.py to get CDS data"
