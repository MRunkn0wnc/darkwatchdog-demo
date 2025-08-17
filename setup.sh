#!/usr/bin/env bash
set -e

# Remove any old virtual environment
rm -rf venv

# Create new venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install -U pip setuptools wheel

# Install CPU-only packages
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers faiss-cpu
pip install scikit-learn pandas  # optional for training

echo "✅ Environment ready. Activate with: source venv/bin/activate"
