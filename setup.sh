#!/usr/bin/env bash
set -e
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers faiss-cpu
pip install scikit-learn pandas 
echo "✅ Environment ready. Activate with: source venv/bin/activate"
