# DarkWatchdog Demo

Small demo for a DarkWatchdog-style pipeline: rule-based classification + RAG-style retrieval using sentence-transformers + FAISS + Flan-T5-small generator.

> WARNING: This demo uses synthetic safe sample posts. Do NOT publish real leaked PII in public repos.

## Quickstart (Linux)

```bash
# clone after you push, or run locally pre-push
git clone https://github.com/MRunkn0wnc/darkwatchdog-demo.git
cd darkwatchdog-demo

# setup
./setup.sh

# run demo
./run_demo.sh
