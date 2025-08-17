DarkWatchdog

DarkWatchdog is RAG + BERT powered for detecting and analyzing potential data leaks on dark web forums.

Features

- Retrieval-Augmented Generation (RAG) for realistic query → post matching
- BERT-based classifier (fine-tuned for leak vs non-leak detection)
- Synthetic dark web–style dataset (safe and realistic)
- Interactive CLI: type a query to get classification and relevant posts. Leak queries show retrieved posts with alert. Non-leak queries show simple classification.

Setup

1. Clone the repo:
 ```bash
git clone https://github.com/MRunkn0wnc/darkwatchdog-demo.git
cd darkwatchdog-demo
```

2. CPU-Only Environment Setup
For non-GPU machines, run the setup script:
```bash
./setup.sh
```
This will remove any old virtual environment, create a new venv, install CPU-only packages (torch, sentence-transformers, faiss-cpu, transformers) and optional packages scikit-learn and pandas for training scripts.
3.Activate the environment:
```bash
source venv/bin/activate
```
Verify installation:
```
python -c "import torch, transformers, sentence_transformers, faiss; print('OK')"
```
4. Run the demo:
```bash
./run_demo.sh
```
or directly from Python module syntax:
```bash
python -m demo.demo_rag
```
Always run from the project root to ensure src imports work correctly.



Disclaimer

This project uses synthetic dark web–style data for demonstration purposes. 

