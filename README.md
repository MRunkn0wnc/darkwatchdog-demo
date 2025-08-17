DarkWatchdog Demo

A RAG + BERT powered demo for detecting and analyzing potential data leaks on dark web forums.

#Features

Retrieval-Augmented Generation (RAG) for realistic query → post matching

BERT-based classifier (fine-tuned sentiment model adapted to leak vs non-leak)

Synthetic dark web–style forum posts dataset (safe + realistic)

Interactive CLI: type a query, get classification and relevant posts

Leak queries show retrieved posts with alert

Non-leak queries show simple classification

###Setup

Clone the repo:
```bash
git clone https://github.com/MRunkn0wnc/darkwatchdog-demo.git
cd darkwatchdog-demo
```

##Create a virtual environment:
```bash
sudo python3 -m venv venv
source venv/bin/activate
```
##Install dependencies:
```bash
pip install -r requirements.txt
```
Run Demo

Option 1 – Direct:
```bash
python -m demo.demo_rag
```
Option 2 – Using helper script:
```bash
./setup.sh # creates venv and installs dependencies
./run_demo.sh # runs the demo
```
###########=============Example Run==========###########

=== DarkWatchdog Demo (BERT + RAG) ===
Enter a query (or type 'exit' to quit): leaked paypal credentials

Potential leak detected!

Top retrieved posts:

[DarkMarket | 2025-08-14] Full PayPal logs available, verified accounts. Contact seller. (dist=0.6423)

[OnionForumX | 2025-08-15] Selling fresh database dump with 5M user emails. BTC only. (dist=0.7432)

Classifier: LEAK (score=0.997)

Enter a query (or type 'exit' to quit): linux tutorials

Query classified as NON-LEAK (score=0.985).

#############===============Project Structure==============#################

darkwatchdog-demo/
├── demo/
│ ├── demo_data.py
│ ├── demo_rag.py # Main interactive demo
├── src/
│ ├── classifier_model.py # BERT classifier
│ ├── utils.py
├── requirements.txt
├── setup.sh # auto-setup venv and install deps
├── run_demo.sh # run demo with venv
└── README.md

Disclaimer--------------------------------------------------------------------

This project uses synthetic dark web–style data for demonstration purposes.
It does not connect to or scrape any real dark web sources.
