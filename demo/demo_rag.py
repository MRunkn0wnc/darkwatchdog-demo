# demo/demo_rag.py
from src.classifier_model import LeakClassifier
from src.retriever import Retriever

def main():
    print("=== DarkWatchdog Demo (BERT + RAG) ===")

    # 1. Load classifier
    classifier = LeakClassifier()

    # 2. Build retriever index (small fake corpus for demo)
    corpus = [
        "Forum post about leaked database dump",
        "Random conversation about movies",
        "Credentials found in public pastebin",
        "Cybercriminal offering ransomware sample"
    ]
    retriever = Retriever()
    retriever.build_index(corpus)

    # 3. User query
    query = "leaked credentials database"
    print(f"\nQuery: {query}")

    # 4. Retrieve top docs
    results = retriever.query(query, top_k=2)
    print("\nTop retrieved posts:")
    for text, score in results:
        print(f" - {text}  (dist={score:.4f})")

    # 5. Classify the query
    pred = classifier.predict(query)
    print(f"\nClassifier: {pred['label']}  (score={pred['score']:.3f})")

if __name__ == "__main__":
    main()
