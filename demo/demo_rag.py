from src.classifier_model import LeakClassifier
from src.retriever import Retriever

def main():
    print("=== DarkWatchdog Demo (BERT + RAG) ===")
    classifier = LeakClassifier()
    corpus = [
    "Selling fresh database dump with 5M user emails (2025). BTC only.",
    "Full PayPal logs available, verified accounts. Contact seller.",
    "New ransomware sample for testing. DM for details.",
    "Leaked government employee contact list posted yesterday.",
    "Fresh combo list: emails + passwords, updated weekly.",
    "Database breach from e-commerce site: 120k accounts.",
    "Selling stolen credit card dumps with CVV, worldwide.",
    "Banking credentials leaked, high-balance accounts only.",
    "Dark web market update: new exploit kits available.",
    "Private VPN logs leaked — IP + timestamps.",
    "Access to compromised RDP servers for sale, cheap rates.",
    "Dump of healthcare records (patients, prescriptions).",
    "New Telegram group sharing cracked software + keys.",
    "Corporate email leaks — internal documents posted.",
    "Fresh dump: student records from university portal.",
    "Anyone here into crypto mining tips?",
    "Looking for football streaming links.",
    "Does anyone know good Linux tutorials?",
    "Selling gaming accounts — Fortnite, Steam, cheap.",
    "New movie recommendations? Just watched Oppenheimer."
]
    retriever = Retriever()
    retriever.build_index(corpus)

    while True:
        query = input("\nEnter a query (or type 'exit' to quit): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting DarkWatchdog . Goodbye!")
            break

        results = retriever.query(query, top_k=2)
        print("\nTop retrieved posts:")
        for text, score in results:
            print(f" - {text}  (dist={score:.4f})")

        pred = classifier.predict(query)
        print(f"\nClassifier: {pred['label']}  (score={pred['score']:.3f})")

if __name__ == "__main__":
    main()
