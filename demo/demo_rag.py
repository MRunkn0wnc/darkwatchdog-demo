from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from src.classifier_model import LeakClassifier

corpus = [

    "[OnionForumX | 2025-08-15] Selling fresh database dump with 5M user emails. BTC only.",
    "[DarkMarket | 2025-08-14] Full PayPal logs available, verified accounts. Contact seller.",
    "[HackersBay | 2025-08-13] New ransomware sample for testing. DM for details.",
    "[GovLeaks | 2025-08-12] Leaked government employee contact list posted yesterday.",
    "[CredHub | 2025-08-11] Fresh combo list: emails + passwords, updated weekly.",
    "[DataDumpers | 2025-08-10] Database breach from e-commerce site: 120k accounts.",
    "[CCMart | 2025-08-09] Selling stolen credit card dumps with CVV, worldwide.",
    "[BankFraudNet | 2025-08-08] Banking credentials leaked, high-balance accounts only.",
    "[ExploitForum | 2025-08-07] Dark web market update: new exploit kits available.",
    "[AnonVPN | 2025-08-06] Private VPN logs leaked — IP + timestamps.",
    "[RDPShop | 2025-08-05] Access to compromised RDP servers for sale, cheap rates.",
    "[HealthLeaks | 2025-08-04] Dump of healthcare records (patients, prescriptions).",
    "[CrackHub | 2025-08-03] New Telegram group sharing cracked software + keys.",
    "[CorpLeaks | 2025-08-02] Corporate email leaks — internal documents posted.",
    "[UniLeaks | 2025-08-01] Fresh dump: student records from university portal.",
    "[CryptoTalk | 2025-07-28] Anyone here into crypto mining tips?",
    "[SportsForum | 2025-07-27] Looking for football streaming links.",
    "[TechHelp | 2025-07-26] Does anyone know good Linux tutorials?",
    "[GameZone | 2025-07-25] Selling gaming accounts — Fortnite, Steam, cheap.",
    "[MoviesClub | 2025-07-24] New movie recommendations? Just watched Oppenheimer."
]

class Retriever:
    def __init__(self, docs):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = docs
        self.embeddings = self.model.encode(docs, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def query(self, q, top_k=2):
        q_emb = self.model.encode([q], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        return [(self.docs[i], float(D[0][j])) for j, i in enumerate(I[0])]
def main():
    print("=== DarkWatchdog Demo (BERT + RAG) ===")
    classifier = LeakClassifier()
    retriever = Retriever(corpus)

    while True:
        query = input("\nEnter a query (or type 'exit' to quit): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting DarkWatchdog. Goodbye!")
            break

        pred = classifier.predict(query)

        if pred["label"] == "LEAK":
            print("\n🚨 Potential leak detected! 🚨")
            results = retriever.query(query, top_k=2)
            print("\nTop retrieved posts:")
            for text, score in results:
                print(f" - {text}  (dist={score:.4f})")
            print(f"\nClassifier: {pred['label']} (score={pred['score']:.3f})")

        else:
            print("\n✅ Query classified as NON-LEAK (score={:.3f}).".format(pred["score"]))


if __name__ == "__main__":
    main()
