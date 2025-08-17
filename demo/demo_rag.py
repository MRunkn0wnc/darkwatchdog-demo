# demo/demo_rag.py
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from demo_data import posts
from src.classifier import simple_rule_classify
from src.utils import format_post_text

def build_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embedder, embs

def build_faiss_index(embs):
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embs)
    return index

def load_generator(model_name="google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return gen

def main():
    print("Preparing demo texts...")
    texts = [format_post_text(p) for p in posts]

    print("Building embeddings (sentence-transformers)...")
    embedder, embs = build_embeddings(texts)

    print("Creating FAISS index...")
    index = build_faiss_index(embs)

    print("Loading small generator model (flan-t5-small). May take time...")
    gen = load_generator()

    # Demo query
    query = "What companies and sensitive types can be found?"
    print("\n=== DEMO QUERY ===")
    print("Query:", query)

    # embed query
    q_emb = embedder.encode([query], convert_to_numpy=True)
    k = 2
    D, I = index.search(q_emb, k)
    retrieved = [texts[i] for i in I[0]]

    print("\nRetrieved fragments:")
    for r in retrieved:
        print("-", r)

    # Construct prompt RAG-style
    prompt = "Using the following fragments, answer succinctly:\n\n"
    for i, r in enumerate(retrieved):
        prompt += f"Fragment {i+1}: {r}\n\n"
    prompt += f"Question: {query}\nAnswer:"

    print("\nCalling generator for RAG answer (this may take a few seconds)...")
    out = gen(prompt, max_length=200, do_sample=False)[0]['generated_text']
    print("\nRAG answer:\n", out)

    print("\n--- Classifications (demo simple rule-based) ---")
    for p in posts:
        label = simple_rule_classify(format_post_text(p))
        print(p["id"], "->", label)

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nDemo completed in {time.time()-start:.1f}s")
