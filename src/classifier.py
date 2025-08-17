# src/classifier.py
from transformers import pipeline

# toy wrapper that can be replaced with a fine-tuned BERT model.
# By default uses a simple rule-based fallback for the demo.

def simple_rule_classify(text: str) -> str:
    t = text.lower()
    if "dump" in t or "database" in t:
        return "Databases"
    if "combolist" in t or "combo" in t:
        return "Combolists"
    if "stealer" in t or "cookies" in t or "tokens" in t:
        return "Stealer-Logs"
    return "Other"

# Example integration with a Hugging Face model if you train one:
def load_model_classification(model_path: str = None):
    if model_path:
        clf = pipeline("text-classification", model=model_path, truncation=True)
        return clf
    return None
