from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
class LeakClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes a HuggingFace text classification pipeline.
        Default: DistilBERT (sentiment), but we remap labels to LEAK/NON-LEAK.
        """
        print(f"[+] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
    def predict(self, text: str) -> dict:
        """
        Returns {"label": "LEAK"/"NON-LEAK", "score": float}
        """
        out = self.pipe(text, truncation=True, max_length=256)[0]
        if out["label"].upper() == "POSITIVE":
            mapped_label = "LEAK"
        else:
            mapped_label = "NON-LEAK"

        return {"label": mapped_label, "score": out["score"]}
if __name__ == "__main__":
    clf = LeakClassifier()
    print(clf.predict("This forum post contains leaked credentials"))
    print(clf.predict("I love watching movies"))
