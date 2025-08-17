# src/classifier_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

class LeakClassifier:
    def __init__(self, model_path="models/bert-leak-classifier"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run train.py first."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

        # label mapping
        id2label = self.model.config.id2label
        self.label_map = id2label if id2label else {0: "non-leak", 1: "leak"}

    def predict(self, text: str) -> dict:
        """
        Returns {"label": <predicted_class>, "score": <probability>}
        """
        out = self.pipe(text, truncation=True, max_length=256)
        return out[0]

if __name__ == "__main__":
    clf = LeakClassifier()
    print(clf.predict("This forum post contains leaked credentials"))
