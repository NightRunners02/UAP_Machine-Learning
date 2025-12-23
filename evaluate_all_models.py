import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BertTokenizerFast,
    BertForSequenceClassification
)

# =========================
# LOAD DATASET
# =========================
labels, texts = [], []

with open("data/spam.csv", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split(";", 1)
        if len(parts) == 2 and parts[0] in ["ham", "spam"]:
            labels.append(0 if parts[0] == "ham" else 1)
            texts.append(parts[1])

df = pd.DataFrame({"text": texts, "label": labels})

X = df["text"]
y = df["label"]

# =========================
# LOAD MODELS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NN Base
nn_model = joblib.load("model_nn_base/model_text_nn.pkl")
vectorizer = joblib.load("model_nn_base/vectorizer.pkl")

# DistilBERT
distil_tok = DistilBertTokenizerFast.from_pretrained("model_distilbert")
distil_model = DistilBertForSequenceClassification.from_pretrained("model_distilbert").to(device)
distil_model.eval()

# BERT
bert_tok = BertTokenizerFast.from_pretrained("model_bert")
bert_model = BertForSequenceClassification.from_pretrained("model_bert").to(device)
bert_model.eval()

# =========================
# PREDICTION FUNCTIONS
# =========================
def predict_nn_base(texts):
    X_vec = vectorizer.transform(texts)
    return nn_model.predict(X_vec)

def predict_distilbert(texts):
    preds = []
    for t in texts:
        enc = distil_tok(t, return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            preds.append(torch.argmax(distil_model(**enc).logits, dim=1).item())
    return preds

def predict_bert(texts):
    preds = []
    for t in texts:
        enc = bert_tok(t, return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            preds.append(torch.argmax(bert_model(**enc).logits, dim=1).item())
    return preds

# =========================
# RUN EVALUATION
# =========================
results = {}

results["NN Base"] = predict_nn_base(X)
results["DistilBERT"] = predict_distilbert(X)
results["BERT"] = predict_bert(X)

# =========================
# METRIC COMPARISON
# =========================
accuracy_scores = {}

for model_name, y_pred in results.items():
    acc = accuracy_score(y, y_pred)
    accuracy_scores[model_name] = acc

    print(f"\n===== {model_name} =====")
    print("Accuracy:", acc)
    print(classification_report(y, y_pred, target_names=["HAM", "SPAM"]))

# =========================
# BAR CHART ACCURACY
# =========================
plt.figure(figsize=(6,4))
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.ylabel("Accuracy")
plt.title("Perbandingan Akurasi Model")
plt.ylim(0,1)
plt.show()

# =========================
# CONFUSION MATRIX (BERT)
# =========================
cm = confusion_matrix(y, results["BERT"])

plt.figure(figsize=(4,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["HAM","SPAM"],
    yticklabels=["HAM","SPAM"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - BERT")
plt.show()
