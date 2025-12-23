import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# =========================
# KONFIGURASI
# =========================
MAX_LEN = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD VOCAB & MODEL
# =========================
vocab = torch.load("model_lstm/vocab.pt")

from train_lstm import LSTMClassifier
model = LSTMClassifier(len(vocab))
model.load_state_dict(torch.load("model_lstm/model_lstm.pt", map_location=device))
model.to(device)
model.eval()

# =========================
# LOAD DATA
# =========================
texts, labels = [], []

with open("data/spam.csv", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split(";", 1)
        if len(parts) == 2 and parts[0] in ["ham", "spam"]:
            texts.append(parts[1])
            labels.append(0 if parts[0] == "ham" else 1)

# =========================
# ENCODING (FIXED LENGTH)
# =========================
def encode(sentence):
    tokens = sentence.lower().split()
    encoded = [vocab.get(w, 1) for w in tokens][:MAX_LEN]
    padded = encoded + [0] * (MAX_LEN - len(encoded))
    return padded

X = torch.tensor([encode(t) for t in texts]).to(device)
y = torch.tensor(labels)

# =========================
# PREDICTION
# =========================
with torch.no_grad():
    outputs = model(X)
    preds = (outputs > 0.5).int().cpu().numpy()

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n=== Classification Report (LSTM) ===")
print(classification_report(y, preds, target_names=["HAM", "SPAM"]))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y, preds)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["HAM", "SPAM"],
    yticklabels=["HAM", "SPAM"]
)
plt.title("Confusion Matrix - LSTM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
