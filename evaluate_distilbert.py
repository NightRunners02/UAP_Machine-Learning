import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# =========================
# LOAD DATA
# =========================
labels, texts = [], []

with open("data/spam.csv", encoding="latin-1") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(";", 1)
        if len(parts) == 2:
            label, text = parts
            if label in ["ham", "spam"]:
                labels.append(0 if label == "ham" else 1)
                texts.append(text)

df = pd.DataFrame({"text": texts, "label": labels})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================
# TOKENIZER & MODEL
# =========================
tokenizer = DistilBertTokenizerFast.from_pretrained("model_distilbert")
model = DistilBertForSequenceClassification.from_pretrained("model_distilbert")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels.values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

test_dataset = SpamDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8)

# =========================
# PREDICTION
# =========================
y_true, y_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n=== CLASSIFICATION REPORT (DISTILBERT) ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Ham", "Spam"]
))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - DistilBERT")
plt.tight_layout()
plt.show()
