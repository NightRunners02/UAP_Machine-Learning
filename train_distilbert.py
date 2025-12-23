import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# =========================
# 1. LOAD DATA
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
print(f"Total data: {len(df)}")

# =========================
# 2. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================
# 3. TOKENIZER
# =========================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

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

train_dataset = SpamDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# =========================
# 4. MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# =========================
# 5. TRAINING
# =========================
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# =========================
# 6. SAVE MODEL (PASTI ADA)
# =========================
SAVE_DIR = "model_distilbert"
os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✅ MODEL DISTILBERT BERHASIL DISIMPAN DI: {SAVE_DIR}/")
