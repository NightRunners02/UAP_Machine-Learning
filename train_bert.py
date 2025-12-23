import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# =========================
# 1. LOAD DATA (AMAN CSV SEMICOLON)
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

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

print("Jumlah data:", len(df))

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
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

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
test_dataset = SpamDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# =========================
# 4. MODEL BERT
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# =========================
# 5. TRAINING
# =========================
EPOCHS = 3
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# =========================
# 6. SAVE MODEL
# =========================
model.save_pretrained("model_bert")
tokenizer.save_pretrained("model_bert")

print("✅ BERT model berhasil disimpan.")
