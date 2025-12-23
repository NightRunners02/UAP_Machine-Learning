import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import Counter

# =========================
# LOAD DATA
# =========================
texts, labels = [], []

with open("data/spam.csv", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split(";", 1)
        if len(parts) == 2 and parts[0] in ["ham", "spam"]:
            texts.append(parts[1])
            labels.append(parts[0])

df = pd.DataFrame({"text": texts, "label": labels})

le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])  # ham=0, spam=1

# =========================
# TOKENIZATION (MANUAL)
# =========================
def tokenize(text):
    return text.lower().split()

tokenized = [tokenize(t) for t in df["text"]]

word_counts = Counter(w for sent in tokenized for w in sent)
vocab = {w: i+2 for i, (w, _) in enumerate(word_counts.most_common(5000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(sentence, max_len=50):
    encoded = [vocab.get(w, 1) for w in sentence][:max_len]
    return encoded + [0] * (max_len - len(encoded))

X = torch.tensor([encode(s) for s in tokenized])
y = torch.tensor(df["label"].values)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# DATASET
# =========================
class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SpamDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(SpamDataset(X_test, y_test), batch_size=32)

# =========================
# LSTM MODEL
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.sigmoid(self.fc(h[-1])).squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(len(vocab)).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAINING
# =========================
EPOCHS = 5
train_losses, train_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.float().to(device)
        optimizer.zero_grad()

        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += ((preds > 0.5) == yb).sum().item()
        total += yb.size(0)

    train_losses.append(total_loss / len(train_loader))
    train_accs.append(correct / total)

    print(f"Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Acc: {train_accs[-1]:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "model_lstm/model_lstm.pt")
torch.save(vocab, "model_lstm/vocab.pt")

# =========================
# PLOT LOSS & ACCURACY
# =========================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses, marker='o')
plt.title("Training Loss (LSTM)")

plt.subplot(1,2,2)
plt.plot(train_accs, marker='o', color='green')
plt.title("Training Accuracy (LSTM)")

plt.tight_layout()
plt.show()
