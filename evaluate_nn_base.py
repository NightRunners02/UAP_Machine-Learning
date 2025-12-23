import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. LOAD DATA (MANUAL PARSER)
# =========================
labels = []
texts = []

with open("data/spam.csv", encoding="latin-1") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(";", 1)
        if len(parts) == 2:
            label, text = parts
            label = label.strip()
            text = text.strip()

            if label in ["ham", "spam"]:
                labels.append(label)
                texts.append(text)

df = pd.DataFrame({
    "label": labels,
    "text": texts
})

df["label"] = df["label"].map({"ham": 0, "spam": 1})

# =========================
# 2. LOAD MODEL & VECTORIZER
# =========================
model = joblib.load("model_nn_base/model_text_nn.pkl")
vectorizer = joblib.load("model_nn_base/vectorizer.pkl")

# =========================
# 3. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

X_test_vec = vectorizer.transform(X_test)

# =========================
# 4. PREDIKSI
# =========================
y_pred = model.predict(X_test_vec)

# =========================
# 5. CLASSIFICATION REPORT
# =========================
print("\n=== CLASSIFICATION REPORT (NN BASE) ===")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

# =========================
# 6. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - NN Base")
plt.tight_layout()
plt.show()
