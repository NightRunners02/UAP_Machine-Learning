import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# =========================
# 1. LOAD DATA MANUAL
# =========================
labels = []
texts = []

with open("data/spam.csv", encoding="latin-1") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # split pakai ; pertama saja
        parts = line.split(";", 1)

        if len(parts) == 2:
            label = parts[0].strip()
            text = parts[1].strip()

            if label in ["ham", "spam"]:
                labels.append(label)
                texts.append(text)

df = pd.DataFrame({
    "label": labels,
    "text": texts
})

print(df.head())
print(df.info())

# =========================
# 2. ENCODE LABEL
# =========================
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# =========================
# 3. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================
# 4. TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 5. NN BASE (NON-PRETRAINED)
# =========================
model = MLPClassifier(
    hidden_layer_sizes=(128,),
    max_iter=20,
    random_state=42
)

model.fit(X_train_vec, y_train)

# =========================
# 6. EVALUASI
# =========================
y_pred = model.predict(X_test_vec)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# =========================
# 7. SIMPAN MODEL
# =========================
joblib.dump(model, "model_nn_base/model_text_nn.pkl")
joblib.dump(vectorizer, "model_nn_base/vectorizer.pkl")

print("✅ Model & vectorizer berhasil disimpan")
