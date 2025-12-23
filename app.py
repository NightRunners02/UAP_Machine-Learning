import streamlit as st
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BertTokenizerFast,
    BertForSequenceClassification
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="UAP Text Classification",
    page_icon="📩",
    layout="centered"
)

# =========================
# HEADER + PROFILE
# =========================
col_title, col_profile = st.columns([8, 1])

with col_title:
    st.markdown(
        """
        <h2 style='margin-bottom:0'>📩 Spam SMS Classification</h2>
        <p style='color:gray'>UAP Pembelajaran Mesin – Text Classification</p>
        """,
        unsafe_allow_html=True
    )

with col_profile:
    with st.popover("👤"):
        st.markdown("### 👨‍🎓 Profil Mahasiswa")
        st.write("**Nama :** Khairy Zhafran H. Kastella")
        st.write("**NIM :** 202210370311439")
        st.write("**Mata Kuliah :** Pembelajaran Mesin")

st.markdown("---")

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_dataset():
    labels, texts = [], []
    with open("data/spam.csv", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split(";", 1)
            if len(parts) == 2 and parts[0] in ["ham", "spam"]:
                labels.append(0 if parts[0] == "ham" else 1)
                texts.append(parts[1])
    return pd.DataFrame({"text": texts, "label": labels})

df = load_dataset()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Pengaturan Model")

st.sidebar.metric("📊 Total Data SMS", len(df))
st.sidebar.metric("✅ HAM", int((df["label"] == 0).sum()))
st.sidebar.metric("🚨 SPAM", int((df["label"] == 1).sum()))

model_choice = st.sidebar.radio(
    "Pilih Model Klasifikasi",
    [
        "Neural Network Base (TF-IDF + RNN)",
        "Pretrained 1 – DistilBERT",
        "Pretrained 2 – BERT"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Petunjuk:**\n\n"
    "- Masukkan teks SMS\n"
    "- Pilih model\n"
    "- Klik **Prediksi**"
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    nn_model = joblib.load("model_nn_base/model_text_nn.pkl")
    vectorizer = joblib.load("model_nn_base/vectorizer.pkl")

    distil_tok = DistilBertTokenizerFast.from_pretrained("model_distilbert")
    distil_model = DistilBertForSequenceClassification.from_pretrained("model_distilbert")

    bert_tok = BertTokenizerFast.from_pretrained("model_bert")
    bert_model = BertForSequenceClassification.from_pretrained("model_bert")

    return nn_model, vectorizer, distil_tok, distil_model, bert_tok, bert_model


nn_model, vectorizer, distil_tok, distil_model, bert_tok, bert_model = load_models()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distil_model.to(device).eval()
bert_model.to(device).eval()

# =========================
# INPUT TEXT
# =========================
st.subheader("✍️ Input Teks SMS")

text_input = st.text_area(
    "Masukkan isi SMS:",
    placeholder="Contoh: Free entry in a weekly competition...",
    height=160
)

# =========================
# PREDICTION
# =========================
if st.button("🚀 Prediksi", use_container_width=True):
    if text_input.strip() == "":
        st.warning("⚠️ Masukkan teks SMS terlebih dahulu.")
    else:
        with st.spinner("🔍 Model sedang memproses..."):
            if model_choice == "Neural Network Base (TF-IDF + MLP)":
                X = vectorizer.transform([text_input])
                pred = nn_model.predict(X)[0]

            elif model_choice == "Pretrained 1 – DistilBERT":
                enc = distil_tok(text_input, return_tensors="pt", truncation=True, padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    pred = torch.argmax(distil_model(**enc).logits, dim=1).item()

            else:
                enc = bert_tok(text_input, return_tensors="pt", truncation=True, padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    pred = torch.argmax(bert_model(**enc).logits, dim=1).item()

        if pred == 1:
            st.error("🚨 **SPAM** – Pesan terindikasi spam")
        else:
            st.success("✅ **HAM** – Pesan normal (bukan spam)")
