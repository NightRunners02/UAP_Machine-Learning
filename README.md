# 📩 Spam SMS Classification – UAP Pembelajaran Mesin

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg"> <br>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch">
  <img src="https://img.shields.io/badge/Framework-Transformers-yellow?logo=huggingface">
  <img src="https://img.shields.io/badge/Web-Streamlit-ff4b4b?logo=streamlit">
  <img src="https://img.shields.io/badge/Status-Completed-success">
</p>

---

Proyek ini merupakan **Ujian Akhir Praktikum (UAP)** mata kuliah **Pembelajaran Mesin**, yang berfokus pada **klasifikasi teks SMS spam dan ham** menggunakan **tiga pendekatan model**, yaitu:

1. **Neural Network Base (Non-Pretrained – LSTM)**
2. **Pretrained Model 1 – DistilBERT**
3. **Pretrained Model 2 – BERT**

Selain pelatihan dan evaluasi model, proyek ini juga dilengkapi dengan **dashboard interaktif menggunakan Streamlit** untuk melakukan inferensi dan analisis performa model.

---

## 👨‍🎓 Informasi Mahasiswa

- **Nama** : Khairy Zhafran H. Kastella  
- **NIM** : 202210370311439  
- **Mata Kuliah** : Pembelajaran Mesin  
- **Universitas** : Universitas Negeri Malang  

---

## 📂 Struktur Repository

```
📦 Praktikum-Text-UAP
│
├── data/
│   └── spam.csv
│
├── file_ipynb/
│  ├── Model_Bert_UAP_Machine_Learning.ipynb
│  ├── Model_Distilbert_UAP_Machine_Learning.ipynb
│  └── Model_LSTM_UAP_Machine_Learning.ipynb
│   
├── model_nn_base/
│   ├── model_lstm.pth
│   └── tokenizer.pkl
│
├── model_distilbert/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
│
├── model_bert/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
│
├── train_lstm.py
├── train_distilbert.py
├── train_bert.py
│
├── evaluate_lstm.py
├── evaluate_distilbert.py
├── evaluate_bert.py
│
├── app.py
├── requirements.txt
├── pyproject.toml
└── README.md

````

---

## 📊 Dataset

Proyek ini menggunakan dataset **SMS Spam Collection Dataset** dari Kaggle.

- **Sumber Dataset**  
  🔗 https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  

- **Jumlah Data** : 5.574 SMS  
- **Label** :
  - `ham` → SMS normal
  - `spam` → SMS spam

### Contoh Data

| Label | Teks |
|------|------|
| ham | Go until jurong point, crazy.. Available only in bugis |
| spam | Free entry in a weekly competition to win FA Cup tickets |

---

## ⚙️ Preprocessing Data

Langkah preprocessing yang dilakukan:

- Encoding label (`ham = 0`, `spam = 1`)
- Pembersihan teks dasar
- Tokenisasi:
  - **TF-IDF** untuk model NN Base
  - **Tokenizer Transformer** untuk DistilBERT dan BERT
- Padding & truncation (max length = 128)

---

## 🧠 Model yang Digunakan

### 1️⃣ Neural Network Base (Non-Pretrained – LSTM)

- Embedding Layer
- LSTM Layer
- Fully Connected Layer
- Sigmoid Output
- Dilatih dari awal tanpa pretrained weight

**Kelebihan**:
- Lebih ringan
- Mudah dipahami
- Cocok untuk baseline

---

### 2️⃣ Pretrained Model 1 – DistilBERT

- Model Transformer ringan
- Transfer learning dari `distilbert-base-uncased`
- Fine-tuning pada dataset SMS Spam

**Kelebihan**:
- Lebih cepat dari BERT
- Akurasi tinggi
- Lebih efisien untuk deployment

---

### 3️⃣ Pretrained Model 2 – BERT

- Model Transformer penuh
- Transfer learning dari `bert-base-uncased`
- Representasi konteks teks lebih kaya

**Kelebihan**:
- Akurasi tertinggi
- Pemahaman konteks lebih baik

---

## 📈 Evaluasi Model

Evaluasi dilakukan menggunakan:

- **Classification Report**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Confusion Matrix**
- **Grafik Loss dan Accuracy**

### Contoh Metrik Evaluasi

| Model | Accuracy |
|------|----------|
| LSTM (NN Base) | ~87% |
| DistilBERT | ~96% |
| BERT | ~97% |

📌 *Model pretrained menunjukkan performa yang lebih baik dibandingkan model non-pretrained.*

---

## 📊 Visualisasi

- Grafik **Training Loss vs Epoch**
- Grafik **Accuracy vs Epoch**
- Confusion Matrix ditampilkan di dashboard Streamlit

---

## 🖥️ Dashboard Streamlit

Dashboard menyediakan fitur:

- Pilih model (LSTM / DistilBERT / BERT)
- Input teks SMS
- Prediksi real-time
- Tampilan hasil dengan:
  - 🟥 Background merah untuk **SPAM**
  - 🟩 Background hijau untuk **HAM**
- Confusion Matrix & Classification Report
- Informasi jumlah total data

---

## ▶️ Cara Menjalankan Project (Local)

### 1️⃣ Clone Repository
```bash
git clone https://github.com/username/Praktikum-Text-UAP.git
cd Praktikum-Text-UAP
````

### 2️⃣ Install Dependency

```bash
pip install -r requirements.txt
```

atau menggunakan **PDM**:

```bash
pdm install
```

### 3️⃣ Jalankan Dashboard

```bash
streamlit run app.py
```

---

## 🌐 Live Demo (Optional)

🔗 *Belum tersedia / Opsional*

---

## 📝 Kesimpulan

* Model **pretrained (DistilBERT & BERT)** memberikan performa terbaik
* **LSTM** tetap layak sebagai baseline
* Streamlit mempermudah analisis dan presentasi model
* Transfer learning sangat efektif untuk klasifikasi teks

---

## 📌 Catatan

Proyek ini dibuat untuk keperluan **akademik** dan **pembelajaran**, bukan untuk penggunaan komersial.

---

⭐ Jangan lupa beri **star** jika repository ini membantu!

---

<details> 
  <summary>
  <h2> ⛓️‍💥 Misc / Lain-lain</h2> 
  </summary>

<p>
<div align="center">
<h3>
  🗣️ Powered By:
</h3>
<img src="https://awesome-svg.vercel.app/card/card_2?name=NightRunners02&summary=Newbie%20Developer&style=nameColor:rgba(223,255,0,1);summaryColor:rgba(57,255,20,1);backgroundColor:rgba(0,0,0,1);" />

---
<h3>
  🌠 Starred:
</h3>
  
[![Stargazers repo roster for @NightRunners02/](https://reporoster.com/stars/NightRunners02/UAP_Machine-Learning)](https://github.com/NightRunners02/UAP_Machine-Learning/stargazers)

---
<h3>
  🪐 Forked:
</h3>

[![Forkers repo roster for @NightRunners02/](https://reporoster.com/forks/NightRunners02/UAP_Machine-Learning)](https://github.com/NightRunners02/UAP_Machine-Learning/network/members)

---
<h3>
  💫 Star History:
</h3>

[![Star History Chart](https://api.star-history.com/svg?repos=NightRunners02/UAP_Machine-Learning&type=Date)](https://star-history.com/#UAP_Machine-Learning/Portofolio_Simple&Date)

</p>
</div>
</details>


