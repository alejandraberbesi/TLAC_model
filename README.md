# 📚 Spanish BERT Book‑Category Classifier (2021)

A project that fine‑tunes **BETO** (Spanish BERT, `dccuchile/bert-base-spanish-wwm-uncased`) to classify Spanish‑language book descriptions into **11 academic categories**.

---

## 🗂  Data Pipeline

| Stage | Details |
|-------|---------|
| **Source** | Google BigQuery table `tlac-vision.book_backend.train_categories` |
| **Cleaning** | • Dedup titles<br>• Language check with `langdetect` (keep `es`) |
| **Class balance** | 2 477 rows → class counts 102–277 |
| **Splits** | Stratified 70 % / 15 % / 15 % (train / val / test) |

---

## 🔢 Pre‑processing

| Step | Library | Notes |
|------|---------|-------|
| Tokenisation | `BertTokenizerFast` | max_len = 250, padding = `max_length` |
| Label Mapping | Dict → 0‑10 | 11 classes |
| Class Weights | `compute_class_weight` | for imbalance, passed to `NLLLoss` |

---

## 🧠 Model

- **Base**: Frozen BETO encoder (12 layers, 768 dim)
- **Head**: `Linear(768→512) + ReLU + Dropout + Linear(512→11) + LogSoftmax`
- **Loss**: Weighted `NLLLoss`
- **Optimiser**: `AdamW`, lr = 1 e‑5
- **Training**: 50 epochs, best weights saved on lowest val‑loss  
  *(Transformers v 3.3, PyTorch 1.7 GPU)*

---

## 📈 Results (Test Set)

| Metric | Score |
|--------|-------|
| Accuracy | **49 %** |
| Macro F1 | 0.47 |

*Per‑class F1 ranges 0.17 – 0.65 (see Colab output).*

---

## 🛠  2021 Tech Stack

- Python 3.6 + PyTorch 1.7 (GPU)
- `transformers` 3.3.1  
- `langdetect` 1.0.8
- Google Colab + BigQuery
